# samplers/slices_sampler_2d.py
"""
## Overview

Generates a comprehensive 2D slice sampling strategy across all parameter pairs.
Creates a grid of samples for each pair of parameters while fixing all other parameters
at their midpoint values. Generates n_pairs × res² total samples where n_pairs is the
number of unique parameter pairs and res is the resolution per parameter.

Produces visualization of slices through the design space including:
- 2D contour plots with static 3D surface plots
- Interactive 3D Plotly visualizations
- Full parameter space grid with diagonal 1D slices

---
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from enchanted_surrogates.samplers.base_sampler import Sampler


class SlicesSampler2D(Sampler):
    """
    2D slice-based sampling strategy for exploring parameter space.

    This sampler generates samples by creating a 2D slice for each pair of parameters,
    with all other parameters fixed at their midpoint values. It provides both sampling
    and visualization capabilities including 2D contours, 3D surfaces, and interactive plots.

    ## Configuration

    To use the 2D slices sampler, specify it in the configuration file as follows:

    ```yaml
    sampler:
        type: SlicesSampler2D
        parameters: ['x', 'y', 'z']
        bounds: [[1, 10], [0, 1], [5, 15]]
        res: 20
        budget: 10000
        fixed: {'x': 5.5, 'y': 0.5, 'z': 10.0}  # optional
        dot_alpha: 0.7
        elev_3d: 70
        azim_3d: 200
    ```

    In this configuration:
    - parameters x, y, z are sampled
    - each parameter pair is sampled at 20 × 20 = 400 points
    - total 3 pairs × 400 = 1200 samples (if under budget)
    - fixed values define reference points for parameters not in the current slice

    ## Attributes

        parameters (list of str): The names of the parameters.
        bounds (list of tuple of float): The bounds of each parameter.
        base_run_dir (str): Base directory for saving plots and datasets.
        res (int): Resolution per parameter in each 2D slice (default 50).
        budget (int): Maximum total number of samples (default 100000).
        fixed (dict): Fixed parameter values for non-varied dimensions.
        batch_number (int): Counter tracking sampling batches.
        dot_alpha (float): Alpha transparency for plotted sample points (0-1).
        elev_3d (float): Elevation angle for 3D surface plots in degrees.
        azim_3d (float): Azimuth angle for 3D surface plots in degrees.
        output_col (str): Name of output column in dataset CSV.
        MARKERS (list): Matplotlib marker styles for plot points.
        COLORS (list): Color names for distinguishing multiple datasets.
        LINESTYLES (list): Line styles for distinguishing multiple curves.
        PLOTLY_MARKERS (list): Plotly marker symbols for interactive plots.

    ## Assumptions and Notes

      - Total samples = (d choose 2) × res², where d is the number of parameters.
      - Fixed parameters default to the midpoint of their bounds.
      - The sampler requires a dataset CSV with results for visualization.
      - Supports plotting multiple datasets simultaneously with custom labels.
      - Plotly HTML plots enable interactive 3D visualization in web browsers.
      - Grid resolution grows quadratically; use moderate res values (20-50).

    ---
    """

    MARKERS = ['o', 'X', '^', '+', 'D', 'v', 'P', '*', 's']
    COLORS = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown']
    LINESTYLES = ['-', '--', '-.', ':']

    PLOTLY_MARKERS = [
        "circle",        # dot
        "x",             # cross
        "triangle-up",   # ^
        "cross",         # +
        "diamond",       # D
        "triangle-down", # v
        "cross-thin",    # P (closest)
        "star",          # *
        "square",        # s
    ]

    def __init__(
        self,
        parameters,
        bounds,
        base_run_dir=None,
        res=50,
        fixed=None,
        budget=100000,
        type='SlicesSampler2D',
        dot_alpha=0.7,
        elev_3d=70,
        azim_3d=200,
        *args,
        **kwargs
    ):
        """
        Initialize the 2D slices sampler.

        Args:
            parameters (list of str): Names of the parameters to sample.
            bounds (list of tuple of float): Lower and upper bounds for each parameter.
            base_run_dir (str, optional): Base directory for output files. Defaults to None.
            res (int, optional): Resolution per parameter per slice. Defaults to 50.
            fixed (dict, optional): Fixed values for parameters. Defaults to midpoints.
            budget (int, optional): Maximum total samples allowed. Defaults to 100000.
            type (str, optional): Sampler type name. Defaults to 'SlicesSampler2D'.
            dot_alpha (float, optional): Alpha transparency for scatter plots (0-1). Defaults to 0.7.
            elev_3d (float, optional): Elevation angle for 3D plots in degrees. Defaults to 70.
            azim_3d (float, optional): Azimuth angle for 3D plots in degrees. Defaults to 200.
            **kwargs: Additional keyword arguments including 'output_col' for CSV column name.
        """
        super().__init__()
        self.parameters = parameters
        self.bounds = bounds
        self.base_run_dir = base_run_dir
        self.res = res
        self.budget = budget
        self.fixed = fixed or {p: 0.5 * (b[0] + b[1]) for p, b in zip(parameters, bounds)}
        self.batch_number = 0

        self.elev_3d = elev_3d
        self.azim_3d = azim_3d
        self.dot_alpha = dot_alpha

        self.output_col = kwargs.get('output_col', None)

    def _normalize_dict_inputs(self, dots_x, predictor):
        """
        Normalize dots and predictor inputs to dict format.

        Converts single arrays or None to dictionaries with default keys.
        This allows flexible input handling for multiple datasets/models.

        Args:
            dots_x: Sample points as array, dict, or None.
            predictor: Predictor function, dict of functions, or None.

        Returns:
            tuple: (dots_dict, pred_dict) both as dictionaries with string keys.
        """
        # Normalize dots_x
        if dots_x is None:
            dots_dict = {}
        elif isinstance(dots_x, dict):
            dots_dict = dots_x
        else:
            dots_dict = {"default": dots_x}

        # Normalize predictor
        if predictor is None:
            pred_dict = {}
        elif isinstance(predictor, dict):
            pred_dict = predictor
        else:
            pred_dict = {"default": predictor}

        return dots_dict, pred_dict

    def get_next_samples(self):
        """
        Get the next batch of samples in the sampling sequence.

        On first call (batch_number=0), returns grid samples for the first 2D slice.
        On subsequent calls, generates plots from the collected dataset and returns None.

        Returns:
            list of dict: Parameter combinations for the current batch, or None.
        """
        if self.batch_number > 0:
            self.make_plots()
            return None
        elif self.batch_number == 0:
            return self.get_samples()

    def get_samples(self):
        """
        Generate all 2D slice samples.

        Creates samples for all unique pairs of parameters by generating a
        res × res grid for each pair while fixing all other parameters.

        Returns:
            list of dict: Each dict maps parameter names to values.

        Raises:
            RuntimeError: If total samples exceed the budget.
        """
        d = len(self.parameters)
        n_pairs = d * (d - 1) // 2
        total_samples = n_pairs * (self.res ** 2)
        print(f'[SlicesSampler2D] DIM {d}, RES {self.res}, N SAMPLES {total_samples}')

        if total_samples > self.budget:
            raise RuntimeError(
                f"Requested {total_samples} samples exceeds budget={self.budget}. "
                f"Reduce resolution or number of parameters."
            )
        self.budget = total_samples
        samples = []

        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(i + 1, d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                for u in range(self.res):
                    for v in range(self.res):
                        row = {}
                        for k, (param, (a, b)) in enumerate(zip(self.parameters, self.bounds)):
                            if k == i:
                                row[param] = Xi[u, v]
                            elif k == j:
                                row[param] = Yi[u, v]
                            else:
                                row[param] = self.fixed[param]
                        samples.append(row)

        self.batch_number += 1
        return samples

    def get_samples_array(self):
        """
        Get samples as a numpy array.

        Calls get_samples() and converts output to (n_samples, n_parameters) array.

        Returns:
            np.ndarray: Shape (n_samples, n_parameters) array of parameter values.
        """
        samples = self.get_samples()
        df = pd.DataFrame(samples)
        return df[self.parameters].to_numpy()

    def make_plots(self, dots_x=None, predictor=None, save_dir=None, name='', white=None):
        """
        Generate all available plot types.

        Creates 2D/3D static slices, full parameter grid, and interactive 3D plots.

        Args:
            dots_x (array or dict, optional): Sample points to overlay on plots.
            predictor (callable or dict, optional): Predictor functions for 1D slices.
            save_dir (str, optional): Directory for saving plots. Uses base_run_dir if None.
            name (str, optional): Prefix for plot filenames. Defaults to ''.
            white (float, optional): Data value to anchor the colour white at. When
                None (default), a standard sequential colormap spanning the data
                range is used. Set e.g. ``white=0`` to make 0 render as white.
        """
        self.plot_slices_from_dataset(dots_x=dots_x, save_dir=save_dir, name=name, white=white)
        self.plot_full_grid(dots_x=dots_x, predictor=predictor, save_dir=save_dir, name=name, white=white)
        self.plot_interactive_3d_slices(dots_x=dots_x, save_dir=save_dir, name=name, white=white)

    def get_output_col(self, df=None, csv_path=None):
        """
        Determine the output column name from dataset.

        Searches for a column containing 'output' in its name. If output_col is
        explicitly set, returns that. Otherwise, looks for it in provided CSV or DataFrame.

        Args:
            df (pd.DataFrame, optional): DataFrame to search for output column.
            csv_path (str, optional): Path to CSV file to read and search.

        Returns:
            str: Name of the output column.

        Raises:
            RuntimeError: If zero or multiple output columns found.
        """
        def output_from_df(df):
            output_col = [col for col in df.columns if 'output' in col]
            if len(output_col) != 1:
                raise RuntimeError(f'Exactly one output column required but found: {output_col}')
            return output_col[0]

        if self.output_col:
            return self.output_col
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
            return output_from_df(df)
        elif df is not None:
            return output_from_df(df)

    # -------------------------------------------------------------------------
    # INTERACTIVE 3D PLOTS
    # -------------------------------------------------------------------------

    def plot_interactive_3d_slices(self, df=None, dots_x=None, save_dir=None, name="", white=None):
        """
        Create interactive 3D surface plots using Plotly.

        Generates one HTML file per parameter pair showing the 2D slice surface
        with sample points overlaid at the floor level. Publication-ready with
        A4-width figure sizing and large fonts.

        Args:
            df (pd.DataFrame, optional): Dataset with results. Reads from CSV if None.
            dots_x (array or dict, optional): Sample points to overlay.
            save_dir (str, optional): Output directory. Uses base_run_dir if None.
            name (str, optional): Prefix for output filenames.
            white (float, optional): Data value to anchor white at (e.g. 0). None
                (default) uses the standard Viridis colorscale.
        """
        import plotly.graph_objects as go

        dots_dict, _ = self._normalize_dict_inputs(dots_x, predictor=None)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        ycol = self.get_output_col(df=df)

        # Colour scale shared across every pair. With white anchoring the matplotlib
        # colormap is sampled into a Plotly colorscale; otherwise plain Viridis.
        ymin, ymax = df[ycol].min(), df[ycol].max()
        cmap_obj, norm = self._color_mapping(ymin, ymax, white=white)
        colorscale = "Viridis" if white is None else self._plotly_colorscale(cmap_obj)

        d = len(self.parameters)

        for i in range(d):
            p1 = self.parameters[i]
            x_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)

            for j in range(i + 1, d):
                p2 = self.parameters[j]
                y_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)

                Xi, Yi = np.meshgrid(x_lin, y_lin)
                Z = np.zeros_like(Xi)

                for u in range(self.res):
                    for v in range(self.res):
                        mask = (
                            (np.isclose(df[p1], Xi[u, v], atol=1e-6)) &
                            (np.isclose(df[p2], Yi[u, v], atol=1e-6))
                        )
                        vals = df.loc[mask, ycol].values
                        Z[u, v] = vals[0] if len(vals) else np.nan

                fig = go.Figure()

                Z = self._zero_to_nan(Z)
                fig.add_trace(go.Surface(
                    x=Xi, y=Yi, z=Z,
                    colorscale=colorscale,
                    cmin=norm.vmin,
                    cmax=norm.vmax,
                    opacity=0.85,
                    name="dataset surface",
                    colorbar=dict(
                        thickness=20,
                        len=0.7,
                        tickfont=dict(size=12),
                        title=dict(text=ycol, font=dict(size=13)),
                    ),
                ))

                z_floor = np.nanmin(Z)

                # Multi-dots support
                for idx, (label, arr) in enumerate(dots_dict.items()):
                    arr = np.asarray(arr)

                    fig.add_trace(go.Scatter3d(
                        x=arr.T[i],
                        y=arr.T[j],
                        z=np.full(arr.shape[0], z_floor),
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=self.COLORS[idx % len(self.COLORS)],
                            symbol=self.PLOTLY_MARKERS[idx % len(self.PLOTLY_MARKERS)],
                            opacity=self.dot_alpha,
                            line=dict(width=1, color='white'),
                        ),
                        name=label,
                    ))

                fig.update_layout(
                    title=dict(
                        text=f"3D Slice: {p1} vs {p2} vs {ycol}",
                        font=dict(size=16, color='black'),
                        x=0.5,
                        xanchor='center',
                    ),
                    scene=dict(
                        xaxis=dict(
                            title=dict(text=p1, font=dict(size=14)),
                            tickfont=dict(size=12),
                        ),
                        yaxis=dict(
                            title=dict(text=p2, font=dict(size=14)),
                            tickfont=dict(size=12),
                        ),
                        zaxis=dict(
                            title=dict(text=ycol, font=dict(size=14)),
                            tickfont=dict(size=12),
                        ),
                    ),
                    width=1000,
                    height=800,
                    font=dict(size=12),
                    legend=dict(
                        font=dict(size=12),
                        x=0.02,
                        y=0.98,
                    ),
                )

                out_name = f"{name}interactive_3d_{p1}_{p2}.html"
                fig.write_html(os.path.join(save_dir, out_name))
                print(f"[Saved] {out_name}")

    # -------------------------------------------------------------------------
    # 2D + 3D STATIC SLICE PLOTS
    # -------------------------------------------------------------------------

    def plot_slices_from_dataset(
        self,
        cmap=None,
        surface_alpha=0.9,
        dataset_path=None,
        df=None,
        dots_x=None,
        save_dir=None,
        name='',
        include_3d=True,
        white=None
    ):
        """
        Create 2D contour and optional 3D surface plots for parameter pairs.

        Generates one figure per parameter pair with left-side contour and right-side
        3D surface when include_3d is True. Sample points are overlaid. Publication-ready
        formatting for full A4 width.

        Args:
            cmap (str or Colormap, optional): Colormap override. Defaults to
                'viridis' when None.
            surface_alpha (float, optional): Opacity for 3D surface (0-1). Defaults to 0.9.
            dataset_path (str, optional): Path to CSV file. Reads from base_run_dir if None.
            df (pd.DataFrame, optional): Pre-loaded dataset. Defaults to None.
            dots_x (array or dict, optional): Sample points to overlay.
            save_dir (str, optional): Output directory. Defaults to base_run_dir.
            name (str, optional): Prefix for filenames. Defaults to ''.
            include_3d (bool, optional): Whether to include 3D surface panel. Defaults to True.
            white (float, optional): Data value to anchor white at (e.g. 0). None
                (default) uses the standard colormap with no white anchoring.
        """
        dots_dict, _ = self._normalize_dict_inputs(dots_x, predictor=None)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        ycol = self.get_output_col(df=df)
        ymin, ymax = df[ycol].min(), df[ycol].max()
        cmap, norm = self._color_mapping(ymin, ymax, cmap, white)

        d = len(self.parameters)

        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(i + 1, d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                Z = np.zeros_like(Xi)

                for u in range(self.res):
                    for v in range(self.res):
                        mask = (
                            (np.isclose(df[self.parameters[i]], Xi[u, v], atol=1e-6)) &
                            (np.isclose(df[self.parameters[j]], Yi[u, v], atol=1e-6))
                        )
                        vals = df.loc[mask, ycol].values
                        Z[u, v] = vals[0] if len(vals) else np.nan

                import matplotlib.gridspec as gridspec
                nc = 3 if include_3d else 2
                fig = plt.figure(figsize=(8.5 * (nc - 1) / 2, 6))
                wr = [1, 1, 0.06] if include_3d else [1, 0.06]
                gs = gridspec.GridSpec(1, nc, width_ratios=wr, figure=fig, hspace=0.3, wspace=0.3)

                ax1 = fig.add_subplot(gs[0])
                Z = self._zero_to_nan(Z)
                cs = ax1.contourf(Xi, Yi, Z, cmap=cmap, norm=norm, levels=20)
                # Square box (display only) without tying data units together, so
                # parameters with very different ranges stay legible.
                ax1.set_box_aspect(1)
                ax1.set_xlabel(self.parameters[i], fontsize=13, fontweight='bold')
                ax1.set_ylabel(self.parameters[j], fontsize=13, fontweight='bold')
                ax1.tick_params(labelsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.set_title(f"{self.parameters[i]} vs {self.parameters[j]}", fontsize=12, fontweight='bold')

                # Multi-dots support (2D)
                legend_handles = []
                for idx, (label, arr) in enumerate(dots_dict.items()):
                    arr = np.asarray(arr)
                    h = ax1.scatter(arr.T[i], arr.T[j],
                                    marker=self.MARKERS[idx % len(self.MARKERS)],
                                    color=self.COLORS[idx % len(self.COLORS)],
                                    s=80,
                                    edgecolors='black',
                                    linewidths=1,
                                    label=label, alpha=self.dot_alpha, zorder=10)
                    legend_handles.append(h)

                if include_3d:
                    ax3d = fig.add_subplot(gs[1], projection='3d')
                    Z = self._zero_to_nan(Z)

                    ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha,
                                      norm=norm, rstride=2, cstride=2)

                    ax3d.set_xlabel(self.parameters[i], fontsize=11, fontweight='bold')
                    ax3d.set_ylabel(self.parameters[j], fontsize=11, fontweight='bold')
                    ax3d.set_zlabel(ycol, fontsize=11, fontweight='bold')
                    ax3d.tick_params(labelsize=10)

                    ax3d.view_init(elev=self.elev_3d, azim=self.azim_3d)

                    z_floor = float(np.nanmin(Z))

                    # Multi-dots support (3D)
                    for idx, (label, arr) in enumerate(dots_dict.items()):
                        arr = np.asarray(arr)
                        ax3d.scatter(arr.T[i], arr.T[j], zs=z_floor, zdir='z',
                                     s=60,
                                     marker=self.MARKERS[idx % len(self.MARKERS)],
                                     color=self.COLORS[idx % len(self.COLORS)],
                                     edgecolors='black',
                                     linewidths=0.5,
                                     label=label, alpha=self.dot_alpha, zorder=10)

                if legend_handles:
                    ax1.legend(handles=legend_handles, fontsize=11, loc='best', framealpha=0.95)

                cb_gs = gs[2] if include_3d else gs[1]
                cax = fig.add_subplot(cb_gs)
                cbar = fig.colorbar(cs, cax=cax)
                cbar.set_label(ycol, fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)

                fig.suptitle(f"{self.parameters[i]} vs {self.parameters[j]} vs {ycol}",
                             fontsize=14, fontweight='bold', y=0.98)
                fig.subplots_adjust(left=0.1, right=0.92, top=0.93, bottom=0.1, wspace=0.3, hspace=0.3)
                fig.savefig(os.path.join(save_dir, name + f"slices_{self.parameters[i]}_{self.parameters[j]}_{ycol}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

    # -------------------------------------------------------------------------
    # FULL GRID PLOT
    # -------------------------------------------------------------------------

    def plot_full_grid(
        self,
        cmap=None,
        surface_alpha=0.9,
        dataset_path=None,
        df=None,
        name='',
        dots_x=None,
        predictor=None,
        save_dir=None,
        white=None
    ):
        """
        Create a d×d grid of plots showing all parameter relationships.

        Layout:
        - Diagonal: 1D slices with optional predictor curves
        - Upper triangle: 2D contour plots
        - Lower triangle: 3D surface plots

        Publication-ready formatting with large fonts and improved visibility.

        Args:
            cmap (str or Colormap, optional): Colormap override. Defaults to
                'viridis' when None.
            surface_alpha (float, optional): 3D surface opacity (0-1). Defaults to 0.9.
            dataset_path (str, optional): Path to CSV file.
            df (pd.DataFrame, optional): Pre-loaded dataset.
            name (str, optional): Filename prefix. Defaults to ''.
            dots_x (array or dict, optional): Sample points to overlay.
            predictor (callable or dict, optional): Predictor functions for 1D slices.
            save_dir (str, optional): Output directory. Defaults to base_run_dir.
            white (float, optional): Data value to anchor white at (e.g. 0). None
                (default) uses the standard colormap with no white anchoring.
        """
        dots_dict, pred_dict = self._normalize_dict_inputs(dots_x, predictor)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        ycol = self.get_output_col(df=df)
        ymin, ymax = df[ycol].min(), df[ycol].max()
        cmap, norm = self._color_mapping(ymin, ymax, cmap, white)

        d = len(self.parameters)
        base_size = 3.5
        fig, axes = plt.subplots(d, d, figsize=(max(8.5, base_size * d), max(8.5, base_size * d)))

        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                ax = axes[i, j]

                # ------------------------------------------------------------------
                # DIAGONAL: 1D SLICES WITH MULTI-PREDICTOR SUPPORT
                # ------------------------------------------------------------------
                if i == j:
                    p = self.parameters[i]

                    fixed_assignments = {}
                    for k, q in enumerate(self.parameters):
                        if k == i:
                            continue
                        mid = 0.5 * (self.bounds[k][0] + self.bounds[k][1])
                        vals_q = df[q].unique()
                        fixed_assignments[q] = float(vals_q[np.argmin(np.abs(vals_q - mid))]) if vals_q.size else mid

                    yvals = []
                    for xv in xi_lin:
                        vals_p = df[p].unique()
                        xv_closest = float(vals_p[np.argmin(np.abs(vals_p - xv))]) if vals_p.size else xv
                        diffs = np.abs(df[p] - xv_closest)
                        for q, val in fixed_assignments.items():
                            diffs += np.abs(df[q] - val)
                        idx = np.argmin(diffs.values)
                        yvals.append(float(df.iloc[idx][ycol]))

                    yvals = np.array(yvals)
                    yvals[yvals == 0.0] = np.nan

                    ax.plot(xi_lin, yvals, '-', color='tab:blue', linewidth=2.5, label="dataset", zorder=5)

                    # Multi-predictor support
                    for idx, (label, pred_fn) in enumerate(pred_dict.items()):
                        to_predict = np.array([
                            [fixed_assignments[q] if q in fixed_assignments else xi for q in self.parameters]
                            for xi in xi_lin
                        ])
                        y_pred = pred_fn(to_predict)

                        ax.plot(
                            xi_lin, y_pred,
                            linestyle=self.LINESTYLES[idx % len(self.LINESTYLES)],
                            color=self.COLORS[idx % len(self.COLORS)],
                            linewidth=2.5,
                            label=label,
                            zorder=5
                        )

                    ax.set_xlabel(p, fontsize=11, fontweight='bold')
                    if i == 0:
                        ax.set_ylabel(ycol, fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.tick_params(labelsize=9)
                    ax.legend(fontsize=8.5, loc='best')

                # ------------------------------------------------------------------
                # UPPER TRIANGLE: 2D CONTOUR
                # ------------------------------------------------------------------
                elif i < j:
                    self._draw_contour(ax, i, j, df, ycol, cmap, norm,
                                       dots_dict, labelsize=9, fontsize=11)

                # ------------------------------------------------------------------
                # LOWER TRIANGLE: 3D SURFACE
                # ------------------------------------------------------------------
                else:
                    ax.remove()
                    ax3d = fig.add_subplot(d, d, i * d + j + 1, projection='3d')

                    Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                    Z = np.zeros_like(Xi)
                    for u in range(self.res):
                        for v in range(self.res):
                            mask = (
                                (np.isclose(df[self.parameters[i]], Xi[u, v], atol=1e-6)) &
                                (np.isclose(df[self.parameters[j]], Yi[u, v], atol=1e-6))
                            )
                            vals = df.loc[mask, ycol].values
                            Z[u, v] = vals[0] if len(vals) else np.nan

                    Z = self._zero_to_nan(Z)
                    ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha,
                                      norm=norm, rstride=2, cstride=2)

                    ax3d.view_init(elev=self.elev_3d, azim=self.azim_3d)

                    z_floor = float(np.nanmin(Z))

                    # Multi-dots support
                    for idx, (label, arr) in enumerate(dots_dict.items()):
                        arr = np.asarray(arr)
                        ax3d.scatter(arr.T[i], arr.T[j], zs=z_floor, zdir='z',
                                     s=40,
                                     marker=self.MARKERS[idx % len(self.MARKERS)],
                                     color=self.COLORS[idx % len(self.COLORS)],
                                     edgecolors='black',
                                     linewidths=0.5,
                                     label=label, alpha=self.dot_alpha, zorder=10)

                    ax3d.set_xlabel(self.parameters[i], fontsize=10, fontweight='bold')
                    ax3d.set_ylabel(self.parameters[j], fontsize=10, fontweight='bold')
                    ax3d.tick_params(labelsize=8)

        # Leave room on the right for a single shared colorbar that spans the
        # vertical extent of the grid, then place it in a dedicated axis.
        fig.subplots_adjust(left=0.06, right=0.88, top=0.95, bottom=0.07,
                            wspace=0.45, hspace=0.45)

        import matplotlib.cm as cm
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar_ax = fig.add_axes([0.905, 0.20, 0.018, 0.60])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(ycol, fontsize=12, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=10)
        cbar.outline.set_linewidth(0.8)

        fig.suptitle(f"Parameter Space Exploration: {ycol}", fontsize=16, fontweight='bold', y=0.99)
        fig.savefig(os.path.join(save_dir, name + f"slices_full_grid_{ycol}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Each 2D contour panel additionally gets its own standalone figure.
        self.plot_contours(df, ycol, cmap, norm, dots_dict, save_dir, name)

    # -------------------------------------------------------------------------
    # COLOUR MAPPING (0 -> white)
    # -------------------------------------------------------------------------

    @staticmethod
    def _white_zero_cmap():
        """
        Sequential colormap that is pure white at its lowest value.

        White anchors the bottom of the scale so a value of 0 reads as white,
        with colour saturating as magnitude grows (a ColorBrewer "Blues"
        progression). A fresh copy is returned each call so per-figure tweaks
        (e.g. the NaN/"bad" colour) never leak between figures.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: White-at-zero colormap.
        """
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#ffffff', '#deebf7', '#c6dbef', '#9ecae1',
                  '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        cmap = LinearSegmentedColormap.from_list('white_zero', colors)
        cmap.set_bad('white')  # missing / NaN cells render white too
        return cmap

    def _color_mapping(self, ymin, ymax, cmap=None, white=None):
        """
        Build a (Colormap, Normalize) pair for the colour scale.

        Default (``white=None``): a standard sequential colormap ('viridis' unless
        overridden) spanning the data range.

        White-anchored (``white`` is a number): the given value maps to white. When
        the data straddles the anchor a diverging map is centred on it; otherwise a
        sequential white-at-anchor map is used with the colour floor pinned to it.

        Args:
            ymin, ymax (float): Data range.
            cmap (str or Colormap, optional): Explicit colormap override.
            white (float, optional): Data value to anchor white at. None disables
                white anchoring (default colormap behaviour).

        Returns:
            tuple: (matplotlib.colors.Colormap, matplotlib.colors.Normalize).
        """
        import matplotlib.colors as mcolors

        if white is None:
            # Default: standard sequential map spanning the data range.
            cmap_obj = plt.get_cmap(cmap if cmap is not None else 'viridis').copy()
            norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
        elif ymin < white < ymax:
            # Data straddles the anchor: diverging map, white pinned at `white`.
            cmap_obj = (plt.get_cmap(cmap).copy() if cmap is not None
                        else plt.get_cmap('RdBu_r').copy())
            norm = mcolors.TwoSlopeNorm(vmin=ymin, vcenter=white, vmax=ymax)
        else:
            # One-sided data: sequential white-at-anchor map, colour floor at `white`.
            cmap_obj = (plt.get_cmap(cmap).copy() if cmap is not None
                        else self._white_zero_cmap())
            norm = mcolors.Normalize(vmin=min(white, ymin), vmax=max(white, ymax))

        cmap_obj.set_bad('white')
        return cmap_obj, norm

    def _plotly_colorscale(self, cmap, n=17):
        """
        Sample a matplotlib Colormap into a Plotly colorscale.

        Args:
            cmap (matplotlib.colors.Colormap): Source colormap.
            n (int): Number of evenly spaced samples.

        Returns:
            list: [[position, "#rrggbb"], ...] suitable for Plotly traces.
        """
        import matplotlib.colors as mcolors
        return [[k / (n - 1), mcolors.to_hex(cmap(k / (n - 1)))] for k in range(n)]

    def _draw_contour(self, ax, i, j, df, ycol, cmap, norm, dots_dict,
                      labelsize=9, fontsize=11):
        """
        Render a single 2D filled-contour panel for a parameter pair.

        Pivots the dataset on parameters i and j, draws the contour, formats the
        axis, and overlays any sample points. Used both inside the full grid and
        in the standalone contour figures so styling stays consistent.

        Args:
            ax (matplotlib.axes.Axes): Target axis.
            i (int): Index of the parameter on the x-axis.
            j (int): Index of the parameter on the y-axis.
            df (pd.DataFrame): Dataset with results.
            ycol (str): Output column name.
            cmap (matplotlib.colors.Colormap): Colormap (0 -> white).
            norm (matplotlib.colors.Normalize): Shared colour normalisation.
            dots_dict (dict): Mapping of label -> sample-point array to overlay.
            labelsize (int): Tick label font size.
            fontsize (int): Axis label font size.

        Returns:
            matplotlib.contour.QuadContourSet: The drawn contour set.
        """
        pivot = df.pivot_table(
            index=self.parameters[i],
            columns=self.parameters[j],
            values=ycol,
            aggfunc='mean'
        )
        Xi_vals = pivot.index.to_numpy()
        Yi_vals = pivot.columns.to_numpy()
        Xi, Yi = np.meshgrid(Xi_vals, Yi_vals, indexing='ij')
        Z = self._zero_to_nan(pivot.to_numpy())

        cs = ax.contourf(Xi, Yi, Z, cmap=cmap, norm=norm, levels=15)
        ax.set_box_aspect(1)
        ax.set_xlabel(self.parameters[i], fontsize=fontsize, fontweight='bold')
        ax.set_ylabel(self.parameters[j], fontsize=fontsize, fontweight='bold')
        ax.tick_params(labelsize=labelsize, direction='out', length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        for idx, (label, arr) in enumerate(dots_dict.items()):
            arr = np.asarray(arr)
            ax.scatter(arr.T[i], arr.T[j],
                       marker=self.MARKERS[idx % len(self.MARKERS)],
                       color=self.COLORS[idx % len(self.COLORS)],
                       s=50,
                       edgecolors='black',
                       linewidths=0.5,
                       label=label,
                       alpha=self.dot_alpha,
                       zorder=10)
        return cs

    def plot_contours(self, df, ycol, cmap, norm, dots_dict,
                      save_dir, name=''):
        """
        Save each 2D contour slice as its own standalone, publication-ready figure.

        One clean figure is produced per parameter pair, each with a properly
        height-matched colorbar attached to the right of the panel. This keeps a
        single contour from looking stranded in a sparse grid.

        Args:
            df (pd.DataFrame): Dataset with results.
            ycol (str): Output column name.
            cmap (matplotlib.colors.Colormap): Colormap (0 -> white).
            norm (matplotlib.colors.Normalize): Shared colour normalisation.
            dots_dict (dict): Mapping of label -> sample-point array to overlay.
            save_dir (str): Output directory.
            name (str, optional): Filename prefix. Defaults to ''.
        """
        d = len(self.parameters)

        for i in range(d):
            for j in range(i + 1, d):
                fig, ax = plt.subplots(figsize=(6.4, 5.2))

                cs = self._draw_contour(ax, i, j, df, ycol, cmap, norm,
                                        dots_dict, labelsize=12, fontsize=14)
                ax.set_title(
                    f"{self.parameters[i]} vs {self.parameters[j]}",
                    fontsize=15, fontweight='bold', pad=12
                )

                # fraction=0.046 / pad=0.04 makes the colorbar height match a
                # square (box_aspect=1) panel exactly.
                cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(ycol, fontsize=14, fontweight='bold', labelpad=10)
                cbar.ax.tick_params(labelsize=11)
                cbar.outline.set_linewidth(0.8)

                if dots_dict:
                    ax.legend(fontsize=11, loc='best', framealpha=0.95)

                fig.savefig(
                    os.path.join(
                        save_dir,
                        name + f"contour_{self.parameters[i]}_{self.parameters[j]}_{ycol}.png"
                    ),
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)

    def _zero_to_nan(self, arr):
        """
        Convert zero values to NaN in an array.

        Useful for handling missing data represented as zeros.

        Args:
            arr (array-like): Input array.

        Returns:
            np.ndarray: Array with zeros replaced by NaN.
        """
        arr = np.asarray(arr, dtype=float).copy()
        arr[arr == 0.0] = np.nan
        return arr

    def register_future(self, future):
        """
        Register a single future result (unused stub).

        Args:
            future: Future object from async execution.

        Returns:
            None
        """
        return None

    def register_futures(self, futures):
        """
        Register multiple future results (unused stub).

        Args:
            futures (list): List of future objects.

        Returns:
            None
        """
        return None
