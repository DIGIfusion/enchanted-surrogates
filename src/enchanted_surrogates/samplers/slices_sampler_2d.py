import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 15})

from enchanted_surrogates.samplers.base_sampler import Sampler


class SlicesSampler2D(Sampler):

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


    def _normalize_dict_inputs(self, dots_x, predictor):
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

    def __init__(self, parameters, bounds, base_run_dir=None, res=50, fixed=None, budget=100000, type='SlicesSampler2D', dot_alpha=0.7, elev_3d=70, azim_3d=200):
        self.parameters = parameters
        self.bounds = bounds
        self.base_run_dir = base_run_dir
        self.res = res
        self.budget = budget
        self.fixed = fixed or {p: 0.5*(b[0]+b[1]) for p, b in zip(parameters, bounds)}
        self.batch_number = 0
        
        self.elev_3d = elev_3d
        self.azim_3d = azim_3d

        self.dot_alpha = dot_alpha
    def get_next_samples(self):
        if self.batch_number > 0:
            self.make_plots()
            return None
        elif self.batch_number == 0:
            return self.get_samples()

    def get_samples(self):
        d = len(self.parameters)
        n_pairs = d * (d - 1) // 2
        total_samples = n_pairs * (self.res ** 2)
        print(f'[SliceSampler] DIM {d}, RES {self.res}, N SAMPLES {total_samples}')

        if total_samples > self.budget:
            raise RuntimeError(
                f"Requested {total_samples} samples exceeds budget={self.budget}. "
                f"Reduce resolution or number of parameters."
            )
        self.budget = total_samples
        samples = []

        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(i+1, d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                for u in range(self.res):
                    for v in range(self.res):
                        row = {}
                        for k, (param, (a, b)) in enumerate(zip(self.parameters, self.bounds)):
                            if k == i:
                                row[param] = Xi[u,v]
                            elif k == j:
                                row[param] = Yi[u,v]
                            else:
                                row[param] = self.fixed[param]
                        samples.append(row)

        self.batch_number += 1
        return samples

    def get_samples_array(self):
        samples = self.get_samples()
        df = pd.DataFrame(samples)
        return df[self.parameters].to_numpy()

    def make_plots(self, dots_x=None, predictor=None, save_dir=None, name=''):
        self.plot_slices_from_dataset(dots_x=dots_x, save_dir=save_dir, name=name)
        self.plot_full_grid(dots_x=dots_x, predictor=predictor, save_dir=save_dir, name=name)
        self.plot_interactive_3d_slices(dots_x=dots_x, save_dir=save_dir, name=name)

    # -------------------------------------------------------------------------
    # INTERACTIVE 3D PLOTS
    # -------------------------------------------------------------------------
    def plot_interactive_3d_slices(self, df=None, dots_x=None, save_dir=None, name=""):
        import plotly.graph_objects as go

        dots_dict, pred_dict = self._normalize_dict_inputs(dots_x, predictor=None)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        ycol = output_col[0]

        d = len(self.parameters)

        for i in range(d):
            p1 = self.parameters[i]
            x_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)

            for j in range(i+1, d):
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

                Z=self._zero_to_nan(Z)
                fig.add_trace(go.Surface(
                    x=Xi, y=Yi, z=Z,
                    colorscale="Viridis",
                    opacity=0.85,
                    name="dataset surface"
                ))

                z_floor = np.nanmin(Z)

                # MULTI-DOTS SUPPORT
                for idx, (label, arr) in enumerate(dots_dict.items()):
                    arr = np.asarray(arr)
                    
                    fig.add_trace(go.Scatter3d(
                        x=arr.T[i],
                        y=arr.T[j],
                        z=np.full(arr.shape[0], z_floor),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=self.COLORS[idx % len(self.COLORS)],
                            symbol=self.PLOTLY_MARKERS[idx % len(self.PLOTLY_MARKERS)],
                            opacity=self.dot_alpha,
                        ),
                        name=label,
                    ))

                fig.update_layout(
                    title=f"Interactive 3D Slice: {p1} vs {p2} vs {ycol}",
                    scene=dict(
                        xaxis_title=p1,
                        yaxis_title=p2,
                        zaxis_title=ycol,
                    ),
                    height=700,
                )

                out_name = f"{name}interactive_3D_{p1}_{p2}.html"
                fig.write_html(os.path.join(save_dir, out_name))
                print(f"[Saved interactive plot] {out_name}")

    # -------------------------------------------------------------------------
    # 2D + 3D STATIC SLICE PLOTS
    # -------------------------------------------------------------------------
    def plot_slices_from_dataset(self, cmap='viridis', surface_alpha=0.9,
                                 dataset_path=None, df=None, dots_x=None,
                                 save_dir=None, name='', include_3D=True):

        dots_dict, pred_dict = self._normalize_dict_inputs(dots_x, predictor=None)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        ycol = output_col[0]
        ymin, ymax = df[ycol].min(), df[ycol].max()

        d = len(self.parameters)

        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(i+1, d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                Z = np.zeros_like(Xi)

                for u in range(self.res):
                    for v in range(self.res):
                        mask = (
                            (np.isclose(df[self.parameters[i]], Xi[u,v], atol=1e-6)) &
                            (np.isclose(df[self.parameters[j]], Yi[u,v], atol=1e-6))
                        )
                        vals = df.loc[mask, ycol].values
                        Z[u,v] = vals[0] if len(vals) else np.nan

                import matplotlib.gridspec as gridspec
                nc = 3 if include_3D else 2
                fig = plt.figure(figsize=(5*(nc-1),4))
                wr = [1,1,0.05] if include_3D else [1,0.05]
                gs = gridspec.GridSpec(1, nc, width_ratios=wr, figure=fig)

                ax1 = fig.add_subplot(gs[0])
                Z=self._zero_to_nan(Z)
                cs = ax1.contourf(Xi, Yi, Z, cmap=cmap, vmin=ymin, vmax=ymax)
                ax1.set_aspect('equal')
                ax1.set_box_aspect(1)
                ax1.set_xlabel(self.parameters[i])
                ax1.set_ylabel(self.parameters[j])

                # MULTI-DOTS SUPPORT (2D)
                legend_handles = []
                for idx, (label, arr) in enumerate(dots_dict.items()):
                    arr = np.asarray(arr)
                    h = ax1.scatter(arr.T[i], arr.T[j],
                                    marker=self.MARKERS[idx % len(self.MARKERS)],
                                    color=self.COLORS[idx % len(self.COLORS)],
                                    label=label, alpha=self.dot_alpha)
                    legend_handles.append(h)

                if include_3D:
                    ax3d = fig.add_subplot(gs[1], projection='3d')
                    Z=self._zero_to_nan(Z)
                    
                    # ax3d.plot_surface(
                    #     Xi, Yi, Z,
                    #     cmap=cmap,
                    #     alpha=surface_alpha,
                    #     vmin=ymin, vmax=ymax,
                    #     rstride=10, cstride=10,          # draw grid lines in both directions
                    #     edgecolor='black',              # grid line color
                    #     linewidth=0.3,                 # thin lines
                    #     antialiased=True
                    # )
                    
                    ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha,
                                    vmin=ymin, vmax=ymax)

                    ax3d.set_xlabel(self.parameters[i])
                    ax3d.set_ylabel(self.parameters[j])
                    ax3d.set_zlabel(ycol)

                    # Set rotation angle here 
                    ax3d.view_init(elev=self.elev_3d, azim=self.azim_3d) # <-- change these numbers

                    z_floor = float(np.nanmin(Z))

                    # MULTI-DOTS SUPPORT (3D)
                    for idx, (label, arr) in enumerate(dots_dict.items()):
                        arr = np.asarray(arr)
                        ax3d.scatter(arr.T[i], arr.T[j], zs=z_floor, zdir='z',
                                    s=20,
                                    marker=self.MARKERS[idx % len(self.MARKERS)],
                                    color=self.COLORS[idx % len(self.COLORS)],
                                    label=label, alpha=self.dot_alpha)

                if legend_handles:
                    ax1.legend(handles=legend_handles)

                cb_gs = gs[2] if include_3D else gs[1]
                cax = fig.add_subplot(cb_gs)
                fig.colorbar(cs, cax=cax)

                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, name + f"slices_{self.parameters[i]}_{self.parameters[j]}_{ycol}.png"))
                plt.close(fig)

    # -------------------------------------------------------------------------
    # FULL GRID PLOT
    # -------------------------------------------------------------------------
    def plot_full_grid(self, cmap='viridis', surface_alpha=0.9,
                       dataset_path=None, df=None, name='',
                       dots_x=None, predictor=None, save_dir=None):

        dots_dict, pred_dict = self._normalize_dict_inputs(dots_x, predictor)

        if save_dir is None:
            save_dir = self.base_run_dir

        if df is None:
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        ycol = output_col[0]
        ymin, ymax = df[ycol].min(), df[ycol].max()

        d = len(self.parameters)
        base_size = 4
        fig, axes = plt.subplots(d, d, figsize=(base_size*d, base_size*d))

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

                    ax.plot(xi_lin, yvals, '-', color='tab:blue', linewidth=1.5, label="dataset")

                    # MULTI-PREDICTOR SUPPORT
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
                            linewidth=2,
                            label=label
                        )

                    # # MULTI-DOTS SUPPORT
                    # for idx, (label, arr) in enumerate(dots_dict.items()):
                    #     arr = np.asarray(arr)
                    #     ax.scatter(arr.T[i], np.full(arr.shape[0], np.nanmin(yvals)),
                    #                marker=self.MARKERS[idx % len(self.MARKERS)],
                    #                color=self.COLORS[idx % len(self.COLORS)],
                    #                label=label)

                    ax.set_xlabel(p)
                    if i == 0:
                        ax.set_ylabel(ycol)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                # ------------------------------------------------------------------
                # UPPER TRIANGLE: 2D CONTOUR
                # ------------------------------------------------------------------
                elif i < j:
                    Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                    Z = np.zeros_like(Xi)
                    for u in range(self.res):
                        for v in range(self.res):
                            mask = (
                                (np.isclose(df[self.parameters[i]], Xi[u,v], atol=1e-6)) &
                                (np.isclose(df[self.parameters[j]], Yi[u,v], atol=1e-6))
                            )
                            vals = df.loc[mask, ycol].values
                            Z[u,v] = vals[0] if len(vals) else np.nan
                    
                    Z=self._zero_to_nan(Z)
                    cs = ax.contourf(Xi, Yi, Z, cmap=cmap, vmin=ymin, vmax=ymax)
                    ax.set_box_aspect(1)
                    ax.set_xlabel(self.parameters[i])
                    ax.set_ylabel(self.parameters[j])

                    # MULTI-DOTS SUPPORT
                    for idx, (label, arr) in enumerate(dots_dict.items()):
                        arr = np.asarray(arr)
                        ax.scatter(arr.T[i], arr.T[j],
                                   marker=self.MARKERS[idx % len(self.MARKERS)],
                                   color=self.COLORS[idx % len(self.COLORS)],
                                   label=label, 
                                   alpha=self.dot_alpha)

                # ------------------------------------------------------------------
                # LOWER TRIANGLE: 3D SURFACE
                # ------------------------------------------------------------------
                else:
                    ax.remove()
                    ax3d = fig.add_subplot(d, d, i*d+j+1, projection='3d')

                    Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                    Z = np.zeros_like(Xi)
                    for u in range(self.res):
                        for v in range(self.res):
                            mask = (
                                (np.isclose(df[self.parameters[i]], Xi[u,v], atol=1e-6)) &
                                (np.isclose(df[self.parameters[j]], Yi[u,v], atol=1e-6))
                            )
                            vals = df.loc[mask, ycol].values
                            Z[u,v] = vals[0] if len(vals) else np.nan

                    Z=self._zero_to_nan(Z)
                    ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha,
                                      vmin=ymin, vmax=ymax)
                    
                    # Set rotation angle here 
                    ax3d.view_init(elev=self.elev_3d, azim=self.azim_3d) # <-- change these numbers


                    z_floor = float(np.nanmin(Z))

                    # MULTI-DOTS SUPPORT
                    for idx, (label, arr) in enumerate(dots_dict.items()):
                        arr = np.asarray(arr)
                        ax3d.scatter(arr.T[i], arr.T[j], zs=z_floor, zdir='z',
                                     s=20,
                                     marker=self.MARKERS[idx % len(self.MARKERS)],
                                     color=self.COLORS[idx % len(self.COLORS)],
                                     label=label, alpha=self.dot_alpha)

                    ax3d.set_xlabel(self.parameters[i])
                    ax3d.set_ylabel(self.parameters[j])

        # COLORBAR
        cbar_ax = fig.add_axes([0.01, 0.02, 0.02, 0.7])
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(ycol)

        fig.subplots_adjust(wspace=0.7, hspace=0.7)
        fig.savefig(os.path.join(save_dir, name + f"slices_full_grid_{ycol}.png"), dpi=300)
        plt.close(fig)

    def _zero_to_nan(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr[arr == 0.0] = np.nan
        return arr

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None
