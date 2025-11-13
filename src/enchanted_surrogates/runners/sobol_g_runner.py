import os
from enchanted_surrogates.runners.base_runner import Runner
import numpy as np
import shutil
import warnings
from enchanted_surrogates.utils.print_stats_table import print_stats_table

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import math

class SobolGRunner(Runner):
    """
    SobolGRunner: evaluates the Sobol G-function for a given input vector and sensitivity parameters.

    Initialization options (provide exactly one of the two):
    - a: list/tuple/ndarray of floats (length = number of dimensions). Each a_i >= 0.
         - smaller a_i -> higher sensitivity for that dimension
         - typical example: a=[0.01, 0.01, 0.5, 0.99, 0.99]
    - S: list/tuple/ndarray of first-order Sobol indices (length = number of dimensions).
         - values must be between 0 and 1
         - at least one S_i must be > 0
         - the runner will compute the corresponding a vector that yields those first-order indices

    Notes:
    - Samples must have parameters like "x1", "x2", ..., matching the number of dimensions.
    - Samples must be in the range [0, 1].
    - The runner computes the function value and returns a dictionary with "output" and "success".
    - The mean is 1 and it has analytically computable variance and Sobol indices which are useful for testing UQ samplers.
    """

    def __init__(self, *args, **kwargs):
        # Accept either 'a' or 'S' (first-order Sobol indices)
        a_input = kwargs.get('a', None)
        S_input = kwargs.get('S', None)

        if (a_input is None) and (S_input is None):
            raise TypeError("Provide either 'a' (sensitivity params) or 'S' (first-order Sobol indices).")

        if (a_input is not None) and (S_input is not None):
            raise TypeError("Provide only one of 'a' or 'S', not both.")

        if a_input is not None:
            if not isinstance(a_input, (list, tuple, np.ndarray)):
                raise TypeError("Parameter 'a' must be a list, tuple, or numpy array of floats.")
            self.a = np.array(a_input, dtype=float)
            if np.any(self.a < 0):
                raise ValueError("All elements of 'a' must be >= 0.")
        else:
            # S provided: compute a from S
            if not isinstance(S_input, (list, tuple, np.ndarray)):
                raise TypeError("Parameter 'S' must be a list, tuple, or numpy array of floats between 0 and 1.")
            S = np.array(S_input, dtype=float)
            if np.any(S < 0) or np.any(S > 0.5):
                raise ValueError("All elements of 'S' must be in [0, 0.5].")
            if not np.any(S > 0):
                raise ValueError("At least one first-order Sobol index in 'S' must be > 0.")

            # Solve for scalar P = prod(1 + V) where V_i = S_i * (P - 1)
            # Equation: P = prod_i (1 + S_i * (P - 1))
            # We'll solve f(P) = P - prod(1 + S*(P-1)) = 0 for P >= 1
            def f(P):
                return P - np.prod(1.0 + S * (P - 1.0))

            # Bisection/root-finding for P in [1, P_max]
            P_lo = 1.0
            f_lo = f(P_lo)  # should be 0 if S all zero, but we've ensured some S>0 so f_lo >= 0?
            # Evaluate behavior: at P = 1, RHS = prod(1) = 1 so f(1) = 0.
            # However for S not all zero, derivative causes f to become negative for P slightly > 1.
            # We need to find P_hi where f(P_hi) >= 0 again (sign change) or bracket differently.
            # We will search for P where f(P) crosses zero with initial bracketing strategy.
            # Start with a small delta above 1 to check sign
            P_hi = 1.0 + 1e-6
            max_hi = 1e12
            f_hi = f(P_hi)

            # Expand upper bound until sign change or until a large maximum
            expand_factor = 2.0
            attempts = 0
            while (f_hi < 0) and (P_hi < max_hi) and (attempts < 200):
                P_hi *= expand_factor
                f_hi = f(P_hi)
                attempts += 1

            if f_hi < 0:
                # If still negative, try larger upper bound but if fails, raise
                raise RuntimeError("Failed to bracket root for P when computing 'a' from 'S'.")

            # If f_hi >= 0 and f(1) == 0, there can be multiple roots; we search in (1, P_hi]
            # Use bisection on [1, P_hi] but avoid trivially zero at P=1; shift lower endpoint slightly
            lo = 1.0 + 1e-12
            hi = P_hi
            flo = f(lo)
            fhi = f(hi)

            # Ensure signs are opposite for bisection. If not, move lo slightly upwards until sign change or fail.
            if flo * fhi > 0:
                # attempt to find a bracket by sampling
                found = False
                for factor in [1.0001, 1.001, 1.01, 1.1, 2.0]:
                    lo_try = 1.0 + 1e-12
                    hi_try = 1.0 + factor
                    if f(lo_try) * f(hi_try) <= 0:
                        lo, hi = lo_try, hi_try
                        found = True
                        break
                if not found:
                    # fallback: if f(hi) == 0, choose that; else error
                    if abs(fhi) < 1e-12:
                        P = hi
                    else:
                        raise RuntimeError("Failed to find suitable bracket to solve for P from S.")
            # If not already set P
            if 'P' not in locals():
                # Bisection
                it = 0
                max_iter = 200
                while it < max_iter:
                    mid = 0.5 * (lo + hi)
                    fmid = f(mid)
                    if abs(fmid) < 1e-12:
                        P = mid
                        break
                    # Choose subinterval
                    if f(lo) * fmid <= 0:
                        hi = mid
                    else:
                        lo = mid
                    it += 1
                else:
                    # fallback: take midpoint
                    P = 0.5 * (lo + hi)

            # Now compute V and then a
            P = float(P)
            V = S * (P - 1.0)  # elementwise
            # Convert V to a: V_i = 1 / (3 (1 + a_i)^2) -> 1 + a_i = sqrt(1 / (3 V_i))
            a_computed = np.zeros_like(V)
            for i, Vi in enumerate(V):
                if Vi <= 0.0:
                    # Very small or zero V -> very large a (negligible sensitivity)
                    warnings.warn(
                        f"S[{i}] == 0 leads to V[{i}] == 0. Setting a[{i}] to a large value (1e6) to represent negligible sensitivity."
                    )
                    a_computed[i] = 1e6
                else:
                    denom = 3.0 * Vi
                    if denom <= 0.0:
                        raise RuntimeError("Computed nonpositive denominator when converting V to 'a'.")
                    a_computed[i] = np.sqrt(1.0 / denom) - 1.0
                    if a_computed[i] < 0.0:
                        # numerical safety: a must be >= 0
                        a_computed[i] = 0.0
            self.a = a_computed

        # Final checks and warnings
        if np.any(np.array(self.a) == 0):
            warnings.warn(
                "IN THE SOBOL G RUNNER ONE OF THE a VALUES IS 0. THIS MEANS IF ANY OF THE INPUTS IS 0.5 THEN THE OUTPUT WILL BE 0. BEWARE THIS CAN CAUSE ISSUES WITH SPARSE GRIDS WHERE MANY OF THE POINTS HAVE AT LEAST ONE DIMENSION WITH 0.5 VALUE."
            )

    def sobol_g(self, x):
        return np.prod([(np.abs(4 * xi - 2) + ai) / (1 + ai) for xi, ai in zip(x, self.a)])

    def analytical_stats(self):
        """
        Returns analytical mean, variance, first-order and total-order Sobol indices.

        Returns:
        - dict with keys: "mean", "variance", "std", "sobol_indices", "sobol_total_indices"
        """
        a = np.array(self.a, dtype=float)
        # individual variances V_i = 1 / (3 (1 + a_i)^2)
        V = 1.0 / (3.0 * (1.0 + a) ** 2)
        D = np.prod(1.0 + V)  # D = 1 + total_variance
        var = D - 1.0

        # First-order Sobol indices
        # If var is zero (e.g., all V_i == 0), then S_i undefined; set to zeros
        if var <= 0.0:
            S = np.zeros_like(V)
        else:
            S = V / var

        # Total-order Sobol indices
        ST = []
        for i in range(len(a)):
            D_wo_i = np.prod(1.0 + np.delete(V, i))
            ST_i = 1.0 - D_wo_i / D
            ST.append(float(ST_i))

        return {
            "mean": 1.0,
            "variance": float(var),
            "std": float(np.sqrt(var)) if var >= 0.0 else 0.0,
            "sobol_indices": S.tolist(),
            "sobol_total_indices": ST
        }

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        # Validate input domain
        exclusive_params = [f"x{i+1}" for i in range(len(self.a))]
        if params is None:
            raise AssertionError(f"Parameters must include at least one of {exclusive_params}.")
        x = [float(params[k]) for k in sorted(params.keys()) if k in exclusive_params]
        assert len(x) > 0, f"Parameters must include at least one of {exclusive_params}."
        for xi in x:
            if not (0.0 <= xi <= 1.0):
                raise ValueError(
                    f"Invalid input {xi}. Sobol G-function expects all inputs in [0, 1]."
                )
        output = self.sobol_g(x)
        self.clean(run_dir)
        return {"output": output, "success": True}

    def print_stats(self):
        stats = self.analytical_stats()
        stats['header'] = 'ANALYTICAL UQ QUANTITIES'
        stats['subheader'] = f'SOBOL G FUNCTION | a:{self.a}'
        table = print_stats_table(stats)
        return table
    
    def light_post_processing(self, base_run_dir, *args, **kwargs):
        self.print_stats()
        outdir = os.path.join(base_run_dir, "true_function_plots")
        os.makedirs(outdir, exist_ok=True)        

        # Plot lower-triangle (shows mirrored pairs) and save
        self.plot_2D_and_3D_slices(res=100, fixed=None,
                                    save_dir=outdir, triangle='lower', cmap='plasma')

        print(f"Plots saved to {outdir}")
        out = os.path.join(outdir,'sobol_slices.html')
        self.export_plotly_slices_html(out, res=120, fixed={'x3':0.3}, triangle='upper')
        print("Saved interactive HTML:", out)
        
        self.plot_1D_slices_compare(save_path=os.path.join(outdir,'sobol_1d_slices.png'))

        self.plot_1D_slices_compare(normalize_profiles=True, n_profiles=4, save_path=os.path.join(outdir,'sobol_1d_slices_normalized.png'))
            
    
    def light_pre_processing(self, base_run_dir, *args, **kwargs):
        table = self.print_stats()
        with open(os.path.join(base_run_dir, 'true_uq_stats.txt'), 'w') as file:
            file.write(table)
            
        outdir = os.path.join(base_run_dir, "true_function_plots")
        os.makedirs(outdir, exist_ok=True)        

        # Plot lower-triangle (shows mirrored pairs) and save
        self.plot_2D_and_3D_slices(res=100, fixed=None,
                                    save_dir=outdir, triangle='lower', cmap='plasma')

        print(f"Plots saved to {outdir}")
        out = os.path.join(outdir,'sobol_slices.html')
        self.export_plotly_slices_html(out, res=120, fixed={'x3':0.3}, triangle='upper')
        print("Saved interactive HTML:", out)
        
        self.plot_1D_slices_compare(save_path=os.path.join(outdir,'sobol_1d_slices.png'))

        self.plot_1D_slices_compare(normalize_profiles=True, n_profiles=4, save_path=os.path.join(outdir,'sobol_1d_slices_normalized.png'))


    def clean(self, run_dir):
        shutil.rmtree(run_dir)

    def plot_2D_and_3D_slices(self, res=120, fixed=None, vmin=None, vmax=None, cmap='viridis',
                             save_dir=None, figsize_per_dim=2.5, triangle='upper', surface_alpha=1.0):
        """
        Plot half-matrix heatmaps+contours for unique pairs and matching 3D surfaces.

        Features:
        - If d == 2 produces a single figure with the heatmap (left, square) and a 3D surface (right).
        - 3D surfaces accept transparency via surface_alpha (0.0 transparent .. 1.0 opaque).
        - If save_dir (or its parent) contains enchanted_dataset.csv with columns x1,x2,...,
          the dataset points are overplotted on heatmaps and projected onto 3D floors.
        - Returns slice_results: dict mapping (i,j) -> {'X','Y','Z'}.
        """
        import os as _os
        import numpy as _np
        import matplotlib.pyplot as _plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        triangle = triangle.lower()
        if triangle not in ('upper', 'lower'):
            raise ValueError("triangle must be 'upper' or 'lower'")

        d = len(self.a)
        if d < 2:
            raise ValueError("Need at least 2 dimensions to plot 2D slices.")

        # Prepare fixed values (default 0.5)
        fixed_vals = {}
        if fixed is not None:
            if not isinstance(fixed, dict):
                raise TypeError("fixed must be a dict like {'x3':0.5}")
            for k, v in fixed.items():
                if not k.startswith('x'):
                    raise ValueError("fixed keys must be like 'x1','x2',...")
                fixed_vals[k] = float(v)
        for i in range(d):
            key = f'x{i+1}'
            if key not in fixed_vals:
                fixed_vals[key] = 0.5

        # Compute all i<j pairs
        pairs = [(i, j) for i in range(d) for j in range(d) if i < j]
        xi_lin = _np.linspace(0.0, 1.0, res)
        Xg, Yg = _np.meshgrid(xi_lin, xi_lin)

        slice_results = {}
        global_min = _np.inf
        global_max = -_np.inf

        # Compute Z for each pair
        for (i, j) in pairs:
            Z = _np.zeros_like(Xg)
            for iu in range(res):
                for ju in range(res):
                    x = []
                    for k in range(d):
                        key = f'x{k+1}'
                        if k == i:
                            x.append(float(Xg[iu, ju]))
                        elif k == j:
                            x.append(float(Yg[iu, ju]))
                        else:
                            x.append(float(fixed_vals[key]))
                    try:
                        Z[iu, ju] = float(self.sobol_g(x))
                    except Exception:
                        Z[iu, ju] = _np.nan
            finite_Z = Z[_np.isfinite(Z)]
            if finite_Z.size:
                local_min = float(_np.nanmin(finite_Z))
                local_max = float(_np.nanmax(finite_Z))
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)
            else:
                global_min = min(global_min, 0.0)
                global_max = max(global_max, 1.0)
            slice_results[(i, j)] = {'X': Xg, 'Y': Yg, 'Z': Z}

        if vmin is None:
            vmin = global_min
        if vmax is None:
            vmax = global_max

        # Prepare text info for opposite triangle
        try:
            stats = self.analytical_stats()
            S_list = stats.get('sobol_indices', [])
        except Exception:
            a_arr = _np.array(self.a, dtype=float)
            V = 1.0 / (3.0 * (1.0 + a_arr) ** 2)
            D = _np.prod(1.0 + V)
            var = D - 1.0
            S_list = (V / var).tolist() if var > 0 else [0.0]*len(a_arr)
        a_list = [float(ai) for ai in _np.array(self.a, dtype=float)]
        sobol_line = "Sobol S: " + ", ".join([f"{s:.4g}" for s in S_list])
        a_line = "a: " + ", ".join([f"{ai:.4g}" for ai in a_list])
        eq_line = "g(x) = ∏ (|4 x_i - 2| + a_i) / (1 + a_i)"
        note_line = "Other dims fixed at provided values"

        # --- Load optional dataset if save_dir provided (look for enchanted_dataset.csv in save_dir or its parent) ---
        dataset_df = None
        if save_dir:
            _candidate_paths = []
            _candidate_paths.append(_os.path.join(save_dir, "enchanted_dataset.csv"))
            _parent = _os.path.abspath(_os.path.join(save_dir, os.pardir))
            if _parent and _parent != '' and _parent != os.path.abspath(save_dir):
                _candidate_paths.append(_os.path.join(_parent, "enchanted_dataset.csv"))
            for _dataset_path in _candidate_paths:
                if _os.path.exists(_dataset_path):
                    try:
                        import pandas as _pd
                        _dataset = _pd.read_csv(_dataset_path)
                        required_cols = [f"x{i+1}" for i in range(d)]
                        if all(c in _dataset.columns for c in required_cols):
                            dataset_df = _dataset
                            break
                        else:
                            dataset_df = None
                    except Exception:
                        try:
                            _data = _np.genfromtxt(_dataset_path, delimiter=',', names=True, dtype=float)
                            required_cols = [f"x{i+1}" for i in range(d)]
                            if _data.size == 0:
                                dataset_df = None
                                continue
                            if all(col in _data.dtype.names for col in required_cols):
                                _dataset = {name: _np.asarray(_data[name], dtype=float) for name in _data.dtype.names}
                                class _Shim:
                                    def __init__(self, data_dict):
                                        self._d = data_dict
                                    def __getitem__(self, key):
                                        return self._d[key]
                                    @property
                                    def columns(self):
                                        return list(self._d.keys())
                                dataset_df = _Shim(_dataset)
                                break
                            else:
                                dataset_df = None
                        except Exception:
                            dataset_df = None
        # dataset_df is either None or an object supporting column access like df['x1']
        # -------------------------------------------------------------------------------

        # Figure sizing
        n = d
        fig_size = max(6, figsize_per_dim * n)

        # --- Special-case d == 2: single figure with left heatmap (square) and right 3D (boxless) ---
        if d == 2:
            pair = pairs[0]
            data = slice_results[pair]
            X = data['X']; Y = data['Y']; Z = data['Z']

            fig, axs = _plt.subplots(1, 2, figsize=(max(8, figsize_per_dim * 3), max(4, figsize_per_dim * 1.5)))
            # Left: heatmap + contours (make square by equal aspect)
            ax0 = axs[0]
            im = ax0.imshow(Z, origin='lower', extent=(0,1,0,1), vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
            try:
                cs = ax0.contour(X, Y, Z, colors='k', linewidths=0.5, levels=6, alpha=0.7)
                ax0.clabel(cs, fmt='%.2f', fontsize=8)
            except Exception:
                pass
            ax0.set_xlabel(f"x{pair[1]+1}")
            ax0.set_ylabel(f"x{pair[0]+1}")

            # Right: 3D surface with transparency
            ax1 = fig.add_subplot(1, 2, 2, projection='3d')
            # Remove any leftover 2D axes that overlap this cell
            from matplotlib.transforms import Bbox as _Bbox
            for a in list(fig.axes):
                if a is ax1 or getattr(a, "name", "").lower().startswith("axes3d"):
                    continue
                if _Bbox.intersection(a.get_position(), ax1.get_position()) is not None:
                    fig.delaxes(a)

            plot_stride = max(1, int(res / 60))
            try:
                # use facecolors approach for nicer alpha control
                from matplotlib import cm as _cm
                norm = _plt.Normalize(vmin=vmin, vmax=vmax)
                m = _cm.ScalarMappable(norm=norm, cmap=cmap)
                fc = m.to_rgba(Z)
                fc[..., 3] = float(surface_alpha)
                ax1.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True,
                                 rcount=max(2, Z.shape[0]//plot_stride),
                                 ccount=max(2, Z.shape[1]//plot_stride), shade=True)
            except Exception:
                try:
                    ax1.plot_surface(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True,
                                     rcount=max(2, Z.shape[0]//plot_stride),
                                     ccount=max(2, Z.shape[1]//plot_stride), alpha=float(surface_alpha))
                except Exception:
                    ax1.plot_wireframe(X, Y, Z, rcount=max(2, Z.shape[0]//plot_stride),
                                        ccount=max(2, Z.shape[1]//plot_stride), color='k')

            ax1.set_xlabel(f"x{pair[1]+1}")
            ax1.set_ylabel(f"x{pair[0]+1}")
            ax1.set_zlabel("g(x)")

            # Overlay dataset points if loaded (2D heatmap + project onto 3D floor)
            if dataset_df is not None:
                # heatmap overlay on ax0: scatter x_j vs x_i
                x_col = f"x{pair[1]+1}"
                y_col = f"x{pair[0]+1}"
                try:
                    _xs_plot = dataset_df[x_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[x_col])
                    _ys_plot = dataset_df[y_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[y_col])
                except Exception:
                    _xs_plot = dataset_df[x_col]
                    _ys_plot = dataset_df[y_col]
                # plot on heatmap
                ax0.scatter(_xs_plot, _ys_plot, s=18, c='k', alpha=0.6, edgecolor='white', linewidth=0.3, zorder=10)

                # 3D projection on floor: project to z_floor (min of Z)
                z_floor = float(_np.nanmin(Z)) if _np.isfinite(_np.nanmin(Z)) else vmin
                # build full inputs to optionally color by g(x)
                _x_full = []
                for idx_row in range(len(_xs_plot)):
                    row = []
                    for k in range(d):
                        if k == pair[1]:
                            row.append(float(_xs_plot[idx_row]))
                        elif k == pair[0]:
                            row.append(float(_ys_plot[idx_row]))
                        else:
                            row.append(float(fixed_vals[f"x{k+1}"]))
                    _x_full.append(row)
                try:
                    _gvals = _np.array([float(self.sobol_g(r)) for r in _x_full])
                except Exception:
                    _gvals = None
                if _gvals is None:
                    ax1.scatter(_xs_plot, _ys_plot, zs=z_floor, zdir='z', s=20, c='k', alpha=0.6, depthshade=False)
                else:
                    from matplotlib import cm as _cm
                    norm = _plt.Normalize(vmin=vmin, vmax=vmax)
                    _colors = _cm.get_cmap(cmap)(norm(_gvals))
                    ax1.scatter(_xs_plot, _ys_plot, zs=z_floor, zdir='z', s=20, c=_colors, alpha=0.8, depthshade=False)

            # place colorbar in an explicit non-overlapping figure axis on the far right
            cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])   # tweak left position if necessary
            norm_mappable = _plt.cm.ScalarMappable(cmap=cmap)
            norm_mappable.set_clim(vmin, vmax)
            fig.colorbar(norm_mappable, cax=cax, label="g(x)")

            if save_dir:
                _os.makedirs(save_dir, exist_ok=True)
                path = _os.path.join(save_dir, "slice_2d_heatmap_3d.png")
                fig.savefig(path, dpi=200, bbox_inches='tight')
                _plt.close(fig)
            else:
                _plt.show()

            return slice_results

        # --- For d > 2: heatmap half-matrix with opposite triangle text and separate 3D figure ---
        fig, axes = _plt.subplots(n, n, figsize=(fig_size, fig_size), squeeze=False)
        _plt.subplots_adjust(hspace=0.5, wspace=0.5)
        for row_idx in range(n):
            for col_idx in range(n):
                ax = axes[row_idx, col_idx]
                ax.set_xticks([])
                ax.set_yticks([])

                plot_slice = (triangle == 'upper' and row_idx < col_idx) or (triangle == 'lower' and row_idx > col_idx)
                if plot_slice:
                    if row_idx < col_idx:
                        pair = (row_idx, col_idx)
                    else:
                        pair = (col_idx, row_idx)
                    data = slice_results.get(pair)
                    if data is None:
                        ax.axis('off')
                        continue
                    X = data['X']; Y = data['Y']; Z = data['Z']
                    im = ax.imshow(Z, origin='lower', extent=(0,1,0,1), vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
                    try:
                        cs = ax.contour(X, Y, Z, colors='k', linewidths=0.5, levels=6, alpha=0.7)
                        ax.clabel(cs, fmt='%.2f', fontsize=6)
                    except Exception:
                        pass
                    ax.set_xlabel(f"x{pair[1]+1}", fontsize=7)
                    ax.set_ylabel(f"x{pair[0]+1}", fontsize=7)

                    # overlay dataset points on heatmap (if available)
                    if dataset_df is not None:
                        x_col = f"x{pair[1]+1}"
                        y_col = f"x{pair[0]+1}"
                        try:
                            _xs_plot = dataset_df[x_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[x_col])
                            _ys_plot = dataset_df[y_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[y_col])
                        except Exception:
                            _xs_plot = dataset_df[x_col]
                            _ys_plot = dataset_df[y_col]
                        ax.scatter(_xs_plot, _ys_plot, s=12, c='k', alpha=0.6, edgecolor='white', linewidth=0.2, zorder=10)

                else:
                    ax.axis('off')
                    txt = sobol_line + "\n" + a_line + "\n" + eq_line + "\n" + note_line
                    ax_text = ax.inset_axes([0.05, 0.05, 0.9, 0.9])
                    ax_text.axis('off')
                    ax_text.text(0.0, 1.0, txt, va='top', ha='left', fontsize=7, family='monospace', wrap=True)

        # shared colorbar for heatmaps
        cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
        norm_mappable = _plt.cm.ScalarMappable(cmap=cmap)
        norm_mappable.set_clim(vmin, vmax)
        fig.colorbar(norm_mappable, cax=cbar_ax)
        fig.suptitle(f"Sobol G 2D slices (heatmaps + contours) | triangle={triangle}", fontsize=14)
        if save_dir:
            _os.makedirs(save_dir, exist_ok=True)
            heatmap_path = _os.path.join(save_dir, f"slices_heatmaps_{triangle}.png")
            fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            _plt.close(fig)
        else:
            _plt.show()

        # 3D surfaces figure (separate)
        fig3d = _plt.figure(figsize=(fig_size, fig_size))
        for row_idx in range(n):
            for col_idx in range(n):
                plot_surface_cell = (triangle == 'upper' and row_idx < col_idx) or (triangle == 'lower' and row_idx > col_idx)
                if not plot_surface_cell:
                    # mirror textual info in the empty cell
                    idx = row_idx * n + col_idx + 1
                    ax_info = fig3d.add_subplot(n, n, idx)
                    ax_info.axis('off')
                    txt = sobol_line + "\n" + a_line + "\n" + eq_line
                    ax_info.text(0.0, 1.0, txt, va='top', ha='left', fontsize=7, family='monospace', wrap=True)
                    continue

                if row_idx < col_idx:
                    pair = (row_idx, col_idx)
                else:
                    pair = (col_idx, row_idx)
                data = slice_results.get(pair)
                if data is None:
                    continue
                idx = row_idx * n + col_idx + 1
                ax3d = fig3d.add_subplot(n, n, idx, projection='3d')
                X = data['X']; Y = data['Y']; Z = data['Z']
                plot_stride = max(1, int(res / 60))
                try:
                    from matplotlib import cm as _cm
                    norm = _plt.Normalize(vmin=vmin, vmax=vmax)
                    m = _cm.ScalarMappable(norm=norm, cmap=cmap)
                    fc = m.to_rgba(Z)
                    fc[..., 3] = float(surface_alpha)
                    ax3d.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True,
                                      rcount=max(2, Z.shape[0]//plot_stride),
                                      ccount=max(2, Z.shape[1]//plot_stride), shade=True)
                except Exception:
                    try:
                        ax3d.plot_surface(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True,
                                          rcount=max(2, Z.shape[0]//plot_stride),
                                          ccount=max(2, Z.shape[1]//plot_stride), alpha=float(surface_alpha))
                    except Exception:
                        ax3d.plot_wireframe(X, Y, Z,
                                            rcount=max(2, Z.shape[0]//plot_stride),
                                            ccount=max(2, Z.shape[1]//plot_stride), color='k')
                ax3d.set_xlabel(f"x{pair[1]+1}", fontsize=7)
                ax3d.set_ylabel(f"x{pair[0]+1}", fontsize=7)
                ax3d.set_zlabel("g(x)", fontsize=7)
                ax3d.set_title(f"x{pair[0]+1} vs x{pair[1]+1}", fontsize=8)

                # overlay dataset points projected to the floor of this 3D subplot
                if dataset_df is not None:
                    x_col = f"x{pair[1]+1}"
                    y_col = f"x{pair[0]+1}"
                    try:
                        _xs_plot = dataset_df[x_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[x_col])
                        _ys_plot = dataset_df[y_col].values if hasattr(dataset_df, "values") else _np.asarray(dataset_df[y_col])
                    except Exception:
                        _xs_plot = dataset_df[x_col]
                        _ys_plot = dataset_df[y_col]
                    z_floor = float(_np.nanmin(Z)) if _np.isfinite(_np.nanmin(Z)) else vmin
                    _x_full = []
                    for idx_row in range(len(_xs_plot)):
                        row = []
                        for k in range(d):
                            if k == pair[1]:
                                row.append(float(_xs_plot[idx_row]))
                            elif k == pair[0]:
                                row.append(float(_ys_plot[idx_row]))
                            else:
                                row.append(float(fixed_vals[f"x{k+1}"]))
                        _x_full.append(row)
                    try:
                        _gvals = _np.array([float(self.sobol_g(r)) for r in _x_full])
                    except Exception:
                        _gvals = None
                    if _gvals is None:
                        ax3d.scatter(_xs_plot, _ys_plot, zs=z_floor, zdir='z', s=15, c='k', alpha=0.6, depthshade=False)
                    else:
                        from matplotlib import cm as _cm
                        norm = _plt.Normalize(vmin=vmin, vmax=vmax)
                        _colors = _cm.get_cmap(cmap)(norm(_gvals))
                        ax3d.scatter(_xs_plot, _ys_plot, zs=z_floor, zdir='z', s=15, c=_colors, alpha=0.85, depthshade=False)

        fig3d.suptitle(f"Sobol G function 3D surfaces | triangle={triangle}", fontsize=14)
        mappable = _plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(vmin, vmax)
        fig3d.subplots_adjust(right=0.9)
        cax2 = fig3d.add_axes([0.92, 0.12, 0.02, 0.76])
        fig3d.colorbar(mappable, cax=cax2)

        if save_dir:
            surf_path = _os.path.join(save_dir, f"slices_3d_{triangle}.png")
            fig3d.savefig(surf_path, dpi=200, bbox_inches='tight')
            _plt.close(fig3d)
        else:
            _plt.show()

        return slice_results
    
    def export_plotly_slices_html(self, html_path, res=120, fixed=None, colorscale='Viridis', triangle='upper', opacity=0.5):
        """
        Create interactive Plotly HTML showing heatmap(s) and 3D surface(s) of Sobol G slices.

        - html_path: path to save standalone HTML file (will be overwritten).
        - res: grid resolution for each slice.
        - fixed: dict of fixed coordinates, e.g. {'x3':0.5}. Missing dims default to 0.5.
        - colorscale: Plotly colorscale name or list.
        - triangle: 'upper' or 'lower' to select which triangle has the slice plots for d>2.
        Returns: path to saved HTML.
        """
        import os
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        triangle = triangle.lower()
        if triangle not in ('upper', 'lower'):
            raise ValueError("triangle must be 'upper' or 'lower'")

        d = len(self.a)
        if d < 2:
            raise ValueError("Need at least 2 dimensions to plot 2D slices.")

        # Prepare fixed values
        fixed_vals = {}
        if fixed is not None:
            if not isinstance(fixed, dict):
                raise TypeError("fixed must be a dict like {'x3':0.5}")
            for k, v in fixed.items():
                if not k.startswith('x'):
                    raise ValueError("fixed keys must be like 'x1','x2',...")
                fixed_vals[k] = float(v)
        for i in range(d):
            key = f'x{i+1}'
            if key not in fixed_vals:
                fixed_vals[key] = 0.5

        # prepare pairs (i<j) and grid
        pairs = [(i, j) for i in range(d) for j in range(d) if i < j]
        xi = np.linspace(0.0, 1.0, res)
        Xg, Yg = np.meshgrid(xi, xi)
        slice_results = {}
        global_min = np.inf
        global_max = -np.inf

        for (i, j) in pairs:
            Z = np.empty_like(Xg)
            for iu in range(res):
                for ju in range(res):
                    x = []
                    for k in range(d):
                        key = f'x{k+1}'
                        if k == i:
                            x.append(float(Xg[iu, ju]))
                        elif k == j:
                            x.append(float(Yg[iu, ju]))
                        else:
                            x.append(float(fixed_vals[key]))
                    try:
                        Z[iu, ju] = float(self.sobol_g(x))
                    except Exception:
                        Z[iu, ju] = np.nan
            finite = Z[np.isfinite(Z)]
            if finite.size:
                global_min = min(global_min, float(np.nanmin(finite)))
                global_max = max(global_max, float(np.nanmax(finite)))
            else:
                global_min = min(global_min, 0.0)
                global_max = max(global_max, 1.0)
            slice_results[(i, j)] = {'X': Xg, 'Y': Yg, 'Z': Z}

        if d == 2:
            # Single figure: 1 row, 2 cols
            pair = pairs[0]
            data = slice_results[pair]
            X = data['X']; Y = data['Y']; Z = data['Z']

            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"type": "heatmap"}, {"type": "surface"}]],
                                subplot_titles=(f"Heatmap: x{pair[0]+1} vs x{pair[1]+1}", f"Surface: x{pair[0]+1} vs x{pair[1]+1}"))

            # Heatmap (Plotly's heatmap expects z with [rows, cols] matching meshgrid)
            fig.add_trace(
                go.Heatmap(z=Z, x=xi, y=xi, colorscale=colorscale, zmin=global_min, zmax=global_max, colorbar=dict(title="g(x)")),
                row=1, col=1
            )

            # Surface (note Plotly's surface interprets z as rows x cols)
            fig.add_trace(
                go.Surface(z=Z, x=xi, y=xi, colorscale=colorscale, cmin=global_min, cmax=global_max,
                        showscale=False, opacity=float(opacity)),
                row=1, col=2
            )

            fig.update_layout(title="Sobol G: 2D slice (heatmap + interactive 3D surface)", autosize=True, height=600, width=1200)
            fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
            return html_path

        # For d > 2: create panels. For many pairs limit panels per HTML page for usability.
        # We'll put one pair per row with 2 columns: heatmap (left) + surface (right).
        # If there are many pairs, we write a single long HTML (scrollable) or you can request paging.
        rows = len(pairs)
        cols = 2
        fig = make_subplots(rows=rows, cols=cols,
                            specs=[[{"type":"heatmap"}, {"type":"surface"}] for _ in range(rows)],
                            subplot_titles=[f"{'Heatmap:':7} x{i+1} vs x{j+1} | {'Surface:':8} x{i+1} vs x{j+1}" for (i,j) in pairs])

        for ridx, (i, j) in enumerate(pairs, start=1):
            data = slice_results[(i, j)]
            Z = data['Z']
            # heatmap on column 1
            fig.add_trace(
                go.Heatmap(z=Z, x=xi, y=xi, colorscale=colorscale, zmin=global_min, zmax=global_max, showscale=(ridx==1)),
                row=ridx, col=1
            )
            # # surface on column 2
            # fig.add_trace(
            #     go.Surface(z=Z, x=xi, y=xi, colorscale=colorscale, cmin=global_min, cmax=global_max, showscale=False),
            #     row=ridx, col=2
            # )
            fig.add_trace(
                go.Surface(z=Z, x=xi, y=xi, colorscale=colorscale, cmin=global_min, cmax=global_max,
                        showscale=False, opacity=float(opacity)),
                row=ridx, col=2
            )

        fig.update_layout(title="Sobol G: interactive slices (heatmap + 3D surface per pair)",
                        height=min(3000, 300*rows), width=1200, autosize=False)
        fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
        return html_path
    
    def plot_1D_slices_compare(self, xs=None, fixed_values=None, n_profiles=5, figsize=(8,5),
                               normalize_profiles=False, title=None, save_path=None):
        """
        Plot 1D slice profiles and label lines inline: x1-varying labels on the left,
        x2-varying labels on the right. Labels include the fixed value of the other dimension.

        Parameters:
        - xs: 1D array-like of x values (default np.linspace(0,1,401))
        - fixed_values: None, scalar, dict, or sequence. If None and d==2, uses n_profiles evenly spaced fixed values.
        - n_profiles: number of profiles when fixed_values is None
        - normalize_profiles: if True divide each profile by its mean
        - title: optional figure title
        - save_path: if provided saves figure and still returns (fig, ax)
        Returns:
        - (fig, ax)
        """
        import numpy as _np
        import matplotlib.pyplot as _plt

        d = len(self.a)
        if d < 2:
            raise ValueError("Need at least 2 dimensions for 1D slice comparison.")

        # x grid
        if xs is None:
            xs = _np.linspace(0.0, 1.0, 401)
        xs = _np.asarray(xs)

        # Prepare fixed values for non-plotted dims
        if fixed_values is None:
            if d == 2:
                fixed_vals = None
            else:
                fixed_vals = {f'x{i+1}': 0.5 for i in range(d)}
        else:
            if isinstance(fixed_values, dict):
                fixed_vals = dict(fixed_values)
            elif _np.isscalar(fixed_values):
                fixed_vals = {f'x{i+1}': float(fixed_values) for i in range(d)}
            else:
                seq = list(fixed_values)
                fixed_vals = {}
                for i in range(d):
                    key = f'x{i+1}'
                    if i < len(seq):
                        fixed_vals[key] = float(seq[i])
                    else:
                        fixed_vals[key] = 0.5

        # Try to get Sobol first-order indices for labels
        try:
            stats = self.analytical_stats()
            S_list = stats.get('sobol_indices', None)
            if S_list is None:
                raise Exception("no sobol_indices")
            S = _np.array(S_list, dtype=float)
        except Exception:
            a_arr = _np.array(self.a, dtype=float)
            V = 1.0 / (3.0 * (1.0 + a_arr) ** 2)
            D = _np.prod(1.0 + V)
            var = D - 1.0
            if var > 0:
                S = V / var
            else:
                S = _np.zeros_like(a_arr)

        # Colors and linestyles
        colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']
        color0 = colors[0 % len(colors)]
        color1 = colors[1 % len(colors)]
        ls_var_x1 = '-'   # x1 varying
        ls_var_x2 = '--'  # x2 varying

        fig, ax = _plt.subplots(figsize=figsize)

        if d == 2:
            # Determine fixed_list for the other dimension
            if fixed_values is None:
                fixed_list = _np.linspace(0.05, 0.95, num=n_profiles)
            elif isinstance(fixed_values, (list, tuple, _np.ndarray)):
                fixed_list = list(fixed_values)
            elif _np.isscalar(fixed_values):
                fixed_list = [float(fixed_values)]
            else:
                if isinstance(fixed_values, dict) and 'x2' in fixed_values:
                    fixed_list = [fixed_values['x2']]
                else:
                    fixed_list = _np.linspace(0.05, 0.95, num=n_profiles)

            # Plot profiles where x1 varies (color of dim1), label on left with fixed x2
            for idx, x2_fixed in enumerate(fixed_list):
                g_vals = _np.array([self.sobol_g([float(x), float(x2_fixed)]) for x in xs])
                if normalize_profiles:
                    g_vals = g_vals / _np.nanmean(g_vals)
                ax.plot(xs, g_vals, color=color0, linestyle=ls_var_x1, alpha=0.9)
                # Inline label on the left (near xs[0])
                x_label_pos = xs[0]
                y_label_pos = g_vals[0]
                label = f"x1 | S1={S[0]:.3g} | x2={x2_fixed:.3g}"
                # offset slightly to avoid overlapping line start
                dx = 0.01 * (xs[-1] - xs[0])
                ax.text(x_label_pos - dx, y_label_pos, label, color=color0, fontsize=8, va='center', ha='right')

            # Plot profiles where x2 varies (color of dim2), label on right with fixed x1
            for idx, x1_fixed in enumerate(fixed_list):
                g_vals = _np.array([self.sobol_g([float(x1_fixed), float(x)]) for x in xs])
                if normalize_profiles:
                    g_vals = g_vals / _np.nanmean(g_vals)
                ax.plot(xs, g_vals, color=color1, linestyle=ls_var_x2, alpha=0.9)
                # Inline label on the right (near xs[-1])
                x_label_pos = xs[-1]
                y_label_pos = g_vals[-1]
                label = f"x2 | S2={S[1]:.3g} | x1={x1_fixed:.3g}"
                dx = 0.01 * (xs[-1] - xs[0])
                ax.text(x_label_pos + dx, y_label_pos, label, color=color1, fontsize=8, va='center', ha='left')

        else:
            # d > 2: vary x1 and x2 while other dims fixed (from fixed_vals)
            base = {}
            for i in range(d):
                key = f'x{i+1}'
                base[key] = float(fixed_vals[key]) if (fixed_vals and key in fixed_vals) else 0.5

            # Profiles varying x1 (label left)
            for idx in range(n_profiles):
                g_vals = []
                for x in xs:
                    inp = [None]*d
                    for k in range(d):
                        key = f'x{k+1}'
                        if k == 0:
                            inp[k] = float(x)
                        else:
                            inp[k] = float(base[key])
                    g_vals.append(self.sobol_g(inp))
                g_vals = _np.array(g_vals)
                if normalize_profiles:
                    g_vals = g_vals / _np.nanmean(g_vals)
                ax.plot(xs, g_vals, color=color0, linestyle=ls_var_x1, alpha=0.9)
                x_label_pos = xs[0]
                y_label_pos = g_vals[0]
                label = f"x1 | S1={S[0]:.3g} | others fixed"
                ax.text(x_label_pos - 0.01*(xs[-1]-xs[0]), y_label_pos, label, color=color0, fontsize=8, va='center', ha='right')

            # Profiles varying x2 (label right)
            for idx in range(n_profiles):
                g_vals = []
                for x in xs:
                    inp = [None]*d
                    for k in range(d):
                        key = f'x{k+1}'
                        if k == 1:
                            inp[k] = float(x)
                        else:
                            inp[k] = float(base[key])
                    g_vals.append(self.sobol_g(inp))
                g_vals = _np.array(g_vals)
                if normalize_profiles:
                    g_vals = g_vals / _np.nanmean(g_vals)
                ax.plot(xs, g_vals, color=color1, linestyle=ls_var_x2, alpha=0.9)
                x_label_pos = xs[-1]
                y_label_pos = g_vals[-1]
                label = f"x2 | S2={S[1]:.3g} | others fixed"
                ax.text(x_label_pos + 0.01*(xs[-1]-xs[0]), y_label_pos, label, color=color1, fontsize=8, va='center', ha='left')

        # Formatting
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.set_title(title if title is not None else "1D profiles — inline Sobol S labels with fixed values")
        ax.grid(True)
        _plt.tight_layout(rect=[0,0,1,0.95])

        if save_path:
            _plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig, ax


if __name__ == "__main__":
    # Minimal example showing how to construct the runner and produce plots.
    # Requirements: matplotlib installed.
    import os

    outdir = os.path.join('.', "sobol_plots_example")
    os.makedirs(outdir, exist_ok=True)

    # Example 1: initialize from sensitivity parameters a
    # a = [0.01, 0.1, 0.5, 0.99]        # 4-dimensional problem
    # runner = SobolGRunner(a=a)
    # Example 2: initialize from first-order Sobol indices S (at least one > 0)
    a = [0.1, 1]          # first-order Sobol indices for 4 dims
    runner = SobolGRunner(a=a)
    runner.print_stats()
    

    # Plot lower-triangle (shows mirrored pairs) and save
    runner.plot_2D_and_3D_slices(res=100, fixed=None,
                                save_dir=outdir, triangle='lower', cmap='plasma')

    print(f"Plots saved to {outdir}")

    # Example evaluation: single run call to verify runner works
    # params = {"x1": 0.2, "x2": 0.5, "x3": 0.7, "x4": 0.1}
    # result = runner.single_code_run(run_dir=os.path.join(tempfile.gettempdir(), "tmp_run_dir"), params=params)
    # print("Single run result:", result)

    # runner = SobolGRunner(a=[0.01,0.1,0.5])    # or S=...
    out = os.path.join(outdir,'sobol_slices.html')
    runner.export_plotly_slices_html(out, res=120, fixed={'x3':0.3}, triangle='upper')
    print("Saved interactive HTML:", out)
    
    runner.plot_1D_slices_compare(save_path=os.path.join(outdir,'sobol_1d_slices.png'))

    runner.plot_1D_slices_compare(normalize_profiles=True, n_profiles=4, save_path=os.path.join(outdir,'sobol_1d_slices_normalized.png'))
    

    # Users can now open the HTML in any modern browser.
