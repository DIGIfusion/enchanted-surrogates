def plot_2D_and_3D_slices(predict_func=None, parameters=None, bounds=None, res=120, fixed=None,
                          vmin=None, vmax=None, cmap='viridis',
                          save_dir=None, figsize_per_dim=2.5, triangle='upper',
                          surface_alpha=1.0, dataset_df=None, sampler=None):
    """
    Plot half-matrix heatmaps+contours for unique parameter pairs and matching 3D surfaces.

    Args:
        predict_func: callable f(x) -> scalar
        parameters: list of parameter names, e.g. ["length","width","height"]
        bounds: list of [lower, upper] pairs, same length as parameters
        res: resolution of grid
        fixed: dict of fixed values {param: value}, defaults to midpoint of bounds
        vmin,vmax: color scale limits
        cmap: colormap
        save_dir: optional directory to save figure
        figsize_per_dim: scaling factor for figure size
        triangle: 'upper' or 'lower'
        surface_alpha: transparency for 3D surface
        dataset_df: optional dataframe with columns matching parameters
        sampler: unused placeholder
    """
    import os as _os
    import numpy as _np
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if sampler:
        parameters = sampler.parameters
        bounds = sampler.bounds
        predict_func = sampler.surrogate_predict

    triangle = triangle.lower()
    if triangle not in ('upper', 'lower'):
        raise ValueError("triangle must be 'upper' or 'lower'")

    d = len(parameters)
    if d < 2:
        raise ValueError("Need at least 2 parameters to plot 2D slices.")

    # Prepare fixed values (default midpoint of bounds)
    fixed_vals = {}
    if fixed is not None:
        if not isinstance(fixed, dict):
            raise TypeError("fixed must be a dict like {'length':0.5}")
        for k, v in fixed.items():
            if k not in parameters:
                raise ValueError(f"fixed key {k} not in parameters")
            fixed_vals[k] = float(v)
    for i, pname in enumerate(parameters):
        if pname not in fixed_vals:
            lo, hi = bounds[i]
            fixed_vals[pname] = 0.5 * (lo + hi)

    # Compute all i<j pairs
    pairs = [(i, j) for i in range(d) for j in range(d) if i < j]

    slice_results = {}
    global_min = _np.inf
    global_max = -_np.inf

    # Compute Z for each pair
    for (i, j) in pairs:
        xi_lin = _np.linspace(bounds[i][0], bounds[i][1], res)
        yj_lin = _np.linspace(bounds[j][0], bounds[j][1], res)
        Xg, Yg = _np.meshgrid(xi_lin, yj_lin)

        Z = _np.zeros_like(Xg)
        for iu in range(res):
            for ju in range(res):
                x = []
                for k in range(d):
                    if k == i:
                        x.append(float(Xg[iu, ju]))
                    elif k == j:
                        x.append(float(Yg[iu, ju]))
                    else:
                        x.append(float(fixed_vals[parameters[k]]))
                try:
                    Z[iu, ju] = float(predict_func(x))
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

    # --- Special-case d == 2: single figure with left heatmap and right 3D ---
    if d == 2:
        pair = pairs[0]
        data = slice_results[pair]
        X, Y, Z = data['X'], data['Y'], data['Z']

        fig, axs = _plt.subplots(1, 2, figsize=(max(8, figsize_per_dim * 3), max(4, figsize_per_dim * 1.5)))
        # Left: heatmap
        ax0 = axs[0]
        im = ax0.imshow(Z, origin='lower',
                        extent=(bounds[pair[1]][0], bounds[pair[1]][1],
                                bounds[pair[0]][0], bounds[pair[0]][1]),
                        vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        try:
            cs = ax0.contour(X, Y, Z, colors='k', linewidths=0.5, levels=6, alpha=0.7)
            ax0.clabel(cs, fmt='%.2f', fontsize=8)
        except Exception:
            pass
        ax0.set_xlabel(parameters[pair[1]])
        ax0.set_ylabel(parameters[pair[0]])

        # Right: 3D surface
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        plot_stride = max(1, int(res / 60))
        from matplotlib import cm as _cm
        norm = _plt.Normalize(vmin=vmin, vmax=vmax)
        m = _cm.ScalarMappable(norm=norm, cmap=cmap)
        fc = m.to_rgba(Z)
        fc[..., 3] = float(surface_alpha)
        ax1.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True,
                         rcount=max(2, Z.shape[0]//plot_stride),
                         ccount=max(2, Z.shape[1]//plot_stride), shade=True)
        ax1.set_xlabel(parameters[pair[1]])
        ax1.set_ylabel(parameters[pair[0]])
        ax1.set_zlabel("g(x)")

        # Colorbar
        cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
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
