import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams.update({'font.size': 10})

from enchanted_surrogates.samplers.base_sampler import Sampler

class SlicesSampler2D(Sampler):
    # Total number of samples = (number of unique parameter pairs) × (res^2)
    #                         = [d * (d - 1) / 2] × (res^2)
    # where d = dimensionality (number of parameters), res = grid resolution per axis

    def __init__(self, parameters, bounds, base_run_dir=None, res=50, fixed=None, budget=100000, type='SlicesSampler2D'):
        self.parameters = parameters
        self.bounds = bounds
        self.base_run_dir = base_run_dir
        self.res = res
        self.budget = budget
        # default fixed values for non-slice dimensions
        self.fixed = fixed or {p: 0.5*(b[0]+b[1]) for p, b in zip(parameters, bounds)}
        self.batch_number = 0
    def get_next_samples(self):
        """
        Generate all samples needed for 2D slice plots.
        Returns a list of dicts mapping parameter -> value.
        """
        if self.batch_number > 0:
            self.make_plots()
            return None
        elif self.batch_number == 0:
            return self.get_samples()
        
    def get_samples(self):
        d = len(self.parameters)
        # number of unique parameter pairs
        n_pairs = d * (d - 1) // 2
        # samples per pair = res^2
        total_samples = n_pairs * (self.res ** 2)
        print(f'[SliceSampler] DIM {d}, RES {self.res}, N SAMPLES {total_samples}')
        
        if total_samples > self.budget:
            raise RuntimeError(
                f"Requested {total_samples} samples exceeds budget={self.budget}. "
                f"Reduce resolution or number of parameters."
            )
        self.budget = total_samples
        samples = []
        # loop over all unique parameter pairs
        
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

    def make_plots(self, dots_x=None):
        self.plot_slices_from_dataset(dots_x=dots_x)
        self.plot_full_grid(dots_x=dots_x)
    
    def plot_slices_from_dataset(self, cmap='viridis', surface_alpha=0.9, dataset_path=None, df=None, dots_x=None):
        """
        Load enchanted_dataset.csv and plot 2D contours + 3D surfaces.
        """
        if df is not None:
            pass
        elif dataset_path is not None:
            dataset_path = dataset_path
            df = pd.read_csv(dataset_path)
        else:
            if not self.base_run_dir:
                raise RuntimeError("base_run_dir must be set to load dataset.")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"{dataset_path} not found.")
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

            
        output_col = [c for c in df.columns if 'output' in c]
        if len(output_col) != 1:
            raise RuntimeError("Dataset must contain exactly one output column.")
        ycol = output_col[0]
        ymin, ymax = df[ycol].min(), df[ycol].max()

        d = len(self.parameters)
        
        for i in range(d):
            xi_lin = np.linspace(self.bounds[i][0], self.bounds[i][1], self.res)
            for j in range(i+1, d):
                yi_lin = np.linspace(self.bounds[j][0], self.bounds[j][1], self.res)
                Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                Z = np.zeros_like(Xi)
                # reconstruct Z from dataset
                for u in range(self.res):
                    for v in range(self.res):
                        mask = (
                            (np.isclose(df[self.parameters[i]], Xi[u,v], atol=1e-6)) &
                            (np.isclose(df[self.parameters[j]], Yi[u,v], atol=1e-6))
                        )
                        vals = df.loc[mask, ycol].values
                        if vals[0] == 0.0:
                            Z[u,v] = np.nan
                        else:
                            Z[u,v] = vals[0] if len(vals) else np.nan
    
                import matplotlib.gridspec as gridspec

                fig = plt.figure(figsize=(10,4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.05], figure=fig)
                # gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])  # equal widths

                ax1 = fig.add_subplot(gs[0])
                # pos = ax1.get_position()   # [left, bottom, width, height]
                # ax1.set_position([pos.x0, pos.y0, pos.width*0.7, pos.height])  # shrink width
                cs = ax1.contourf(Xi, Yi, Z, cmap=cmap, vmin=ymin, vmax=ymax)
                ax1.set_aspect('equal')       # equal data scaling
                ax1.set_box_aspect(1)         # force the axes box to be square

                # fig.colorbar(cs, ax=ax1)
                ax1.set_xlabel(self.parameters[i])
                ax1.set_ylabel(self.parameters[j])

                ax3d = fig.add_subplot(gs[1], projection='3d')
                ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha, vmin=ymin, vmax=ymax)
                ax3d.set_xlabel(self.parameters[i])
                ax3d.set_ylabel(self.parameters[j])
                ax3d.set_zlabel(ycol)
                z_floor = float(np.nanmin(Z)) if np.isfinite(np.nanmin(Z)) else 0
                if dots_x:
                    ax3d.scatter(dots_x.T[i], dots_x.T[j], zs=z_floor, zdir='z', s=15, c='k', alpha=0.6, depthshade=False)
                fname = f"slices_{self.parameters[i]}_{self.parameters[j]}.png"                

                cax = fig.add_subplot(gs[2])   # colorbar axis
                cs = ax1.contourf(Xi, Yi, Z, cmap=cmap, vmin=ymin, vmax=ymax)
                if dots_x:
                    ax1.scatter(dots_x.T[i], dots_x.T[j])
                fig.colorbar(cs, cax=cax)
                fig.tight_layout()
                fig.savefig(os.path.join(self.base_run_dir, fname))
                plt.close(fig)

    def plot_full_grid(self, cmap='viridis', surface_alpha=0.9, dataset_path=None, df=None, name=''):
                
        if df is not None:
            pass
        elif dataset_path is not None:
            dataset_path = dataset_path
            df = pd.read_csv(dataset_path)
        else:
            if not self.base_run_dir:
                raise RuntimeError("base_run_dir must be set to load dataset.")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"{dataset_path} not found.")
            dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        if len(output_col) != 1:
            raise RuntimeError("Dataset must contain exactly one output column.")
        ycol = output_col[0]
        ymin, ymax = df[ycol].min(), df[ycol].max()

        d = len(self.parameters)
        base_size = 6
        fig, axes = plt.subplots(d, d, figsize=(base_size*d, base_size*d),
                                subplot_kw={'projection': None})

        for i in range(d):
            a, b = self.bounds[i]
            xi_lin = np.linspace(a, b, self.res)
            for j in range(d):
                a, b = self.bounds[j]
                yi_lin = np.linspace(a, b, self.res)
                ax = axes[i, j]
                if i == j:
                    # --- Diagonal: 1D slice with others fixed at nearest midpoint ---
                    p = self.parameters[i]
                    # Precompute nearest midpoint values for all other parameters
                    fixed_assignments = {}
                    for k, q in enumerate(self.parameters):
                        if k == i:
                            continue
                        mid = 0.5 * (self.bounds[k][0] + self.bounds[k][1])
                        vals_q = df[q].unique()
                        if vals_q.size > 0:
                            fixed_assignments[q] = float(vals_q[np.argmin(np.abs(vals_q - mid))])
                        else:
                            fixed_assignments[q] = mid

                    yvals = []
                    for xv in xi_lin:
                        # snap xv to nearest available value in dataset for p
                        vals_p = df[p].unique()
                        if vals_p.size > 0:
                            xv_closest = float(vals_p[np.argmin(np.abs(vals_p - xv))])
                        else:
                            xv_closest = xv

                        # now find the single closest row in the dataset across all parameters
                        diffs = np.abs(df[p] - xv_closest)
                        for q, val in fixed_assignments.items():
                            diffs += np.abs(df[q] - val)
                        idx = np.argmin(diffs.values)
                        yvals.append(float(df.iloc[idx][ycol]))
                    yvals = np.array(yvals)
                    yvals[yvals==0.0] = np.nan
                    ax.plot(xi_lin, yvals, '-', color='tab:blue', linewidth=1.5)
                    ax.set_xlabel(p)
                    ax.set_ylabel(ycol)
                    ax.grid(True, alpha=0.3)

                elif i < j:
                    # --- Upper triangle: 2D contour ---
                    Xi, Yi = np.meshgrid(xi_lin, yi_lin)
                    Z = np.zeros_like(Xi)
                    for u in range(self.res):
                        for v in range(self.res):
                            mask = (
                                (np.isclose(df[self.parameters[i]], Xi[u,v], atol=1e-6)) &
                                (np.isclose(df[self.parameters[j]], Yi[u,v], atol=1e-6))
                            )
                            vals = df.loc[mask, ycol].values
                            if vals[0] == 0.0:
                                Z[u,v] = np.nan
                            else:
                                Z[u,v] = vals[0] if len(vals) else np.nan
                    cs = ax.contourf(Xi, Yi, Z, cmap=cmap, vmin=ymin, vmax=ymax)
                    ax.set_box_aspect(1)   # matplotlib ≥ 3.3
                    ax.set_xlabel(self.parameters[i])
                    ax.set_ylabel(self.parameters[j])
                    if dots_x:
                        ax.scatter(dots_x.T[i], dots_x.T[j])
                else:
                    # --- Lower triangle: 3D surface ---
                    ax.remove()  # remove 2D axis
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
                            if vals[0] == 0.0:
                                Z[u,v] = np.nan
                            else:
                                Z[u,v] = vals[0] if len(vals) else np.nan
                    ax3d.plot_surface(Xi, Yi, Z, cmap=cmap, alpha=surface_alpha, vmin=ymin, vmax=ymax)
                    z_floor = float(np.nanmin(Z)) if np.isfinite(np.nanmin(Z)) else 0
                    if dots_x:
                        ax3d.scatter(dots_x.T[i], dots_x.T[j], zs=z_floor, zdir='z', s=15, c='k', alpha=0.6, depthshade=False)

                    ax3d.set_xlabel(self.parameters[i])
                    ax3d.set_ylabel(self.parameters[j])
                    ax3d.set_zlabel(ycol)
        
        # Add a new axes for the colorbar at custom coordinates
        cbar_ax = fig.add_axes([0.01, 0.02, 0.02, 0.7])  # [x, y, width, height]
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Define the global normalization you want
        norm = mcolors.Normalize(vmin=ymin, vmax=ymax)

        # Use the same colormap as your plots (pick one, or enforce consistency)
        cmap = cs.cmap   # or explicitly: plt.get_cmap("viridis")

        # Create a dummy ScalarMappable
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])   # required placeholder
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(ycol)

        # fig.tight_layout(pad=3.0)
        fig.subplots_adjust(wspace=0.4, hspace=0.2)
        fig.savefig(os.path.join(self.base_run_dir, name+"slices_full_grid.png"), dpi=300)
 
        plt.close(fig)

    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None

if __name__ == "__main__":
    import sys
    from enchanted_surrogates.utils.load_configuration import load_from_dir
    _, base_run_dir = sys.argv
    config = load_from_dir(base_run_dir)
    sampler_config = config.executor['sampler_config']
    sampler_config['base_run_dir'] = base_run_dir
    sampler = SlicesSampler2D(**sampler_config)
    sampler.make_plots()