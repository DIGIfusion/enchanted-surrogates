import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enchanted_surrogates.samplers.base_sampler import Sampler

plt.rcParams.update({'font.size': 15})


class SlicesSampler1D(Sampler):
    """
    1D slice sampler:
    For each parameter p_i, generate a 1D sweep of length `res`,
    holding all other parameters fixed at midpoint or user-provided values.
    """

    def __init__(self, parameters, bounds, base_run_dir=None,
                 res=100, fixed=None, budget=100000, type='SlicesSampler1D'):
        self.parameters = parameters
        self.bounds = bounds
        self.base_run_dir = base_run_dir
        self.res = res
        self.budget = budget

        # default fixed values for non-slice dimensions
        self.fixed = fixed or {p: 0.5*(b[0]+b[1]) for p, b in zip(parameters, bounds)}

        self.batch_number = 0

    # ------------------------------------------------------------------
    # SAMPLING
    # ------------------------------------------------------------------

    def get_next_samples(self):
        """
        First call: return samples.
        Second call: make plots and return None.
        """
        if self.batch_number > 0:
            self.make_plots()
            return None
        else:
            return self.get_samples()

    def get_samples(self):
        """
        Generate 1D slices for each parameter.
        Total samples = d * res
        """
        d = len(self.parameters)
        total_samples = d * self.res

        print(f'[SliceSampler1D] DIM {d}, RES {self.res}, N SAMPLES {total_samples}')

        if total_samples > self.budget:
            raise RuntimeError(
                f"Requested {total_samples} samples exceeds budget={self.budget}. "
                f"Reduce resolution or number of parameters."
            )

        self.budget = total_samples
        samples = []

        for i, (p, (a, b)) in enumerate(zip(self.parameters, self.bounds)):
            xi = np.linspace(a, b, self.res)
            for val in xi:
                row = {}
                for q, (qa, qb) in zip(self.parameters, self.bounds):
                    if q == p:
                        row[q] = val
                    else:
                        row[q] = self.fixed[q]
                samples.append(row)

        self.batch_number += 1
        return samples

    def get_samples_array(self):
        samples = self.get_samples()
        df = pd.DataFrame(samples)
        return df[self.parameters].to_numpy()

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------

    def make_plots(self, dots_x=None, predictor=None, save_dir=None, name=''):
        self.plot_slices_from_dataset(
            dots_x=dots_x,
            predictor=predictor,
            save_dir=save_dir,
            name=name
        )
    
    def plot_slices_from_dataset(self, cmap='viridis', dataset_path=None,
                                 df=None, dots_x=None, predictor=None,
                                 save_dir=None, name=''):
        """
        Plot 1D slices: y vs parameter_i.
        If predictor is provided, overlay predicted curve.
        """
        if save_dir is None:
            save_dir = self.base_run_dir

        # Load dataset
        if df is None:
            if dataset_path is None:
                if not self.base_run_dir:
                    raise RuntimeError("base_run_dir must be set to load dataset.")
                dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        if len(output_col) != 1:
            raise RuntimeError(f"Dataset must contain exactly one output column. Found: {output_col}")
        ycol = output_col[0]

        d = len(self.parameters)

        for i, p in enumerate(self.parameters):
            a, b = self.bounds[i]
            xi = np.linspace(a, b, self.res)

            # --- reconstruct dataset slice ---
            vals_p = df[p].unique()
            yvals = []
            for xv in xi:
                xv_closest = float(vals_p[np.argmin(np.abs(vals_p - xv))])
                mask = np.isclose(df[p], xv_closest, atol=1e-6)
                yvals.append(float(df.loc[mask, ycol].iloc[0]))
            yvals = np.array(yvals)
            yvals[yvals == 0.0] = np.nan

            # --- predictor curve ---
            if predictor is not None:
                # Build full-dimensional input array
                Xpred = np.zeros((self.res, d))
                for j, q in enumerate(self.parameters):
                    if j == i:
                        Xpred[:, j] = xi
                    else:
                        Xpred[:, j] = self.fixed[q]

                ypred = predictor(Xpred)
                ypred = np.array(ypred).reshape(-1)

            # --- plotting ---
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xi, yvals, '-', color='tab:blue', linewidth=2, label='dataset')

            if predictor is not None:
                ax.plot(xi, ypred, '--', color='tab:red', linewidth=2, label='predictor')

            ax.set_xlabel(p)
            ax.set_ylabel(ycol)
            ax.grid(True, alpha=0.3)
            ax.legend()

            if dots_x is not None:
                ax.scatter(dots_x, np.full_like(dots_x, np.nanmin(yvals)),
                           s=20, c='k', alpha=0.5)

            fname = f"slice_1D_{p}_{ycol}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, name + fname), dpi=200)
            plt.close(fig)

    # ------------------------------------------------------------------
    # FUTURE REGISTRATION (unused)
    # ------------------------------------------------------------------

    def register_future(self, future):
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
    sampler = SlicesSampler1D(**sampler_config)
    sampler.make_plots()
