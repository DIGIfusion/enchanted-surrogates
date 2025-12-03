import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams.update({'font.size': 10})

from enchanted_surrogates.samplers.base_sampler import Sampler

class MainDiagonalSampler(Sampler):
    def __init__(self, parameters, bounds, base_run_dir=None, res=50, fixed=None, budget=100000, type='MainDiagonalSampler'):
        self.parameters = parameters
        self.bounds = bounds
        self.base_run_dir = base_run_dir
        self.res = res
        self.budget = res
        # default fixed values for non-slice dimensions
        self.batch_number = 0
        
        # store scaling factors
        self._lb = np.array([b[0] for b in self.bounds], dtype=float)
        self._ub = np.array([b[1] for b in self.bounds], dtype=float)
        self._range = self._ub - self._lb

    def to_unit(self, X):
        """Map real-bounds inputs to [0,1]."""
        return (np.asarray(X) - self._lb) / self._range
    
    def from_unit(self, X_unit):
        """Map unit inputs back to real bounds."""
        return self._lb + np.asarray(X_unit) * self._range
    
    def get_next_samples(self):
        """
        Generate all samples needed for 2D slice plots.
        Returns a list of dicts mapping parameter -> value.
        """
        if self.batch_number > 0:
            self.make_plots()
            return None
        elif self.batch_number == 0:
            d = len(self.parameters)
            samples = []
            coordiante = np.linspace(0,1,self.res)
            for co in coordiante:
                X_unit = np.repeat(co, d)
                X_real = self.from_unit(X_unit)
                sample = {parameter: X_real[i] for i, parameter in enumerate(self.parameters)}
                samples.append(sample)

            self.batch_number += 1
            return samples

    def make_plots(self):
        self.plot_main_diagonal()

    def plot_main_diagonal(self):
        """
        Load enchanted_dataset.csv and plot 2D contours + 3D surfaces.
        """
        if not self.base_run_dir:
            raise RuntimeError("base_run_dir must be set to load dataset.")
        dataset_path = os.path.join(self.base_run_dir, "enchanted_dataset.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{dataset_path} not found.")

        df = pd.read_csv(dataset_path)

        output_col = [c for c in df.columns if 'output' in c]
        if len(output_col) != 1:
            raise RuntimeError("Dataset must contain exactly one output column.")
        ycol = output_col[0]

        X_real = df[self.parameters].to_numpy()
        X_unit = self.to_unit(X_real)
        print('debug x_unit', X_unit[0:3])
        x = np.array([X_unit_i[0] for X_unit_i in X_unit])
        y = df[ycol].to_numpy()
        sorted_ind = np.argsort(x)        
        x = x[sorted_ind]
        y = y[sorted_ind]
        fig = plt.figure()
        plt.plot(x,y)
        plt.title('main diagonal')
        plt.xlabel('main diagonal normalised co-ordinate')        
        plt.ylabel(ycol)
        fig.savefig(os.path.join(self.base_run_dir,'main_diagonal.png'))
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
    sampler = MainDiagonalSampler(**sampler_config)
    sampler.make_plots()