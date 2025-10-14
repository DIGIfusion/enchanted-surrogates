import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# from .base import Runner
import warnings
import re
import os
import shutil
from scipy.stats.qmc import Sobol


from dask.distributed import print
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats.qmc import Sobol
import warnings
import re
import os
import shutil
from .base_runner import Runner

class MaxMultipleGaussiansRunner(Runner):
    """
    Runner class for evaluating the MaxOfManyGaussians function.

    Parameters:
    - num_dim (int): Number of input dimensions.
    - bounds (list of [float, float]): Bounds for each input dimension.
    - scale (float, optional): Scaling factor for the output.
    - round (int, optional): Decimal rounding for output.
    - rm_run_dir (bool, optional): Whether to delete run directory after execution.
    - random_gaussians (dict, optional): Parameters to randomly generate Gaussian components.
    - specify_gaussians (dict, optional): Explicitly specify Gaussian means and stds.
    """
    def __init__(self, *args, **kwargs):
        num_dim = kwargs['num_dim']
        bounds = kwargs['bounds']
        self.mmg = MaxOfManyGaussians(num_dim, bounds)
        self.mmg.scale = kwargs.get('scale',1)
        self.mmg.round = kwargs.get('round', None)
        self.rm_run_dir = kwargs.get('rm_run_dir', False)
        if 'random_gaussians' in kwargs:
            num_gaussians = kwargs['random_gaussians'].get('num_gaussians', 2)
            seed = kwargs['random_gaussians'].get('seed', None)
            self.mmg.num_flat_dim = kwargs['random_gaussians'].get('num_flat_dim', 0)
            kwargs['random_gaussians']['num_dim'] = num_dim
            mean_bounds = kwargs['random_gaussians'].get('mean_bounds', bounds)
            std_bounds = kwargs['random_gaussians'].get('std_bounds', bounds)
            self.mmg.random_gaussians(num_dim-self.mmg.num_flat_dim, num_gaussians, bounds, mean_bounds, std_bounds, seed)
        elif 'specify_gaussians' in kwargs:
            means = kwargs['specify_gaussians']['means']
            stds = kwargs['specify_gaussians']['stds']
            self.mmg.num_flat_dim = len(bounds)-len(means)
            self.mmg.specify_gaussians(means, stds)

    def single_code_run(self, run_dir: str=None, params:dict=None, *args, **kwargs):
        """
        Evaluate the function at a given parameter dictionary.

        Parameters:
        - run_dir (str, optional): Directory to simulate output file writing.
        - params (dict): Dictionary of parameter values keyed by 'x0', 'x1', ..., 'xn'.

        Returns:
        - str: Comma-separated input values and function output.
        """
        if params is None:
            raise ValueError("params cannot be None")

        if run_dir is not None:
            warnings.warn("The MMMGrunner does not need to read or write to file, yet a run_dir was passed.")
            function_out_file_path = os.path.join(run_dir, 'function_out.txt')
            with open(function_out_file_path, 'w') as file:
                file.write('THIS IS WHERE THE SIMULATION WOULD PUT SOME OUTPUT')

        def find_first_int(string):
            return int(re.search(r'\d+', string).group())

        keys = list(params.keys())
        ordinals = [find_first_int(key) for key in keys]
        indicies = np.argsort(ordinals)
        keys = [keys[index] for index in indicies]
        pos = np.array([params[key] for key in keys])

        params_values = ','.join([str(v) for v in params.values()])
        out = self.mmg.evaluate(pos)

        if self.rm_run_dir:
            shutil.rmtree(run_dir)
        return params_values + ',' + str(out)

class MaxOfManyGaussians():
    """
    Defines a multimodal function composed of multiple multivariate Gaussian distributions.

    Parameters:
    - num_dim (int): Number of input dimensions.
    - bounds (list of [float, float]): Bounds for each input dimension.
    """
    def __init__(self, num_dim, bounds):
        self.num_dim = num_dim
        self.bounds = bounds
        self.all_means = None
        self.all_stds = None
        self.num_flat_dim = 0
        self.scale = 1
        self.round = None

    def specify_gaussians(self, means, stds):
        """
        Manually specify Gaussian components.

        Parameters:
        - means (list of list): Mean vectors for each Gaussian.
        - stds (list of list): Standard deviations for each Gaussian.
        """
        self.all_means = np.array(means)
        self.all_stds = np.array(stds)
        self.gaussians = []
        for mean, std in zip(means, stds):
            cov = np.diag(np.array(std)**2)
            self.gaussians.append(multivariate_normal(mean, cov))

    def random_gaussians(self, num_dim, num_gaussians, bounds=None, mean_bounds=None, std_bounds=None, seed=42):
        """
        Randomly generate Gaussian components.

        Parameters:
        - num_dim (int): Number of dimensions for Gaussians.
        - num_gaussians (int): Number of Gaussian components.
        - bounds, mean_bounds, std_bounds (list): Bounds for sampling.
        - seed (int): Random seed.
        """
        self.num_dim = num_dim
        self.num_gaussians = num_gaussians
        self.bounds = np.array(bounds)
        self.std_bounds = np.array(std_bounds) if std_bounds is not None else self.bounds.copy()
        self.mean_bounds = mean_bounds if mean_bounds is not None else self.bounds
        self.rg = np.random.default_rng(seed=seed)
        self.gaussians = self.generate_gaussians()

    def generate_gaussians(self):
        gaussians = []
        all_means = []
        all_stds = []
        n_gaussian_dim = len(self.bounds) - self.num_flat_dim
        for _ in range(self.num_gaussians):
            mean = [self.rg.uniform(*b, 1)[0] for b in self.mean_bounds[0:n_gaussian_dim]]
            cov_bounds = self.std_bounds**2
            cov_bounds = cov_bounds[0:n_gaussian_dim].T
            cov = np.diag(self.rg.uniform(*cov_bounds, (self.num_dim, self.num_dim)))
            gaussians.append(multivariate_normal(mean, cov))
            all_means.append(mean)
            all_stds.append(np.sqrt(cov))
        self.all_means = np.array(all_means)
        self.all_stds = np.array(all_stds)
        return gaussians

    def evaluate(self, pos):
        """
        Evaluate the function at a given position.

        Parameters:
        - pos (array-like): Input vector or batch of vectors.

        Returns:
        - float or np.ndarray: Function value(s).
        """
        pos = np.array(pos)
        Z = [g.pdf(pos[..., :len(self.bounds) - self.num_flat_dim]) for g in self.gaussians]
        Zmax = np.max(np.stack(Z), axis=0)

        flat_portion = 0
        if self.num_flat_dim > 0:
            flat_portion = np.sum(pos[..., -self.num_flat_dim:]) * 10**-3

        out = (Zmax + flat_portion) * self.scale
        return np.round(out, self.round) if self.round is not None else out

    def get_expectation(self, num_eval_power_of_2=14, do_gaussian=False):
        """
        Estimate the expected value of the function using Sobol sampling.

        Parameters:
        - num_eval_power_of_2 (int): Number of samples = 2^power.
        - do_gaussian (bool): Whether to use truncated Gaussian sampling.

        Returns:
        - float: Estimated expectation.
        """
        dim = len(self.bounds)
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]
        sobol = Sobol(d=dim, scramble=False)
        points = sobol.random_base2(m=num_eval_power_of_2)

        if do_gaussian:
            from scipy.stats import truncnorm
            a, b = -2, 2
            trunc_gaussian = truncnorm(a=a, b=b, loc=0, scale=1)
            points = trunc_gaussian.ppf(points)
            mu, sigma = 0.5, 0.5
            points = mu + sigma * points
            gauss_min = mu + sigma * a
            gauss_max = mu + sigma * b
            scaled_points = lower_bounds + (points - gauss_min) / (gauss_max - gauss_min) * (upper_bounds - lower_bounds)
        else:
            scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)

        return np.mean(self.evaluate(scaled_points))

if __name__ == '__main__':
    # Example usage
    num_dim = 12
    bounds = [[0, 1]] * num_dim
    random_gaussians = {
        'num_gaussians': 3,
        'seed': 42
    }

    runner = MMMGrunner(num_dim=num_dim, bounds=bounds, random_gaussians=random_gaussians)

    # Evaluate at midpoint

class MMMGrunner():
    def __init__(self, *args, **kwargs):
        num_dim = kwargs['num_dim']
        bounds = kwargs['bounds']
        self.mmg = MaxOfManyGaussians(num_dim, bounds)
        self.mmg.scale = kwargs.get('scale',1)
        self.mmg.round = kwargs.get('round', None)
        self.rm_run_dir = kwargs.get('rm_run_dir', False)
        if 'random_gaussians' in kwargs:
            # the last bounds will be used as flat dim if flat_dim>0
            num_gaussians = kwargs['random_gaussians'].get('num_gaussians', 2)
            seed = kwargs['random_gaussians'].get('seed', None)
            self.mmg.num_flat_dim = kwargs['random_gaussians'].get('num_flat_dim', 0)
            kwargs['random_gaussians']['num_dim'] = num_dim
            # gaussian_bounds = bounds[0:num_dim-self.mmg.num_flat_dim]
            mean_bounds = kwargs['random_gaussians'].get('mean_bounds', bounds)
            std_bounds = kwargs['random_gaussians'].get('std_bounds', bounds)
            print('debug std bounds', std_bounds)
            self.mmg.random_gaussians(num_dim-self.mmg.num_flat_dim, num_gaussians, bounds, mean_bounds, std_bounds, seed)
            # self.mmg.random_gaussians(**kwargs['random_gaussians'])
            
        elif 'specify_gaussians' in kwargs:
            # if you want flat dim then just specify more bounds than means and stds
            means = kwargs['specify_gaussians']['means']
            stds = kwargs['specify_gaussians']['stds']
            self.mmg.num_flat_dim = len(bounds)-len(means)
            self.mmg.specify_gaussians(means, stds)
            
    def single_code_run(self, run_dir: str=None, params:dict=None, *args, **kwargs):
        if type(params)==(None):
            raise ValueError('''
                             params cannot be None: The MMMGrunner needs to be provided parameters. It is a python function that does not read from input files
                             ''')
        
        if run_dir != None:
            warnings.warn('''
                             The MMMGrunner does not need to read or write to file, yet a run_dir was passed. 
                             ''')
            function_out_file_path = os.path.join(run_dir, 'function_out.txt')
            #Just make a file like a real simulation would. For testing purposes
            with open(function_out_file_path, 'w') as file:
                file.write('THIS IS WHERE THE SIMULATION WOULD PUT SOME OUTPUT')
                
                    
        def find_first_int(string):
            return int(re.search(r'\d+',string).group())
        
        keys = list(params.keys())
        ordinals = [find_first_int(key) for key in keys]
        indicies = np.argsort(ordinals)
        keys = [keys[index] for index in indicies]
        pos = np.array([params[key] for key in keys])
        
        params_values = ','.join([str(v) for v in params.values()])
        out = self.mmg.evaluate(pos)
        
        if self.rm_run_dir:
            shutil.rmtree(run_dir)
        return params_values + ',' + str(out)
        
        
class MaxOfManyGaussians():
    def __init__(self, num_dim, bounds):
        self.num_dim = num_dim
        self.bounds = bounds
        self.all_means = None
        self.all_stds = None
        self.num_flat_dim=0
        self.scale=1
        self.grad_scale=None
    def specify_gaussians(self, means, stds):
        # print('SET GAUSSIANS HAVE-:\nMEANS OF:', means, '\n','STDs:', stds)
        self.all_means = np.array(means)
        self.all_stds = np.array(stds)
        self.gaussians = []
        for mean, std in zip(means, stds):
            cov = np.diag(np.array(std)**2)
            self.gaussians.append(multivariate_normal(mean, cov))   

    def random_gaussians(self, num_dim, num_gaussians, bounds=None, mean_bounds=None, std_bounds=None, seed=42):
        print('random gaussians', '+'*100)
        self.num_dim = num_dim
        self.num_gaussians = num_gaussians
        
        if type(bounds) != type(None):
            self.bounds = np.array(bounds)
        
        if type(std_bounds) == type(None):
            self.std_bounds = self.bounds.copy()
        else:
            self.std_bounds = np.array(std_bounds)
            
        if type(mean_bounds) == type(None):
            self.mean_bounds=self.bounds
        else: self.mean_bounds = mean_bounds
                
        self.rg = np.random.default_rng(seed=seed)
        self.gaussians = self.generate_gaussians()
    def generate_gaussians(self):
        gaussians = []
        # Generate multiple Gaussians
        all_means = []
        all_stds = []
        n_gaussian_dim = len(self.bounds) - self.num_flat_dim
        for i, _ in enumerate(range(self.num_gaussians)):
            mean = []
            for b in self.mean_bounds[0:n_gaussian_dim]:
                mean.append(self.rg.uniform(*b, 1)[0])
            mean = np.array(mean)
            cov_bounds = self.std_bounds**2
            cov_bounds = cov_bounds[0:n_gaussian_dim]
            cov_bounds = cov_bounds.T
            # cov = self.rg.uniform(*cov_bounds, (self.num_dim, self.num_dim))
            # cov = np.dot(cov, cov.T)  # Ensure the covariance matrix is positive semi-definite
            cov = np.diag(self.rg.uniform(*cov_bounds, (self.num_dim,self.num_dim)))
            # mean=np.repeat(0, len(self.std_bounds))
            gaussians.append(multivariate_normal(mean, cov))
            # print(f'RANDOM GAUSSIAN {i} HAS-:\nMEANS OF:', mean, '\n','STDs:', np.sqrt(cov))
            all_means.append(mean)
            all_stds.append(np.sqrt(cov))
        self.all_means = np.array(all_means)
        self.all_stds = np.array(all_stds)
        # print(f'RANDOM GAUSSIANS HAVE-:\nMEANS OF:', all_means, '\n','STDs:', all_stds)
        return gaussians

    def evaluate(self, pos):
        pos = np.array(pos)
        # print('debug pos shape', np.array(pos).shape)
        Z = []
        n_gaussian_dim = len(self.bounds) - self.num_flat_dim
        for g in self.gaussians:
            Z.append(g.pdf(pos[...,0:n_gaussian_dim]))
        
        Z = np.array(Z)
        Zmax = np.max(np.stack(Z), axis = 0)

        if self.round!=None:
            flat_scale = self.round
        else:
            flat_scale = 3
        def flat_func(x):
            # strength_scaling = np.arange(0.1,1,self.num_flat_dim)
            return np.sum(x)*10**-flat_scale
        
        flat_portion = 0
        if self.num_flat_dim > 0:
            flat_portion = flat_func(pos[...,n_gaussian_dim:])
        #set out of bounds to 0
        # for i in range(self.num_dim):
        #     b1 = self.bounds[i][0]
        #     b2 = self.bounds[i][1]
        #     if isinstance(Zmax, np.ndarray):
        #         p = pos[...,i]
        #         Zmax[(p < b1) | (p > b2)] = 0
        #     elif isinstance(Zmax, int):
        #         if pos[i] > b1 or pos[i] < b2: 
        #             Zmax = 0
        out = (Zmax + flat_portion) * self.scale
        if self.round != None:
            return np.round(out, self.round)
        else:
            return out
    
    def plot_slices(self, grid_size=200, nominals=None, not_vectorised=False, function=None, dimension_labels=None, compare_function=None):
        if type(nominals) == type(None):
            nominals = [np.mean(b) for b in self.bounds]
        h=3
        w=3
        r=1
        c=len(self.bounds)
        fig, AX = plt.subplots(r,c,figsize=(w*c, h*r), dpi = 200, sharey=True)
        self.slices = []
        for i, b in enumerate(self.bounds):
            p = np.stack([nominals for i in range(grid_size)])
            x = np.linspace(b[0],b[1], grid_size)
            p[:,i] = x
            if not_vectorised:
                y = []
                for pi in p:
                    y.append(self.evaluate(pi))
            else:
                if type(function)!=type(None):
                    y = function(p)
                else:
                    y = self.evaluate(p)
                    if compare_function != None:
                        y_compare = compare_function(p)
            self.slices.append((x,y))
            AX[i].plot(x,y, color ='black')
            if compare_function != None:
                AX[i].plot(x,y_compare, color ='blue', label='Comparison Function')
            # AX[i].legend()
            if type(dimension_labels) != type(None):
                AX[i].set_xlabel(dimension_labels[i])
            else:
                AX[i].set_xlabel(f'dimension {i}')
            AX[i].set_ylabel('function value')
            fig.tight_layout()
            fig.show()
        return fig

    def get_slices(self, *args, **kwargs):
        self.plot_slices(*args, **kwargs)
        return self.slices

    def plot_matrix_contour(self, points=None, function=None, dimension_labels=None):
        w=2
        h=2
        dim = len(self.bounds)
        figure, AX = plt.subplots(dim, dim, figsize=(w*dim, h*dim), sharex=True, sharey=True)
        for i in range(dim):
            for j in range(dim):
                if j>=i:
                    figure.delaxes(AX[i,j])
                    # break
                else:
                    self.plot_2D_of_many(which2=(j,i), points=points ,ax=AX[i,j], style='contour', grid_size=50, function=function)
                    if j==0:
                        if type(dimension_labels) != type(None):
                            AX[i,j].set_ylabel(dimension_labels[i])
                        else:
                            AX[i,j].set_ylabel(f'{i}')
                    if i==dim-1:
                        if type(dimension_labels) != type(None):
                            AX[i,j].set_xlabel(dimension_labels[j])
                        else:
                            AX[i,j].set_xlabel(f'{j}')
        figure.show()
        return figure

        
    def plot_2D_of_many(self, which2, points=None, extra=0, plot_bounds=None, nominals=None, grid_size=100, style='3D', ax=None, function=None):
        # which2 is a sequence that specifies which dimensions to plot, the rest are kept nominal, example which2 = (0,2) to plot the 1st and 3rd dimensions. 
        if type(points)!=type(None):
            points = np.array(points) # assumes shape num_points,num_dim
            points_2d = np.array([points.T[which2[0]],points.T[which2[1]]])
            
        if plot_bounds == None:
            plot_bounds = self.bounds
        if type(nominals) == type(None):
            nominals = [np.mean(b) for b in self.bounds]
        
        xlow, xhigh = plot_bounds[which2[0]][0]-extra, plot_bounds[which2[0]][1]+extra
        ylow, yhigh = plot_bounds[which2[1]][0]-extra, plot_bounds[which2[1]][1]+extra
        x = np.linspace(xlow, xhigh, grid_size)
        y = np.linspace(ylow, yhigh, grid_size)
        X, Y = np.meshgrid(x, y)
        
        arrays2stack = []
        for i, n in enumerate(nominals):
            arrays2stack.append(np.full_like(X,n))
        arrays2stack[which2[0]] = X
        arrays2stack[which2[1]] = Y
        pos = np.vstack(np.dstack(arrays2stack))
        
        # print(pos.shape)
        Z = np.zeros(shape=(grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                p = nominals
                p[which2[0]] = X[i,j]
                p[which2[1]] = Y[i,j]
                if type(function) == type(None):
                    Z[i,j] = self.evaluate(p)
                else:
                    Z[i,j] = function(p)

        if style == '3D':
            if type(ax) == type(None):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')    
            ax.plot_surface(X, Y, Z, cmap='viridis')
        else:
            if type(ax) == type(None):
                fig = plt.figure(figsize=(2,2), dpi=200)
                ax = fig.add_subplot(111)
            ax.contour(X,Y,Z, cmap='viridis')
            if type(points)!=type(None):
                ax.scatter(points_2d[0], points_2d[1], marker='+', color='black')
        
        if type(ax) == type(None):
            ax.set_xlabel(f'{which2[0]}')
            ax.set_ylabel(f'{which2[1]}')
            fig.show()
            
    def plot_2d_gaussians(self, ax, grid_size=100, onlyContour=False, plot_bounds=None, extra=0, sample_points=None, title=None):
        if plot_bounds == None:
            plot_bounds = self.bounds
        if self.num_dim != 2:
            raise ValueError('Daniel Says: n_dim must equil 2')
        xlow, xhigh = plot_bounds[0][0]-extra, plot_bounds[0][1]+extra
        ylow, yhigh = plot_bounds[1][0]-extra, plot_bounds[1][1]+extra
        x = np.linspace(xlow, xhigh, grid_size)
        y = np.linspace(ylow, yhigh, grid_size)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
                
        
        Zmax = self.evaluate(pos)
        
        z = []
        yy = 0.5
        x_at_y = [(xi, yy) for xi in x]
        for g in self.gaussians:
            z.append(g.pdf(x_at_y))
        zmax = np.max(np.stack(z), axis = 0)
        #slice
        if not onlyContour:
            plt.figure()
            plt.plot(x, zmax)
            plt.show()
            
            fig = plt.figure()
            ax_3d = fig.add_subplot(111, projection='3d')
            ax_3d.plot_surface(X, Y, Zmax, cmap='viridis')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Function Value')
            # ax_3d.set_title('2D Surface of Multimodal Multivariate Gaussian Distribution')
            ax_3d.view_init(elev=30, azim=30-90)
                    
        if title != None:
            ax.set_title(title)
        if type(sample_points) != type(None):
            ax.scatter(*sample_points, marker='.')
        
        ax.contour(X,Y,Zmax)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    def get_expectation(self, num_eval_power_of_2=14, do_gaussian=False):
        # Define the dimensionality
        dim = len(self.bounds)  # Change this for the number of dimensions

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        sobol = Sobol(d=dim, scramble=False)

        print(f'''
              GENERATING SOBOL SEQUENCE SAMPLES, NUM SAMPLES: {2**num_eval_power_of_2}\n
              BOUNDS:{self.bounds}''')
        
        # Generate points in the unit hypercube [0, 1]^d
        points = sobol.random_base2(m=num_eval_power_of_2)  # Generates 2^power points
        
        if do_gaussian:
            from scipy.stats import norm, truncnorm
    
            # Define bounds and standard normal parameters
            a, b = -2, 2  # in standard normal units
            trunc_gaussian = truncnorm(a=a, b=b, loc=0, scale=1)
            points = trunc_gaussian.ppf(points)
            mu = 0.5
            sigma = 0.5
            points = mu + sigma * points
                        
            # Now scale to custom bounds
            gauss_min = mu + sigma * a  # -0.5
            gauss_max = mu + sigma * b  # 1.5

            scaled_points = lower_bounds + (points - gauss_min) / (gauss_max - gauss_min) * (upper_bounds - lower_bounds)
        else:
            # Scale the points to the desired bounds
            scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)
        return np.mean(self.evaluate(scaled_points))
        
if __name__=='__main__':
    # Example 1: Create a runner with 12 dimensions and 3 random Gaussians
    num_dim = 12
    bounds = [[0,1]] * num_dim
    random_gaussians = {
        'num_gaussians': 3,
        'seed': 42
    }

    runner = MMMGrunner(num_dim=num_dim, bounds=bounds, random_gaussians=random_gaussians)

    # Example 2: Evaluate the function at a specific point
    params = {f'x{i}': 0.5 for i in range(num_dim)}
    result = runner.single_code_run(params=params)
    print("Function evaluation:", result)

    # Example 3: Estimate the expectation using Sobol sampling
    expectation = runner.mmg.get_expectation(num_eval_power_of_2=10)
    print("Estimated expectation:", expectation)

    # Example 4: Plot slices of the function
    fig = runner.mmg.plot_slices()
    fig.savefig('data_store/slices_plot.png')
    
    # Example: Using specified Gaussians
    num_dim = 2
    bounds = [[0, 1], [0, 1]]

    # Define two Gaussian components manually
    means = [
        [0.3, 0.3],
        [0.7, 0.7]
    ]
    stds = [
        [0.05, 0.05],
        [0.1, 0.1]
    ]

    runner = MMMGrunner(
        num_dim=num_dim,
        bounds=bounds,
        specify_gaussians={
            'means': means,
            'stds': stds
        },
        scale=1.0,
        round=4
    )

    # Evaluate at a specific point
    params = {'x0': 0.5, 'x1': 0.5}
    result = runner.single_code_run(params=params)
    print("Function evaluation at center:", result)

    # Plot slices
    fig = runner.mmg.plot_slices()
    fig.savefig('specified_gaussians_slices.png')

    # Estimate expectation
    expectation = runner.mmg.get_expectation(num_eval_power_of_2=10)
    print("Estimated expectation:", expectation)


