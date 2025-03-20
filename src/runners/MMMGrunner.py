import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from .base import Runner
import warnings
import re

class MMMGrunner(Runner):
    def __init__(self, *args, **kwargs):
        num_dim = kwargs['num_dim']
        bounds = kwargs['bounds']
        self.mmg = MaxOfManyGaussians(num_dim, bounds)
        if 'random_gaussians' in kwargs:
            num_gaussians = kwargs['random_gaussians'].get('num_gaussians', 2)
            mean_bounds = kwargs['random_gaussians'].get('mean_bounds', bounds)
            std_bounds = kwargs['random_gaussians'].get('std_bounds', bounds) 
            seed = kwargs['random_gaussians'].get('seed', None)
            self.mmg.random_gaussians(num_dim, num_gaussians, bounds, mean_bounds, std_bounds, seed)

        elif 'specify_gaussians' in kwargs:
            means = kwargs['specify_gaussians']['means']
            stds = kwargs['specify_gaussians']['stds']
            self.mmg.specify_gaussians(means, stds)
    
    def single_code_run(self, run_dir: str=None, out_dir:str=None, params:dict=None):
        if type(params)==(None):
            raise ValueError('''
                             params cannot be None: The MMMGrunner needs to be provided parameters. It is a python function that does not read from input files
                             ''')
        
        if run_dir != None or out_dir != None:
            warnings.warn('''
                             The MMMGrunner does not read or write to file, yet a run_dir or out_dir was passed. These will not be used. 
                             ''') 
        def find_first_int(string):
            return int(re.search(r'\d+',string).group())
        
        keys = params.keys()
        ordinals = [find_first_int(key) for key in keys]
        indicies = np.argsort(ordinals)
        keys = [keys[index] for index in indicies]
        pos = np.array([params[key] for key in keys])
        
        return self.mmg.evaluate(pos), params
        

class MaxOfManyGaussians():
    def __init__(self, num_dim, bounds):
        self.num_dim = num_dim
        self.bounds = bounds
            
    def specify_gaussians(self, means, stds):
        means = np.array(means)
        stds = np.array(stds)
        self.gaussians = []
        for mean, std in zip(means, stds):
            print('debug tuple', type(np.array(std)**2))
            cov = np.diag(np.array(std)**2)
            self.gaussians.append(multivariate_normal(mean, cov))   

    def random_gaussians(self, num_dim, num_gaussians, bounds=None, mean_bounds=None, std_bounds=None, seed=42):
        self.num_dim = num_dim
        self.num_gaussians = num_gaussians
        if type(bounds) == type(None):
            self.bounds = np.repeat((0,1),num_dim)
        else: self.bounds = np.array(bounds)
        if type(std_bounds) == type(None):
            self.std_bounds = np.array([(b[0], b[0]+(b[1]-b[0])/2 ) for b in bounds])
        else: self.std_bounds = np.array(std_bounds)
        if type(mean_bounds) == type(None):
            self.mean_bounds=self.bounds
        else: self.mean_bounds = mean_bounds
        
        self.rg = np.random.default_rng(seed=seed)
        self.gaussians = self.generate_gaussians()
    def generate_gaussians(self):
        gaussians = []
        # Generate multiple Gaussians
        for _ in range(self.num_gaussians):
            mean = []
            for b in self.mean_bounds:
                mean.append(self.rg.uniform(*b, 1)[0])
            mean = np.array(mean)
            cov_bounds = self.std_bounds**2
            cov_bounds = cov_bounds.T
            # cov = self.rg.uniform(*cov_bounds, (self.num_dim, self.num_dim))
            # cov = np.dot(cov, cov.T)  # Ensure the covariance matrix is positive semi-definite
            cov = np.diag(self.rg.uniform(*cov_bounds, (self.num_dim,self.num_dim)))
            gaussians.append(multivariate_normal(mean, cov))
        return gaussians

    def evaluate(self, pos):
        Z = []
        for g in self.gaussians:
            Z.append(g.pdf(pos))
        Z = np.array(Z)
        Zmax = np.max(np.stack(Z), axis = 0)

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
        return Zmax
    
    def plot_slices(self, grid_size=200, nominals=None, not_vectorised=False):
        print('PLOT SLICES')
        if type(nominals) == type(None):
            nominals = [np.mean(b) for b in self.bounds]
        
        h=3
        w=3
        r=1
        c=len(self.bounds)
        fig, AX = plt.subplots(r,c,figsize=(w*c, h*r), dpi = 200)
        for i, b in enumerate(self.bounds):
            p = np.stack([nominals for i in range(grid_size)])
            print('d', p.shape)
            x = np.linspace(b[0],b[1], grid_size)
            p[:,i] = x
            if not_vectorised:
                y_true = []
                for pi in p:
                    print('DEBUG',pi)
                    y_true.append(self.evaluate(pi))
            else:
                y_true = self.evaluate(p)
            AX[i].plot(x,y_true, color ='black', label='True Function')
            AX[i].legend()
            AX[i].set_xlabel(f'dimension {i}')
            AX[i].set_ylabel('function value')
            fig.tight_layout()
            fig.show()

    def plot_matrix_contour(self):
        w=2
        h=2
        figure, AX = plt.subplots(self.num_dim, self.num_dim, figsize=(w*self.num_dim, h*self.num_dim), sharex=True, sharey=True)
        for i in range(self.num_dim):
            for j in range(self.num_dim):
                if j>=i:
                    figure.delaxes(AX[i,j])
                    # break
                else:
                    self.plot_2D_of_many(which2=(j,i),ax=AX[i,j], style='contour', grid_size=50)
                    if j==0:
                        AX[i,j].set_ylabel(f'{i}')
                    if i==self.num_dim-1:
                        AX[i,j].set_xlabel(f'{j}')
        figure.show()

        
    def plot_2D_of_many(self, which2, extra=0, plot_bounds=None, nominals=None, grid_size=100, style='3D', ax=None):
        # which2 is a sequence that specifies which dimensions to plot, the rest are kept nominal, example which2 = (0,2) to plot the 1st and 3rd dimensions. 
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
        pos = np.dstack(arrays2stack)
        
        # print(pos.shape)
        Z = np.zeros(shape=(grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                p = nominals
                p[which2[0]] = X[i,j]
                p[which2[1]] = Y[i,j]
                Z[i,j] = self.evaluate(p)

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
