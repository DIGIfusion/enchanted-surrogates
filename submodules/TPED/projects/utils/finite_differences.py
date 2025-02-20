import numpy as np
from .interpolation_tools import interp
import matplotlib.pyplot as plt

def get_mat_fd_d1_o4(size,dx,plot_matrix=False):
    """Creates matrix for centered finite difference, first derivative, 4th order.
    size: size of (number of elements in) quantity to be differentiated
    dx: grid spacing (for constant grid)."""

    prefactor=1.0/(12.0*dx)
    mat=np.zeros((size,size),dtype='float')    
    for i in range(size):
        if i-1 >= 0:
            mat[i,i-1]=-8
        if i-2 >= 0:
            mat[i,i-2]=1
        if i+1 <= size-1:
            mat[i,i+1]=8
        if i+2 <= size-1:
            mat[i,i+2]=-1
   
    mat=prefactor*mat

    if plot_matrix:
        plt.contourf(mat,50)
        plt.colorbar()
        plt.show()

    return mat

    
def fd_d1_o4(var,grid,mat=False):
    """Centered finite difference, first derivative, 4th order.
    var: quantity to be differentiated.
    grid: grid for var 
    mat: matrix for the finite-differencing operator. if mat=False then it is created"""

    if not mat:
        mat=get_mat_fd_d1_o4(len(var),grid[1]-grid[0])

    dvar=-np.dot(mat,var)
    dvar[0]=0.0
    dvar[1]=0.0
    #dvar[2]=0.0
    dvar[-1]=0.0
    dvar[-2]=0.0
    #dvar[-3]=0.0
    return -dvar 

def fd_d1_o4_uneven(var,grid,mat=False,return_new_grid = False):
    """Centered finite difference, first derivative, 4th order.  Evenly spaced grid is created and var is interpolated onto this grid.  Derivative is interpolated back onto original grid.
    var: quantity to be differentiated.
    grid: grid for var 
    mat: matrix for the finite-differencing operator. if mat=False then it is created"""

    N = 2*len(grid)
    grid0 = np.linspace(grid[0],grid[-1],N)
    var0 = interp(grid,var,grid0)

    if not mat:
        mat=get_mat_fd_d1_o4(len(var0),grid0[1]-grid0[0])

    dvar0=-np.dot(mat,var0)
    dvar0[0]=0.0
    dvar0[1]=0.0
    dvar0[-1]=0.0
    dvar0[-2]=0.0

    if return_new_grid:
        return grid0,-dvar0
    else:
        dvar = np.zeros(len(grid))
        dvar[2:-2] = interp(grid0[2:-2],dvar0[2:-2],grid[2:-2])
        return -dvar 

def fd_d1_o4_smoothend(var,grid,mat=False):
    """Centered finite difference, first derivative, 4th order using extrapolation to get boundary points
    var: quantity to be differentiated.
    grid: grid for var 
    mat: matrix for the finite-differencing operator. if mat=False then it is created"""

    dx = grid[1]-grid[0]
    grid0 = np.linspace(grid[0]-2*dx,grid[-1]+2*dx,len(grid)+4)
    var0 = interp(grid,var,grid0)

    if not mat:
        mat=get_mat_fd_d1_o4(len(var0),grid0[1]-grid0[0])

    dvar0=-np.dot(mat,var0)
    dvar_out=dvar0[2:-2]

    return -dvar_out 
 
    
def invert_fd_d1_o4(var,grid,mat=False):
    """Invert cenntered finite difference, first derivative, 4th order.
    var: quantity to be integrated.
    grid: grid for var 
    mat: matrix for the finite-differencing operator. if mat=False then it is created
    note--mat will be inverted for this operation"""

    if not mat:
        mat=get_mat_fd_d1_o4(len(var),grid[1]-grid[0])

    imat=np.linalg.inv(mat)

    ivar=np.dot(imat,var)
    return ivar 


