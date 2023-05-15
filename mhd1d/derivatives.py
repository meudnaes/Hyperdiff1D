"""
Script for calculating derivatives
"""

# Global imports
import numpy as np

from numba import njit, prange

# derivative constants
c = 3/640
b = (-1 - 120*c)/24
a = 1 - 3*b - 5*c

@njit(parallel=True)
def deriv_x(x, var, shift=-1, bnd_type='wrap'):
    r"""
    6th order Bifrost spatial derivative in x-direction
    
    $$\partial^+_x(f_{i,j,k}) = f'_{i+1/2, j, k} =
        ( a(f_{i, j, k} - f_{i+1, j, k}) +
          b(f_{i-1, j, k} - f_{i+2, j, k}) + 
          c(f_{i-2, j, k} - f_{i+3, j, k}) )/dx$$

    Parameters
    ----------
    x : `1D array`
        x-axis
    var : `1D array`
        variable

    Returns
    -------
    out : `1D array`
        Derivative in x-direction
    """

    assert c == 3/640, "Wrong derivation constants"
    assert b == (-1 - 120*c)/24, "Wrong derivation constants"
    assert a == 1 - 3*b - 5*c, "Wrong derivation constants"

    start = 2 - shift
    end = 3 + shift

    nx = var.shape[0]

    out = np.zeros_like(var)

    # x-axis is defined in opposite direction
    dx = x[0] - x[1]

    # Calculate derivatives in the inner grid points
    for i in prange(start, nx - end):
        out[i] = ( 
                  a*(var[i  +shift] - var[i+1+shift]) +
                  b*(var[i-1+shift] - var[i+2+shift]) +
                  c*(var[i-2+shift] - var[i+3+shift])
                 )

    # Fix derivatives at the edges from the boundary condition
    if bnd_type == 'wrap':
        # Periodic boundaries, wrap around to the other side of the grid
        if shift==0:
            # a*(var[i]-var[i+1]) + b*(var[i-1]-var[i+2]) + c*(var[i-2] - var[i+3])
            
            # Left side
            out[0] = a*(var[0]-var[1]) + b*(var[-1]-var[2]) + c*(var[-2]-var[3])
            out[1] = a*(var[1]-var[2]) + b*(var[0]-var[3]) + c*(var[-1]-var[4])
        
            # Right side
            out[-1] = a*(var[-1]-var[0]) + b*(var[-2]-var[1]) + c*(var[-3]-var[2])
            out[-2] = a*(var[-2]-var[-1]) + b*(var[-3]-var[0]) + c*(var[-4]-var[1])
            out[-3] = a*(var[-3]-var[-2]) + b*(var[-4]-var[-1]) + c*(var[-5]-var[0])
        
        elif shift==-1:
            # a*(var[i-1]-var[i]) + b*(var[i-2]-var[i+1]) + c*(var[i-3] - var[i+2])
            
            # Left side
            out[0] = a*(var[-1]-var[0]) + b*(var[-2]-var[1]) + c*(var[-3]-var[2])
            out[1] = a*(var[0]-var[1]) + b*(var[-1]-var[2]) + c*(var[-2]-var[3])
            out[2] = a*(var[1]-var[2]) + b*(var[0]-var[3]) + c*(var[-1]-var[4])
        
            # Right side
            out[-1] = a*(var[-2]-var[-1]) + b*(var[-3]-var[0]) + c*(var[-4]-var[1])
            out[-2] = a*(var[-3]-var[-2]) + b*(var[-4]-var[-1]) + c*(var[-5]-var[0])
    
    elif bnd_type == 'edge':
        # Fixed boundaries, all points 'outside' the grid are constant
        if shift==0:
            # Left side
            out[0] = a*(var[0]-var[1]) + b*(var[0]-var[2]) + c*(var[0]-var[3])
            out[1] = a*(var[1]-var[2]) + b*(var[0]-var[3]) + c*(var[0]-var[4])
    
            # Right side
            out[-1] =                       b*(var[-2]-var[-1]) + c*(var[-3]-var[-1])
            out[-2] = a*(var[-2]-var[-1]) + b*(var[-3]-var[-1]) + c*(var[-4]-var[-1])
            out[-3] = a*(var[-3]-var[-2]) + b*(var[-4]-var[-1]) + c*(var[-5]-var[-1])
    
        elif shift==-1:
            # Left side
            out[0] =                     b*(var[0]-var[1]) + c*(var[0]-var[2])
            out[1] = a*(var[0]-var[1]) + b*(var[0]-var[2]) + c*(var[0]-var[3])
            out[2] = a*(var[1]-var[2]) + b*(var[0]-var[3]) + c*(var[0]-var[4])
    
            # Right side
            out[-1] = a*(var[-2]-var[-1]) + b*(var[-3]-var[-1]) + c*(var[-4]-var[-1])
            out[-2] = a*(var[-3]-var[-2]) + b*(var[-4]-var[-1]) + c*(var[-5]-var[-1])
    
    return out/dx