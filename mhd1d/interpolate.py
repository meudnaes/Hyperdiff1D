"""
Script for interpolating along axis on the staggered mesh
"""

# Global imports
import numpy as np

from numba import njit, prange

## Interpolation constants
c = 3/256
b = -25/256
a = 1/2 - b - c

@njit(parallel=True)
def x_shift(var, shift=0, bnd_type='wrap'):
    """
    5th-order interpolation operator to shift quantity a half grid point
    in the x-direction
    
    $$T^+_x(f_{i,j,k}) = f_{i+1/2, j, k} =
        a(f_{i, j, k} + f_{i+1, j, k}) +
        b(f_{i-1, j, k} + f_{i+2, j, k}) +
        c(f_{i-2, j, k} + f_{i+3, j, k})$$
    
    Parameters
    ----------
    var : `1D Array`
        Quantity defined at half grid shifted
    shift : `integer`
        Which direction to interpolate variable. Defaults to -1 to shift a half
        grid point to the left. Use 0 to shift a half grid point to the right.

    Returns
    -------
    out : `1D Array`
        Quantity defined at cell center of grid
    """

    assert c == 3/256, "Wrong interpolation constants"
    assert b == -25/256, "Wrong interpolation constants"
    assert a == 1/2 - b - c, "Wrong interpolation constants"

    start = 2 - shift
    end = 3 + shift

    nx = var.shape[0]

    out = np.zeros_like(var)

    # Calculate the interpolations for the inner grid points
    for i in prange(start, nx - end):
        out[i] = ( 
                  a*(var[i  +shift] + var[i+1+shift]) +
                  b*(var[i-1+shift] + var[i+2+shift]) +
                  c*(var[i-2+shift] + var[i+3+shift])
                 )

    # Fix interpolation at the edges from the boundary condition
    if bnd_type == 'wrap':
        # Periodic boundaries, wrap around to the other side of the grid
        if shift==0:
            # a*(var[i]+var[i+1]) + b*(var[i-1]+var[i+2]) + c*(var[i-2]+var[i+3])
            
            # Left side
            out[0] = a*(var[0]+var[1]) + b*(var[-1]+var[2]) + c*(var[-2]+var[3])
            out[1] = a*(var[1]+var[2]) + b*(var[0]+var[3]) + c*(var[-1]+var[4])
        
            # Right side
            out[-1] = a*(var[-1]+var[0]) + b*(var[-2]+var[1]) + c*(var[-3]+var[2])
            out[-2] = a*(var[-2]+var[-1]) + b*(var[-3]+var[0]) + c*(var[-4]+var[1])
            out[-3] = a*(var[-3]+var[-2]) + b*(var[-4]+var[-1]) + c*(var[-5]+var[0])
        
        elif shift==-1:
            # a*(var[i-1]+var[i]) + b*(var[i-2]+var[i+1]) + c*(var[i-3]+var[i+2])
            
            # Left side
            out[0] = a*(var[-1]+var[0]) + b*(var[-2]+var[1]) + c*(var[-3]+var[2])
            out[1] = a*(var[0]+var[1]) + b*(var[-1]+var[2]) + c*(var[-2]+var[3])
            out[2] = a*(var[1]+var[2]) + b*(var[0]+var[3]) + c*(var[-1]+var[4])
        
            # Right side
            out[-1] = a*(var[-2]+var[-1]) + b*(var[-3]+var[0]) + c*(var[-4]+var[1])
            out[-2] = a*(var[-3]+var[-2]) + b*(var[-4]+var[-1]) + c*(var[-5]+var[0])   
    
    elif bnd_type == 'edge':
        # Fixed boundaries, all points 'outside' the grid are constant
        if shift==0:
            # Left side
            out[0] = a*(var[0]+var[1]) + b*(var[0]+var[2]) + c*(var[0]+var[3])
            out[1] = a*(var[1]+var[2]) + b*(var[0]+var[3]) + c*(var[0]+var[4])
    
            # Right side
            out[-1] = a*(2*var[-1])       + b*(var[-2]+var[-1]) + c*(var[-3]+var[-1])
            out[-2] = a*(var[-2]+var[-1]) + b*(var[-3]+var[-1]) + c*(var[-4]+var[-1])
            out[-3] = a*(var[-3]+var[-2]) + b*(var[-4]+var[-1]) + c*(var[-5]+var[-1])
    
        elif shift==-1:
            # Left side
            out[0] = a*(2*var[0])      + b*(var[0]+var[1]) + c*(var[0]+var[2])
            out[1] = a*(var[0]+var[1]) + b*(var[0]+var[2]) + c*(var[0]+var[3])
            out[2] = a*(var[1]+var[2]) + b*(var[0]+var[3]) + c*(var[0]+var[4])
            # Right side
            out[-1] = a*(var[-2]+var[-1]) + b*(var[-3]+var[-1]) + c*(var[-4]+var[-1])
            out[-2] = a*(var[-3]+var[-2]) + b*(var[-4]+var[-1]) + c*(var[-5]+var[-1])
    
    return out