"""
File for calculating the viscous heating term used in Bifrost with the 
quenching factor/quench parameter
"""

# Global imports
import numpy as np

from numba import njit, prange

# Defined in Bifrost, do not change
qmax = 8.0

@njit(parallel=True)
def quenchx(var, bnd_type='wrap'):
    """
    Quench in x direction

    Parameters
    ----------
    var : `Array`
        Variable to be quenched

    Returns
    -------
    out : `Array`
        Result from the quenching of `var`
    """

    nx = var.shape[0]

    out = np.zeros_like(var)
    qq = np.zeros_like(var)

    if qmax == 1.:
        out[:] = 1.
        return out

    # Calculate quenching for the inner grid points  
    for i in prange(1,nx-1):
        dd = abs(var[i+1] - 2*var[i] + var[i-1])
        qq[i]= dd/(abs(var[i]) + (1/qmax)*dd + 1e-20)

    for i in prange(2,nx-2):
        out[i] = max(qq[i-1], qq[i], qq[i+1])

    # Fix quench factor at the edges from the boundary condition
    if bnd_type=='wrap':
        # Left
        dd = abs(var[1] - 2*var[0] + var[-1])
        qq[0]= dd/(abs(var[0]) + (1.0/qmax)*dd + 1e-20)      
        out[0] = max(qq[-1], qq[0], qq[1])
        out[1] = max(qq[0], qq[1], qq[2])
        
        # Right
        dd = abs(var[0] + 2*var[-1] + var[-2])
        qq[-1]= dd/(abs(var[-1]) + (1/qmax)*dd + 1e-20)
        out[-1] = max(qq[-2], qq[-1], qq[0])
        out[-2] = max(qq[-3], qq[-2], qq[-1])

    elif bnd_type=='edge':
        # Left
        dd = abs(var[1] - var[0])
        qq[0]= dd/(abs(var[0]) + (1.0/qmax)*dd + 1e-20)
        out[0] = max(qq[0], qq[1])
        out[1] = max(qq[0], qq[1], qq[2])
        
        # Right
        dd = abs(-var[-1] + var[-2])
        qq[-1]= dd/(abs(var[-1]) + (1/qmax)*dd + 1e-20)
        out[-2] = max(qq[-3], qq[-2], qq[-1])
        out[-1] = max(qq[-2], qq[-1])

    return out