"""
Relations for ideal gas
"""

import numpy as np

gamma = 5/3

def ideal_gas_law(rho, e):
    """
    Equation of state for ideal gas
    
    Parameters
    ----------
    rho : float or array
        mass density
    e : float or array
        energy density
    
    Returns
    -------
    P_g : float or array
        gas pressure
    """

    return rho*(gamma - 1)*e

def sound_speed(rho, P):
    """
    Sound speed in an ideal gas

    Parameters
    ----------
    rho : float or array
        mass density
    P_g : float or array
        gas pressure
    
    Returns
    -------
    c_s : float or array
        sound speed
    """

    return np.sqrt(gamma*P/rho)
