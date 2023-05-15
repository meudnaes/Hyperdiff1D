"""
Script for solving rhs of the mhd equations
"""

# Global imports
import numpy as np

# Local imports
from interpolate import x_shift
from derivatives import deriv_x
from quench import Qvisc

def step_mass(x, rho, u, ddx=deriv_x, bnd_type='wrap', bnd_limits=[3,3]):
    """
    Computes the rhs of the mass conservation equation, i.e.
    $$- \frac{\partial }{\partial x} \left( \rho u \right)$$
    
    Parameters
    ----------
    x : `1D array`
        spatial axis
    rho : `1D array`
        density
    u : `1D array`
        velocity vector of all grid points
    
    Returns
    -------
    dfdx : `1D array`
        Right-hand-side of mass conservation equation

    """

    # Shift cell-centered quantities to the edge
    rho = x_shift(rho)

    # Periodic boundaries
    rho = np.pad(rho, bnd_type, bnd_limits)

    # returns derivative at cell centre
    dfdx = -ddx(x, rho*u)

    return dfdx

def step_momentum(x, rho, u, P_g, ddx=deriv_x, bnd_type='wrap', bnd_limits=[3,3]):
    """
    Computes the rhs of the momentum equation, i.e.
    $$- \frac{\partial }{\partial x} \left( \rho u^2 \right)
      - \tau_{visc.} - \frac{\partial P_g}{\partial x}$$
    
    Parameters
    ----------
    x : `1D array`
        spatial axis
    rho : `1D array`
        density
    u : `1D array`
        velocity
    P_g : `1D array`
        gas pressure

    Returns
    -------
    src : `1D array`
        Right-hand-side of momentum equation

    """

    # Shift cell-centered quantities to the edge
    rho = x_shift(rho)
    P_g = x_shift(P_g)

    # Periodic boundaries
    rho = np.pad(rho, bnd_type, bnd_limits)
    u = np.pad(u, bnd_type, bnd_limits)
    P_g = np.pad(P_g, bnd_type, bnd_limits)

    # first term, d rho u^2 / dx
    src1 = - ddx(x, rho*u^2)

    # second term, viscosity
    # temporary, fix later
    tau_visc = 0
    src2 = - tau_visc

    # third term, d P_g / dx
    src3 = - ddx(x, P_g)

    return src1 + src2 + src3

def step_energy(x, rho, u, e, P_g, ddx=deriv_x, bnd_type='wrap', bnd_limits=[3,3]):
    """
    Computes the rhs of the energy equation, i.e.
    $$- \frac{\partial }{\partial x} \left( e u \right)
      - P_g \frac{\partial u}{\partial x}
      + \mu (\frac{\partial u}{\partial x})^2 + Q_\tau $$
    
    Parameters
    ----------
    x : `1D array`
        spatial axis
    rho : `1D array`
        density
    u : `1D array`
        velocity
    e : `1D array`
        internal energy
    P_g : `1D array`
        gas pressure

    Returns
    -------
    src : `1D array`
        Right-hand-side of energy equation

    """

    # Shift cell-centered quantities to the edge
    rho = x_shift(rho)
    e = x_shift(e)
    
    # Periodic boundaries
    rho = np.pad(rho, bnd_type, bnd_limits)
    e = np.pad(e, bnd_type, bnd_limits)


    # first term, d e u / dx
    src1 = - ddx(e*u)

    # second term, P_g d u / dx
    src2 = - P_g*ddx(x, u)

    # third term, mu (d u / dx)^2
    # temporary, fix later
    mu = 0.1
    src3 = mu * (ddx(x, u))**2

    # fourth term
    src4 = Qvisc(rho)

    return src1 + src2 + src3 + src4

