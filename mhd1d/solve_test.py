"""
Script for class SolveMHD1D
"""

# Global imports
import numpy as np

# Local imports
from interpolate import x_shift
from derivatives import deriv_x
from quench import qvisc, quenchx
from ideal_gas import ideal_gas_law, sound_speed

class SolveMHD1D:
    """
    Attributes
    ----------
    x : array
        spatial axis
    rho : array
        mass density
    u : array
        velocity
    e : array
        internal energy
    dx : float
        grid spacing, delta x
    equation_of_state: callable
        equation of state. Options are 'ideal_gas_law' (default)
    c_s: callable
        sound speed
    mu : float
        viscosity coefficient
    parameters : dictionary
        diffusion parameters
    bnd_type : string
        boundary condition for padding
    bnd_limits : string
        ghost cells to pad on each side of the domain

    Methods
    -------
    continuity_equation
        Computes the rhs of the mass conservation equation
    momentum_equation
        Computes the rhs of the momentum equation
    energy_equation
        Computes the rhs of the energy equation


    """

    def __init__(self, x, rho0, u0, e0, boundaries='periodic', order='sixth', 
                 equation_of_state=ideal_gas_law, c_s=sound_speed,
                 mu=0.001, nu=1, nu1=0.1, nu2=0.1, nu3=0.1):
        """
        Parameters
        ----------
        x : array
            spatial axis
        rho0 : array
            initial mass density
        u0 : array
            initial velocity
        e0 : array
            initial energy density
        boundaries : string (optional)
            boundary conditions. Options are 'periodic' (default), or 'fixed'
            (boundaries are held constant)
        order : string (optional)
            derivation order, determines differentiation method and number
            of ghost cells. Options are 'sixth' (default)
        equation_of_state: callable
            equation of state. Options are 'ideal_gas_law' (default)
        c_s: callable
            sound speed
        mu : float (optional)
            viscosity coefficient
        nu1, nu2, nu3 : float (optional)
            diffusion parameters
        """

        if boundaries == 'periodic':
            self.bnd_type = 'wrap'
        elif boundaries == 'fixed':
            self.bnd_type = 'edge'
        else:
            raise NotImplementedError(f"Boundary condition '{boundaries}' is not implemented")
        
        if order == 'sixth':
            self.ddx = deriv_x
            self.shift = x_shift
            self.bnd_limits = [3, 3]
        else:
            raise NotImplementedError(f"Derivation order '{order}' is not implemented")

        self.x = x
        self.dx = x[1] - x[0]

        # Set the MHD variables to the initial condition
        self.rho = rho0
        self.u = u0
        self.e = e0

        self.equation_of_state = equation_of_state
        self.c_s = c_s

        # Viscosity coefficient
        self.mu = mu

        # Diffusion paremeters
        self.parameters = {'nu': nu,
                           'nu1': nu1,
                           'nu2': nu2,
                           'nu3': nu3}

        # Check if correct range
        param_vals = np.array(list(self.parameters.values()))
        if any((param_vals < 0) | (param_vals > 1)):
            raise ValueError(f"Expected diffusion parameters to be in" +
                             f"range [0,1], got {self.parameters}")


    def continuity_equation(self, rho, u):
        """
        Computes the rhs of the mass conservation equation, i.e.
        $$- \frac{\partial }{\partial x} \left( \rho u \right)$$
        
        Parameters
        ----------
        rho : array
            gas mass density
        u : array
            velocity

        Returns
        -------
        rhs : array
            Right-hand-side of mass conservation equation

        """

        deriv = -self.ddx(self.x, rho*u, bnd_type=self.bnd_type)
        
        return self.shift(deriv, bnd_type=self.bnd_type)

    def momentum_equation(self, rho, u, e):
        """
        Computes the rhs of the momentum equation, i.e.
        $$- \frac{\partial }{\partial x} \left( \rho u^2 \right)
        - \tau_{visc.} - \frac{\partial P_g}{\partial x}$$
        
        Parameters
        ----------
        rho : array
            gas mass density
        u : array
            velocity
        e : array
            energy density

        Returns
        -------
        rhs : array
            Right-hand-side of momentum equation

        """

        P_g = self.equation_of_state(rho, e)

        # first term, d rho u**2 / dx
        src1 = - self.ddx(self.x, rho*u**2, bnd_type=self.bnd_type)

        # second term, hyperdiffusion
        src2 = - self.hyperdiffusion(rho, u, P_g)

        # third term, d P_g / dx
        src3 = - self.ddx(self.x, P_g, bnd_type=self.bnd_type)

        rhs = src1 + src2 + src3

        return self.shift(rhs, bnd_type=self.bnd_type)

    def energy_equation(self, rho, u, e):
        """
        Computes the rhs of the energy equation, i.e.
        $$- \frac{\partial }{\partial x} \left( e u \right)
        - P_g \frac{\partial u}{\partial x}
        + \mu (\frac{\partial u}{\partial x})^2 + Q_\tau $$

        Parameters
        ----------
        rho : array
            mass density
        u : array
            velocity
        e : array
            internal energy

        Returns
        -------
        rhs : array
            Right-hand-side of energy equation

        """

        P_g = self.equation_of_state(rho, e)
        
        # first term, d e u / dx
        src1 = - self.ddx(self.x, e*u, bnd_type=self.bnd_type)
        #src1 = self.shift(src1, bnd_type=self.bnd_type)

        # second term, P_g d u / dx
        src2 = - P_g*self.ddx(self.x, u)
        #src2 = self.shift(src2, bnd_type=self.bnd_type)

        # third term, mu (d u / dx)**2
        src3 = self.mu * self.ddx(self.x, u, bnd_type=self.bnd_type)**2
        #src3 = self.shift(src3, bnd_type=self.bnd_type)

        # fourth term
        src4 = 0 #qvisc(self.x, rho, shift=False, bnd_type=self.bnd_type)
        #src4 = self.shift(src4, bnd_type=self.bnd_type)

        rhs = src1 + src2 + src3 + src4

        return self.shift(rhs, bnd_type=self.bnd_type)

    def step_forward(self):
        """
        Evolves the mhd equations by forward in time method
        """

        dt = self.cfl_condition()

        # Solve rho from equation of mass
        rho_new = self.rho + dt*self.continuity_equation(self.rho, self.u)

        # Solve u from equation of momentum
        u_new = ( self.rho*self.u +
                  dt*self.momentum_equation(self.rho, self.u, self.e) )/rho_new

        # Solve e from equation of energy
        e_new = self.e + dt*self.energy_equation(self.rho, self.u, self.e)

        # Update the variables
        self.rho = rho_new
        self.u = u_new
        self.e = e_new

        return rho_new, u_new, e_new
    
    def __call__(self):
        """
        Evolves the system one time-step
        """

        rho, u, e = self.step_forward()

        return rho, u, e
        
    def cfl_condition(self, cfl_cut=0.98):
        """
        Calculate the timestep from the cfl condition

        Parameters
        ----------
        cfl_cut : float
            Reduce time-step for stability. Should be in the range (0, 1),
            defaults to 0.98

        Returns
        -------
        dt : float
            timestep
        """

        dt = 1e-4

        return dt

    def hyperdiffusion(self, rho, u, e):
        """
        Calculate hyperdiffusion
        """

        P_g = self.equation_of_state(rho, e)

        drho = self.ddx(self.x, rho, bnd_type=self.bnd_type)

        # becomes shifted
        Q = qvisc(self.x, rho, shift=False, bnd_type=self.bnd_type)
        Q = self.shift(Q, bnd_type=self.bnd_type)

        src1 = self.parameters['nu1']*self.c_s(rho, P_g)
        
        src2 = self.parameters['nu2']*np.abs(u)

        src3 = self.parameters['nu3']*self.dx*self.ddx(self.x, u, bnd_type=self.bnd_type)
        src3 = self.shift(src3, bnd_type=self.bnd_type)

        inner = self.parameters['nu']*self.dx*(src1 + src2 + src3)*drho*Q

        return self.ddx(self.x, inner, bnd_type=self.bnd_type)