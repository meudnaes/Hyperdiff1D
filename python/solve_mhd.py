"""
Script for class SolveMHD1D
"""

# Global imports
import warnings
import numpy as np

# Local imports
from interpolate import x_shift
from derivatives import deriv_6th
from quench import quenchx
from ideal_gas import ideal_gas_law, sound_speed

class SolveMHD1D:
    """
    Attributes
    ----------
    x : array
        spatial axis
    rho : array
        mass density, defined at cell centre (x_i)
    u : array
        velocity, defined at cell faces (x_i+1/2)
    e : array
        internal energy, defined at cell centre (x_i)
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
    cfl_cut : float
        Factor to reduce time-step for stability

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
                 nu_p=1.0, nu_e=0.3, nu_1=0.2, nu_2=0.2, nu_3=0.3,
                 cfl_cut=0.5, step='forward'):
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
        nu_p: float (optional)
            weighting parameter for hyperdiffusion on momentum
        nu_e: float (optional)
            weighting parameter for hyperdiffusion on energy
        nu_1, nu_2, nu_3 : float (optional)
            diffusion 
        cfl_cut : float (optional)
            Reduce time-step for stability. Should be in the range (0, 1),
            defaults to 0.98
        """

        if boundaries == 'periodic':
            self.bnd_type = 'wrap'
        elif boundaries == 'fixed':
            self.bnd_type = 'edge'
        else:
            raise NotImplementedError(f"Boundary condition '{boundaries}' is not implemented")
        
        if order == 'sixth':
            self.ddx = deriv_6th
            self.shift = x_shift
        else:
            raise NotImplementedError(f"Derivation order '{order}' is not implemented")

        self.step = step

        self.x = x
        self.dx = x[1] - x[0]

        # Set the MHD variables to the initial condition
        self.rho = rho0
        self.u = u0
        self.e = e0

        # Initial time is zero
        self.t = 0.0

        self.equation_of_state = equation_of_state
        self.c_s = c_s

        # Diffusion paremeters
        self.parameters = {'nu_p': nu_p,
                           'nu_e': nu_e,
                           'nu_1': nu_1,
                           'nu_2': nu_2,
                           'nu_3': nu_3}
        # Check if correct range
        param_vals = np.array(list(self.parameters.values()))
        if any((param_vals < 0) | (param_vals > 1)):
            raise ValueError(f"Expected diffusion parameters to be in " +
                             f"range [0,1], got {self.parameters}")

        self.cfl_cut = cfl_cut
        if step=='forward' and not 0 < cfl_cut <= 1:
            warnings.warn(f"cfl_cut={cfl_cut} is outside of range (0,1]. This might be" +
                           "okay with Runge-Kutta 4 timestepping scheme...")


    def continuity_equation(self, rho, u):
        r"""
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

        # Shift rho to cell face
        rho = self.shift(rho, shift=0, bnd_type=self.bnd_type)
        
        # momentum
        p = rho*u
        rhs = -self.ddx(self.x, p, shift=-1, bnd_type=self.bnd_type)
        
        return rhs

    def momentum_equation(self, rho, u, e):
        r"""
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

        # first term, d rho u**2 / dx, at i+1/2
        u_shift = self.shift(u, shift=-1, bnd_type=self.bnd_type)
        src1 = - self.ddx(self.x, rho*u_shift**2, shift=0, bnd_type=self.bnd_type)

        # second term, hyperdiffusion, at i+1/2
        rho_shift = self.shift(rho, shift=0, bnd_type=self.bnd_type)
        f = u*rho_shift
        src2 = self.parameters['nu_p']*self.hyperdiffusion(rho, u, e, f)

        # third term, d P_g / dx, at i+1/2
        P_g = self.equation_of_state(rho, e)
        src3 = - self.ddx(self.x, P_g, shift=0, bnd_type=self.bnd_type)

        rhs = src1 + src2 + src3

        return rhs

    def energy_equation(self, rho, u, e):
        r"""
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
        # Units J/m^3

        # first term, d e u / dx, returns in cell centre
        e_shift = self.shift(e, shift=0, bnd_type=self.bnd_type)
        src1 = - self.ddx(self.x, e_shift*u, shift=-1, bnd_type=self.bnd_type)

        # second term, P_g d u / dx, returns in cell centre
        P_g = self.equation_of_state(rho, e)
        src2 = - P_g*self.ddx(self.x, u, shift=-1, bnd_type=self.bnd_type)

        # third term, viscosity: mu (d u / dx)**2
        src3 = 0

        # fourth term, source term
        src4 = 0

        rhs = src1 + src2 + src3 + src4

        return rhs

    def step_forward(self):
        """
        Evolves the mhd equations by forward in time method
        """

        dt = self.cfl_condition()
        t_new = self.t + dt

        # Solve rho from equation of mass
        rho_new = self.rho + dt*self.continuity_equation(self.rho, self.u)

        # Solve u from equation of momentum
        u_new = ( self.shift(self.rho, shift=0, bnd_type=self.bnd_type)*self.u +
                  dt*self.momentum_equation(self.rho, self.u, self.e) 
                ) / self.shift(rho_new, shift=0, bnd_type=self.bnd_type)

        # Solve e from equation of energy
        e_new = self.e + dt*self.energy_equation(self.rho, self.u, self.e)

        # Update the variables
        self.rho = rho_new
        self.u = u_new
        self.e = e_new
        self.t = t_new

        return t_new, rho_new, u_new, e_new
    
    def step_simple(self, rho, u, e):
        """
        Evolves the mhd equations by forward in time method
        """

        # Solve rho from equation of mass
        rho_step = self.continuity_equation(rho, u)

        # Solve u from equation of momentum
        u_step = (self.momentum_equation(rho, u, e) - u*rho_step)/self.shift(rho, shift=0, bnd_type=self.bnd_type)

        # Solve e from equation of energy
        e_step = self.energy_equation(rho, u, e)

        return np.array([rho_step, u_step, e_step])

    def step_rk4(self):
        """
        Runge-Kutta 4 timestepping scheme, where f(y, t) is the rhs of the 
        fluid equations
        
        k1 = f(y(t0), t0)
        k2 = f(y(t0)+k1*dt/2, t0+dt/2)
        k3 = f(y(t0)+k2*dt/2, t0+dt/2)
        k4 = f(y(t0)+k3*dt, t0+dt)
        
        y(t0 + dt) = y(t0) + (k1/6 + k2/3 + k3/3 + k4/6)*dt
        """

        # self.cfl_cut = 1.0
        dt = self.cfl_condition()
        t_new = self.t + dt

        y0 = np.array([self.rho, self.u, self.e])

        k1 = np.copy(y0)
        k2 = self.step_simple(*(y0+k1*dt/2))
        k3 = self.step_simple(*(y0+k2*dt/2))
        k4 = self.step_simple(*(y0+k3*dt))

        ynew = y0 + (k1/6 + k2/3 + k3/3 + k4/6)*dt

        # Update the variables
        self.rho = ynew[0]
        self.u = ynew[1]
        self.e = ynew[2]
        self.t = t_new

        return t_new, ynew[0], ynew[1], ynew[2]
    
    def __call__(self, centered=True):
        """
        Evolves the system one time-step, and returns variables
        """

        if self.step == 'forward':
            t, rho, u, e = self.step_forward()
        elif self.step == 'rk4':
            t, rho, u, e = self.step_rk4()
        else:
            raise NotImplementedError(f'Stepping scheme {self.step} is not implemented')

        if centered:
            u = self.shift(u, shift=-1, bnd_type=self.bnd_type)

        return t, rho, u, e
        
    def cfl_condition(self):
        """
        Calculate the timestep from the cfl condition

        Returns
        -------
        dt : float
            timestep
        """

        P_g = self.equation_of_state(self.rho, self.e)

        # Information cannot propagate more than one cell
        c_fast = self.c_s(self.rho, P_g)
        dt1 = self.dx/np.max(np.abs(self.u) + c_fast)

        # Time-step from hyper-diffusion
        dt2 = self.dx**2/(2*np.max(np.abs(self.u)) + 1e-20)

        dt = min(dt1, dt2)

        if np.all(self.u == 0):
            dt = 1e-5

        return self.cfl_cut*dt

    def hyperdiffusion(self, rho, u, e, f):
        """
        Calculate hyperdiffusion, f defined at i+1/2
        """

        # centered
        P_g = self.equation_of_state(rho, e)

        # centered
        src1 = self.parameters['nu_1']*self.c_s(rho, P_g)
        
        # shifted i+1/2
        src2 = self.parameters['nu_2']*np.abs(u)
        # centered
        src2 = self.shift(src2, shift=-1, bnd_type=self.bnd_type)

        # centered
        src3 = self.parameters['nu_3']*self.dx*self.ddx(self.x, u, shift=-1, bnd_type=self.bnd_type)

        # centered
        df = self.ddx(self.x, f, shift=-1, bnd_type=self.bnd_type)
        
        # centered
        Q = quenchx(df, bnd_type=self.bnd_type)

        # centered
        inner = self.dx*(src1 + src2 + src3)*df*Q

        # Derivative returns at cell face, i+1/2
        return self.ddx(self.x, inner, shift=0, bnd_type=self.bnd_type)