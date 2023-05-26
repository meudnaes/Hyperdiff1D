@doc raw"""
    continuity_equation(solver::Solver, rho::AbstractVector, u::AbstractVector)

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
function continuity_equation(solver::Solver,
                             rho::AbstractVector,
                             u::AbstractVector)

    # Shift rho to cell face
    rho_shift = x_shift(solver, rho, shift=0)

    # momentum
    p = rho_shift .* u
    rhs = -deriv_6th(solver, p, shift=-1)
    
    return rhs
end

@doc raw"""
    momentum_equation(solver::Solver, rho::AbstractVector, u::AbstractVector, e::AbstractVector)

Computes the rhs of the momentum equation, i.e.
$$- \frac{\partial }{\partial x} \left( \rho u^2 \right)
- \tau_{visc.} - \frac{\partial P}{\partial x}$$

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
function momentum_equation(solver::Solver,
                           rho::AbstractVector, 
                           u::AbstractVector, 
                           e::AbstractVector)

    # first term, d rho u**2 / dx, at i+1/2
    u_shift = x_shift(solver, u, shift=-1)
    src1 = - deriv_6th(solver, rho .* u_shift .^ 2, shift=0)

    # second term, hyperdiffusion, at i+1/2
    rho_shift = x_shift(solver, rho, shift=0)
    f = u .* rho_shift
    src2 = solver.parameters["nu_p"]*hyperdiffusion(solver, rho, u, e, f)

    # third term, d P / dx, at i+1/2
    P = solver.eos.(e, mode=:pressure)
    src3 = - deriv_6th(solver, P, shift=0)

    rhs = src1 + src2 + src3

    return rhs
end

@doc raw"""
    energy_equation(solver::Solver, rho::AbstractVector, u::AbstractVector, e::AbstractVector)

Computes the rhs of the energy equation, i.e.
$$- \frac{\partial }{\partial x} \left( e u \right)
- P \frac{\partial u}{\partial x}
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
function energy_equation(solver::Solver,
                         rho::AbstractVector, 
                         u::AbstractVector, 
                         e::AbstractVector)


    # first term, d e u / dx, returns in cell centre
    e_shift = x_shift(solver, e, shift=0)
    src1 = - deriv_6th(solver, e_shift .* u, shift=-1)

    # second term, P d u / dx, returns in cell centre
    P = solver.eos.(e, mode=:pressure)
    src2 = - P .* deriv_6th(solver, u, shift=-1)

    # third term, viscosity: mu (d u / dx)**2
    # src3 = ...

    # fourth term, source term
    # src4 = ...

    rhs = src1 + src2

    return rhs
end