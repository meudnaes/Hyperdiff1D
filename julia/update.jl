"""
Evolves the mhd equations by forward in time method
"""
function step_forward(solver::Solver,
                      t::AbstractFloat, 
                      rho::AbstractVector, 
                      u::AbstractVector, 
                      e::AbstractVector)

    dt = cfl_condition(solver, rho, u, e)
    t_new = t + dt

    # Solve rho from equation of mass
    rho_new = rho + dt*continuity_equation(solver, rho, u)

    # Solve u from equation of momentum
    rho_shift = x_shift(solver, rho, shift=0)
    rho_new_shift = x_shift(solver, rho_new, shift=0)

    u_new = (rho_shift .* u + dt*momentum_equation(solver, rho, u, e)) ./ rho_new_shift

    # Solve e from equation of energy
    e_new = e + dt*energy_equation(solver, rho, u, e)

    # Update the variables
    rho = rho_new
    u = u_new
    e = e_new
    t = t_new

    return t, rho, u, e
end

function step_simple(solver::Solver, 
                     rho::AbstractVector, 
                     u::AbstractVector, 
                     e::AbstractVector)
    """
    Evolves the mhd equations by forward in time method
    """

    # Solve rho from equation of mass
    rho_step = continuity_equation(solver, rho, u)

    # Solve u from equation of momentum
    u_step = (momentum_equation(solver, rho, u, e) - u .* rho_step) ./ x_shift(solver, rho, shift=0)

    # Solve e from equation of energy
    e_step = energy_equation(solver, rho, u, e)

    return [rho_step, u_step, e_step]
end


"""
Runge-Kutta 4 timestepping scheme, where f(y, t) is the rhs of the 
fluid equations

k1 = f(y(t0), t0)
k2 = f(y(t0)+k1*dt/2, t0+dt/2)
k3 = f(y(t0)+k2*dt/2, t0+dt/2)
k4 = f(y(t0)+k3*dt, t0+dt)

y(t0 + dt) = y(t0) + (k1/6 + k2/3 + k3/3 + k4/6)*dt
"""
function step_rk4(solver::Solver,
                  t::AbstractFloat,
                  rho::AbstractVector, 
                  u::AbstractVector, 
                  e::AbstractVector)

    # cfl_cut = 1.0
    dt = cfl_condition(solver, rho, u, e)
    t_new = t + dt

    y0 = [rho, u, e]

    k1 = copy(y0)
    k2 = step_simple(solver, (y0 + k1*dt/2)...)
    k3 = step_simple(solver, (y0 + k2*dt/2)...)
    k4 = step_simple(solver, (y0 + k3*dt)...)

    @. ynew = y0 + (k1/6 + k2/3 + k3/3 + k4/6)*dt

    # Update the variables
    rho = ynew[0]
    u = ynew[1]
    e = ynew[2]
    t = t_new

    return t, rho, u, e
end


"""
    cfl_condition(solver::Solver,
                  u::AbstractVector, 
                  e::AbstractVector)

Calculate the timestep from the cfl condition

Returns
-------
dt : float
    timestep
"""
function cfl_condition(solver::Solver,
                       rho::AbstractVector,
                       u::AbstractVector, 
                       e::AbstractVector)

    # Information cannot propagate more than one cell
    P = solver.eos.(e, mode=:pressure)
    c_fast = solver.c_s.(P, rho)
    dt1 = solver.dx/maximum(abs.(u) + c_fast)

    # Time-step from hyper-diffusion
    dt2 = solver.dx^2/(2*maximum(abs.(u)) + 1e-20)

    dt = min(dt1, dt2)

    if all(u == 0)
        dt = 1e-5
    end

    return solver.cfl_cut*dt
end