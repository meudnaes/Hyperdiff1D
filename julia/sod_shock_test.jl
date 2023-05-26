include("utils.jl")

using .SolveMHD

function sod_shock(; nx::Integer=526, nt::Integer=50, nsave::Integer=1)
    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0
    xf = 1
    x = collect(LinRange(x0, xf, nx))

    rho_R = 0.125
    rho_L = 1.0

    P_R = 0.125/gamma
    P_L = 1.0/gamma

    # Introduce a slope instead of disontinuity
    slope = 400.

    # Mass density
    rho_0 = sigmoid.(x, rho_L, rho_R, s=slope)

    # Gas pressure
    P_0 = sigmoid.(x, P_L, P_R, s=slope)

    # Velocity
    u_0 = zero(x)

    solver = Solver(x, P_0, rho_0, u_0, boundaries="fixed", cfl_cut=0.8,
                    nu_p=1.0, nu_e=0.3, nu_1=0.1, nu_2=0.1, nu_3=0.3)

    t = Vector{Float64}(undef, nt)
    rho = Matrix{Float64}(undef, nt, nx)
    u = zero(rho)
    e = zero(rho)

    t_i = 0.0
    rho_i = solver.rho0
    u_i = solver.u0
    e_i = solver.e0

    for i in 1:Int(nt*nsave)
        t_i, rho_i, u_i, e_i = step_forward(solver, t_i, rho_i, u_i, e_i)
        if i%nsave==0
            idt=Int(i/nsave)
            t[idt] = t_i
            rho[idt, :] .= rho_i[4:end-3]
            u[idt, :] .= u_i[4:end-3]
            e[idt, :] .= e_i[4:end-3]
        end
    end

    return x, t, rho, u, e

end