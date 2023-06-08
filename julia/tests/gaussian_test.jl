include("../utils/utils.jl")
include("../solver.jl")

import .SolveMHD: gamma, Ng
using .SolveMHD

using Plots
using ProgressBars

function travelling_wave(; nu_1::AbstractFloat=0.15,
                   nu_2::AbstractFloat=0.1, 
                   nu_3::AbstractFloat=0.1, 
                   nx::Integer=526,
                   nt::Integer=50, 
                   nsave::Integer=1)
    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0
    xf = 1
    x = collect(LinRange(x0, xf, nx))

    rho_h = 1.0
    rho_b = 0.125

    # Mass density
    rho_0 = gaussian.(x, rho_h, rho_b, xm=0.5, s=0.1)

    # Gas pressure
    P_0 = ones(Float64, nx)

    # Velocity
    u_0 = ones(Float64, nx)*0.7

    # Fill initial condition with fixed boundaries
    rho_0 = wrap_boundary(rho_0)
    P_0 = wrap_boundary(P_0)
    u_0 = wrap_boundary(u_0)

    solver = Solver(x, P_0, rho_0, u_0, boundaries="periodic", cfl_cut=0.8,
                    nu_p=1.0, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3)

    t = Vector{Float64}(undef, nt)
    rho = Matrix{Float64}(undef, nt, nx)
    u = zero(rho)
    e = zero(rho)

    t_i = 0.0
    rho_i = solver.rho0
    u_i = solver.u0
    e_i = solver.e0

    # for i in ProgressBar(1:Int(nt*nsave))
    for i in ProgressBar(1:Int(nt*nsave))
        t_i, rho_i, u_i, e_i = step_forward(solver, t_i, rho_i, u_i, e_i)
        #println("Be")
        #println(u_i)
        if i%nsave==0
            idt=Int(i/nsave)
            t[idt] = t_i
            rho[idt, :] .= rho_i[Ng+1:end-Ng]
            u[idt, :] .= u_i[Ng+1:end-Ng]
            e[idt, :] .= e_i[Ng+1:end-Ng]
        end
        # Boundary conditions
        solver.pad!(rho_i)
        solver.pad!(u_i)
        solver.pad!(e_i)
        #println("Af")
        #println(u_i)
    end

    println("-- animating solution")
    anim = @animate for i in 1:nt
        plot(x, rho[i, :],
             dpi=250, 
             title="t=$(round(t[i], digits=1))",
             xlabel="x",
             ylabel="rho",
             ylim=(minimum(rho)-0.1, maximum(rho)+0.1),
             legend=false)
    end

    gif(anim, "../figs/gaussian.gif", fps = 25);

end