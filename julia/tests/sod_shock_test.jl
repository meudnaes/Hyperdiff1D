include("../utils/utils.jl")
include("../solver.jl")

import .SolveMHD: gamma, Ng
using .SolveMHD

using Plots
using PyCall
using ProgressBars

function sod_shock(; nu_1::AbstractFloat=0.08,
                   nu_2::AbstractFloat=0.1, 
                   nu_3::AbstractFloat=0.1, 
                   nx::Integer=526,
                   nt::Integer=50, 
                   nsave::Integer=1,
                   plott=true)
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

    # Fill initial condition with fixed boundaries
    rho_0 = reflect_boundary(rho_0, rho_L, rho_R)
    P_0 = reflect_boundary(P_0, P_L, P_R)
    u_0 = reflect_boundary(u_0, 0.0, 0.0)

    solver = Solver(x, P_0, rho_0, u_0, boundaries="reflect", cfl_cut=0.8,
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
    for i in 1:Int(nt*nsave)
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
        solver.pad!(rho_i, rho_L, rho_R)
        solver.pad!(u_i, 0.0, 0.0)
        solver.pad!(e_i, solver.eos(P_L, mode=:energy), solver.eos(P_R, mode=:energy))
        #println("Af")
        #println(u_i)
    end


    if plott
        # Compare with analytical, add sod_shock_exact.py to PYTHONPATH
        pushfirst!(pyimport("sys")."path", "")
        py"""
        from sod_shock_exact import SodShockTube
        """

        exact = py"""SodShockTube(gamma=5/3, p_r=0.125/(5/3), p_l=1.0/(5/3), rho_r=0.125, rho_l=1.0)"""
        x_e, p_e, rho_e, u_e = exact(t[end], 5)


        # Create the subplots
        plot1 = plot(x, rho[end,:], 
                    title="Sod shock tube after t = $(round(t[end]; digits=2))",
                    label="numerical", grid=false, size=(800,500), xlabel="x", ylabel="ρ")
        scatter!(x_e, rho_e, label="exact", marker=:x, color="red", ms=2)
        plot2 = plot(x, u[end,:], label="numerical", grid=false, xlabel="x", ylabel="u")
        scatter!(x_e, u_e, label="exact",marker=:x, color="red", ms=2)
        plot3 = plot(x, e[end,:], label="numerical", grid=false, xlabel="x", ylabel="e")
        scatter!(x_e, solver.eos.(p_e, mode=:energy), label="exact", marker=:x, color="red", ms=2)
        
        # Combine the subplots
        plot_all = plot(plot1, 
                        plot(plot2, plot3, layout=(1, 2)), 
                        layout=(3, 1), 
                        size=(800, 800))

        # Save the plot as an image
        savefig(plot_all, "../figs/sod_shock.pdf")
        # return x, t, rho, u, e
    end

end
sod_shock(nt=1, nsave=120_000, plott=true)

# sod_shock(nt=1,nsave=25_000,nx=256,nu_3=0.1,nu_1=0.06,nu_2=0.04)
# sod_shock(nt=1,nsave=20_000,nu_3=0.1,nu_1=0.08,nu_2=0.1)
