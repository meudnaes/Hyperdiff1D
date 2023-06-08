# include("../solver.jl")

import .SolveMHD: Ng, gamma

using .SolveMHD
using Plots
using ProgressBars

function drive_wave(; nu_1::AbstractFloat=0.1,
                      nu_2::AbstractFloat=0.1, 
                      nu_3::AbstractFloat=0.3, 
                      nx::Integer=256,
                      nt::Integer=200, 
                      nsave::Integer=100,
                      animate::Bool=true)

    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0
    xf = 1
    x = collect(LinRange(x0, xf, nx))

    rho_const = 1.0

    rho_0 = typeof(x)(undef, nx)
    rho_0 .= rho_const

    e_const = rho_const/gamma

    # Velocity
    u_0 = zero(x)

    # Set up driver
    f = 1e-3
    ω = 2π*f
    λ = 1.0
    k = 2π/λ
    t0 = 0.0

    # boundary coords for driver
    x_b = collect(LinRange((Ng-1)*(x[1]-x[2]), x0, Ng))

    # Fill initial condition with fixed and driven boundary
    # rho_0 = reflect_boundary(rho_0, 1.0, 1.0)
    rho_0 = drive_boundary(rho_0,x_b,t0,k,ω,rho_const,rho_const)
    u_0 = reflect_boundary(u_0, 0.0, 0.0)
    # P_0 = reflect_boundary(P_0, 1.0/gamma, 1.0/gamma)
    e_0 = rho_0*e_const#drive_boundary(e_0,x_b,t0,k,ω,e_const,e_const)

    println("-- initialising perturbation")
    solver = Solver(x, e_0, rho_0, u_0, boundaries="reflect", cfl_cut=0.8,
                    nu_p=1.0, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, state=:energy)

    t = Vector{Float64}(undef, nt)
    rho = Matrix{Float64}(undef, nt, nx+Ng)
    u = Matrix{Float64}(undef, nt, nx+Ng)
    e = Matrix{Float64}(undef, nt, nx+Ng)

    t_i = 0.0
    rho_i = solver.rho0
    u_i = solver.u0
    e_i = solver.e0

    println("-- simulating system")

    for i in ProgressBar(1:Int(nt*nsave))
        t_i, rho_i, u_i, e_i = step_forward(solver, t_i, rho_i, u_i, e_i)
        # Boundary conditions
        solver.pad!(u_i, 0.0, 0.0)
        # solver.pad!(rho_i, 1.0, 1.0)
        drive_boundary!(rho_i, x_b, t_i, k, ω, rho_const, rho_const)
        solver.pad!(e_i, e_const, e_const)
        e_i[1:Ng] = rho_i[1:Ng]*e_const

        #drive_boundary!(e_i, x_b, t_i, k, ω, e_const, e_const)
        if i%nsave==0
            idt=Int(i/nsave)
            t[idt] = t_i
            #print(size(rho[idt,:]))
            #print(size(rho_i[1:end-Ng]))
            rho[idt, :] .= rho_i[1:end-Ng]
            u[idt, :] .= u_i[1:end-Ng]
            e[idt, :] .= e_i[1:end-Ng]
        end
    end

    if animate
        println("-- animating solution")
        anim = @animate for i in 1:nt
            plot1 = plot(vcat(x_b, x), e[i, :],
                title="t=$(round(t[i], digits=1))",
                xlabel="x",
                ylabel="e",
                ylim=(minimum(e), maximum(e)),
                legend=false)
            plot2 = plot(vcat(x_b, x), rho[i, :],
                xlabel="x",
                ylabel="rho",
                ylim=(minimum(rho), maximum(rho)),
                legend=false)
            plot3 = plot(vcat(x_b, x), u[i, :],
                xlabel="x",
                ylabel="u",
                ylim=(minimum(u), maximum(u)),
                legend=false)
            T = ideal_gas.(e[i,:],rho[i,:])
            plot4 = plot(vcat(x_b, x), T,
                xlabel="x",
                ylabel="T",
                ylim=(minimum(T), maximum(T)),
                legend=false)
             # Combine the subplots
            plot(plot(plot1, plot2, layout=(1, 2)), 
                 plot(plot3, plot4, layout=(1, 2)), 
                 layout=(4, 1), 
                 size=(800, 800),
                 dpi=200)
        end

        gif(anim, "../figs/waves.gif", fps = 5)
    else
        plot(vcat(x_b, x), e[end, :],
                dpi=150, 
                title="t=$(round(t[end], digits=1))",
                xlabel="x",
                ylabel="e",
                ylim=(minimum(e), maximum(e)),
                legend=false)
    end
end