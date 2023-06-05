using .SolveMHD
using Plots
using ProgressBars

function random_perturbation(; nu_1::AbstractFloat=0.1,
                             nu_2::AbstractFloat=0.1, 
                             nu_3::AbstractFloat=0.3, 
                             nx::Integer=526,
                             nt::Integer=50, 
                             nsave::Integer=100)

    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0
    xf = 1
    x = collect(LinRange(x0, xf, nx))

    rho0 = typeof(x)(undef, nx)
    rho0 .= 0.5

    P0 = (rand(nx) .- 0.5)/3 .+ 0.5

    # Velocity
    u0 = zero(x)

    println("-- initialising perturbation")
    solver = Solver(x, P0, rho0, u0, boundaries="periodic", cfl_cut=0.8,
                    nu_p=1.0, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3)

    t = Vector{Float64}(undef, nt)
    rho = Matrix{Float64}(undef, nt, nx)
    u = zero(rho)
    e = zero(rho)

    t_i = 0.0
    rho_i = solver.rho0
    u_i = solver.u0
    e_i = solver.e0

    println("-- simulating system")

    for i in ProgressBar(1:Int(nt*nsave))
        t_i, rho_i, u_i, e_i = step_forward(solver, t_i, rho_i, u_i, e_i)
        if i%nsave==0
            idt=Int(i/nsave)
            t[idt] = t_i
            rho[idt, :] .= rho_i[4:end-3]
            u[idt, :] .= u_i[4:end-3]
            e[idt, :] .= e_i[4:end-3]
        end
    end

    println("-- animating solution")
    anim = @animate for i in 1:200
        plot(x, rho[i, :],
             dpi=250, 
             title="t=$(round(t[i], digits=1))",
             xlabel="x",
             ylabel="rho",
             ylim=(minimum(rho), maximum(rho)),
             legend=false)
    end

    gif(anim, "../figs/perturbations.gif", fps = 25)

end