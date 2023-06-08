# include("../solver.jl")

import .SolveMHD: Ng

using .SolveMHD
using FFTW
using Plots
using ProgressBars
using Statistics

function random_perturbation(; nu_1::AbstractFloat=0.1,
                             nu_2::AbstractFloat=0.1, 
                             nu_3::AbstractFloat=0.3, 
                             nx::Integer=526,
                             nt::Integer=50, 
                             dt_save::AbstractFloat=0.01,
                             animate::Bool=false,
                             k_diagram::Bool=false)

    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0
    xf = 1
    x = collect(LinRange(x0, xf, nx))

    P_0 = (rand(nx) .- 0.5)/3 .+ 0.5

    rho_0 = P_0*5/3

    # Velocity
    u_0 = zero(x)

    # Fill initial condition with fixed boundaries
    rho_0 = wrap_boundary(rho_0)
    P_0 = wrap_boundary(P_0)
    u_0 = wrap_boundary(u_0)

    println("-- initialising perturbation")
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

    t[1] = t_i
    rho[1, :] .= rho_i[Ng+1:end-Ng]
    u[1, :] .= u_i[Ng+1:end-Ng]
    e[1, :] .= e_i[Ng+1:end-Ng]

    println("-- simulating system")

    t_sum = 0.0
    for i in ProgressBar(2:nt)
        while t_i/i < dt_save
            t_i, rho_i, u_i, e_i = step_forward(solver, t_i, rho_i, u_i, e_i)
            # Boundary conditions
            solver.pad!(rho_i)
            solver.pad!(u_i)
            solver.pad!(e_i)
        end
        t_sum = 0.0
        t[i] = t_i
        rho[i, :] .= rho_i[Ng+1:end-Ng]
        u[i, :] .= u_i[Ng+1:end-Ng]
        e[i, :] .= e_i[Ng+1:end-Ng]
    end

    if animate
        println("-- animating solution")
        anim = @animate for i in 1:nt
            plot(x, rho[i, :],
                dpi=150, 
                title="t=$(round(t[i], digits=1))",
                xlabel="x",
                ylabel="rho",
                ylim=(minimum(rho), maximum(rho)),
                legend=false)
        end

        gif(anim, "../figs/perturbations.gif", fps = 25)
    end

    if k_diagram
        # Do a FFT in x and t and see which wave modes survive, change nu's
        F = fft(e.-mean(e))

        freq = fftfreq(nt, 1/dt_save)
        kx = fftfreq(nx, 1/solver.dx)

        idf = freq .> 0
        idk = kx .> 0

        freq = freq[idf]
        kx = kx[idk]

        F = F[idf,:]
        F = F[:, idk]

        #=
        heatmap(transpose(abs.(F).^2),
                x=freq,
                y=kx,
                xlabel="frequency (Hz)",
                ylabel="wavenumber (1/m)")
        =#

        F1 = fft(e[1,:] .- mean(e[1,:]))
        F1 = F1[idk]

        Fend = fft(e[end,:] .- mean(e[end,:]))
        Fend = Fend[idk]

        plot(kx, abs.(Fend).^2, label="end")
        plot!(kx, abs.(F1).^2, label="start", yscale=:log10, 
              xlabel="wavenumber (1/m)",
              ylabel="fourier power")

    end

end