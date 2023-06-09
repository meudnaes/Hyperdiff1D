# include("../solver.jl")

import .SolveMHD: Ng

using .SolveMHD
using FFTW
using Plots
using PyPlot
using PyCall
using ProgressBars
using Random
using LsqFit

function random_perturbation(; nu_1::AbstractFloat=0.1,
                             nu_2::AbstractFloat=0.1, 
                             nu_3::AbstractFloat=0.3, 
                             nx::Integer=526,
                             nt::Integer=50, 
                             dt_save::AbstractFloat=0.01,
                             animate::Bool=false,
                             k_diagram::Bool=false)

    Random.seed!(1998)
    # ================== # 
    # Initial conditions
    # ================== #
    x0 = 0.0
    xf = 10.0
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
            Plots.plot(x, rho[i, :],
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
        py"""
       from mpl_toolkits.axes_grid1 import make_axes_locatable
       """
        make_axes_locatable = py"""make_axes_locatable"""

        kx = fftfreq(nx, 1/solver.dx)

        idk = kx .> 0
        kx = kx[idk]

        PSD = Matrix{Float64}(undef, nt, length(kx))

        for i in 1:nt
            F = fft(e[i,:] .- mean(e[i,:]))
            F = 2*abs.(F[idk]).^2/solver.dx

            PSD[i, :] .= F
        end

        # Do the curve fitting for decaying of wave...
        # We have the square of the decaying wave!
        @. function decaying_wave(t, p)
                   
           p[1]^2*exp(-2*p[2]*t)*cos(p[3]*t-p[4])^2

        end
 
        p0 = [10.0, 1.0, 60.0, 0.0]

        # Get decay as function of wavenumber!
        λ = Vector{Float64}(undef, length(kx))
        for j in 1:length(kx)
            fit = curve_fit(decaying_wave, t, PSD[:, j], p0);
            λ[j] = coef(fit)[2]
        end

        fig, ax = subplots()

        ax.plot(kx, λ)
        ax.set_xlabel("wavenumber (1/m)")
        ax.set_ylabel("decay")
        ax.set_title("Decay as function of wavenumber")
        fig.savefig("../figs/decay_$(nu_1)_$(nu_2)_$nu_3.pdf")

        extent = [t[1], t[end], kx[1], kx[end]]

        fig, ax = subplots()
        
        im = ax.imshow(transpose(PSD), 
                       origin="lower",
                       extent=extent, 
                       aspect="auto",
                       vmax=5000,
                       vmin=0)
        
        div = make_axes_locatable(ax)
        cax = div.new_vertical(size ="5%", pad = 0.5)
        fig.add_axes(cax)
    
        cb = fig.colorbar(im, 
                          cax=cax, 
                          orientation="horizontal", 
                          label="power")

        # fig.colorbar(im, label="power", pad=0.04, fraction=0.046)
        #ax.set_xscale("log")
        ax.set_ylim([minimum(kx),12.5])
        ax.set_xlim([t[1], 2.5])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("wavenumber (1/m)")
        ax2 = ax.secondary_yaxis("right")
        ax2.set_ylabel("wavelength (m)")
        y_vals = ax.get_yticks()
        y2vals = ["$(round((1000/60/f);digits=2))" for f in y_vals]
        # y2vals[1] = nothing
        ax2.set_yticklabels(y2vals)
        
        fig.savefig("../figs/k-diagram$(nu_1)_$(nu_2)_$nu_3.pdf")
        close(fig);

        return t, kx, PSD

    end

end