struct Solver{T <: AbstractFloat}
    # Computational grid
    x::Vector{T}
    dx::T
    # Initial conditions
    rho0::Vector{T}
    u0::Vector{T}
    e0::Vector{T}
    # Equations of state
    eos::Function
    c_s::Function
    # Boundary conditions
    pad::Function
    pad!::Function
    # Hyperdiffusion parameters
    parameters::Dict
    cfl_cut::T

    function Solver(x::Vector{T},
                    p0::Vector{T}, 
                    rho0::Vector{T}, 
                    u0::Vector{T};
                    boundaries::String="periodic", 
                    equation_of_state::String="ideal",
                    nu_p::T=1.0, 
                    nu_1::T=0.1, 
                    nu_2::T=0.1, 
                    nu_3::T=0.3,
                    cfl_cut::T=0.8) where T <: AbstractFloat

        dx = x[2] - x[1]

        if boundaries=="periodic"
            pad = wrap_boundary
            pad! = wrap_boundary!
        elseif boundaries=="reflect"
            pad = reflect_boundary
            pad! = reflect_boundary!
        else
            throw(ErrorException("Boundary condition $boundary not implemented"))
        end

        if equation_of_state=="ideal"
            eos = ideal_gas
            c_s = sound_speed
        else
            throw(ErrorException("Equation of state $equation_of_state not implemented"))
        end

        # Fill boundaries with three ghost zones
        p0 = pad(0.0, p0)
        rho0 = pad(0.0, rho0)
        u0 = pad(0.0, u0)

        e0 = eos.(p0, mode=:energy)

        parameters = Dict("nu_p" => nu_p,
                          "nu_1" => nu_1,
                          "nu_2" => nu_2,
                          "nu_3" => nu_3)

        new{T}(x, dx, rho0, u0, e0, eos, c_s, pad, pad!, parameters, cfl_cut)
    end
end