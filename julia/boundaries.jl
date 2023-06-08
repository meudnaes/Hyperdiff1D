include("utils/utils.jl")

# number of ghost cells
const Ng=12

"""
    wrap_boundary(a::AbstractVector)

Fills the ghost cells of `a` to make periodic boundary conditions
"""
function wrap_boundary(a::AbstractVector)

    b = typeof(a)(undef, length(a)+2*Ng)

    b[Ng+1:end-Ng] .= a

    b[1:Ng] .= a[end-Ng+1:end]
    b[end-Ng+1:end] .= a[1:Ng]

    return b

end

function wrap_boundary!(a::AbstractVector)

    a[1:Ng] .= a[end-2*Ng+1:end-Ng]
    a[end-Ng+1:end] .= a[Ng+1:2*Ng]

end


"""
    reflect_boundary(a::AbstractVector, a_L::AbstractFloat, a_R::AbstractFloat)

Fills the ghost cells of `a` to make fixed boundary conditions.
"""
function reflect_boundary(a::AbstractVector, a_L::AbstractFloat, a_R::AbstractFloat)

    b = typeof(a)(undef, length(a)+2*Ng)

    b[Ng+1:end-Ng] .= a

    b[1:Ng] .= a_L
    b[end-Ng+1:end] .= a_R

    return b

end

function reflect_boundary!(a::AbstractVector, a_L::AbstractFloat, a_R::AbstractFloat)

    a[1:Ng] .= a_L
    a[end-Ng+1:end] .= a_R

end

"""
    drive_boundary(x::AbstractVector, t::AbstractFloat, a::AbstractVector)

Fills the ghost cells of `a` to make a wave driven boundary condition at the
left side and a fixed boundary at the right side
"""
function drive_boundary(a::AbstractVector,
                        x::AbstractVector,
                        t::AbstractFloat, 
                        k::AbstractFloat,
                        ω::AbstractFloat, 
                        a_L::AbstractFloat,
                        a_R::AbstractFloat)

    b = typeof(a)(undef, length(a)+2*Ng)

    b[Ng+1:end-Ng] .= a

    b[1:Ng] = sine_wave.(x, t, k, ω, 0.0)/75.0 .+ a_L
    b[end-Ng+1:end] .= a_R

    return b
end

function drive_boundary!(a::AbstractVector,
                         x::AbstractVector,
                         t::AbstractFloat, 
                         k::AbstractFloat,
                         ω::AbstractFloat, 
                         a_L::AbstractFloat,
                         a_R::AbstractFloat)

    a[1:Ng] = sine_wave.(x, t, k, ω, 0.0)/75.0 .+ a_L
    a[end-Ng+1:end] .= a_R

end