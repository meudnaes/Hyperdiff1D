"""
    wrap_boundary(a::AbstractVector)

Fills the ghost cells of `a` to make periodic boundary conditions
"""
function wrap_boundary(t::AbstractFloat, a::AbstractVector)

    b = typeof(a)(undef, length(a)+6)

    b[4:end-3] .= a

    b[1:3] .= a[end-2:end]
    b[end-2:end] .= a[1:3]

    return b

end

function wrap_boundary!(solver::Solver, t::AbstractFloat, a::AbstractVector, mode::Symbol)

    a[1:3] .= a[end-5:end-3]
    a[end-2:end] .= a[4:6]

end


"""
    reflect_boundary(a::AbstractVector)

Fills the ghost cells of `a` to make fixed boundary conditions. Does not work
perfectly
"""
function reflect_boundary(t::AbstractFloat, a::AbstractVector)

    b = typeof(a)(undef, length(a)+6)

    b[4:end-3] .= a

    b[1:3] .= a[1]
    b[end-2:end] .= a[end]

    return b

end

function reflect_boundary!(solver::Solver, t::AbstractFloat, a::AbstractVector, mode::Symbol)

    if mode==:derivative
        a[1:3] .= 0
        a[end-2:end] .= 0
    end

    a[1:3] .= a[4]
    a[end-2:end] .= a[end-3]

end

"""
    drive_boundary(a::AbstractVector)

Fills the ghost cells of `a` to make a wave driven boundary condition at the
left side and a fixed boundary at the right side
"""
function drive_boundary(t::AbstractFloat, a::AbstractVector)

    b = typeof(a)(undef, length(a)+6)

    b[4:end-3] .= a

end


function drive_boundary!(solver::Solver, t::AbstractFloat, a::AbstractVector, mode::Symbol)

    b = typeof(a)(undef, length(a)+6)

    b[4:end-3] .= a

end