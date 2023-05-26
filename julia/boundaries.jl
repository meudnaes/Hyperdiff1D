function wrap_boundary(a::AbstractVector, gzones::Integer)

    b = typeof(a)(undef, length(a)+2*gzones)

    b[4:end-3] .= a

    b[1:3] .= a[end-2:end]
    b[end-2:end] .= a[1:3]

    return b

end

function wrap_boundary(a::AbstractVector)

    a[1:3] .= a[end-5:end-3]
    a[end-2:end] .= a[4:6]

end

function fixed_boundary(a::AbstractVector, gzones::Integer)

    b = typeof(a)(undef, length(a)+2*gzones)

    b[4:end-3] .= a

    b[1:3] .= a[1]
    b[end-2:end] .= a[end]

    return b

end

function fixed_boundary(a::AbstractVector)

    a[1:3] .= a[4]
    a[end-2:end] .= a[end-3]

end