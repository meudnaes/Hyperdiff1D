"""
File for calculating the viscous heating term used in Bifrost with the 
quenching factor/quench parameter
"""

# Defined in Bifrost, do not change
const qmax = 8.0

"""
    quenchx(solver::Solver, var::AbstractVector)

Quench in x direction

Parameters
----------
var : `Array`
    Variable to be quenched

Returns
-------
out : `Array`
    Result from the quenching of `var`
"""
function quenchx(solver::Solver, var::AbstractVector)

    nx = length(var)

    out = typeof(var)(undef, nx)
    qq = typeof(var)(undef, nx)

    if qmax == 1.
        out .= 1.
        return out
    end

    # Calculate quenching for the inner grid points  
    for i in 3:nx-2
        dd = abs(var[i+1] - 2*var[i] + var[i-1])
        qq[i]= dd/(abs(var[i]) + (1/qmax)*dd + 1e-20)
    end

    for i in 4:nx-3
        out[i] = max(qq[i-1], qq[i], qq[i+1])
    end

    solver.pad(out)

    return out
end