"""
Script for calculating derivatives on staggered mesh
"""

# Derivative constants from Bifrost
const c_d = 3/640
const b_d = (-1 - 120*c_d)/24
const a_d = 1 - 3*b_d - 5*c_d


@doc raw"""
    deriv_6th(solver::Solver, var::AbstractVector; shift::Integer=-1)

6th order Bifrost spatial derivative in x-direction

$$\partial^+_x(f_{i,j,k}) = f'_{i+1/2, j, k} =
    ( a(f_{i, j, k} - f_{i+1, j, k}) +
        b(f_{i-1, j, k} - f_{i+2, j, k}) + 
        c(f_{i-2, j, k} - f_{i+3, j, k}) )/dx$$

Parameters
----------
x : `1D array`
    x-axis
var : `1D array`
    variable
shift : `integer`
    Which direction to take the derivative. Defaults to -1 to shift a half
    grid point to the left. Use 0 to shift a half grid point to the right.

Returns
-------
out : `1D array`
    Derivative in x-direction
"""
function deriv_6th(solver::Solver, t::AbstractFloat, var::AbstractVector; shift::Integer=-1)

    out = typeof(var)(undef, length(var))

    # Calculate derivatives in the inner grid points
    for i in 4:length(var)-3
        out[i] = ( 
                a_d*(var[i+shift] - var[i+1+shift]) +
                b_d*(var[i-1+shift] - var[i+2+shift]) +
                c_d*(var[i-2+shift] - var[i+3+shift])
                )
    end

    solver.pad!(solver,t,out,:derivative)
    
    # divide by dx, which is defined in opposite direction
    return out./(-solver.dx)

end