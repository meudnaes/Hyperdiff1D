"""
Script for interpolating along axis on the staggered mesh
"""

# Interpolation constants from Bifrost
const c_i = 3/256
const b_i = -25/256
const a_i = 1/2 - b_i - c_i


@doc raw"""
    x_shift(solver::Solver, var::AbstractVector; shift::Integer=-1)

5th-order interpolation operator to shift quantity a half grid point
in the x-direction

$$T^+_x(f_{i,j,k}) = f_{i+1/2, j, k} =
    a(f_{i, j, k} + f_{i+1, j, k}) +
    b(f_{i-1, j, k} + f_{i+2, j, k}) +
    c(f_{i-2, j, k} + f_{i+3, j, k})$$

Parameters
----------
var : `1D Array`
    Quantity defined at half grid shifted
shift : `integer`
    Which direction to interpolate variable. Defaults to -1 to shift var half
    grid point to the left. Use 0 to shift var half grid point to the right.

Returns
-------
out : `1D Array`
    Quantity defined at cell center of grid
"""
function x_shift(var::AbstractVector; shift::Integer=-1)

    out = typeof(var)(undef, length(var)-6)

    # Calculate the interpolations for the inner grid points
    for i in 4:length(var)-3
        @inbounds out[i-3] = ( 
                a_i*(var[i+shift] + var[i+1+shift]) +
                b_i*(var[i-1+shift] + var[i+2+shift]) +
                c_i*(var[i-2+shift] + var[i+3+shift])
                )
    end
    
    return out
end