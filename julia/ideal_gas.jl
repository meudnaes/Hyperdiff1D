const gamma = 5/3

"""
    ideal_gas(value::AbstractFloat, mode=:pressure::Symbol)

Equation of state for ideal gas
"""
function ideal_gas(x::AbstractFloat; mode::Symbol=:pressure)

    if mode==:pressure
        # input is internal energy, calculates the gas pressure
        P = (gamma - 1)*x
        return P

    elseif mode==:energy
        # input is pressure, calculates the internal energy
        e = x/(gamma - 1)
        return e
    
    else
        throw(ErrorException("Mode $mode does not exist"))
    end

end

"""
    sound_speed(e::AbstractVector)

Sound speed in an ideal gas
"""
function sound_speed(P::AbstractFloat, rho::AbstractFloat)

    sqrt(gamma*P/rho)

end