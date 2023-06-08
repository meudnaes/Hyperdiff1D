import PhysicalConstants.CODATA2018: m_u, k_B

using Unitful

const gamma = 5/3
# Average particle mass
const μ = 1.0

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
    ideal_gas(e::AbstractFloat, rho::AbstractFloat)

Calculates temperature in an ideal gas
"""
function ideal_gas(e::AbstractFloat, rho::AbstractFloat)

    (gamma - 1)*e/rho*μ*ustrip(m_u/k_B)

end

"""
    sound_speed(P::AbstractFloat, rho::AbstractFloat)

Sound speed in an ideal gas
"""
function sound_speed(P::AbstractFloat, rho::AbstractFloat)

    sqrt(gamma*P/rho)

end