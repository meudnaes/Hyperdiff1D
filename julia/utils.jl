"""
Utility functions
"""

"""
Sigmoid function

Parameters
----------
x : float or array
    axis
A_L : float
    Height of left side
A_R :
    Height of right side
xm : float
    midpoint
s : float
    slope of sigmoid
"""
function sigmoid(x::T, A_L::T, A_R::T;
                 xm::T=0.5, s::T=100.0) where T <: AbstractFloat

    f = 1 / (1 + exp(-s*(x-xm)))

    return f*(A_R - A_L) + A_L
end

"""
Gaussian envelope function

Parameters
----------
x : float or array
    axis
A_c : float
    Height envelope
A_R :
    Bottom of envelope
xm : float
    midpoint of envelope
s : float
    standard deviation of envelope
"""
function gaussian(x::T, A_h::T, A_b::T;
                  xm::T=0.5, s::T=1.0) where T <: AbstractFloat

    f = exp(-(x-xm)^2/s^2)

    return f*(A_h - A_b) + A_b
end