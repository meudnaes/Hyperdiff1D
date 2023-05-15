"""
Utility functions
"""

import numpy as np

def sigmoid(x, A_L, A_R, xm=0.5, s=100.0):
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

    f = 1 / (1 + np.exp(-s*(x-xm)))

    return f*(A_R - A_L) + A_L

def gaussian(x, A_h, A_b, xm=0.5, s=1.0):
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

    f = np.exp(-(x-xm)**2/s**2)

    return f*(A_h - A_b) + A_b