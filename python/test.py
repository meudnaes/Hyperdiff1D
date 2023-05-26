import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, pi

from derivatives import deriv_6th
from interpolate import x_shift


x = np.linspace(0, 2*pi, 101)
x = x[:-1]
h = sin(x)

dhdx_analytic = cos(x)
dhdx_analytic = x_shift(dhdx_analytic, shift=-1, bnd_type='wrap')

dhdx_up = deriv_6th(x, h, shift=-1, bnd_type='wrap')
# dhdx_up = x_shift(dhdx_up, shift=-1, bnd_type='wrap')

fig, ax = plt.subplots()

ax.plot(x, dhdx_analytic, ls='-', color='k')
ax.plot(x, dhdx_up, ls='--', color='r')
ax.plot(x, h, alpha=0.4, color='g')

print(h[0], h[-1])

plt.show()

step = np.zeros(101)
step[50:] = 1

step = x_shift(step, shift=0, bnd_type='edge')

plt.plot(np.pad(h, [1,1], 'wrap'))
plt.show()

