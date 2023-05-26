"""
Test code with Sod shock tube test
"""

# Local imports
from solve_mhd import SolveMHD1D
from ideal_gas import ideal_gas_law, gamma
from utils import gaussian
from sod_shock_test import test_params

# Global modules
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Mesh
x0 = 0
xf = 50
xm = 50/2 + 2   
nx = 256
x = np.linspace(x0, xf, nx)

rho_b = 0.125
rho_h = 1.0

Pg_b = 0.125/gamma
Pg_h = Pg_b #1.0/gamma

# Allocate arrays for solution
nt = 200
u_n = np.zeros((nt, nx))
rho_n = np.zeros_like(u_n)
e_n = np.zeros_like(u_n)

# ================== # 
# Initial conditions
# ================== #

# Mass density
# rho0 = np.zeros_like(x)
# rho0[:nx//2] = rho_L
# rho0[nx//2:] = rho_R
rho0 = gaussian(x, rho_h, rho_b, xm=xm, s=3.0)
rho_n[0, :] = rho0[:]

# Gas pressure
# P0 = np.zeros_like(x)
# P0[:nx//2] = Pg_L
# P0[nx//2:] = Pg_R
P0 = gaussian(x, Pg_h, Pg_b, xm=xm)

# Velocity
u0 = np.zeros_like(x)
u_n[0, :] = u0[:]

# Energy density
e0 = P0/(rho0 * (gamma - 1))
e_n[0, :] = e0[:]

# time
t = np.zeros(nt)

# Set up solver
solver = SolveMHD1D(x, rho0, u0, e0, boundaries='periodic', nu_p=1.0, nu_e=0.0, nu_1=0.2, nu_2=0.3, nu_3=0.5, cfl_cut=0.1)

# Save every 10 timestep
nsave = 1000
for i in range(1, nt*nsave):
    
    t_i, rho_i, u_i, e_i = solver()

    if i%nsave == 0:

        j = i//nsave

        t[j] = t_i
        u_n[j] = u_i
        rho_n[j] = rho_i
        e_n[j] = e_i

fig, ax = plt.subplots(2, 2, figsize=(8,5))

skipframes = 1

def init(): 
    ax[0, 0].plot(x, rho0)
    ax[0, 1].plot(x, u0)
    ax[1, 0].plot(x, e0)
    ax[1, 1].plot(x, P0)

    ax[0, 0].set_label('rh0')
    ax[0, 1].set_label('u')
    ax[1, 0].set_label('e')
    ax[1, 1].set_label('P')

    fig.suptitle('t=%.3f'%t[0])

def animate(i):

    P_i = ideal_gas_law(rho_n[i*skipframes], e_n[i*skipframes])

    # ax.clear()
    
    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()

    ax[0, 0].plot(x, rho_n[i*skipframes], color='k')
    ax[0, 1].plot(x, u_n[i*skipframes], color='k')
    ax[1, 0].plot(x, e_n[i*skipframes], color='k')
    ax[1, 1].plot(x, P_i, color='k')

    ax[0, 0].set_title('rho')
    ax[0, 1].set_title('u')
    ax[1, 0].set_title('e')
    ax[1, 1].set_title('P')

    fig.suptitle('t=%.3f'%t[i*skipframes])

ani = FuncAnimation(fig, animate, frames=nt//skipframes, init_func=init, interval=1)

plt.show()
