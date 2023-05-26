"""
Test code with Sod shock tube test
"""

# Local imports
from solve_mhd import SolveMHD1D
from ideal_gas import ideal_gas_law, gamma
from sod_shock_exact import SodShockTube
from utils import sigmoid

# Global modules
import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# ================== # 
# Initial conditions
# ================== #
x0 = 0
xf = 1
nx = 526
x = np.linspace(x0, xf, nx)

rho_R = 0.125
rho_L = 1.0

Pg_R = 0.125/gamma
Pg_L = 1.0/gamma

# Introduce a slope instead of disontinuity
slope = 400

# Mass density
rho0 = sigmoid(x, rho_L, rho_R, s=slope)

# Gas pressure
P0 = sigmoid(x, Pg_L, Pg_R, s=slope)

# Velocity
u0 = np.zeros_like(x)

# Energy density
e0 = P0/(gamma - 1)

# ============================= #
# Calculate analytical solution
# ============================= #
exact = SodShockTube(x_0=0.5, gamma=gamma, rho_l=rho_L, rho_r=rho_R,
                     p_l=Pg_L, p_r=Pg_R, u_l=0.0, u_r=0.0)


def test_params(nu_p, nu_e, nu_1, nu_2, nu_3):

    # Set up solver
    solver = SolveMHD1D(x, rho0, u0, e0, boundaries='fixed',
                        nu_p=nu_p, nu_e=nu_e, nu_1=nu_1, nu_2=nu_2, nu_3=nu_3,
                        step='forward', cfl_cut=0.7)

    # Evolve for 500 steps
    for _ in range(50):
        
        _, _, _, _ = solver()

    # Save solution
    t, rho, u, e = solver()

    return t, rho, u, e

        
def animate():

    # Allocate arrays for solution
    nt = 200
    u_n = np.zeros((nt, nx))
    rho_n = np.zeros_like(u_n)
    e_n = np.zeros_like(u_n)

    rho_n[0, :] = rho0[:]
    u_n[0, :] = u0[:]
    e_n[0, :] = e0[:]

    # time
    t = np.zeros(nt)

    # Set up solver
    solver = SolveMHD1D(x, rho0, u0, e0, boundaries='fixed',
                        nu_p=1.0, nu_e=0.0, nu_1=0.1, nu_2=0.1, nu_3=0.2,
                        cfl_cut=0.7, step='forward')

    # Save every 10 timestep
    nsave = 150
    for i in range(1, nt*nsave):
        
        t_i, rho_i, u_i, e_i = solver()

        if i%nsave == 0:

            j = i//nsave

            t[j] = t_i
            u_n[j] = u_i
            rho_n[j] = rho_i
            e_n[j] = e_i

    fig, ax = plt.subplots(2, 2, figsize=(8,5))

    skipframes = 2

    def init(): 
        ax[0, 0].plot(x, rho0)
        ax[0, 1].plot(x, u0)
        ax[1, 0].plot(x, e0)
        ax[1, 1].plot(x, P0)

        ax[0, 0].set_title('rh0')
        ax[0, 1].set_title('u')
        ax[1, 0].set_title('e')
        ax[1, 1].set_title('P')

        fig.suptitle('t=%.3f'%t[0])

    def animate(i):
        # exact solution
        x_e, P_e, rho_e, u_e = exact(t[i*skipframes], N=25)
        e_e = P_e/(gamma - 1)

        P_i = ideal_gas_law(rho_n[i*skipframes], e_n[i*skipframes])

        # ax.clear()
        
        ax[0, 0].clear()
        ax[0, 1].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()

        ax[0, 0].plot(x, rho_n[i*skipframes], color='k', label='numerical')
        ax[0, 1].plot(x, u_n[i*skipframes], color='k', label='numerical')
        ax[1, 0].plot(x, e_n[i*skipframes], color='k', label='numerical')
        ax[1, 1].plot(x, P_i, color='k', label='numerical')
        
        ax[0, 0].scatter(x_e, rho_e, color='r', marker='x', linewidth=1.0, label='analytical')
        ax[0, 1].scatter(x_e, u_e, color='r', marker='x', linewidth=1.0, label='analytical')
        ax[1, 0].scatter(x_e, e_e, color='r', marker='x', linewidth=1.0, label='analytical')
        ax[1, 1].scatter(x_e, P_e, color='r', marker='x', linewidth=1.0, label='analytical')

        ax[0, 0].set_title('rho')
        ax[0, 1].set_title('u')
        ax[1, 0].set_title('e')
        ax[1, 1].set_title('P')

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()

        fig.suptitle('t=%.3f'%t[i*skipframes])

    ani = FuncAnimation(fig, animate, frames=nt//skipframes, init_func=init, interval=1)

    plt.show()


if __name__ == "__main__":
    animate()


    if True:
        from itertools import product

        # nu_1 and nu_2 has to be 0.1 or smaller
        # nu_3 ish okay at 0.1 and 0.5

        nu_params = {
                        'nu_p' : [1.0],
                        'nu_e' : [0.3],
                        'nu_1': [0.1],
                        'nu_2': [0.1],
                        'nu_3': [0.3],
                    }

        fig, ax = plt.subplots(2, 2, figsize=(8,5))

        for params in list(product(*nu_params.values())):
            
            nu_p, nu_e, nu_1, nu_2, nu_3 = params

            start = time.time()
            for i in range(50):
                t, rho, u, e = test_params(nu_p, nu_e, nu_1, nu_2, nu_3)

            elapsed = time.time() - start

            print(f"Avg time: {elapsed/50*1e3} ms")

            if not np.any(np.isnan(np.array([rho, u, e]))):
                ls='-'
                if nu_1 == -0.05:
                    ls='--'
                elif nu_1 == -0.1:
                    ls='-'
                elif nu_1 == -0.2:
                    ls='-.'
                P = ideal_gas_law(rho, e)
                ax[0, 0].plot(x, rho, label=f'{params}', lw=0.8, ls=ls)
                ax[0, 1].plot(x, u, label=f'{params}', lw=0.8, ls=ls)
                ax[1, 0].plot(x, e, label=f'{params}', lw=0.8, ls=ls)
                ax[1, 1].plot(x, P, label=f'{params}', lw=0.8, ls=ls)
            else:
                print(f'{params} gave NaN at t={t}')

        # exact solution
        x_e, P_e, rho_e, u_e = exact(t, N=15)
        e_e = P_e/(gamma - 1)

        ax[0, 0].scatter(x_e, rho_e, color='r', marker='x', linewidth=0.5)
        ax[0, 1].scatter(x_e, u_e, color='r', marker='x', linewidth=0.5)
        ax[1, 0].scatter(x_e, e_e, color='r', marker='x', linewidth=0.5)
        ax[1, 1].scatter(x_e, P_e, color='r', marker='x', linewidth=0.5)

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()

        plt.show()