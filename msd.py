import numpy as np
import matplotlib.pyplot as plt
import os
from potentials import *
from RWMH import RWMH

def lp_constraint(d, I, mu_arr, p):
    """
    Compute the L^p constraint for the diffusion coefficient d

    Args
    d (list): the diffusion coefficient values for the point in the mesh
    I (int): number of point in the mesh (length of vector d)
    mu_arr (list): the approximation of the unnormalized density of the Gibbs measure
    p (float): L^p constraint.
    """
    return (np.dot(np.power(d, p), np.power(mu_arr, p)) / I) ** (1 / p)

if __name__ == "__main__":

    # MSD simulation parameters
    N_sim = 10**3
    N_it = 10**6
    dt = 10**(-4)
    x0 = 0
    beta = 1

    # Plot preparation
    linestyles=["solid", "dashed", "dashdot"]
    linewidths=[3,2,2]
    colors = ["red", "blue", "black"]
    fig, ax = plt.subplots()

    # Specifying the potential
    V = sin_two_wells

    # Get optimal diffusion from optimization algorithm
    I = 1000
    p = 2
    a = 0.0
    b = np.inf
    a_rounded = np.round(a, 2)
    b_rounded = np.round(b, 2)
    dir_string = "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b_rounded}/"
    d_opt = np.loadtxt(dir_string + "d_opt.txt")

    # Set optimal diffusion in the homogenized limit
    XX = [i / I for i in range(0, I)]
    mu_arr = np.fromiter(map(lambda x: np.exp(-V(x)), XX), float)
    d_homog = np.fromiter(map(lambda x: np.exp(V(x)), XX), float)

    # constant diffusion
    d_constant = np.ones(I) / lp_constraint(np.ones(I), I, mu_arr, p)

    for (idx, (d, d_string)) in enumerate(zip(
        [
            d_opt,
            d_homog,
            d_constant
        ],
        [
            "d_opt",
            "d_homog",
            "d_constant"
        ]
    )):

        if not os.path.isfile("data/" + V.__name__ + f"/msd_{d_string}.txt"):
            msd = np.zeros((N_sim, N_it))

            for n in range(N_sim):
                print(f"Running msd iteration {n}/{N_sim} for {d_string}", flush=True)
                trajectory, _ = RWMH(d, I, dt, N_it, x0, V, beta)
                for i in range(1, N_it):
                    msd[n, i] = (msd[n,i-1]*(i-1) + trajectory[i]**2)/i

            np.savetxt("data/" + V.__name__ + f"/msd_{d_string}.txt", msd)
        
        else: 
            msd = np.loadtxt("data/" + V.__name__ + f"/msd_{d_string}.txt")

        ax.plot(
            np.arange(N_it),
            np.mean(msd, axis=0),
            label=d_string,
            linestyle=linestyles[idx],
            linewidth = linewidths[idx],
            color=colors[idx]
        )

    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Mean squared distance")
    ax.legend()
    fig.savefig("data/" + V.__name__ + "/msd.png")
