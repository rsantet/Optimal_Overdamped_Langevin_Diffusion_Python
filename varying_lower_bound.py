import numpy as np
import matplotlib.pyplot as plt
from potentials import *
from main import optim_algo
import os

if __name__ == "__main__":

    # Parameters for the optimization algorithm
    I = 1000
    p = 2
    b = np.inf
    V = sin_two_wells
    def mu(x):
        return np.exp(-V(x))

    XX = [i / I for i in range(0, I)]
    mu_arr = np.fromiter(map(mu, XX), float)
    Z = np.sum(mu_arr) / I
    pi_arr = mu_arr / Z

    # Preparing plots
    fig, ax = plt.subplots()
    gaps = []

    a_range = [0.05*i for i in range(0, 21)]
    for (idx, a) in enumerate(a_range):
        
        a_rounded = np.round(a, 2)
        b_rounded = np.round(b, 2)
        dir_string = "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b_rounded}/"

        if not os.path.isfile(dir_string + "first_eigenvalue.txt"):
    
            optim_algo(
                V,
                I,
                pi_arr,
                Z,
                p=p,
                a=a,
                b=b,
            )
        
        d_opt = np.loadtxt(dir_string + "d_opt.txt")
        ax.plot(XX, d_opt, label=f"a={a_rounded}")

        d_opt_gap = np.loadtxt(dir_string + "d_opt_gap.txt")
        gaps.append(d_opt_gap)

    ax.legend()
    fig.savefig("data/" + V.__name__ + "/varying_lower_bound_optimal_diffusions.png")

    plt.clf()
    plt.plot(a_range, gaps)
    plt.xlabel("a")
    plt.ylabel("Spectral gaps")
    plt.savefig("data/" + V.__name__ + "/varying_lower_bound_spectral_gaps.png")
