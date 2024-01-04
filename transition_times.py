import numpy as np
import os
from pathlib import Path
from potentials import *

def d_interp(d, I, x):
    """
    Outputs the value of the diffusion coefficient d at point x using a piecewise linear approximation

    Args:
        d (list): the diffusion coefficient values for the point in the mesh
        I (int): number of point in the mesh (length of vector d)
        x0 (float): position where to compute the diffusion coefficient
    """

    q = x % 1
    k = np.floor(I * q).astype(int)
    k1 = k + 1
    if k == I - 1:
        k1 = 0
    elif k == I:
        return d[0]
    return (d[k] * (q - k / I) + d[k1] * ((k + 1) / I - q)) * I

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

def get_one_transition(
    d, I, dt, x0, V
):
    i = 0
    x = x0
    sqrt_2_dt = np.sqrt(2 * dt)
    MH_counter = 0
    while x0 - 1 <= x <= x0 + 1:
        G = np.random.randn()
        d_x = d_interp(d, I, x)
        proposal = x + sqrt_2_dt * np.sqrt(d_x) * G
        d_proposal = d_interp(d, I, proposal)
        V_x = V(x)
        V_proposal = V(proposal)
        sqrt_d_ratio = np.sqrt(d_x / d_proposal)
        G_proposal = sqrt_d_ratio * G
        alpha = np.log(sqrt_d_ratio) - (V_proposal - V_x) - (G_proposal**2 - G**2) / 2
        if np.log(np.random.rand()) > alpha:
            MH_counter += 1
        else:
            x = proposal
        i += 1
    return i * dt, MH_counter / i

def transition_times(
    d, I, dt, N, x0, V
    ):

    times = []
    MH = []
    for i in range(N):
        if i % 100 == 0:
            print(f"Iteration {i}/{N}", flush=True)

        T, MH_ratio = get_one_transition(
            d, I, dt, x0, V
        )
        times.append(T)
        MH.append(MH_ratio)

    return times, MH

if __name__ == '__main__':

    V = sin_two_wells
    # global min of potential is 0.3654418277735119 #
    # local min of potential is 0.8971356821573657 # 
    x0 = 0.3654418277735119
    N = 10000
    
    # Optimal diffusion
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

    # Constant diffusion
    d_constant = np.ones(I) / lp_constraint(np.ones(I), I, mu_arr, p)


    dt_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    for (idx, dt) in enumerate(dt_range):
        for (d, d_string) in zip(
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
        ):
            file_string = "data/" + V.__name__ + f"/transition_times_dt_{dt}_N_{N}_{d_string}"
            if not os.path.isfile(file_string + ".txt"):
                print(f"Running for dt = {dt} and {d_string}", flush=True)
                times, MH_ratios = transition_times(d, I, dt, N, x0, V)
                np.savetxt(file_string + ".txt", times)
                np.savetxt(file_string + "_MH_ratios.txt", MH_ratios)

