import numpy as np


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


def RWMH(d, I, dt, N_it, x0, V, beta, Gs=[]):
    """
    Performs a Random Walk Metropolis-Hastings algorithm using the diffusion coefficient d

    Args
        d (list): the diffusion coefficient values for the point in the mesh
        I (int): number of point in the mesh (length of vector d)
        dt (float): time step for the simulation
        N_it (int): number of time steps for the simulation
        x0 (float): initial position
        V (function): potential energy function defined on the torus
        beta (float): inverse temperature
        Gs (list, optional): The gaussian increments used for the simulation. Must be of length N_it. Default to np.random.randn(N_it).
    """

    trajectory = []
    x = x0
    sqrt_2_dt = np.sqrt(2 * dt / beta)

    if len(Gs) != N_it:
        Gs = np.random.randn(N_it)

    MH_counter = 0
    for i in range(N_it):
        if i % 10000 == 0:
            print(f"{i}/{N_it}", flush=True)

        G = Gs[i]
        d_x = d_interp(d, I, x)
        proposal = x + sqrt_2_dt * np.sqrt(d_x) * G
        d_proposal = d_interp(d, I, proposal)
        V_x = V(x)
        V_proposal = V(proposal)
        sqrt_d_ratio = np.sqrt(d_x / d_proposal)
        G_proposal = sqrt_d_ratio * G
        alpha = (
            np.log(sqrt_d_ratio)
            - beta * (V_proposal - V_x)
            - (G_proposal**2 - G**2) / 2
        )

        if np.log(np.random.rand()) > alpha:
            MH_counter += 1
        else:
            x = proposal

        trajectory.append(x)

    MH_rejection_probability = MH_counter / N_it
    return trajectory, MH_rejection_probability
