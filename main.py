"""One-dimensional optimization procedure to compute numerically the optimal diffusion for the overdamped Langevin process on the torus.

This script uses SLSQP for the optimization procedure.
"""

import numpy as np
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import scipy.optimize as opt
from scipy.optimize import Bounds
import os
from potentials import *

# Variables optimized in the code are x[i] =  exp(-\beta V(q_i))D(q_i)

# Generalized eigenvalue problem: Au=\lambda Mu


def optim_algo(V, I, beta=1.0, p=2.0, a=0.0, b=np.inf, save=True, rewrite_save=True):
    """Optimization algorithm to find optimal diffusion

    Args:
        V (function): potential energy function defined on the torus
        I (int): number of hat functions in P1 Finite Elements basis
        beta
        p (float, optional): L^p constraint. Defaults to 2..
        a (float, optional): lower bound for the variable constraint. Defaults to 0..
        b (float, optional): upper bound for the variable constraint. Defaults to np.inf.
        save (bool, optional): if saving first, second, third and fourth eigenvalues and the constraint value during the optimization procedure. Defaults to True.
        rewrite_save (bool, optional): if forcing save by rewriting previously saved data. Defaults to True.

    Returns:
        OptimizeResult: the optimization result represented as a OptimizeResult object. See scipy.optimize.minimize page for available attributes.
    """

    def lp_constraint(x, I, p):
        """Compute the L^p constraint

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            p (_type_): L^p constraint.

        Returns:
            float: value of the constraint for x
        """
        return (np.sum(x**p) / I) ** (1 / p)

    def construct_M(I, mu_arr):
        """Construct the matrix appearing on the right hand side of the generalized eigenvalue problem

        Args:
            I (int): number of hat functions in P1 Finite Elements basis
            mu_arr (list): discrete approximation of non-normalized target of the Boltzmann-Gibbs measure

        Returns:
            csr_matrix: sparse tri-diagonal matrix (with periodic boundary conditions implemented)
        """
        M = np.zeros((I, I))
        for i in range(I - 1):
            M[i, i] = M[i, i] + mu_arr[i] / (3 * I)
            M[i + 1, i + 1] = M[i + 1, i + 1] + mu_arr[i] / (3 * I)
            M[i, i + 1] = M[i, i + 1] + mu_arr[i] / (6 * I)
            M[i + 1, i] = M[i + 1, i] + mu_arr[i] / (6 * I)
        M[I - 1, I - 1] = M[I - 1, I - 1] + mu_arr[I - 1] / (3 * I)
        M[0, 0] = M[0, 0] + mu_arr[I - 1] / (3 * I)
        M[I - 1, 0] = M[I - 1, 0] + mu_arr[I - 1] / (6 * I)
        M[0, I - 1] = M[0, I - 1] + mu_arr[I - 1] / (6 * I)
        M = csr_matrix(M)
        return M

    def construct_A(x, I):
        """Construction of the matrix appearing on the left hand side of the generalized eigenvalue problem

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            Z (float): partition function, equal to \int exp(-V)

        Returns:
            csr_matrix: sparse tri-diagonal matrix (with periodic boundary conditions implemented)
        """
        A = np.zeros((I, I))
        for i in range(I - 1):
            A[i, i] = A[i, i] + x[i] * I
            A[i + 1, i + 1] = A[i + 1, i + 1] + x[i] * I
            A[i, i + 1] = A[i, i + 1] - x[i] * I
            A[i + 1, i] = A[i + 1, i] - x[i] * I
        A[I - 1, I - 1] = A[I - 1, I - 1] + x[I - 1] * I
        A[0, 0] = A[0, 0] + x[I - 1] * I
        A[I - 1, 0] = A[I - 1, 0] - x[I - 1] * I
        A[0, I - 1] = A[0, I - 1] - x[I - 1] * I
        A = csr_matrix(A)
        return A

    def objective_function(x, I, M):
        """Compute second eigenvalue (spectral gap)

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            M (csr_matrix): matrix appearing on the right hand side of the generalized eigenvalue problem

        Returns:
            float: first positive eigenvalue
        """

        # Construct the matrix depending on D
        A = construct_A(x, I)

        # Obtain eigenvalues
        # Note that we solve for Mu=\sigma (A+M)u
        # and retrieve \lambda = (1-\sigma)/\sigma
        valp = eigsh(M, k=4, M=A + M, return_eigenvectors=False)
        idx = np.argsort(valp)
        valp = valp[idx]
        np.divide(1 - valp, valp, out=valp)

        return valp[-2]

    def objective_function_gradient(x, I, M):
        """Compute the gradient of the second eigenvalue with respect to x

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            M (csr_matrix): matrix appearing on the right hand side of the generalized eigenvalue problem

        Returns:
            tuple: (gradient (list), first eigenvalue (float), second eigenvalue (float), third eigenvalue (float), fourth eigenvalue (float), constraint value (float))
        """

        # Construct the matrix depending on D
        A = construct_A(x, I)

        # Obtain eigenvalues and eigenvectors
        # Note that we solve for Mu=\sigma (A+M)u
        # and retrieve \lambda = (1-\sigma)/\sigma
        # the eigenvectors stay the same
        valp, vecp = eigsh(M, k=4, M=A + M)

        idx = np.argsort(valp)
        valp = valp[idx]

        np.divide(1 - valp, valp, out=valp)

        val1 = valp[-1]
        val2 = valp[-2]
        val3 = valp[-3]
        val4 = valp[-4]

        # retrieve eigenvector corresponding to first positive eigenvalue
        vecp2 = np.real(vecp[:, idx[-2]])
        # Normalize in case it is not done properly by the algorithm
        vecp2 = vecp2 / np.sqrt(np.dot(vecp2, M.dot(vecp2)))

        # retrieve eigenvector corresponding to second positive eigenvalue
        vecp3 = np.real(vecp[:, idx[-3]])
        # Normalize in case it is not done properly by the algorithm
        vecp3 = vecp3 / np.sqrt(np.dot(vecp3, M.dot(vecp3)))

        # for ergonomic reasons for gradient computation
        vp2 = np.zeros(I + 1)
        vp2[1:] = vecp2
        vp2[0] = vp2[-1]
        # for ergonomic reasons for saving
        vp3 = np.zeros(I + 1)
        vp3[1:] = vecp3
        vp3[0] = vp3[-1]

        # compute the gradient
        gradD = np.zeros(I)

        for i in range(I):
            gradD[i] = (vp2[i+1]-vp2[i])**2 * I
        # compute the constraint
        constraint = lp_constraint(x, I, p)
        return gradD, val1, val2, val3, val4, constraint, vp2, vp3

    def f0(x):
        """Function to be minimized during the optimization procedure

        Args:
            x (list): variable to be optimized, equal to mu*D

        Returns:
            float: negative spectral gap
        """
        f_val = objective_function(x, I, M)
        print(f_val, flush=True)
        return -f_val

    # Saving process
    if save:
        a_rounded = np.round(a, 2)
        b_rounded = np.round(b, 2)
        dir_string = (
            "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b_rounded}_beta_{beta}/"
        )

        if not rewrite_save:
            if os.path.isfile(dir_string + "d_opt.txt"):
                print(
                    f"Optimal diffusion already computed for these parameters, check folder {dir_string}"
                )
                return

        first_eigenvalue = []
        second_eigenvalue = []
        global second_eigenvector
        second_eigenvector = np.zeros(I + 1)
        third_eigenvalue = []
        global third_eigenvector
        third_eigenvector = np.zeros(I + 1)
        fourth_eigenvalue = []
        constraints = []
        global old_d
        old_d = 0.0
        diff_norm_d = []

    def Df0(x):
        """Gradient of the function which is minimized during the optimization procedure

        Args:
            x (list): variable to be optimized, equal to mu*D

        Returns:
            list: negative gradient of the spectral gap with respect to x
        """
        tup = objective_function_gradient(x, I, M)
        if save:
            global old_d
            diff_norm_d.append(np.max(np.abs(x - old_d)))
            old_d = x
            first_eigenvalue.append(tup[1])
            second_eigenvalue.append(tup[2])
            third_eigenvalue.append(tup[3])
            fourth_eigenvalue.append(tup[4])
            constraints.append(tup[5])
            global second_eigenvector
            second_eigenvector = tup[6]
            global third_eigenvector
            third_eigenvector = tup[7]
        return -tup[0]

    def mu(V, beta, x):
        return np.exp(-beta * V(x))

    ##### Optimization procedure

    # Construct the approximation of the non-normalized density of the Boltzmann-Gibbs measure
    XX = [i / I for i in range(0, I)]
    mu_arr = np.fromiter(map(lambda x: mu(V, beta, x), XX), float)

    # Construct the matrix on the right hand side once and for all
    M = construct_M(I, mu_arr)

    # First guess for the optimization procedure, D=1/mu (homogenized diffusion)
    x_init = np.ones(I)
    old_d = x_init

    # Constraints on the values of mu*D
    bounds = Bounds(a, b)

    # L^p constraint on mu*D
    ineq_cons_norm = {
        "type": "ineq",
        "fun": lambda x: 1 - np.sum(x**p) / I,
        "jac": lambda x: -p * x ** (p - 1) / I,
    }

    # Optimize
    res = opt.minimize(
        f0,
        x_init,
        method="SLSQP",
        jac=Df0,
        constraints=ineq_cons_norm,
        bounds=bounds,
        options={"disp": True, "maxiter": 5000, "ftol": 1e-15},
        tol=1e-15,
    )

    # If failure
    if not res.success:
        print(res.message, flush=True)
        print("\n", flush=True)
        print("Not saving", flush=True)
        return

    # If success, saves
    if save:
        x_opt = res.x
        d_opt = np.divide(x_opt, mu_arr)
        gap_opt = -res.fun
        min_d = np.min(d_opt)

        Path(dir_string).mkdir(parents=True, exist_ok=True)

        first_eigenvalue = np.asarray(first_eigenvalue)
        second_eigenvalue = np.asarray(second_eigenvalue)
        third_eigenvalue = np.asarray(third_eigenvalue)
        fourth_eigenvalue = np.asarray(fourth_eigenvalue)
        np.savetxt(dir_string + "first_eigenvalue.txt", first_eigenvalue)
        np.savetxt(dir_string + "second_eigenvalue.txt", second_eigenvalue)
        np.savetxt(dir_string + "third_eigenvalue.txt", third_eigenvalue)
        np.savetxt(dir_string + "fourth_eigenvalue.txt", fourth_eigenvalue)
        np.savetxt(dir_string + "d_opt.txt", d_opt)
        np.savetxt(dir_string + "d_opt_gap.txt", np.asarray([gap_opt]))
        np.savetxt(dir_string + "d_opt_min.txt", np.asarray([min_d]))
        np.savetxt(dir_string + "constraint.txt", constraints)
        np.savetxt(dir_string + "second_eigenvector.txt", second_eigenvector)
        np.savetxt(dir_string + "third_eigenvector.txt", third_eigenvector)
        np.savetxt(dir_string + "diff_norm_d.txt", diff_norm_d)

    # return res object
    return res


if __name__ == "__main__":
    I = 1000
    p = 2
    a = 0.0
    b = np.inf
    beta = 1.0
    V = sin_two_wells
    # D_constant = (np.sum(mu_arr**p)/I)**(-1/p)

    I_range = [1000]

    for I in I_range:
        print(I)
        optim_algo(V, I, beta, p, a, b, save=True, rewrite_save=True)
