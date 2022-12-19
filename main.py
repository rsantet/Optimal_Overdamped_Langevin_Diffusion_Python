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

def optim_algo(V, I, pi, Z, p=2., a=0., b=np.inf, save=True, rewrite_save=True):
    """Optimization algorithm to find optimal diffusion

    Args:
        V (function): potential energy function defined on the torus
        I (int): number of hat functions in P1 Finite Elements basis
        pi (list): discrete approximation of the Boltzmann-Gibbs measure
        Z (float): partition function, equal to \int exp(-V)
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
        return (np.sum(x**p) / I)**(1/p)
    
    def construct_M(I, pi):
        """Construct the matrix appearing on the right hand side of the generalized eigenvalue problem

        Args:
            I (int): number of hat functions in P1 Finite Elements basis
            pi (list): discrete approximation of the Boltzmann-Gibbs measure

        Returns:
            csr_matrix: sparse tri-diagonal matrix (with periodic boundary conditions implemented)
        """
        M = np.zeros((I, I))
        for i in range(I - 1):
            M[i, i] = M[i, i] + pi[i] / (3 * I)
            M[i + 1, i + 1] = M[i + 1, i + 1] + pi[i] / (3 * I)
            M[i, i + 1] = M[i, i + 1] + pi[i] / (6 * I)
            M[i + 1, i] = M[i + 1, i] + pi[i] / (6 * I)
        M[I - 1, I - 1] = M[I - 1, I - 1] + pi[I - 1] / (3 * I)
        M[0, 0] = M[0, 0] + pi[I - 1] / (3 * I)
        M[I - 1, 0] = M[I - 1, 0] + pi[I - 1] / (6 * I)
        M[0, I - 1] = M[0, I - 1] + pi[I - 1] / (6 * I)
        M = csr_matrix(M)
        return M

    def construct_A(x, I, Z):
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
            A[i, i] = A[i, i] + x[i] / Z * I
            A[i + 1, i + 1] = A[i + 1, i + 1] + x[i] / Z * I
            A[i, i + 1] = A[i, i + 1] - x[i] / Z * I
            A[i + 1, i] = A[i + 1, i] - x[i] / Z * I
        A[I - 1, I - 1] = A[I - 1, I - 1] + x[I - 1] / Z * I
        A[0, 0] = A[0, 0] + x[I - 1] / Z * I
        A[I - 1, 0] = A[I - 1, 0] - x[I - 1] / Z * I
        A[0, I - 1] = A[0, I - 1] - x[I - 1] / Z * I
        A = csr_matrix(A)
        return A

    def objective_function(x, I, Z, M):
        """Compute second eigenvalue (spectral gap)

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            Z (float): partition function, equal to \int exp(-V)
            M (csr_matrix): matrix appearing on the right hand side of the generalized eigenvalue problem
            
        Returns:
            float: first positive eigenvalue
        """

        # Construct the matrix depending on D
        A = construct_A(x, I, Z)

        # Obtain eigenvalues
        # Note that we solve for Mu=\sigma (A+M)u
        # and retrieve \lambda = (1-\sigma)/\sigma
        valp = eigsh(M, k=4, M=A + M, return_eigenvectors=False)
        idx = np.argsort(valp)
        valp = valp[idx]
        np.divide(1 - valp, valp, out=valp)
        
        return valp[-2]

    def objective_function_gradient(x, I, Z, M):
        """Compute the gradient of the second eigenvalue with respect to x

        Args:
            x (list): variable to be optimized, equal to mu*D
            I (int): number of hat functions in P1 Finite Elements basis
            Z (float): partition function, equal to \int exp(-V)
            M (csr_matrix): matrix appearing on the right hand side of the generalized eigenvalue problem
            
        Returns:
            tuple: (gradient (list), first eigenvalue (float), second eigenvalue (float), third eigenvalue (float), fourth eigenvalue (float), constraint value (float))
        """
        
        # Construct the matrix depending on D
        A = construct_A(x, I, Z)

        # Obtain eigenvalues and eigenvectors
        # Note that we solve for Mu=\sigma (A+M)u
        # and retrieve \lambda = (1-\sigma)/\sigma
        # the eigenvectors stay the same
        valp, vecp = eigsh(M, k=4, M=A + M)

        idx = np.argsort(valp)
        valp = valp[idx]

        np.divide(1 - valp, valp, out=valp)

        # retrieve eigenvector corresponding to first positive eigenvalue
        vecp = np.real(vecp[:, idx[-2]])
        # Normalize in case it is not done properly by the algorithm
        vecp = vecp / np.sqrt(np.dot(vecp, M.dot(vecp)))
        
        # for ergonomic reasons for gradient computation
        vp = np.zeros(I + 1)
        vp[range(I)] = vecp
        vp[I] = vp[0]
        # matrix corresponding to the tri-diagonal matrix A
        mat = np.array(([1, -1], [-1, 1]))
        
        # compute the gradient
        gradD = np.zeros(I)
        for i in range(I):
            gradD[i] = np.dot(np.dot(mat, vp[range(i, i + 2)]), vp[i : (i + 2)]) * I / Z
            
        # compute the constraint 
        constraint = lp_constraint(x, I, p)
        return gradD, valp[-1], valp[-2], valp[-3], valp[-4], constraint
    
    def f0(x):
        """Function to be minimized during the optimization procedure

        Args:
            x (list): variable to be optimized, equal to mu*D

        Returns:
            float: negative spectral gap
        """
        f_val = objective_function(x, I, Z, M)
        print(f_val)
        return -f_val

    def Df0(x):
        """Gradient of the function which is minimized during the optimization procedure

        Args:
            x (list): variable to be optimized, equal to mu*D

        Returns:
            list: negative gradient of the spectral gap with respect to x
        """
        tup = objective_function_gradient(x, I, Z, M)
        if save:
            first_eigenvalue.append(tup[1])
            second_eigenvalue.append(tup[2])
            third_eigenvalue.append(tup[3])
            fourth_eigenvalue.append(tup[4])
            constraints.append(tup[5])
        return -tup[0]
    
    ##### Optimization procedure
    
    # Saving process
    if save:
        a_rounded = np.round(a,2)
        b_rounded = np.round(b,2)
        dir_string = "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b_rounded}/"
        
        if not rewrite_save:
            if os.path.isfile(dir_string + "first_eigenvalue.txt"):
                return
        
        first_eigenvalue = []
        second_eigenvalue = []
        third_eigenvalue = []
        fourth_eigenvalue = []
        constraints = []

    # Construct the matrix on the right hand side once and for all
    M = construct_M(I, pi)
    
    # First guess for the optimization procedure, D=1/mu (homogenized diffusion)
    x_init = np.ones(I)

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
        options={"disp": True, "maxiter": 1000},
        tol=1e-8,
    )
    
    # If failure
    if not res.success:
        print(res.message)
        print("\n")
        print("Not saving")
        return
    
    # If success, saves
    if save:
        x_opt = res.x
        XX = [i / I for i in range(0, I)]
        mu_arr = np.fromiter(map(lambda x:np.exp(V(x)), XX), float)
        d_opt = x_opt * mu_arr
        gap_opt = -res.fun
        min_D = np.min(x_opt)

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
        np.savetxt(dir_string + "D_opt_min.txt", np.asarray([min_D]))
        np.savetxt(dir_string + "constraint.txt", constraints)

    # return res object
    return res



if __name__ == '__main__':
    
    I = 100
    p=2
    a = 0.
    b = np.inf
    V = cos_4
        
    def mu(x):
        return np.exp(-V(x))
    XX = [i / I for i in range(0, I)]
    mu_arr = np.fromiter(map(mu, XX), float)
    Z = np.sum(mu_arr) / I
    pi_arr = mu_arr / Z
    
    #D_constant = (np.sum(mu_arr**p)/I)**(-1/p)
    
    optim_algo(
        V, I, 
        pi_arr,
        Z,
        p=p,
        a=a, b=b,
    )