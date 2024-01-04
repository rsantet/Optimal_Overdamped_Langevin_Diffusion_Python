# Optimal Overdamped Langevin diffusion

This repository contains Python code to optimize the diffusion of overdamped Langevin dynamics. The method is described in the article "Optimizing the diffusion of overdamped Langevin dynamics" by T. Lelièvre, G. Pavliotis, G. Robin, R. Santet and G. Stoltz.

## Method

Overdamped Langevin dynamics are used in many MCMC sampling algorithms to produce new states. An important parameter of such dynamics is the diffusion function which has fundamental impact on the convergence rate of Langevin dynamics. The method developed in this repository aims to compute the optimal diffusion function to maximize the convergence speed of Langevin dynamics towards the target measure. The resulting optimal diffusion can be plugged into MCMC algorithms (RWMH, MALA, etc.) to accelerate their convergence towards equilibrium. The code available here concerns only *one-dimensional* problems on the *torus*.

## Files

The repository contains:
- main.py: the optimization algorithm
- potentials.py: examples of one-dimensional potentials (minus log of target measure)
- plot_optimized_diffusion.py: plot the optimized diffusion obtain from the main.py script
- RWMH.py: run the RWMH algorithm
- plot_RWMH.py: plot a sample trajectory of the RWMH algorithm
- msd.py: compute and plot the Mean Square Displacement for various diffusion coefficients (optimal, homogenized, constant)
- transition_times.py: compute the duration time to transition between two metastable states of a 1D double well potential for various diffusion coefficients (optimal, homogenized, constant)
- plot_transition_times.py: plot the results obtained from the previous script
- varying_lower_bound.py: compute the optimal diffusion coefficient for various lower bounds $a\in[0,1]$.