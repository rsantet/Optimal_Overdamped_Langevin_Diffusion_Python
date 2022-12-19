# Optimal Langevin diffusion

This repository contains Python code to optimize the diffusion of overdamped Langevin dynamics. The method is described in the article "Optimizing the diffusion of overdamped Langevin dynamics" by T. Lelièvre, G. Pavliotis, G. Robin, R. Santet and G. Stoltz.

## Method

Overdamped Langevin dynamics are used in many MCMC sampling algorithms to produce new states. An important parameter of such dynamics is the diffusion function which has fundamental impact on the convergence rate of Langevin dynamics. The method developed in this repository aims to compute the optimal diffusion function to maximize the convergence speed of Langevin dynamics towards the target measure. The resulting optimal diffusion can be plugged into MCMC algorithms (RWMH, MALA, etc.) to accelerate their convergence towards equilibrium. The code available here concerns only *one-dimensional* problems on the *torus*.

## Files

The repository contains:
- main.py: the optimization algorithm.
- potentials.py: examples of one-dimensional potentials (minus log of target measure).
- plot_optimized_diffusion.py: plot the optimized diffusion obtain from the main.py script

## TODO
- Propose RWMH and MALA algorithms where the optimal diffusion computed with main.py can be used as parameter. Propose a visualization of the effect of the optimal diffusion in MCMC sampling algorithm.
- Propose a visualization of the behaviour of the optimal diffusion in the asymptotic regime of periodic homogenization