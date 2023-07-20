"""Plot the a trajectory using the RWMH algorithm
"""

import matplotlib.pyplot as plt
import numpy as np
from potentials import *
from RWMH import RWMH

I = 1000
a = 0.0
b = np.inf
beta = 1.0
a_rounded = np.round(a, 2)
b_rounded = np.round(b, 2)
XX = [i / I for i in range(0, I)]
V = cos_2
dir_string = "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b}_beta_{beta}/"
d_opt = np.loadtxt(dir_string + "d_opt.txt")

dt = 0.01
x0 = np.random.rand()
N_it = 10**4

trajectory, MH_rejection_probability = RWMH(d_opt, I, dt, N_it, x0, V, beta)

print(f"MH Rejection probability is {MH_rejection_probability}")

plt.plot(np.arange(N_it), trajectory)
plt.show()
