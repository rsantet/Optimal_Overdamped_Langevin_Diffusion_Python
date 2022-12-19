"""Plot the optimal diffusion obtained from the main.py script.
"""

import matplotlib.pyplot as plt
import numpy as np
from potentials import *

I = 500
a = 0.0
b = np.inf
a_rounded = np.round(a, 2)
b_rounded = np.round(b, 2)
XX = [i / I for i in range(0, I)]
V = sin_two_wells
dir_string = "data/" + V.__name__ + f"/I_{I}_a_{a_rounded}_b_{b}/"
d_opt = np.loadtxt(dir_string + "d_opt.txt")
d_homog = np.fromiter(map(lambda x: np.exp(V(x)), XX), float)

plt.plot(XX, d_opt, label="Optimal diffusion")
plt.plot(XX, d_homog, label="Homogenized diffusion")
plt.legend()
plt.savefig(dir_string + "d_opt.png")
plt.show()
