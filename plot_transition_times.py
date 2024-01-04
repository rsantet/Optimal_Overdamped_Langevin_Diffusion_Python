import numpy as np
import matplotlib.pyplot as plt
from potentials import *
import os

dt_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
linestyles = ["solid", "dashed", "dashdot", "dotted"]
linewidths = [3, 2, 2]
colors = ["red", "blue", "black"]
labels=["optimal diffusion", "homogenized diffusion", "constant diffusion"]
V = sin_two_wells

N = 10000

fig_T, ax_T = plt.subplots()
fig_MH, ax_MH = plt.subplots()

for (idx, d_string) in enumerate(
    [
    "d_opt",
    "d_homog",
    "d_constant"
]
):

    if not os.path.isfile("data/" + V.__name__ + f"/means_transition_times_N_{N}_{d_string}.txt"):
        T_means = []
        MH_means = []
        T_stds = []
        MH_stds = []

        for dt in dt_range:
            file_string = "data/" + V.__name__ + f"/transition_times_dt_{dt}_N_{N}_{d_string}"
            times = np.loadtxt(
                file_string + ".txt"
            )
            MH_ratios = np.loadtxt(
                file_string + "_MH_ratios.txt"
            )
            T_mean = np.mean(times)
            MH_mean = np.mean(MH_ratios)
            T_std = np.std(times)
            MH_std = np.std(MH_ratios)
            T_means.append(T_mean)
            T_stds.append(T_std)
            MH_means.append(MH_mean)
            MH_stds.append(MH_std)

        np.savetxt("data/" + V.__name__ + f"/means_transition_times_N_{N}_{d_string}.txt", T_means)
        np.savetxt("data/" + V.__name__ + f"/means_MH_transition_times_N_{N}_{d_string}.txt", MH_means)
        np.savetxt("data/" + V.__name__ + f"/stds_transition_times_N_{N}_{d_string}.txt", T_stds)
        np.savetxt("data/" + V.__name__ + f"/stds_MH_transition_times_N_{N}_{d_string}.txt", MH_stds)

    else:
        T_means = np.loadtxt("data/" + V.__name__ + f"/means_transition_times_N_{N}_{d_string}.txt")
        MH_means = np.loadtxt("data/" + V.__name__ + f"/means_MH_transition_times_N_{N}_{d_string}.txt")
        T_stds = np.loadtxt("data/" + V.__name__ + f"/stds_transition_times_N_{N}_{d_string}.txt")
        MH_stds = np.loadtxt("data/" + V.__name__ + f"/stds_MH_transition_times_N_{N}_{d_string}.txt")
    
    ax_T.errorbar(
        dt_range,
        T_means,
        yerr=np.array(T_stds) * 1.96 / np.sqrt(N),
        label=labels[idx],
        color=colors[idx],
        linewidth=linewidths[idx],
        linestyle=linestyles[idx],
    )
    

    ax_MH.errorbar(
        dt_range,
        MH_means,
        yerr=np.array(MH_stds) * 1.96 / np.sqrt(N),
        label=labels[idx],
        color=colors[idx],
        linewidth=linewidths[idx],
        linestyle=linestyles[idx],
    )

ax_MH.plot(
    dt_range,
    np.power(10 * np.array(dt_range), 0.5),
    label="dt^{1/2}"
)

ax_T.set_xscale("log")
ax_T.set_xlabel("dt")
ax_T.set_ylabel("Physical time")
ax_T.legend()

ax_MH.set_xscale("log")
ax_MH.set_yscale("log")
ax_MH.set_ylim(10**(-2),1)
ax_MH.set_xlabel("dt")
ax_MH.set_ylabel("Rejection probability")
ax_MH.legend()
fig_T.savefig(f"data/" + V.__name__ + f"/transition_times_N_{N}.png")
fig_MH.savefig(f"data/" + V.__name__ + f"/transition_times_MH_ratios_N_{N}.png")