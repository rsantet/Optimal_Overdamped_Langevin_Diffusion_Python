"""Examples of potential energy functions defined on the torus.
"""
import numpy as np


def cos_2(x):
    return np.cos(2 * np.pi * x)


def cos_4(x):
    return np.cos(4 * np.pi * x)


def cos_6(x):
    return np.cos(6 * np.pi * x)


def cos_8(x):
    return np.cos(8 * np.pi * x)


def cos_10(x):
    return np.cos(10 * np.pi * x)


def cos_16(x):
    return np.cos(16 * np.pi * x)


def sin_squared_cubed(x):
    return (
        np.sin(np.pi / 2 + np.pi * x) ** 2
        + np.sin(np.pi / 2 + 2 * np.pi * (x - 0.2)) ** 3
    )


def sin_two_wells(x):
    return np.sin(4 * np.pi * x) * (2 + np.sin(2 * np.pi * x))


def sin_four_wells(x):
    return np.sin(8 * np.pi * x) * (2 + np.sin(4 * np.pi * x))
