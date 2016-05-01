import numpy as np


def basis_fun(timesteps, num_basis, h=1):
    t = np.linspace(0, 1, timesteps)
    n = np.linspace(0, 1, num_basis)
    N, T = np.meshgrid(n, t)

    basis = np.exp(-(T - N)**2 / (2 * h))
    return basis / basis.sum(axis=-1, keepdims=True)

print basis_fun(3, 2)