"How does a gaussian change when passed through a nonlinearity like tanh?"

import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal



fig = plt.figure()
ax_gaus1d = fig.add_subplot(2, 2, 1)
ax_gaus2d = fig.add_subplot(2, 2, 3)
ax_trans1d = fig.add_subplot(2, 2, 2)
ax_trans2d = fig.add_subplot(2, 2, 4)


x, y = np.mgrid[-1:1:.1, -1:1:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
# rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
rv = multivariate_normal(mean=None, cov=[[1, 0], [0, 1]])
pdf = 50 * rv.pdf(pos) - 5


cs_gaus = ax_gaus2d.contourf(x, y, pdf)
cs_trans = ax_trans2d.contourf(x, y, np.sin(pdf) + np.cos(pdf))
plt.colorbar(cs_gaus, ax=ax_gaus2d)
plt.colorbar(cs_trans, ax=ax_trans2d)

plt.show()