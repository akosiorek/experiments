"""Computes PmP-like basis functions, samples weight vectors and plots resulting curves. I used it to get an idea
of what shapes can I get from the pmp-prior"""

import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import theano
import theano.tensor as T


n_basis = 10
n_feature = 1024
n_time_steps = 128
w = 1
dt = 1. / n_time_steps

inpt = T.tensor3('inpt')

n_time_steps, _, _ = inpt.shape
timesteps = T.arange(0, 1, 1. / n_time_steps)
basis_indices = T.constant(np.linspace(-1, 1, n_basis), dtype=theano.config.floatX)

weights = T.constant(random.randn(1, n_basis, n_feature), 'weights', dtype=theano.config.floatX)
widths = T.constant(random.rand(n_basis), 'widhts', dtype=theano.config.floatX)


def times_basis(dt, w, b, width):
    basis = T.exp(-(dt - b) ** 2 / (2 * width))
    norm_constant = basis.sum()
    return T.dot(basis / norm_constant, w)


feature, _ = theano.scan(times_basis, sequences=[timesteps],
                         non_sequences=[weights, basis_indices, widths ** 2])


foo = theano.function([inpt], feature)


i = random.rand(15, 10, 10).astype(theano.config.floatX)
result = foo(i)
print result.shape

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result.squeeze(), 'k', alpha=0.2)
plt.show()