import numpy as np
from numpy import random
from matplotlib import pyplot as plt


# def basis_fun(timesteps, num_basis, h=1):
#     t = np.linspace(0, 1, timesteps)
#     n = np.linspace(0, 1, num_basis)
#     N, T = np.meshgrid(n, t)
#
#     basis = np.exp(-(T - N)**2 / (2 * h))
#     dt = t[1] - t[0]
#     return basis / basis.sum(axis=0, keepdims=True) / dt
#
#
# dims = 50
# num_basis = 10
# timesteps = 100
#
#
# basis = basis_fun(timesteps, num_basis, h=0.005)
# w_mean = random.randn(num_basis, dims) + basis.mean(axis=0, keepdims=True).T
# w_diag = random.randn(num_basis, dims) + basis.mean(axis=0, keepdims=True).T
# w_cov1 = random.randn(num_basis, dims) + basis.mean(axis=0, keepdims=True).T
# w_cov2 = random.randn(num_basis, dims) + basis.mean(axis=0, keepdims=True).T
#
# mean = basis.dot(w_mean) / basis.sum(axis=1, keepdims=True)
# diag = basis.dot(w_diag) / basis.sum(axis=1, keepdims=True)
# cov1 = basis.dot(w_cov1) / basis.sum(axis=1, keepdims=True)
# cov2 = basis.dot(w_cov2) / basis.sum(axis=1, keepdims=True)
#
# plt.ion()
# fig = plt.figure()
# mean_ax = fig.add_subplot(2, 1, 1)
# mean_ax.set_xlim([0, dims-1])
# mean_ax.set_ylim([-6, 6])
# cov_ax = fig.add_subplot(2, 1, 2)
# bases_line = mean_ax.plot([], [], 'r', label='bases')[0]
# foo_line = mean_ax.plot([], [], 'b', label='foo')[0]
# mean_ax.legend()
# mean_ax.grid()
#
# # mean = random.randn(*mean.shape)
#
# for t in xrange(timesteps):
#     c1, c2, d = cov1[[t]], cov2[[t]], diag[t]
#     cov = c1.T.dot(c2) + np.diagflat(d**2)
#     cov -= mean[[t]].T.dot(mean[[t]])
#     # cov = np.max(cov, 0)
#     # print cov.shape
#     foo_line.set_data(xrange(mean[t].size), mean[t])
#     cov_ax.matshow(cov)
#     plt.pause(0.00001)
#
# plt.show()
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