import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from scipy.stats.stats import normaltest

def basis_fun(timesteps, num_basis, h=1):
    t = np.linspace(0, 1, timesteps)
    n = np.linspace(0, 1, num_basis)
    N, T = np.meshgrid(n, t)

    basis = np.exp(-(T - N)**2 / (2 * h))
    dt = t[1] - t[0]
    return basis / basis.sum(axis=0, keepdims=True) / dt


def sample(basis, dims):
    num_basis = basis.shape[-1]
    eta = random.randn(num_basis, dims)
    m = basis.dot(eta)
    cov = m.T.dot(m) + np.eye(dims)
    L = np.linalg.cholesky(cov)
    s = m + L.dot(random.randn(dims)).squeeze()
    eta = eta.flatten()
    # print s.shape, eta.shape, eta
    return np.concatenate((s, eta))


def sample_n(n_samples, basis, dims):
    a = np.zeros((dims + basis.shape[0], n_samples))
    for i in xrange(n_samples):
        a[:, i] = sample(basis, dims)
    return a


def create_hist(trials, bins, dims, basis):
    a = np.zeros((dims, N))
    for i in xrange(N):
        a[:, i] = sample(basis, dims)

    hist, bins = np.histogram(a[0], bins=1000, normed=True)
    return hist

dims = 1
num_basis = 1
timesteps = 100
N = 10000
bins = 100
t = 67
basis = basis_fun(timesteps, num_basis, h=1)

data = sample_n(N, basis[t], dims)
print 'data shape:', data.shape


mean = data.mean(axis=1, keepdims=True)
data -= mean

sigma = data.dot(data.T)
U, S, V = np.linalg.svd(sigma)
eps = 1e-8

ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + eps))), U.T)
data = np.dot(ZCAMatrix, data)

cov = np.diag(np.cov(data))
data = data / cov.reshape(2, 1)

print data.mean(axis=1)
print np.cov(data)

print normaltest(data.T)


# hists = np.zeros((bins, timesteps))
# for t in xrange(timesteps):
#     hists[:, t] = create_hist(100, bins, dims, basis[t])
#
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# line = ax.plot(hists[:, 0])[0]
# for t in xrange(1, timesteps):
#     line.set_ydata(hists[:, t])
#     plt.pause(0.1)
#
# plt.show()
#

#   scatterplot
x = data[0, :]
y = data[1, :]

print data.shape, x.shape, y.shape

from matplotlib.ticker import NullFormatter

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y, alpha=0.2)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
lim = (int(xymax/binwidth) + 1) * binwidth

axHistx.hist(x, bins=bins, normed=True)
axHisty.hist(y, bins=bins, normed=True, orientation='horizontal')


plt.show()