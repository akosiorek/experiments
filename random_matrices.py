import numpy as np
from numpy.random import randn
from numpy.linalg import matrix_rank, det, cond, eigvals
from scipy.stats.mstats import gmean


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


N = 100
n = 49
m = 16
# A = randn(N, n, n)


A = np.empty((N, n, n))
scale = 0.9 ** (1.0/n)
for i in xrange(N):
    parts = randn(m, n, n) * 0.1

    alphas = softmax(randn(m, 1, 1))
    # alphas = np.abs(randn(m, 1, 1))
    # alphas /= alphas.sum()

    A[i] = (alphas * parts).sum(0) + np.eye(n) * scale


    # precond = np.sqrt((A[i] ** 2).sum(axis=-1, keepdims=True))
    # print A[i].shape, precond.shape
    # print np.diag(1/np.diag(A[i])).shape
    # A[i] = A[i].dot()

    # A[i] = parts.mean(0)

print scale
conds = [cond(a) for a in A]
dets = np.abs([det(a) for a in A])
ranks = [matrix_rank(a) for a in A]
eigs = [gmean(np.abs(eigvals(a))) for a in A]


print 'cond:', gmean(conds)
print 'det:', gmean(dets)
print 'min rank:', min(ranks)
print 'avg eig:', gmean(eigs)


# for a, d, c in zip(A, dets, conds):
#     ord = 1.0 / a.shape[-1]
#     div = np.abs(d) ** ord
#     print d, div, det(a / div), c, cond(a / div)