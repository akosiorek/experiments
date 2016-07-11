import numpy as np
from numpy.random import randn
from numpy.linalg import matrix_rank, det, cond, eigvals
from scipy.stats.mstats import gmean


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


N = 10
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
    A[i] = (alphas * parts).sum(0)
    A[i] = np.triu(A[i])
    print np.diag(A[i])
    A[i][np.diag_indices(n)] = 1#abs(np.diag(A[i]))



    A[i] = A[i].dot(A[i].T)
    # # A[i] -= np.eye(n)

    # parts = np.zeros(m, n, n)
    # for j in xrange(m):
    #     parts = randn(m, n * (n + 1) / 2)





print scale
conds = [cond(a) for a in A]
dets = np.abs([det(a) for a in A])
ranks = [matrix_rank(a) for a in A]
eigs = [np.mean(np.abs(eigvals(a))) for a in A]


print 'cond:', gmean(conds)
print 'det:', gmean(dets)
print 'min rank:', min(ranks)
print 'gavg eig:', gmean(eigs)
print ' avg eig:', np.mean(eigs)


# for a, d, c in zip(A, dets, conds):
#     ord = 1.0 / a.shape[-1]
#     div = np.abs(d) ** ord
#     print d, div, det(a / div), c, cond(a / div)

print A[i]