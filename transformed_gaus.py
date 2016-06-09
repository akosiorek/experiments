import numpy as np
from numpy import random, sqrt
from numpy.linalg import inv, cholesky


d = 2
n = 100000


X = random.randn(d, n)

# u = random.rand(d, 1)
# D = np.diag(random.rand(d))

u = np.asarray([1, 1])
D = np.diag([1, 1])

original_cov = u.dot(u.T) + D
R = cholesky(original_cov)

sample_cov = np.cov(X)
expected_cov = R.dot(sample_cov).dot(R.T)

invD = inv(D)
eta = 1 / (u.T.dot(invD).dot(u) + 1)
a = (1 - sqrt(eta)) / u.T.dot(invD).dot(u)
print eta, a
# R = (np.eye(d) - a * invD.dot(u).dot(u.T)).dot(invD ** 0.5)
# R = np.linalg.inv(R).T
# R = sqrt(D).dot(inv(np.eye(d) - a * invD.dot(u).dot(u.T))).T

R = np.linalg.solve((np.eye(d) - a * invD.dot(u).dot(u.T)).T, sqrt(D))

transformed_cov = np.cov(R.dot(X))

print expected_cov
print np.allclose(expected_cov, transformed_cov), ((expected_cov - transformed_cov)**2).sum()
print expected_cov
print transformed_cov