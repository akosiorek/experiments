import numpy as np
from numpy import random
from scipy.linalg import cholesky, lu


def mse(X, Y):
    return((X - Y)**2).sum()

A = random.rand(10, 10)
x = random.randn(10, 100)

A = A.dot(A.T)
print A.shape
L, U = lu(A, permute_l=True)
C = cholesky(A)

print mse(L.dot(U), C.T.dot(C))

i = 0

a = x[:, [i]]

x_A = A.dot(x)
x_LU = L.dot(U).dot(x)
print mse(x_A, x_LU)


