import numpy as np
from numpy import trace
from numpy.random import randn
from numpy.linalg import det, cond


def norm(x, n=2):
    return (x**n).sum() ** (1.0/n)

n = 3

A = randn(n, n)
x = randn(n, 1)



print norm(x)
print det(A)
print (abs(trace(A))) * norm(x, 1)
print norm(A.dot(x), 1)