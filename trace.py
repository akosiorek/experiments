import numpy as np
from numpy import random

x = random.randn(5, 1)
y = random.randn(5, 1)
z = random.randn(5, 1)

X = x * x.T
Y = np.diag(y.squeeze())
Z = np.diag(z.squeeze())
print X
print Y
print Z

A = Z * Y * X * Y
a = z * y * x * x * y

print A.trace(), a.sum()