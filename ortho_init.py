from numpy import random
import numpy as np
from numpy.linalg import qr
from climin.initialize import orthogonal

def ortho_initialize(arr):
    if arr.shape[-1] > arr.shape[0]:
        q, _ = np.linalg.qr(arr.T)
        q = q.T
    else:
        q, _ = np.linalg.qr(arr)

    return q


n, m = 2, 3


A = random.randn(n, m)
print A
B = ortho_initialize(A)
print B

C = A.copy().reshape(-1)
orthogonal(C, n)

U, S, V = np.linalg.svd(B)
print S
print A.shape
print B.shape

print np.linalg.cond(B)

print C

print np.linalg.cond(C.reshape(n, m))