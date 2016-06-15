import numpy as np
from numpy import random
from numpy.linalg import qr, det
from matplotlib import pyplot as plt


n = 10
m = 10
dims = 10

x = random.rand(dims, n)

dets = []
for _ in xrange(10000):
    alphas = random.rand(m)
    alphas /= np.sqrt((alphas ** 2).sum()) * 0.5
    As = [qr(random.rand(dims, dims))[0] for _ in xrange(m)]
    A = sum([alphas[i] * As[i] for i in xrange(m)])
    dets.append(np.abs(det(A)))

print np.mean(dets)
print np.var(dets)

plt.hist(dets, bins=100)
plt.show()
