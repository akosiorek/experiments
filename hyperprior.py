import numpy as np
from numpy import random as rd
from matplotlib import pyplot as plt


def gauss(x, mean=0, s=1):
    if all((hasattr(i, '__len__') for i in (x, mean))):
        v = np.zeros([x.size] + list(mean.shape))
        for i in xrange(x.size):
            v[i] = gauss(x[i], mean, s)
        print v.shape
        return v

    return np.exp(-(x-mean)**2 / 2 / s*2) / np.sqrt(2 * np.pi) / s


def foo(a, b, c, aa=1, bb=1, cc=1):
    x = np.linspace(-10, 10, 100)
    a, b, c = (gauss(x, *i) for i in ((a, aa), (b, bb), (c, cc)))
    return np.sum(a * np.log(b / c), axis=0)


u = np.mgrid[-5:5:0.1, -5:5:0.1]
v = foo(u[0], u[1], 1)
print v.shape


plt.figure()
# CS = plt.contour(u[0], u[1], v, 100)
# plt.clabel(CS, inline=1, fontsize=10)

extent = [-5, 5, -5, 5]
im = plt.imshow(v, extent=extent)
plt.plot(1, 1, 'r.', markersize=15)
plt.colorbar(im)
plt.show()



