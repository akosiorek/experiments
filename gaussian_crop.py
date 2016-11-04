import numpy as np
import matplotlib.pyplot as plt



M, N = 10, 10
A, B = 5, 5

# x = np.zeros((M, N))
x = abs(np.random.randn(M, N)) * .3
x[3:6, 3:6] = 1


def gaussian_mask_naive(u, s, d, R, C):
    mask = np.zeros((R, C))
    for r in xrange(R):
        ur = u + r * d
        for c in xrange(C):
            mask[r, c] = np.exp(-.5 * ((c - ur) / s) ** 2)

    return mask / mask.sum(axis=1, keepdims=True)


def gaussian_mask(u, s, d, R, C):
    ur = u + np.arange(R).reshape((R, 1)) * d
    mask = np.arange(C).reshape((1, C)) - ur
    mask = np.exp(-.5 * (mask / s) ** 2)
    return mask / mask.sum(axis=1, keepdims=True)


# Fy = gaussian_mask_naive(u=2, s=.5, d=1, R=A, C=M)
Fy = gaussian_mask(u=2, s=.5, d=1, R=A, C=M)
Fx = gaussian_mask(u=2, s=.5, d=1, R=B, C=N)

assert np.equal(gaussian_mask(u=2, s=.5, d=1, R=A, C=M), gaussian_mask_naive(u=2, s=.5, d=1, R=A, C=M)).all()



cropped = Fy.dot(x).dot(Fx.T)

cmap = plt.get_cmap('gray')
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(x, cmap=cmap)
axes[0, 1].imshow(cropped, cmap=cmap)
axes[1, 0].imshow(Fy, cmap=cmap)
axes[1, 1].imshow(Fx.T, cmap=cmap)

plt.show()
