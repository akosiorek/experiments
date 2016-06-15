import numpy as np
from numpy.random import randn




def float32(x):
    return x.astype(np.float32)

n = 3

As = float32(randn(2, n, n))
xs = float32(randn(2, n))


multiply_sum = (xs[:, :, np.newaxis] * As).sum(1)
dot = np.concatenate([a.dot(x)[np.newaxis, :] for (a, x) in zip(As, xs)], axis=0)


print multiply_sum.shape
print multiply_sum
print dot.shape
print dot

As_m_1 = As - np.eye(3)[np.newaxis, :, :]
print As.shape, As_m_1.shape
print 'for'
for i in xrange(As.shape[0]):
    print As_m_1[i] - As[i]
    print
#


# import theano
# import theano.tensor as T
#
# AA = T.tensor3()
# xx = T.matrix()
#
#
# multiply_sum_foo = theano.function([AA, xx], (xx[:, :, np.newaxis] * AA).sum(1))
# dot_foo = theano.function([AA, xx], T.batched_dot(AA, xx))
#
#
# multiply_sum_theano = multiply_sum_foo(As, xs)
# dot_theano = dot_foo(As, xs)
#
#
# print multiply_sum_theano.shape
# print multiply_sum_theano
# print dot_theano.shape
# print dot_theano