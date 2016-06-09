import theano
import theano.tensor as T
import numpy as np

xx = T.matrix('x')
AA = T.tensor3('A')

foo = T.dot(AA, xx)
f = theano.function([AA, xx], foo)
floatx = np.float32


A = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape((2, 2, 2))
x = np.asarray([1, 2, 3, 4], dtype=floatx).reshape((2, 1, 2))

print 'A:', A.shape
print 'x:', x.shape
# Ax = x.dot(A)
# print 'A * x:', Ax.shape
print f(A, x).shape
# print A
# print x
# print np.equal(A.dot(x), f(x, A)).all()