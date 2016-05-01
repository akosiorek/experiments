import theano
import theano.tensor as T
import numpy as np

M = T.matrix('M')
N = T.tensor3('N')

foo = T.dot(M, N)
f = theano.function([N, M], foo)


m = np.array([[1, 2], [3, 4]], dtype=theano.config.floatX)
n = np.asarray([1, 2, 3, 4], dtype=theano.config.floatX).reshape((2, 2, 1))

print m.shape, n.shape
print m
print n
print np.equal(m.dot(n), f(n, m)).all()