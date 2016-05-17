import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
os.environ['THEANO_FLAGS'] = 'device=cpu,' + theano_flags
os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'allow'


import theano
import theano.tensor as T
from breze.arch.construct.layer.distributions import RankOneGauss
import numpy as np


floatx = theano.config.floatX

dims = 2

mean = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(dims, dims, dims)
var = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(dims, dims, dims)
uu = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(dims, dims, dims)

# real_cov = uu[0, ...].T.dot(uu[0, ...]) + np.diagflat(var[0])
#
# print real_cov

m, v, u = (T.tensor3(i) for i in ('mean', 'var', 'u'))


gaus = RankOneGauss(m, v, u)
foo = theano.function([m, v, u], gaus.sample(), on_unused_input='warn')

# result = foo(mean, var, uu)
# print result.shape
# print result

n = int(1e4)
X = np.zeros((dims ** 3, n))
for i in xrange(n):
    X[:, i] = foo(mean, var, uu).flatten()

print 'cov:'
print np.cov(X)
print 'diag:', list(np.diag(np.cov(X)))
print 'mean:', list(X.mean(axis=1))


