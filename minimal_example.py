import theano
import theano.tensor as T
import numpy as np

from breze.arch.construct.layer.distributions import DiagGauss

n, m = 10, 5
A, B = T.matrices('A', 'B')
X = np.empty((3, 4), dtype=np.float32)
print 'X shape =', X.shape

# # this works
# S = theano.shared(np.random.randn(n, m))
# sample = T.tile(S[np.newaxis, :, :], (A.shape[0], 1, 1), ndim=3)



# this does not work
mean_val, var_val = (np.random.randn(n, m) for _ in xrange(2))
var_val = var_val**2 + 1e-5
mean_raw, var_raw = (theano.shared(v) for v in (mean_val, var_val))

mean, var = (T.tile(v[np.newaxis, :, :], (A.shape[0], 1, 1), ndim=3) for v in (mean_raw, var_raw))
gaus = DiagGauss(mean, var)
sample = gaus.sample()


foo_sample_A = theano.function([A], sample)
print 'foo_sample_A.shape =', foo_sample_A(X).shape     # (3, 10, 5)

sample += B[0, 0] * 0   # to avoid unused input errorr
sample = theano.clone(sample, {A: B})   # this should fix the missing input error, but doesn't

foo_sample_B = theano.function([B], sample)
print 'foo_sample_B.shape =', foo_sample_B(X).shape     # error
