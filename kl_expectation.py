import theano
import theano.tensor as T
import numpy as np
from numpy import random

from breze.arch.construct.layer.distributions import NormalGauss, RankOneGauss, DiagGauss
from breze.arch.construct.layer.kldivergence import kl_div
from breze.learn.sgvb.movement_prior import ProbabilisticMovementPrimitive

n_trials = 100

n_bases = 10
n_timesteps = 160
n_samples = 25
n_dims = 49

floatx = theano.config.floatX
dmean, dvar = [T.tensor3(i) for i in ['dmean', 'dvar']]

timesteps, samples, _ = dmean.shape

n_mean_par = n_dims * n_bases
n_pars = n_mean_par + n_dims
shape = (1, samples, n_pars)

pmp_input = NormalGauss(shape)
sample = pmp_input.sample()
sample = T.tile(sample, (timesteps, 1, 1), ndim=len(shape))

mean = sample[:, :, :n_mean_par]
var = sample[:, :, n_mean_par:] ** 2
uu = mean

pmp = ProbabilisticMovementPrimitive(n_bases, mean, var**2, uu)

diag = DiagGauss(dmean, dvar**2)

kl_coord_wise = kl_div(diag, pmp)
kl_sample_wise = kl_coord_wise.sum(axis=2)
kl = kl_sample_wise.mean()

grad_kl = T.grad(kl, [dmean, dvar])


foo_kl = theano.function([dmean, dvar], kl)
foo_grad = theano.function([dmean, dvar], grad_kl)


kls = []
grads = []

for _ in xrange(n_trials):
# for _ in xrange(1):
    dm, dv = (random.randn(n_timesteps, n_samples, n_dims).astype(floatx) for _ in xrange(2))
    kl = foo_kl(dm, dv)
    kls.append(kl)
    grad = foo_grad(dm, dv)
    grad = [g.mean(axis=(0, 1)) for g in grad]
    grads.append(grad)


print
print 'mean =', np.mean(kls)
print 'std =', np.std(kls)
print 'var =', np.var(kls)


grads = np.asarray(grads)
print 'grad shape=', grads.shape

v = np.var(grads, axis=0)
print 'var shape=', v.shape
print v
print
print v.max()