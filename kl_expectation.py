import theano
import theano.tensor as T
import numpy as np
from numpy import random

from breze.arch.construct.layer.distributions import NormalGauss, RankOneGauss, DiagGauss
from breze.arch.construct.layer.kldivergence import kl_div
from breze.learn.sgvb.movement_prior import ProbabilisticMovementPrimitive

n_trials = 100
n_repeats = 1


width = 0.025
n_bases = 5
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

pmp = ProbabilisticMovementPrimitive(n_bases, mean, var**2, uu, width=width, eps=1e-5)

diag = DiagGauss(dmean, dvar**2)

kl_coord_wise = kl_div(diag, pmp)
kl_sample_wise = kl_coord_wise.sum(axis=2)
kl = kl_sample_wise.mean()

grad_kl = T.grad(kl, [dmean, dvar])


foo_kl = theano.function([dmean, dvar], kl)
foo_grad = theano.function([dmean, dvar], grad_kl)


def evaluate(trials, repeats_per_trial, batch_size):
    kls = []
    grads = []

    for _ in xrange(trials):
        dm, dv = (random.randn(n_timesteps, batch_size, n_dims).astype(floatx) for _ in xrange(2))
        kl = foo_kl(dm, dv)
        grad = foo_grad(dm, dv)

        for _ in xrange(repeats_per_trial - 1):
            kl += foo_kl(dm, dv)
            new_grad = foo_grad(dm, dv)
            grad = [g + gg for g, gg in zip(grad, new_grad)]

        kl /= n_repeats
        grad = [g / n_repeats for g in grad]
        grad = [g.mean(axis=(0, 1)) for g in grad]
        kls.append(kl)
        grads.append(grad)

    kl_mean, kl_std = [f(kls) for f in (np.mean, np.std)]

    grads = np.asarray(grads)
    grad_norm = np.sqrt((grads**2).sum() / np.prod(grads.shape[:2]))
    grad_std = np.std(grads, axis=0).max()

    return kl_mean, kl_std, grad_norm, grad_std


# print evaluate(n_trials, n_repeats, n_samples)


batch_sizes = np.concatenate([[1], np.linspace(10, 100, 10)])
vals = [evaluate(n_trials, n_repeats, bs) for bs in batch_sizes]
vals = np.asarray(vals)
print vals.shape

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(batch_sizes, vals[:, 3])
ax.grid()
ax.set_xlabel('batch size')
ax.set_ylabel('gradient std')
plt.show()

print batch_sizes


