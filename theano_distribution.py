import os

theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
os.environ['THEANO_FLAGS'] = 'device=cpu,' + theano_flags

import numpy as np
import theano
import theano.tensor as T

from breze.arch.construct.layer.distributions import NormalGauss, RankOneGauss

# inpt = T.vector('inpt')
# ng = NormalGauss(inpt.shape)
# normal_sample = theano.function([inpt], ng.sample())

mean = T.dtensor3('mean')
var = T.dtensor3('var')
uu = T.dtensor3('u')
rog = RankOneGauss(mean, var, uu)


rank_sample = theano.function([mean, var, uu], rog.sample(), on_unused_input='warn')

# v = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=theano.config.floatX).reshape(2, 2, 2)
m = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=theano.config.floatX).reshape(2, 2, 2)
v = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=theano.config.floatX).reshape(2, 2, 2)
u = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=theano.config.floatX).reshape(2, 2, 2)

# m = np.asarray([1, 1], dtype=theano.config.floatX).reshape(1, 1, 2)
# v = np.asarray([1, 1], dtype=theano.config.floatX).reshape(1, 1, 2)
# u = np.asarray([1, 1], dtype=theano.config.floatX).reshape(1, 1, 2)

# print normal_sample([1, 2, 3])


samples = [[], [], [], []]
for i in xrange(10000):
    sample = rank_sample(m, v, u)
    for j in xrange(2):
        for k in xrange(2):
            samples[j*2+k].append(sample[j, k, :])

for j in xrange(2):
        for k in xrange(2):
            sample = np.vstack(samples[j*2+k]).T
            print sample.shape, sample.mean()
            print np.cov(sample)


