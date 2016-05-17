import theano
import theano.tensor as T
import numpy as np

from breze.learn.sgvb.prior import LearnableDiagGauss

n_features = 1
shape = (2, 2, n_features)

# gaus = LearnableDiagGauss(shape, n_features)
# sample = gaus.sample()
#
# s = T.scalar('s')
# foo = theano.function([s], sample, on_unused_input='warn')
# print foo(1)

rng = T.shared_randomstreams.RandomStreams()
noise_tensor = rng.normal(size=(1, 1, shape[-1]))
noise_tensor2 = rng.normal(size=(1, 1, shape[-1]))
tiled_noise = T.tile(noise_tensor, (2, 2, 1), ndim=len(shape))

# noise_tensor2 = rng.normal(size=(1, 1, shape[-1]))
# noise_tensor2 = T.tile(noise_tensor2, (2, 2, 1), ndim=len(shape))
# noise_tensor2 = theano.clone(noise_tensor, copy_inputs=True)

final_noise = T.concatenate((tiled_noise, theano.clone(tiled_noise, {noise_tensor:noise_tensor2})), axis=-1)

noise_foo = theano.function([], final_noise)
noise = noise_foo()


print noise.shape
print noise