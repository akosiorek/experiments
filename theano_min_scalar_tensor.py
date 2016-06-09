import theano
from theano import scan
from theano import tensor as T

import numpy as np



i = T.tensor3('i')
r = T.min(T.concatenate([T.ones_like(i), i]))
foo = theano.function([i], r)


def s(x):
    return np.asarray(x).reshape(1, 1, 1).astype(theano.config.floatX)

print foo(s(0))
print foo(s(1))
print foo(s(5))
