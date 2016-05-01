import theano
from theano import scan
from theano import tensor as T

import numpy as np


def mult(a, b):
    return a * b

parts = T.vector('parts')
x = T.scalar('x')

results, updates = scan(mult, sequences=parts, non_sequences=x)


products = theano.function(inputs=[parts, x], outputs=results)


test = np.arange(0, 10, dtype=theano.config.floatX)

print products(test, 2)
print test * 2