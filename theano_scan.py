<<<<<<< HEAD
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
=======
import os

theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
os.environ['THEANO_FLAGS'] = 'device=cpu,' + theano_flags

import numpy as np
import theano
import theano.tensor as T

# print 'Vector multiplication:',
# m = T.scalar('x')
# vec = T.vector('vec')
#
# output, updates = theano.scan(lambda vec, m: m ** 2 * vec, sequences=vec, non_sequences=m)
# foo = theano.function([m, vec], output)
# inpt = np.array([1, 2, 3], dtype=theano.config.floatX)
#
# print np.equal(foo(2, inpt), 2**2 * inpt).all(), foo(2, inpt), 2**2 * inpt

# print 'Matrix-Tensor Multiplication:',
# mat = np.array([1, 2, 3, 4, 5, 6], dtype=theano.config.floatX).reshape(3, 1, 2)
# mat1 = mat[0]
# mat2 = mat[1]
# mat3 = mat[2]
# print mat1, mat2, mat3, mat1.shape
#
#
# mat = mat.reshape(1, 3, 1, 2)
# mat = np.concatenate([mat]*3)
# print 'mat shape:', mat.shape
# mult = np.asarray([1, 2], dtype=theano.config.floatX).reshape(2, 1)
# # mult = np.concatenate([mult]*3, axis=-1)
# print 'mult shape:', mult.shape
#
# MULT = T.fmatrix('mult')
# TENS = T.ftensor4('tens')
# foo = theano.function([MULT, TENS], T.dot(TENS, MULT))
# theano_result = foo(mult, mat)
# print theano_result.shape
# print theano_result

import arm_movement.data


def basis_fun(timesteps, num_basis, h=1):
        t = np.linspace(0, 1, timesteps)
        n = np.linspace(0, 1, num_basis)
        N, TT = np.meshgrid(n, t)

        basis = np.exp(-(TT - N) ** 2 / (2 * h)).astype(theano.config.floatX)
        return basis / basis.sum(axis=-1, keepdims=True)


X, VX, TX = arm_movement.data.load_data(arm_movement.data.DATA_PATH, fraction=0.05, dims=49)

num_basis = 7
basis = basis_fun(X.shape[0], num_basis)

X = X.astype(theano.config.floatX)
X = X.reshape(X.shape[0], X.shape[1], num_basis, X.shape[-1]/num_basis)
print 'X shape:', X.shape, ' basis shape:', basis.shape

TENS = T.ftensor4('tens')
MULT = theano.shared(basis, 'basis')

# output, updates = theano.scan(lambda tens, mat: T.dot(tens, mat), sequences=[TENS, MULT])

basis_indices = T.constant(np.linspace(0, 1, num_basis), dtype=theano.config.floatX)
t = T.fscalar('t')
timesteps = T.arange(0, 1, 1. / X.shape[0])


def foo(tens, tt, b):
    basis = T.exp(-(tt-b)**2/2)
    basis /= basis.sum()
    return T.dot(basis, tens)

output, updates = theano.scan(foo, sequences=[TENS, timesteps], non_sequences=basis_indices)


foo = theano.function([TENS], output)
theano_result = foo(X)
print theano_result.shape


>>>>>>> some more work
