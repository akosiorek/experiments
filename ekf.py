import numpy as np
from numpy import sin, cos, sqrt, random

import quaternion as quat
from quaternion import norm2


def fd_approx(foo, wrt, *args):
    eps = 1e-6
    y = foo(wrt, *args)
    J = np.zeros((y.size, wrt.size))

    w = wrt.ravel()
    for i in xrange(wrt.size):
        w[i] += eps
        yp = foo(w.reshape(wrt.shape), *args)
        w[i] -= 2 * eps
        ym = foo(w.reshape(wrt.shape), *args)
        w[i] += eps
        J[:, i] = (yp - ym) / (2 * eps)

    return J


def extract(x):
    return x[:4], x[4:7], x[7:]


def f(x, dt):
    p, w, e = extract(x.copy())

    v = dt * w + dt ** 2 * e / 2
    q = quat.q_from_omega(v)

    p = quat.prod(q, p)
    w += dt * e
    return np.concatenate((p, w, e))


def f_qw(x, dt):
    p, w, e = extract(x)
    v = dt * w + dt ** 2 * e / 2

    v, p = v[:, np.newaxis], p[:, np.newaxis]
    v_norm = norm2(v)
    theta = v_norm / 2
    q0w = - 0.5 * dt / v_norm * sin(theta) * v
    qw = dt / v_norm ** 2 * (sin(theta) * (v_norm * np.eye(3) - v.dot(v.T) / v_norm) + 0.5 * cos(theta) * v.dot(v.T))

    return np.vstack((q0w.T, qw))


def v_from_state(x, dt):
    p, w, e = extract(x)
    v = dt * w + dt ** 2 * e / 2
    return quat.q_from_omega(v)


def dq_dv_approx(x, dt):
    p, w, e = extract(x.copy())
    v = dt * w + dt ** 2 * e / 2
    return fd_approx(quat.q_from_omega, v)


def cross_approx(x, dt):
    p, w, e = extract(x.copy())
    v = dt * w + dt ** 2 * e / 2

    def f(v, p):
        p = p[1:]
        q = quat.q_from_omega(v)[1:]
        return np.cross(q, p)

    return fd_approx(f, v, p)


def Jf(x, dt):
    p, w, e = extract(x)

    v = dt * w + dt ** 2 * e / 2
    q = quat.q_from_omega(v)

    J = np.eye(x.size)
    J[0, :4] = q
    J[1, :4] = -q[1], q[0], q[3], -q[2]
    J[2, :4] = -q[2], -q[3], q[0], q[1]
    J[3, :4] = -q[3], q[2], -q[1], q[0]
    J[:4, :4] = J[:4, :4].T

    J[4:7, 7:] = np.eye(3) * dt

    v, p = v[:, np.newaxis], p[:, np.newaxis]
    v_norm = norm2(v)
    theta = v_norm / 2
    q0dv = - 0.5 / v_norm * sin(theta) * v.T
    qdv = (sin(theta) * (v_norm * np.eye(3) - v.dot(v.T) / v_norm) + 0.5 * cos(theta) * v.dot(v.T)) / v_norm ** 2

    dcross_dv = np.asarray(((qdv[1] * p[3] - qdv[2] * p[2]).T,
                            (qdv[2] * p[1] - qdv[0] * p[3]).T,
                            (qdv[0] * p[2] - qdv[1] * p[1]).T))

    dfp0_dv = p[0] * q0dv - qdv.dot(p[1:]).T
    dfp_dv = p[1:].dot(q0dv) + p[0] * qdv + dcross_dv

    J[0, 4:7] = dt * dfp0_dv
    J[1:4, 4:7] = dt * dfp_dv

    J[0, 7:] = dt**2 / 2 * dfp0_dv
    J[1:4, 7:] = dt**2 / 2 * dfp_dv
    return J


def pretty(row, width=16, decimals=8):
    format_str = '%{}.{}f'.format(width, decimals)
    return '|' + ' |'.join([format_str % f for f in row]) + ' |'


if __name__ == '__main__':
    dt = 0.5
    random.seed(0)
    x = random.randn(10)
    x[:4] = quat.normalize(x[:4])
    J = Jf(x, dt)#[1:4, 4:7]
    Jd = fd_approx(f, x, dt)#[1:4, 4:7]
    print np.greater((J - Jd) ** 2, 1e-8).astype(int)