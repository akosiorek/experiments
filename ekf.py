import numpy as np
from numpy import sin, cos, sqrt, random

import quaternion as quat
from quaternion import norm2


def extract(x):
    return x[:4], x[4:7], x[7:]


def f(x, dt):
    p, w, e = extract(x)

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


def f_qw_approx(x, dt, eps=1e-4):
    p, w, e = extract(x)

    def q(ww):
        v = dt * ww + dt ** 2 * e / 2
        return quat.q_from_omega(v)

    dq = np.zeros((4, 3))
    for i in xrange(w.size):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        dq[:, i] = (q(wp) - q(wm)) / (2 * eps)
    return dq


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
    q0w = - 0.5 * dt / v_norm * sin(theta) * v
    qw = dt / v_norm ** 2 * (sin(theta) * (v_norm * np.eye(3) - v.dot(v.T) / v_norm) + 0.5 * cos(theta) * v.dot(v.T))

    dcross_dw = np.asarray(((qw[1] * p[3] - qw[2] * p[2]).T,
                            (qw[2] * p[1] - qw[0] * p[3]).T,
                            (qw[0] * p[2] - qw[1] * p[1]).T))

    dp_dw_part1 = p[0] * q0w - qw.dot(p[1:])
    dp_dw_part2 = q0w.T.dot(p[1:]) + p[0] * qw + dcross_dw

    J[0, 4:7] = dp_dw_part1.squeeze()
    J[1:4, 4:7] = dp_dw_part2


    # dqdw = f_qw(x, dt)
    # print dqdw
    # print f_qw_approx(x, dt)

    return J


def Jf_approx(x, dt, eps=1e-4):

    J = np.zeros((x.size, x.size))
    for i in xrange(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (f(xp, dt) - f(xm, dt)) / (2 * eps)
    return J


if __name__ == '__main__':
    dt = 0.5
    random.seed(0)
    x = random.randn(10)
    x[:4] = quat.normalize(x[:4])
    J = Jf(x, dt)
    Jd = Jf_approx(x, dt)
    print np.greater((J - Jd) ** 2, 1e-8).astype(int)
    for i in xrange(x.size):
        print list(Jd[i, :])


    # print J[0:4, 4:7]
    #
    print J[1:4, 4:7]
    print Jd[1:4, 4:7]
