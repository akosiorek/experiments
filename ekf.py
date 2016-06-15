import numpy as np
from numpy import sin, cos, sqrt, random
from matplotlib import pyplot as plt

import quaternion as quat
from kalman import add_noise, mse, grad


def fd_approx(foo, wrt, *args, **kwargs):
    eps = kwargs.get('eps', 1e-6)
    y = foo(wrt, *args)
    J = np.zeros((y.size, wrt.size))

    w = wrt.ravel()
    for i in xrange(wrt.size):
        w[i] += eps
        yp = foo(w.reshape(wrt.shape), *args)
        w[i] -= 2 * eps
        ym = foo(w.reshape(wrt.shape), *args)
        w[i] += eps
        J[:, i] = (yp - ym).squeeze() / (2 * eps)

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
    w_norm = quat.norm2(v)
    theta = w_norm / 2
    q0w = - 0.5 * dt / w_norm * sin(theta) * v
    qw = dt / w_norm ** 2 * (sin(theta) * (w_norm * np.eye(3) - v.dot(v.T) / w_norm) + 0.5 * cos(theta) * v.dot(v.T))

    return np.vstack((q0w.T, qw))


def w_from_state(x, dt):
    p, w, e = extract(x)
    v = dt * w + dt ** 2 * e / 2
    return quat.q_from_omega(v)


def dq_dw_approx(x, dt):
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
    q = quat.q_from_omega(v).squeeze()
    assert q.ndim == 1, 'Jacobian supported only for one point at a time'

    J = np.eye(x.size)
    J[0, :4] = q
    J[1, :4] = -q[1], q[0], q[3], -q[2]
    J[2, :4] = -q[2], -q[3], q[0], q[1]
    J[3, :4] = -q[3], q[2], -q[1], q[0]
    J[:4, :4] = J[:4, :4].T

    J[4:7, 7:] = np.eye(3) * dt

    w_norm = quat.norm2(v)
    theta = w_norm / 2
    q0dv = - 0.5 / w_norm * sin(theta) * v.T
    qdv = (sin(theta) * (w_norm * np.eye(3) - v.dot(v.T) / w_norm) + 0.5 * cos(theta) * v.dot(v.T)) / w_norm ** 2

    dcross_dv = np.asarray(((qdv[1] * p[3] - qdv[2] * p[2]).T,
                            (qdv[2] * p[1] - qdv[0] * p[3]).T,
                            (qdv[0] * p[2] - qdv[1] * p[1]).T))

    dfp0_dv = p[0] * q0dv - qdv.dot(p[1:]).T
    dfo_dv = p[1:].dot(q0dv) + p[0] * qdv + dcross_dv

    J[0, 4:7] = dt * dfp0_dv
    J[1:4, 4:7] = dt * dfo_dv

    J[0, 7:] = dt**2 / 2 * dfp0_dv
    J[1:4, 7:] = dt**2 / 2 * dfo_dv
    return J


def pretty(row, width=16, decimals=8):
    format_str = '%{}.{}f'.format(width, decimals)
    return '|' + ' |'.join([format_str % f for f in row]) + ' |'


if __name__ == '__main__':
    dt = 0.5
    random.seed(0)
    # x = random.randn(10, 1)
    # x[:4] = quat.normalize(x[:4])
    x = np.asarray([0.57996866097729827, 0.13155995106814494, 0.32178033684031654, 0.73673994488019956, 1.8675579901499673, -0.9772778798764109, 0.95008841752558948, -0.15135720829769789, -0.10321885179355784, 0.41059850193837227]).reshape(10, 1)

    J = Jf(x, dt)
    Jd = fd_approx(f, x, dt)
    print np.greater((J - Jd) ** 2, 1e-8).astype(int)

    print J.flatten().tolist()

    print dt
    print list(x.flatten())
    print list(f(x, dt).flatten())

    t = np.linspace(0, 2 * np.pi, 200)
    dt = t[1] - t[0]

    axis = np.array([0, 0, 1])
    o = np.pi * np.sin(t)

    o = quat.from_axisangle((axis, o))
    euler = quat.to_euler(o)
    w = grad(euler, dt)
    e = grad(w, dt)

    o_sigma, w_sigma, e_sigma = 0.05, 0.5, 0.01
    o_noisy, w_noisy, e_noisy = (add_noise(i, s) for (i, s) in zip((o, w, e), (o_sigma, w_sigma, e_sigma)))
    # o_noisy = add_noise(p, o_sigma)
    # w_noisy = grad(o_noisy, dt)
    # e_noisy = grad(w_noisy, dt)

    x = np.hstack((o_noisy[:, 0], w_noisy[:, 0], e_noisy[:, 0]))[:, np.newaxis]   # initial state vector
    z = np.vstack((o_noisy, w_noisy, e_noisy))  # noisy measurements

    P = np.zeros((x.shape[0], x.shape[0]))   # initial state coviarance, it's zero since we know the position and velocity=0
    # Q = np.eye(x.shape[0]) * 0.01  # enviornment noise covariance
    Q = np.eye(x.shape[0]) * 0.002  # enviornment noise covariance
    H = np.eye(x.shape[0])  # sensor and measurement space are the same
    R = np.diag([o_sigma] * o.shape[0] + [w_sigma] * w.shape[0] + [e_sigma] * e.shape[0])   # measurement noise covariance matrix

    o_prediction = np.zeros_like(o)
    o_estimate = np.zeros_like(o)

    # all starts equal
    o_prediction[:, 0] = o[:, 0]
    o_estimate[:, 0] = o[:, 0]

    for i in xrange(1, len(t)):
        # prediction
        F = Jf(x, dt)
        x = f(x, dt)
        # x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q

        o_prediction[:, [i]] = extract(x)[0]

        # estimate
        correction = z[:, [i]] - H.dot(x)
        K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))

        x = x + K.dot(correction)
        x[:o.shape[0]] = quat.normalize(x[:o.shape[0]])
        P = P - K.dot(H).dot(P)
        o_estimate[:, [i]] = extract(x)[0]

    o, o_noisy, o_prediction, o_estimate = (quat.to_euler(i) for i in (o, o_noisy, o_prediction, o_estimate))
    # o_noisy[:2, :] = o_prediction[:2, :] = o_estimate[:2, :] = o[:2, :]

    mse_noise, mse_prediction, mse_estimate = (mse(o, i) for i in (o_noisy, o_prediction, o_estimate))
    print 'noisy error:', mse_noise
    print 'prediction error:', mse_prediction
    print 'estimate error:', mse_estimate

    fig, axes = plt.subplots(o.shape[0], 1, sharex=True, sharey=False)

    fig.tight_layout()
    axes[0].set_title('MSE: Noise={:.3f}, Estimate={:.3f}'.format(mse_noise, mse_estimate))
    for d, ax in enumerate(axes):
        # ax = fig.add_subplot(, d + 1)
        ax.plot(t, o[d, :], 'r-', linewidth='1', label='true')
        ax.plot(t, o_noisy[d, :], 'b', linewidth='1', label='noisy', alpha=0.2)
        ax.plot(t, o_prediction[d, :], 'g', linewidth='1', label='prediction')
        ax.plot(t, o_estimate[d, :], 'm', linewidth='1', label='estimate')

    ax.set_ylim([-1.1 * np.pi, 1.1 * np.pi])
    ax.set_xlim([t[0], t[-1]])
    ax.legend(loc='best')

    plt.show()