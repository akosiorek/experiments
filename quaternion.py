import numpy as np


def norm2(q):
    return np.sqrt((q ** 2).sum(keepdims=True))


def normalize(q):
    return q / norm2(q)


def _prod(q1, q2):
    assert q1.shape == q2.shape or (q1.shape[0] == q2.shape[0] and (q1.shape[1] == 1 or q2.shape[1] == 1))

    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    q = np.zeros_like(q1)
    q[0] = w1 * w2 - (v1 * v2).sum()

    q[1:] = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return q


def prod(q1, q2):
    nq1, nq2 = q1.shape[1], q2.shape[1]
    assert q1.shape == q2.shape or (q1.shape[0] == q2.shape[0] and (nq1 == 1 or nq2 == 1))
    q = np.empty((q1.shape[0], max(nq1, nq2)))

    if nq1 == nq2:
        for i in xrange(q.shape[1]):
            q[:, i] = _prod(q1[:, i], q2[:, i])
    elif nq1 > nq2:
        for i in xrange(q.shape[1]):
            q[:, i] = _prod(q1[:, i], q2[:, 0])
    else:
        for i in xrange(q.shape[1]):
            q[:, i] = _prod(q1[:, 0], q2[:, i])

    return q


def conjugate(q):
    q = q.copy()
    q[1:] = -q[1:]
    return q


def rot(p, q):
    pp = np.zeros_like(q)
    pp[1:] = p
    return prod(prod(q, pp), conjugate(q))[1:]


def from_axisangle(p):
    if len(p) == 2:
        n, theta = p
    else:
        n, theta = p[:-1], p[[-1]]

    n = n.reshape(3, -1)
    theta /= 2
    q = np.zeros((4, theta.shape[-1]))
    q[0, :] = np.cos(theta)
    q[1:, :] = np.sin(theta) * normalize(n)
    return q


def to_axiangle(q):
    v = np.zeros_like(q)
    angle = np.arccos(q[0])
    v[:-1] = q[1:]  # don't have to div by sin(angle) since it doesn't change the axis
    v[-1] = angle * 2
    return v


def to_euler(q):
    e = np.zeros((3, q.shape[1]))
    # q0, q1, q2, q3 = (q[i] for i in xrange(q.shape[0]))
    q0, q1, q2, q3 = q
    e[0, :] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    e[1, :] = np.arcsin(2 * (q0 * q2 - q3 * q1))
    e[2, :] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

    for i in xrange(1, e.shape[1]):
        diff = e[:, i] - e[:, i-1]
        for j in xrange(e.shape[0]):
            if diff[j] > 1.8 * np.pi:
                e[j, i] -= 2 * np.pi
            elif diff[j] < -1.8 * np.pi:
                e[j, i] += 2 * np.pi


    return e


def q_from_omega(w):
    theta = norm2(w)
    return from_axisangle(np.concatenate((w, theta)))


def identity():
    return np.asarray([1, 0, 0, 0])


if __name__ == '__main__':
    point = np.asarray([0, 0, 1])
    axis = np.asarray([0, 1, 0])
    angle = np.pi / 2
    axisangle = np.concatenate((axis, [angle]))
    q = from_axisangle(axisangle)

    dt = 30
    w = dt * angle * axis
    qw = q_from_omega(w)

#   q_{k+1} = dt * dq + q_k works only for small time steps and requires renormalizaiton
    n = 300 * dt
    q_from_derivative = identity()
    ww = np.concatenate(([0], w))
    for _ in xrange(n):
        dq = 0.5 * prod(ww, q_from_derivative)
        q_from_derivative = normalize(q_from_derivative + dq / n)


    rotated_axis = rot(point, q)
    rotated_w = rot(point, qw)
    rotated_d = rot(point, q_from_derivative)

    print rotated_d
    print 'qw:', qw, norm2(qw)
    print 'qd:', q_from_derivative, norm2(q_from_derivative)
    print 'original:', point
    print 'rotated around axis:', rotated_axis
    print 'roted by w*dt:', np.allclose(rotated_axis, rotated_w), rotated_w
    print 'roted by derivative:', np.allclose(rotated_axis, rotated_d), rotated_d, norm2(rotated_d)
