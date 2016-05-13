from matplotlib import pyplot as plt
from numpy import random
import numpy as np


def grad(x, dt):
    return np.vstack((np.gradient(x[i, :], dt) for i in xrange(x.shape[0])))


def add_noise(x, sigma):
    return x + sigma * random.randn(*x.shape)


def prediction_matrix(dt):
    return np.asarray([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])


def control_matrix(dt):
    return np.asarray([[dt**2/2, 0],
                      [0, dt**2/2],
                      [dt, 0],
                      [0, dt]])


def mse(x, y):
    return ((x - y)**2).sum()

if __name__ == '__main__':
    t = np.linspace(0, 2 * np.pi, 200)
    dt = t[1] - t[0]

    p = np.vstack((np.sin(t), np.cos(t)))
    v = grad(p, dt)
    a = grad(v, dt)

    p_sigma, v_sigma, a_sigma = 0.05, 0.5, 0.01
    p_noisy, v_noisy, a_noisy = (add_noise(i, s) for (i, s) in zip((p, v, a), (p_sigma, v_sigma, a_sigma)))
    # p_noisy = add_noise(p, p_sigma)
    # v_noisy = grad(p_noisy, dt)
    # a_noisy = grad(v_noisy, dt)


    x = np.hstack((p_noisy[:, 0], v_noisy[:, 0]))[:, np.newaxis]   # initial state vector
    u = a_noisy.copy()  # control vector
    z = np.vstack((p_noisy, v_noisy))  # noisy measurements

    P = np.zeros((x.shape[0], x.shape[0]))   # initial state coviarance, it's zero since we know the position and velocity=0
    Q = np.eye(x.shape[0]) * a_sigma    # enviornment noise covariance; here modeled as acceleration noise
    H = np.eye(x.shape[0])  # sensor and measurement space are the same
    R = np.diag([p_sigma, p_sigma, v_sigma, v_sigma])   # measurement noise covariance matrix

    F = prediction_matrix(dt)
    B = control_matrix(dt)


    p_prediction = np.zeros_like(p)
    p_estimate = np.zeros_like(p)

    # all starts equal
    p_prediction[:, 0] = p[:, 0]
    p_estimate[:, 0] = p[:, 0]


    for i in xrange(1, len(t)):
        # prediction
        x = F.dot(x) + B.dot(u[:, [i-1]])
        P = F.dot(P).dot(F.T) + Q
        p_prediction[:, [i]] = x[:2]

        # estimate
        correction = z[:, [i]] - H.dot(x)
        K = P.dot(H.T) * np.linalg.inv(H.dot(P).dot(H.T) + R)

        x = x + K.dot(correction)
        P = P - K.dot(H).dot(P)
        p_estimate[:, [i]] = x[:2]


    print 'noisy error:', mse(p, p_noisy)
    print 'prediction error:', mse(p, p_prediction)
    print 'estimate error:', mse(p, p_estimate)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p[0, :], p[1, :], 'r', linewidth='2', label='input', alpha=0.5)
    ax.plot(p_noisy[0, :], p_noisy[1, :], 'b', linewidth='1', label='noisy', alpha=0.2)
    ax.plot(p_prediction[0, :], p_prediction[1, :], 'g', linewidth='2', label='prediction')
    ax.plot(p_estimate[0, :], p_estimate[1, :], 'm', linewidth='2', label='estimate')
    #
    # state = np.vstack((y_noisy, v_noisy))
    #
    #
    # x = np.zeros_like(state)
    #
    # x[:, 0] = state[:, 0]   # starting state
    # # prediction convariance matrix
    # P = np.eye(x.shape[0]) * np.diag([y_sigma, y_sigma, v_sigma, v_sigma])**2
    #
    # # measurement covariance matrix
    # R = P.copy()
    # f
    #
    # F = prediction_matrix(dt)
    # B = control_matrix(dt)
    #
    # # known external influence, accounting for acceleration noise (general other factors)
    # Q = np.eye(x.shape[0]) * a_sigma**2
    #
    # for i in xrange(1, x.shape[1]):
    #     # assume y_noisy and v_noisy are noisy measurements
    #     # a_noisy is a noisy control signal
    #     x[:, [i]] = F.dot(x[:, [i-1]]) + B.dot(a_noisy[:, [i-1]])
    #     P = F.dot(P).dot(F.T) + Q
    #
    #
    # ax.plot(x[0, :], x[1, :], 'g', label='prediction')
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.legend(loc='best')
    #
    # # ax.plot(xrange(len(x1)), x1)
    # # ax.plot(xrange(len(x1)), np.gradient(x1, dz))
    #
    #
    #
    #
    plt.show()