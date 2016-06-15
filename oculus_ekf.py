import numpy as np
from numpy import sin, cos, sqrt, random
from matplotlib import pyplot as plt

import quaternion as quat
from kalman import add_noise, mse, grad
from ekf import f, Jf, extract
from oculus_data import load_data

if __name__ == '__main__':


    datas = load_data('mat.txt')
    total_mse_prediction = total_mse_estimate = 0
    total_points = 0
    for data in datas:
        o = data['oculus'].T
        w = data['velocity']
        e = data['acceleration']
        prime = data['prime'].T
        time = data['timestamp']
        dt = np.diff(time)

        x = np.hstack((o[:, 0], w[:, 0], e[:, 0]))[:, np.newaxis]   # initial state vector
        z = np.vstack((o, w, e))  # measurements
        # z = np.vstack((o, w, e))  # measurements

        # assumed noise
        o_sigma, w_sigma, e_sigma = 0.05, 0.05, 0.01

        # initial state coviarance
        P = np.diag([o_sigma] * o.shape[0] + [w_sigma] * w.shape[0] + [e_sigma] * e.shape[0])
        Q = np.eye(x.shape[0]) * 0.01  # enviornment noise covariance
        H = np.eye(x.shape[0])  # sensor and measurement space are the same
        R = P.copy()   # measurement noise covariance matrix; same as P since P comes from measurements

        Hp = H[:4, :].copy()
        Rp = P[:4, :4].copy() / 4

        o_prediction = np.zeros_like(o)
        o_estimate = np.zeros_like(o)

        # all starts equal
        o_prediction[:, 0] = o[:, 0]
        o_estimate[:, 0] = o[:, 0]

        for i in xrange(1, len(time)):
            # prediction
            F = Jf(x, dt[i-1])
            x = f(x, dt[i-1])
            P = F.dot(P).dot(F.T) + Q

            o_prediction[:, [i]] = extract(x)[0]

            # estimate
            correction = z[:, [i]] - H.dot(x)
            K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))

            x = x + K.dot(correction)
            x[:o.shape[0]] = quat.normalize(x[:o.shape[0]])
            P = P - K.dot(H).dot(P)

            # estimate
            correction = prime[:, [i]] - Hp.dot(x)
            K = P.dot(Hp.T).dot(np.linalg.inv(Hp.dot(P).dot(Hp.T) + Rp))

            x = x + K.dot(correction)
            x[:o.shape[0]] = quat.normalize(x[:o.shape[0]])
            P = P - K.dot(Hp).dot(P)
            o_estimate[:, [i]] = extract(x)[0]

        o, o_prediction, o_estimate = (quat.to_euler(i) for i in (o, o_prediction, o_estimate))
        prime = quat.to_euler(prime)

        # prime[0, :], prime[1, :] = prime[1, :].copy(), prime[0, :].copy()

        # o_noisy[:2, :] = o_prediction[:2, :] = o_estimate[:2, :] = o[:2, :]

        mse_prediction, mse_estimate = (mse(prime, i) for i in (o_prediction, o_estimate))
        print 'prediction error:', mse_prediction
        print 'estimate error:', mse_estimate

        total_mse_estimate += mse_estimate * len(prime)
        total_mse_prediction += mse_prediction * len(prime)
        total_points += len(prime)

    #     fig, axes = plt.subplots(o.shape[0], 1, sharex=True, sharey=False)
    #     fig.tight_layout()
    #     axes[0].set_title('MSE: Estimate={:.3f}'.format(mse_estimate))
    #     for d, ax in enumerate(axes):
    #         # ax = fig.add_subplot(, d + 1)
    #         ax.plot(time, o[d, :], 'r-', linewidth='1', label='oculus')
    #         ax.plot(time, o_prediction[d, :], 'g', linewidth='1', label='prediction')
    #         ax.plot(time, o_estimate[d, :], 'm', linewidth='1', label='estimate')
    #         ax.plot(time, prime[d, :], 'b', linewidth='1', label='prime')
    #
    #     ax.set_ylim([-1.1 * np.pi, 1.1 * np.pi])
    #     ax.set_xlim([time[0], time[-1]])
    #     ax.legend(loc='best')
    #     plt.waitforbuttonpress()
    #     plt.close()
    #
    # plt.show()
    print 'mean prediction error:', total_mse_prediction / total_points
    print 'mean estimate error:', total_mse_estimate / total_points
