from matplotlib import pyplot as plt
from matplotlib import animation
from numpy import random, sin, cos, pi
import numpy as np


def grad(x, dt):
    return np.vstack((np.gradient(x[i, :], dt) for i in xrange(x.shape[0])))


def rot(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]])


def pred_mat(dt):
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


sigma = 0.1
small_sigma = sigma / 2
interval = 10

t = np.linspace(0, 2 * pi, 1000)
noisy_t = t + sigma * random.randn(*t.shape)
better_t = t + small_sigma * random.randn(*t.shape)

pp = np.array([0, 1]).reshape(2, 1)
dt = t[1] - t[0]

p, noisy_p, better_p, predicted_p = (np.zeros((2, len(t))) for i in xrange(4))

P = pred_mat(dt)

for i in xrange(1, len(t)):
    p[:, i] = rot(t[i]).dot(pp).squeeze()
    noisy_p[:, i] = rot(noisy_t[i]).dot(pp).squeeze()
    better_p[:, i] = rot(better_t[i]).dot(pp).squeeze()

noisy_d = grad(noisy_p, dt)
better_p = better_p[:, ::interval]
better_d = grad(better_p, interval * dt)

fig = plt.figure()
global_ax = fig.add_subplot(1, 2, 1)
global_ax.plot(t, t, 'g', label='true', alpha=0.7)
global_ax.plot(t, noisy_t, 'r', label='noisy', alpha=0.2)
global_ax.plot(t, better_t, 'm', label='better', alpha=0.5)
time_line = global_ax.plot([0, 0], [t[0] -1, t[-1] + 1], linewidth=2, label='time')[0]

global_ax.legend(loc='best')
global_ax.grid()
global_ax.set_xlim([t[0], t[-1]])
global_ax.set_ylim([t[0] - 1, t[-1] + 1])


orient_ax = fig.add_subplot(1, 2, 2)

orient_ax.plot(0, 0, 'b.', markersize=15)   # origin
pos_marker = orient_ax.plot(0, 0, 'k.', markersize=10)[0]
true_line = orient_ax.plot([], [], 'g', linewidth=2, alpha=0.7, label='true')[0]
noisy_line = orient_ax.plot([], [], 'r', linewidth=1, alpha=0.2, label='noisy')[0]
better_line = orient_ax.plot([], [], 'm', linewidth=1, alpha=0.5, label='better')[0]
predicted_line = orient_ax.plot([], [], 'c', linewidth=1, alpha=0.7, label='predicted')[0]
corrected_line = orient_ax.plot([], [], 'k', linewidth=1, alpha=1, label='corrected')[0]

orient_ax.legend(loc='best')
orient_ax.grid()
orient_ax.set_xlim([-1.1, 1.1])
orient_ax.set_ylim([-1.1, 1.1])
orient_ax.set_aspect('equal', adjustable='box')
orient_ax.set_xticklabels([])
orient_ax.set_yticklabels([])


def animate(i):
    time_line.set_data((t[i], t[i]), [t[0] -1, t[-1] + 1])
    pos_marker.set_data(p[0, i], p[1, i])
    # true_orient_line.set_data(p[0, :i], p[1, :i])
    noisy_line.set_data(noisy_p[0, :i], noisy_p[1, :i])

    # if i % 10 == 0:
    #     better_line.set_data(better_p[0, :i / interval + 1], better_p[1, :i / interval + 1])


anim = animation.FuncAnimation(fig, animate,
                               frames=len(t), interval=20)

plt.show()
