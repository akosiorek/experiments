import numpy as np
from transforms import quaternion_from_matrix, quaternion_inverse, quaternion_multiply, quaternion_from_euler
from quaternion import to_euler, from_axisangle, from_euler
from matplotlib import pyplot as plt
from kalman import mse, grad


def to_quaternion(m):
    q = np.zeros((m.shape[-1], 4))
    for i in xrange(m.shape[-1]):
        q[i, :] = quaternion_from_matrix(m[:, i].reshape(3, 3).T)
    return q


def parse(run, min_length=100):
    d = {}
    for line in run:
        k, v = line.split(':')
        if k in ('oculus', 'velocity', 'prime', 'acceleration'):
            v = np.array([float(f) for f in v.split()])
        else:
            v = float(v.strip())
        if k not in d:
            d[k] = []
        d[k].append(v)

    current_min_length = min(len(v) for v in d.values())
    if current_min_length >= min_length:

        for k in d.keys():
            d[k] = np.asarray(d[k][:current_min_length]).T

        for i in ('oculus', 'prime'):
            d[i] = to_quaternion(d[i])
            # reset orientation
            reference = d[i][0]
            inverse = quaternion_inverse(reference)
            for j in xrange(len(d[i])):
                d[i][j] = quaternion_multiply(inverse, d[i][j])

        #   reset time
        d['timestamp'] -= d['timestamp'][0]

        #   negate xy axis for oculus:
        oculus = d['oculus']
        oculus = to_euler(oculus.T)
        oculus[0, :] *= -1
        # oculus[1, :] *= -1
        oculus[2, :] *= -1
        d['oculus'] = from_euler(oculus).T

        return d
    return None


def load_data(filename, runs='all', min_length=100):
    data = open(filename).read().split('restart')
    data = [[line.strip() for line in run.strip().split('\n')] for run in data if run]
    data = [run for run in data if len(run) > 1]

    d = []
    for run in data:
        parsed = parse(run, min_length)
        if parsed:
            d.append(parsed)

    if runs == 'all':
        return d

    if not hasattr(runs, '__iter__'):
        runs = [runs]

    return [d[i] for i in runs]


if __name__ == '__main__':

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.tight_layout()
    run_number = 5
    # for run_number in xrange(len(data)):

    data = load_data('mat.txt', run_number)[0]
    time = data['timestamp']
    oculus = to_euler(data['oculus'].T)
    prime = to_euler(data['prime'].T)

    print run_number, mse(oculus, prime)
    for d, ax in enumerate(axes):
        ax.cla()
        ax.plot([0, time[-1]], [0, 0], 'k-', linewidth=2)
        ax.plot(time, oculus[d, :], 'r', linewidth=2, label='oculus')
        ax.plot(time, prime[d, :], 'b', linewidth=2, label='prime')

    ax.set_xlim([0, time[-1]])
    ax.legend(loc='best')
    plt.pause(1)
    plt.show()