import numpy as np


class EAP:

    def __init__(self, score, a, b, model):
        z_val = model
        model = model(a, b, self.x_nodes)
        p = model.prob_values
        self.lik_values = np.prod(p**score*(1.0 - p)**(1-score), axis=1)

    @property
    def g(self):
        x = self.x_nodes
        weight = self.weights
        return np.sum(x[:, 0]*weight*self.lik_values)

    @property
    def h(self):
        weight = self.weights
        return np.sum(weight * self.lik_values)

    @property
    def res(self):
        return self.g / self.h

    x_nodes = np.array([[-7.84938289],
                        [-6.75144472],
                        [-5.82938201],
                        [-4.99496394],
                        [-4.21434398],
                        [-3.46984669],
                        [-2.75059298],
                        [-2.04910247],
                        [-1.35976582],
                        [-0.67804569],
                        [0],
                        [0.67804569],
                        [1.35976582],
                        [2.04910247],
                        [2.75059298],
                        [3.46984669],
                        [4.21434398],
                        [4.99496394],
                        [5.82938201],
                        [6.75144472],
                        [7.84938289]])

    weights = np.array([3.72E-14, 8.82E-11, 2.57E-08, 2.17E-06, 7.48E-05, 0.001254982, 0.011414066, 0.060179647,
                        0.192120324, 0.381669074, 0.479023703, 0.381669074, 0.192120324, 0.060179647, 0.011414066,
                        0.001254982, 7.48E-05, 2.17E-06, 2.57E-08, 8.82E-11, 3.72E-14])

if __name__ == '__main__':
    import time
    s = time.clock()
    error = np.zeros(10000)
    for i in range(10000):
        theta = np.random.normal(size=1)
        a0 = np.random.uniform(1, 3, 10)
        b0 = np.random.normal(size=10)
        score0 = np.random.binomial(1, LogisticModel(a0, b0, theta).prob_values, 10)
        eap = EAP(score0, a0, b0)
        error[i] = np.abs(theta - eap.res)
    print np.mean(error)
    e = time.clock()
    print e - s
