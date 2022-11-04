import pytest

import informant
import numpy as np


def f(x):
    return np.where(x < 0.5, 2 * x, 2 - 2 * x)


def compute_lattice(eps, n_steps=100000, n_maps=100):
    x_0 = np.random.rand(n_maps)

    x_n = np.zeros((n_steps, n_maps))
    x_n[0] = x_0
    for t in range(1, n_steps):
        x_n_m_minus1 = np.roll(x_n[t - 1], 1)
        x_n[t] = (f(eps * x_n_m_minus1 + (1 - eps) * x_n[t - 1]))

    return x_n

class Data:
    def __init__(self):
        self.eps = 0.03
        self.partition = 0.5
        self.n_trails = 20

        self.true_value_TE = 0.77**2 * np.square(self.eps) / np.log(2)

@pytest.fixture(scope="module")
def data() -> Data:
    return Data()

def test_TE_simple(data):
    """
    Test if estimator matches true value as presented in Schreiber, PRL, 2000.
    :return:
    """

    d = 0.
    for _ in range(data.n_trails):
        x_n = compute_lattice(data.eps, n_steps=100000)

        X = (x_n[:, 44] > data.partition).astype(int)
        Y = (x_n[:, 45] > data.partition).astype(int)

        p = informant.MSMProbabilities(reversible=False, msmlag=1)
        e = informant.TransferEntropy(p)
        d += e.estimate(X, Y).d

    np.testing.assert_allclose(d/data.n_trails, data.true_value_TE, rtol=.15)

def test_TE_disconnected(data):
    """
    Test if estimator matches true value as presented in Schreiber, PRL, 2000.
    Test case: single transition matrices are connected,
    combinatorial state (2, 2) is disconnected.
    :return:
    """

    d = 0.

    for _ in range(data.n_trails):
        x_n = compute_lattice(data.eps, n_steps=100000)

        X = (x_n[:, 44] > data.partition).astype(int)
        Y = (x_n[:, 45] > data.partition).astype(int)

        X[100:110] = 2
        Y[200:213] = 2
        X[-6:] = 2
        Y[-6:] = 2
        p = informant.MSMProbabilities(reversible=False, msmlag=1)
        e = informant.TransferEntropy(p)
        d += e.estimate(X, Y).d

    np.testing.assert_allclose(d/data.n_trails, data.true_value_TE, rtol=.15)
