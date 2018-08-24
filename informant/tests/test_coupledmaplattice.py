import unittest
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

class TestCoupledMapLattice(unittest.TestCase):
    """
    Coupled map lattice as described by Schreiber, PRL, 2000.
    """
    @classmethod
    def setUpClass(cls):
        cls.eps = 0.03
        cls.partition = 0.5

        cls.true_value_TE = 0.77**2 * np.square(cls.eps) / np.log(2)

    def test_TE_simple(self):
        """
        Test if estimator matches true value as presented in Schreiber, PRL, 2000.
        :return:
        """

        n_trails = 10
        d = 0.
        for _ in range(n_trails):
            x_n = compute_lattice(self.eps, n_steps=100000)

            X = (x_n[:, 44] > self.partition).astype(int)
            Y = (x_n[:, 45] > self.partition).astype(int)

            p = informant.MSMProbabilities(reversible=False, msmlag=1)
            e = informant.TransferEntropy(p)
            d += e.estimate(X, Y).d

        self.assertAlmostEqual(d/n_trails * 1e3, self.true_value_TE * 1e3, places=1)
