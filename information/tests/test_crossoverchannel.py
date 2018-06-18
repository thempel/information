import unittest
import information
import numpy as np
import msmtools


def entropy1D(x):
    return - x * np.log2(x) - (1 - x) * np.log2(1 - x)


class TestCrossover(unittest.TestCase):
    """
    Cross-over channel test. Directed information can be computed analytically.
    Class sets up a simple binary channel with cross-overs, no delay.
    """
    @classmethod
    def setUpClass(cls):
        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        cls.X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        cls.Y = cls.X.copy()
        cls.Y[_errbits] = 1 - cls.Y[_errbits]

        cls.true_value_DI = entropy1D(p) - (((1 - p) * (1 - eps) + p * eps) * entropy1D(p * eps / ((1 - p) * (1 - eps) + p * eps)) + \
                    ((p) * (1 - eps) + (1 - p) * eps) * entropy1D((1 - p) * eps / ((p) * (1 - eps) + (1 - p) * eps)))


    def test_MSMInfo(self):
        """
        Test MSM probability and I4 estimator.
        :return:
        """
        prob = information.MSMProbabilities().estimate(self.X, self.Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMEnsembleInfo(self):
        """
        Test MSM probability and I4 ensemble estimator.
        :return:
        """
        prob = information.MSMProbabilities().estimate(self.X, self.Y)
        estimator = information.JiaoI4Ensemble(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWInfo(self):
        """
        Test CTW probabilities and I4 estimator.
        :return:
        """
        prob = information.CTWProbabilities(5).estimate(self.X, self.Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMInfo_multi(self):
        """
        Tests if polluting one state of the second trajectory with random noise alters
        the I4 result for MSM probabilities.
        :return:
        """
        Y = self.Y.copy()
        idx = Y == 1
        Y[idx] = self.Y[idx] + np.random.randint(0, 2, size=self.Y[idx].shape[0])
        prob = information.MSMProbabilities().estimate(self.X, Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(self.X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWInfo_multi(self):
        """
        Tests if polluting one state of the second trajectory with random noise alters
        the I4 result for CTW probabilities.
        :return:
        """

        Y = self.Y.copy()
        idx = Y == 1
        Y[idx] = self.Y[idx] + np.random.randint(0, 2, size=self.Y[idx].shape[0])
        prob = information.CTWProbabilities(5).estimate(self.X, Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(self.X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMEnsembleInfo_multi(self):
        """
        Tests if polluting one state of the second trajectory with random noise alters
        the I4 ensemble estimator result for MSM probabilities.
        :return:
        """
        Y = self.Y.copy()
        idx = Y == 1
        Y[idx] = self.Y[idx] + np.random.randint(0, 2, size=self.Y[idx].shape[0])
        prob = information.MSMProbabilities().estimate(self.X, Y)
        estimator = information.JiaoI4Ensemble(prob)
        estimator.estimate(self.X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)
