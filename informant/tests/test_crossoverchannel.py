import unittest
import informant
import numpy as np
import msmtools


def entropy1D(x):
    return - x * np.log2(x) - (1 - x) * np.log2(1 - x)


class TestCrossover(unittest.TestCase):
    """
    Cross-over channel test. Directed informant can be computed analytically.
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
        prob = informant.MSMProbabilities().estimate(self.X, self.Y)
        estimator = informant.JiaoI4(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMEnsembleInfo(self):
        """
        Test MSM probability and I4 ensemble estimator.
        :return:
        """
        prob = informant.MSMProbabilities().estimate(self.X, self.Y)
        estimator = informant.JiaoI4Ensemble(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWInfo(self):
        """
        Test CTW probabilities and I4 estimator.
        :return:
        """
        prob = informant.CTWProbabilities(5).estimate(self.X, self.Y)
        estimator = informant.JiaoI4(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWInfoI3(self):
        """
        Test CTW probabilities and I4 estimator.
        :return:
        """
        prob = informant.CTWProbabilities(5).estimate(self.X, self.Y)
        estimator = informant.JiaoI3(prob)
        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)
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
        prob = informant.MSMProbabilities().estimate(self.X, Y)
        estimator = informant.JiaoI4(prob)
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
        prob = informant.CTWProbabilities(5).estimate(self.X, Y)
        estimator = informant.JiaoI4(prob)
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
        prob = informant.MSMProbabilities().estimate(self.X, Y)
        estimator = informant.JiaoI4Ensemble(prob)
        estimator.estimate(self.X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_congruenceI3I4_CTW(self):

        prob_est = informant.CTWProbabilities(5)
        prob_est.estimate(self.X, self.Y)

        estimator4 = informant.JiaoI4(prob_est)
        estimator4.estimate(self.X, self.Y)

        estimator3 = informant.JiaoI3(prob_est)
        estimator3.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator3.d, estimator4.d, places=2)
        self.assertAlmostEqual(estimator3.r, estimator4.r, places=2)
        self.assertAlmostEqual(estimator3.m, estimator4.m, places=2)

    def test_congruenceI3I4_MSM(self):

        prob_est = informant.MSMProbabilities(reversible=False)
        prob_est.estimate(self.X, self.Y)

        estimator4 = informant.JiaoI4(prob_est)
        estimator4.estimate(self.X, self.Y)

        estimator3 = informant.JiaoI3(prob_est)
        estimator3.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator3.d, estimator4.d, places=1)
        self.assertAlmostEqual(estimator3.r, estimator4.r, places=1)
        self.assertAlmostEqual(estimator3.m, estimator4.m, places=1)

    def test_symmetric_estimate(self):
        prob_est = informant.MSMProbabilities().estimate(self.X, self.Y)
        estimator = informant.JiaoI4(prob_est).symmetrized_estimate(self.X, self.Y)

        self.assertGreater(estimator.d, estimator.r)

    def test_MSMI4Multitraj(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """
        prob = informant.MSMProbabilities().estimate([self.X, self.X], [self.Y, self.Y])
        estimator = informant.JiaoI4(prob)
        estimator.estimate([self.X, self.X], [self.Y, self.Y])

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWI4Multitraj(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """
        prob = informant.CTWProbabilities(5).estimate([self.X, self.X], [self.Y, self.Y])
        estimator = informant.JiaoI4(prob)
        estimator.estimate([self.X, self.X], [self.Y, self.Y])

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMI4EnsembleMultitraj(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """
        prob = informant.MSMProbabilities().estimate([self.X, self.X], [self.Y, self.Y])
        estimator = informant.JiaoI4Ensemble(prob)
        estimator.estimate([self.X, self.X], [self.Y, self.Y])

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_CTWI3Multitraj(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """
        prob = informant.CTWProbabilities(5).estimate([self.X, self.X], [self.Y, self.Y])
        estimator = informant.JiaoI3(prob)
        estimator.estimate([self.X, self.X], [self.Y, self.Y])

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMI3Multitraj(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """
        prob = informant.MSMProbabilities().estimate([self.X, self.X], [self.Y, self.Y])
        estimator = informant.JiaoI3(prob)
        estimator.estimate([self.X, self.X], [self.Y, self.Y])

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def test_MSMI4re(self):
        """
        Test MSM probability and I4 estimator for multiple trajs.
        :return:
        """

        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        X_list, Y_list = [], []
        for _ in range(20):
            X = msmtools.generation.generate_traj(T, N)
            _errbits = np.random.rand(N) < eps
            Y = X.copy()
            Y[_errbits] = 1 - Y[_errbits]
            X_list.append(X)
            Y_list.append(Y)

        prob = informant.MSMProbabilities().estimate(X_list, Y_list)
        estimator = informant.JiaoI4(prob)
        estimator.estimate(X_list, Y_list, traj_eq_reweighting=True)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)
