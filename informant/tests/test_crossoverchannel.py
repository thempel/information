import unittest
import informant
import numpy as np
import msmtools
import six
import itertools
from utils import GenerateTestMatrix

def entropy1D(x):
    return - x * np.log2(x) - (1 - x) * np.log2(1 - x)

class TestCrossover(six.with_metaclass(GenerateTestMatrix, unittest.TestCase)):
    """
    Cross-over channel test. Directed informant can be computed analytically.
    Class sets up a simple binary channel with cross-overs, no delay.
    """

    di_estimators = (informant.JiaoI4, informant.JiaoI3)
    all_estimators = (informant.JiaoI4, informant.JiaoI3, informant.TransferEntropy)
    p_estimators = (informant.MSMProbabilities, informant.CTWProbabilities)

    params = {
        '_test_simple': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)],
        '_test_polluted_state': [dict(di_est=d, p_est=informant.MSMProbabilities) for d in di_estimators],
        '_test_congruency': [dict(di_est1=informant.JiaoI3, di_est2=informant.JiaoI4, p_est=p) for p in p_estimators],
        '_test_symmetric_estimate': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)] +
        [dict(di_est=informant.TransferEntropy, p_est=informant.MSMProbabilities)],
        '_test_multitraj_support': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)]
    }

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

    def _test_simple(self, di_est, p_est):
        estimator = di_est(p_est())

        estimator.estimate(self.X, self.Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def _test_polluted_state(self, di_est, p_est):
        """
        Tests if polluting one state of the second trajectory with random noise alters results.
        """
        Y = self.Y.copy()
        idx = Y == 1
        Y[idx] = self.Y[idx] + np.random.randint(0, 2, size=self.Y[idx].shape[0])

        estimator = di_est(p_est())

        estimator.estimate(self.X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    def _test_congruency(self, di_est1, di_est2, p_est):
        est1 = di_est1(p_est())
        est2 = di_est2(p_est())

        est1.estimate(self.X, self.Y)
        est2.estimate(self.X, self.Y)

        self.assertAlmostEqual(est1.d, est2.d, places=2)
        self.assertAlmostEqual(est1.r, est2.r, places=2)
        self.assertAlmostEqual(est1.m, est2.m, places=2)

    def _test_symmetric_estimate(self, di_est, p_est):

        estimator = di_est(p_est()).symmetrized_estimate(self.X, self.Y)
        self.assertGreater(estimator.d, estimator.r)

    def _test_multitraj_support(self, di_est, p_est):
        estimator = di_est(p_est()).estimate([self.X, self.X], [self.Y, self.Y])

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

        prob = informant.MSMProbabilities()
        estimator = informant.JiaoI4(prob)
        estimator.estimate(X_list, Y_list, traj_eq_reweighting=True)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, self.true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * self.true_value_DI)

    @unittest.skip("Need to derive true result to compare to!")
    def test_delayed(self):
        N = int(1e4)
        shift = 10
        p = 0.01
        eps = .33
        T = np.array([[1-p, p], [p, 1-p]])
        X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        X = X[shift:]
        Y = Y[:-shift]

        prob = informant.MSMProbabilities(shift, reversible=False).estimate(X, Y)
        estimator = informant.JiaoI4(prob)
        estimator.estimate(X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        # TODO: derive relationship to delaytime and add actual value.
        self.assertAlmostEqual(estimator.d, 0.008, places=1)
        self.assertLess(estimator.r, .1 * 0.008)

    def test_schreiber_MI(self):
        prob_est = informant.MSMProbabilities()
        estimator = informant.MutualInfoStationaryDistribution(prob_est).estimate(self.X, self.Y)

        # TODO: compute actual MI value for binary delayed channel.
        self.assertGreaterEqual(estimator.m, self.true_value_DI)


class TestTriplet(unittest.TestCase):
    """
    Cross-over channel test. Directed information can be computed analytically.
    Class sets up a simple binary channel with cross-overs and delay.
    """

    @classmethod
    def setUpClass(cls):
        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        _errbits = np.random.rand(N) < eps
        Z = Y.copy()
        Z[_errbits] = 1 - Z[_errbits]

        cls.X = X[2:]
        cls.Y = Y[1:-1]
        cls.Z = Z[:-2]

    def _test_simple(self):

        # test if direct link is detected with causally cond entropy > 0
        estimator = informant.CausallyConditionedDI(informant.NetMSMProbabilities())
        estimator.estimate(self.X, self.Y, self.Z)

        self.assertGreater(estimator.causally_conditioned_di, 0.)

        # test if indirect link is detected with causally cond entropy ~ 0.
        estimator = informant.CausallyConditionedDI(informant.NetMSMProbabilities())
        estimator.estimate(self.X, self.Z, self.Y)

        self.assertAlmostEquals(estimator.causally_conditioned_di, 0.)


if __name__ == '__main__':
    unittest.main()