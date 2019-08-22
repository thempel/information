import unittest
import informant
import numpy as np
import six
import itertools
from utils import GenerateTestMatrix
import msmtools


class TestSimple(six.with_metaclass(GenerateTestMatrix, unittest.TestCase)):
    """
    Simple and cross-over channel tests. Directed information can be computed analytically.
    Class sets up a simple binary channel with cross-overs and delay.
    """

    di_estimators = (informant.CausallyConditionedDIJiaoI3, informant.CausallyConditionedDIJiaoI4)
    p_estimators = (informant.MSMProbabilities, )
    default_test_grid = [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)]
    params = {
        '_test_simple': default_test_grid,
        '_test_simple_multiconditionals': default_test_grid,
        '_test_simple_raises_disconnectedXW': default_test_grid,
        '_test_proxy': default_test_grid,
        '_test_cascade': default_test_grid
    }

    @classmethod
    def setUpClass(cls):
        cls.A_binary = np.random.randint(0, 2, 5000)
        cls.B_binary = np.random.randint(0, 2, 5000)

        cls.A_nonbinary = np.random.randint(0, 3, 5000)
        cls.B_nonbinary = np.random.randint(0, 4, 5000)

        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        cls.X = X[2:]
        cls.Y = Y[1:-1]

        # proxy configuration: X -> Y -> Z
        _errbits = np.random.rand(N) < eps
        Z = Y.copy()
        Z[_errbits] = 1 - Z[_errbits]
        cls.Z_proxy = Z[:-2]

        # cascade configuration: X -> Y; X -> Z
        _errbits = np.random.rand(N) < eps
        Z = X.copy()
        Z[_errbits] = 1 - Z[_errbits]

        cls.Z_cascade = Z[1:-1]

    def _test_simple(self, di_est, p_est):
        est = di_est(p_est())
        est.estimate(self.A_nonbinary, self.B_nonbinary, self.A_binary)

        self.assertAlmostEqual(est.causally_conditioned_di[0], 0, places=0)

    def _test_simple_multiconditionals(self, di_est, p_est):
        est = di_est(p_est())
        est.estimate(self.A_binary, self.B_binary, [self.B_nonbinary, self.A_nonbinary])

        self.assertAlmostEqual(est.causally_conditioned_di[0], 0, places=0)

    def _test_simple_raises_disconnectedXW(self, di_est, p_est):
        est = di_est(p_est())

        est.estimate(np.array([0, 1, 0, 1, 1]),
                     np.array([0, 1, 0, 1, 0]),
                     [np.array([0, 0, 0, 1, 1])])

        self.assertFalse(np.isfinite(est.causally_conditioned_di[0]))

    def _test_proxy(self, di_est, p_est):
        # TODO:
        # This is a non-Markovian system. Seems to work qualitatively. Find out why.

        # test if direct link is detected with causally cond entropy > 0
        estimator = di_est(p_est())
        estimator.estimate(self.X, self.Y, self.Z_proxy)

        self.assertGreater(estimator.causally_conditioned_di[0], 0.)

        # test if indirect link is detected with causally cond entropy ~ 0.
        estimator = di_est(p_est())
        estimator.estimate(self.X, self.Z_proxy, self.Y)

        self.assertAlmostEqual(estimator.causally_conditioned_di[0], 0., places=2)

    def _test_cascade(self, di_est, p_est):
        # test if direct link is detected with causally cond entropy > 0
        estimator = di_est(p_est())
        estimator.estimate(self.X, self.Y, self.Z_cascade)

        self.assertGreater(estimator.causally_conditioned_di, 0.)

        # test if indirect link is detected with causally cond entropy ~ 0.
        estimator = di_est(p_est())
        estimator.estimate(self.Y, self.Z_cascade, self.X)

        self.assertAlmostEqual(estimator.causally_conditioned_di[0], 0., places=2)

    def test_compare_I4_I3(self):

        prob_est = informant.MSMProbabilities(msmlag=1)
        est1 = informant.CausallyConditionedDIJiaoI3(prob_est)
        est1.estimate(self.A_nonbinary, self.B_nonbinary, self.A_binary)

        est2 = informant.CausallyConditionedDIJiaoI4(prob_est)
        est2.estimate(self.A_nonbinary, self.B_nonbinary, self.A_binary)

        self.assertAlmostEqual(est2.causally_conditioned_di[0], est1.causally_conditioned_di[0], places=1)
