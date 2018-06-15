import unittest
import information
import numpy as np
import msmtools


class TestSuperSimple(unittest.TestCase):
    def test_MSMProb(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_x))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_y))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_xy))

    def test_MSMInfo(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        estimator = information.JiaoI4(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_MSMInfo_multistate(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 3, 200)
        B = np.random.randint(0, 4, 200)
        prob_est.estimate(A, B)

        estimator = information.JiaoI4(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_MSMInfoEnsemble_multistate(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 3, 200)
        B = np.random.randint(0, 4, 200)
        prob_est.estimate(A, B)

        estimator = information.JiaoI4Ensemble(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_compare_ensemble_timeav(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 3, 200)
        B = np.random.randint(0, 4, 200)
        prob_est.estimate(A, B)

        ensemble_estimator = information.JiaoI4Ensemble(prob_est)
        ensemble_estimator.estimate(A, B)

        estimator = information.JiaoI4(prob_est)
        estimator.estimate(A, B)

        self.assertAlmostEqual(estimator.d, ensemble_estimator.d)
        self.assertAlmostEqual(estimator.r, ensemble_estimator.r)
        self.assertAlmostEqual(estimator.m, ensemble_estimator.m)