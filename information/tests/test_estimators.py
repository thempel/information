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

    def test_1(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        info_est_instance = information.JiaoI4(prob_est)

        estimator = information.Information(prob_est, info_est_instance)
        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)