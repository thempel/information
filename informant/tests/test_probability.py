import unittest
import informant
import numpy as np
import msmtools


class TestProbabilitySimple(unittest.TestCase):
    def test_MSMProb(self):
        prob_est = informant.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_x))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_y))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_xy))

        self.assertAlmostEqual(prob_est.pi_xy.sum(), 1)
        self.assertAlmostEqual(prob_est.pi_x.sum(), 1)
        self.assertAlmostEqual(prob_est.pi_y.sum(), 1)

    def test_CTWProb(self):
        prob_est = informant.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(np.alltrue(prob_est.px > np.zeros_like(prob_est.px)))
        self.assertTrue(np.alltrue(prob_est.py > np.zeros_like(prob_est.py)))
        self.assertTrue(np.alltrue(prob_est.pxy > np.zeros_like(prob_est.pxy)))
