import unittest
import information
import numpy as np
import msmtools


class TestProbabilitySimple(unittest.TestCase):
    def test_MSMProb(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_x))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_y))
        self.assertTrue(msmtools.analysis.is_transition_matrix(prob_est.tmat_xy))

    def test_CTWProb(self):
        prob_est = information.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(np.alltrue(prob_est.px > np.zeros_like(prob_est.px)))
        self.assertTrue(np.alltrue(prob_est.py > np.zeros_like(prob_est.py)))
        self.assertTrue(np.alltrue(prob_est.pxy > np.zeros_like(prob_est.pxy)))
