import unittest
import informant
import numpy as np
from deeptime.markov.tools.analysis import is_transition_matrix


class TestProbabilitySimple(unittest.TestCase):
    def test_MSMProb(self):
        prob_est = informant.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(is_transition_matrix(prob_est.tmat_x))
        self.assertTrue(is_transition_matrix(prob_est.tmat_y))
        self.assertTrue(is_transition_matrix(prob_est.tmat_xy))

        self.assertAlmostEqual(prob_est.pi_xy.sum(), 1)
        self.assertAlmostEqual(prob_est.pi_x.sum(), 1)
        self.assertAlmostEqual(prob_est.pi_y.sum(), 1)

    def test_set_transition_matrices(self):

        prob_est = informant.MSMProbabilities(msmlag=1)

        # manually set transition matrices
        tmat = np.array([[.9, .1], [.9, .1]])
        prob_est.set_transition_matrices(tmat_x=tmat, tmat_y=tmat, tmat_xy=np.kron(tmat, tmat))

        # do some estimation
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        est = informant.DirectedInformation(prob_est).estimate(A, B)

        # check if still custom transition matrices
        np.testing.assert_array_equal(prob_est.tmat_x, tmat)
        np.testing.assert_array_equal(prob_est.tmat_y, tmat)
        np.testing.assert_array_equal(prob_est.tmat_xy, np.kron(tmat, tmat))

        self.assertAlmostEqual(est.d, 0)
        self.assertAlmostEqual(est.r, 0)

    def test_CTWProb(self):
        prob_est = informant.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(np.alltrue(prob_est.px > np.zeros_like(prob_est.px)))
        self.assertTrue(np.alltrue(prob_est.py > np.zeros_like(prob_est.py)))
        self.assertTrue(np.alltrue(prob_est.pxy > np.zeros_like(prob_est.pxy)))
