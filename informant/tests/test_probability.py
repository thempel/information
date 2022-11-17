
from numpy.testing import assert_raises, assert_, assert_almost_equal

import informant
import numpy as np
from deeptime.markov.tools.analysis import is_transition_matrix


def test_MSMProb():
    prob_est = informant.MSMProbabilities(msmlag=1)
    A = np.random.randint(0, 2, 100)
    B = np.random.randint(0, 2, 100)
    prob_est.estimate(A, B)

    assert_(is_transition_matrix(prob_est.tmat_x))
    assert_(is_transition_matrix(prob_est.tmat_y))
    assert_(is_transition_matrix(prob_est.tmat_xy))

    assert_almost_equal(prob_est.pi_xy.sum(), 1)
    assert_almost_equal(prob_est.pi_x.sum(), 1)
    assert_almost_equal(prob_est.pi_y.sum(), 1)

def test_is_estimated_decorator():
    prob_est = informant.MSMProbabilities(msmlag=1)
    with assert_raises(RuntimeError):
        prob_est.pi_x

def test_set_transition_matrices():

    prob_est = informant.MSMProbabilities(msmlag=1)

    # manually set transition matrices
    tmat = np.array([[.9, .1], [.9, .1]])
    prob_est.set_transition_matrices(tmat_x=tmat, tmat_y=tmat, tmat_xy=np.kron(tmat, tmat))

    # do some estimation
    A = [np.random.randint(0, 2, 100)]
    B = [np.random.randint(0, 2, 100)]
    prob_est.estimate(A, B)
    est = informant.DirectedInformation(prob_est)
    est.Nx = 2
    est.Ny = 2
    d = est.stationary_estimate(A, B)

    # check if still custom transition matrices
    np.testing.assert_array_equal(prob_est.tmat_x, tmat)
    np.testing.assert_array_equal(prob_est.tmat_y, tmat)
    np.testing.assert_array_equal(prob_est.tmat_xy, np.kron(tmat, tmat))

    assert_almost_equal(d, 0)


def test_CTWProb():
    prob_est = informant.CTWProbabilities(D=3)
    A = np.random.randint(0, 2, 100)
    B = np.random.randint(0, 2, 100)
    prob_est.estimate(A, B)

    assert_(np.alltrue(prob_est.px > np.zeros_like(prob_est.px)))
    assert_(np.alltrue(prob_est.py > np.zeros_like(prob_est.py)))
    assert_(np.alltrue(prob_est.pxy > np.zeros_like(prob_est.pxy)))
