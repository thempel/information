import pytest
from numpy.testing import assert_almost_equal, assert_array_less

import informant
import numpy as np
from deeptime.markov.msm import MarkovStateModel

class CoChannel:
    def __init__(self):
        p = 0.3
        eps = .2
        N = int(1e5)
        T = np.array([[1 - p, p], [p, 1 - p]])


        msm = MarkovStateModel(transition_matrix=T)
        self.X = msm.simulate(N)
        _errbits = np.random.rand(N) < eps
        self.Y = self.X.copy()
        self.Y[_errbits] = 1 - self.Y[_errbits]

        self.true_value_DI = self.entropy1D(p) - (
                    ((1 - p) * (1 - eps) + p * eps) * self.entropy1D(p * eps / ((1 - p) * (1 - eps) + p * eps)) +
                    ((p) * (1 - eps) + (1 - p) * eps) * self.entropy1D(
                (1 - p) * eps / ((p) * (1 - eps) + (1 - p) * eps)))

    @staticmethod
    def entropy1D(x):
        return - x * np.log2(x) - (1 - x) * np.log2(1 - x)


@pytest.fixture(scope="module")
def co_channel() -> CoChannel:
    return CoChannel()


di_estimators = (informant.JiaoI4, informant.JiaoI3)
all_estimators = (informant.JiaoI4, informant.JiaoI3, informant.TransferEntropy, informant.DirectedInformation)
# TODO: find more pythonic solution to exclude slow CTW-based tests on CI but not locally
p_estimators = (informant.MSMProbabilities,
                #informant.CTWProbabilities
                )


def test_verysimple(co_channel):
    estimator = informant.DirectedInformation(informant.MSMProbabilities())

    estimator.estimate(co_channel.X, co_channel.Y)

    assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=2)
    assert_almost_equal(estimator.d, co_channel.true_value_DI, decimal=1)
    assert_almost_equal(estimator.r, 0., decimal=1)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_simple(co_channel, di_est, p_est):
    estimator = di_est(p_est())

    estimator.estimate(co_channel.X, co_channel.Y)

    assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=2)
    assert_almost_equal(estimator.d, co_channel.true_value_DI, decimal=1)
    assert_almost_equal(estimator.r, 0., decimal=1)


@pytest.mark.parametrize('di_est', di_estimators)
def test_polluted_state(co_channel, di_est):
    """
    Tests if polluting one state of the second trajectory with random noise alters results.
    """
    p_est = informant.MSMProbabilities
    Y = co_channel.Y.copy()
    idx = Y == 1
    Y[idx] = co_channel.Y[idx] + np.random.randint(0, 2, size=co_channel.Y[idx].shape[0])

    estimator = di_est(p_est())

    estimator.estimate(co_channel.X, Y)

    assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=2)
    assert_almost_equal(estimator.d, co_channel.true_value_DI, decimal=1)
    assert_almost_equal(estimator.r, 0., decimal=1)


@pytest.mark.parametrize('di_est1,di_est2,p_est',
                         [(informant.JiaoI3, informant.JiaoI4, p) for p in p_estimators] +
                        [(informant.DirectedInformation, informant.JiaoI4, informant.MSMProbabilities)]
                         )
def test_congruency(co_channel, di_est1, di_est2, p_est):
    est1 = di_est1(p_est())
    est2 = di_est2(p_est())

    est1.estimate(co_channel.X, co_channel.Y)
    est2.estimate(co_channel.X, co_channel.Y)

    assert_almost_equal(est1.d, est2.d, decimal=2)
    assert_almost_equal(est1.r, est2.r, decimal=2)
    assert_almost_equal(est1.m, est2.m, decimal=2)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_multitraj_support(co_channel, di_est, p_est):
    estimator = di_est(p_est()).estimate([co_channel.X, co_channel.X], [co_channel.Y, co_channel.Y])

    assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=2)
    assert_almost_equal(estimator.d, co_channel.true_value_DI, decimal=1)
    assert_almost_equal(estimator.r, 0., decimal=1)


def test_MSMI4re(co_channel):
    """
    Test MSM probability and I4 estimator for multiple trajs.
    :return:
    """

    p = 0.3
    eps = .2
    N = int(1e4)
    T = np.array([[1 - p, p], [p, 1 - p]])
    X_list, Y_list = [], []
    for _ in range(50):
        msm = MarkovStateModel(transition_matrix=T)
        X = msm.simulate(N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]
        X_list.append(X)
        Y_list.append(Y)

    prob = informant.MSMProbabilities()
    estimator = informant.JiaoI4(prob)
    estimator.estimate(X_list, Y_list, traj_eq_reweighting=True)

    assert_almost_equal(estimator.d + estimator.r, estimator.m)
    assert_almost_equal(estimator.d, co_channel.true_value_DI, decimal=1)
    assert_almost_equal(estimator.r, 0., decimal=1)


@pytest.mark.skip("Need to derive true result to compare to!")
def test_delayed():
    N = int(1e4)
    shift = 10
    p = 0.01
    eps = .33
    T = np.array([[1-p, p], [p, 1-p]])

    msm = MarkovStateModel(transition_matrix=T)
    X = msm.simulate(N)
    _errbits = np.random.rand(N) < eps
    Y = X.copy()
    Y[_errbits] = 1 - Y[_errbits]

    X = X[shift:]
    Y = Y[:-shift]

    prob = informant.MSMProbabilities(shift, reversible=False).estimate(X, Y)
    estimator = informant.JiaoI4(prob)
    estimator.estimate(X, Y)

    assert_almost_equal(estimator.d + estimator.r, estimator.m)
    # TODO: derive relationship to delaytime and add actual value.
    assert_almost_equal(estimator.d, 0.008, decimal=1)
    assert_array_less(.1 * 0.008, estimator.r)


def test_schreiber_MI(co_channel):
    prob_est = informant.MSMProbabilities()
    estimator = informant.MutualInfoStationaryDistribution(prob_est).estimate(co_channel.X, co_channel.Y)

    # TODO: compute actual MI value for binary delayed channel.
    assert_array_less(co_channel.true_value_DI, estimator.m)
