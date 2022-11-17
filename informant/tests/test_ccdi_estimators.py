import pytest
from numpy.testing import assert_almost_equal, assert_, assert_array_less

import informant
import numpy as np
from deeptime.markov.msm import MarkovStateModel


class SimpleData:
    def __init__(self):
        self.A_binary = np.random.randint(0, 2, 5000)
        self.B_binary = np.random.randint(0, 2, 5000)

        self.A_nonbinary = np.random.randint(0, 3, 5000)
        self.B_nonbinary = np.random.randint(0, 4, 5000)

        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        msm = MarkovStateModel(transition_matrix=T)
        X = msm.simulate(N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        self.X = X[2:]
        self.Y = Y[1:-1]

        # proxy configuration: X -> Y -> Z
        _errbits = np.random.rand(N) < eps
        Z = Y.copy()
        Z[_errbits] = 1 - Z[_errbits]
        self.Z_proxy = Z[:-2]

        # cascade configuration: X -> Y; X -> Z
        _errbits = np.random.rand(N) < eps
        Z = X.copy()
        Z[_errbits] = 1 - Z[_errbits]

        self.Z_cascade = Z[1:-1]


@pytest.fixture(scope="module")
def simple_data() -> SimpleData:
    return SimpleData()


di_estimators = (informant.CausallyConditionedDI, informant.CausallyConditionedDIJiaoI3,
                 informant.CausallyConditionedDIJiaoI4)
p_estimators = (informant.MSMProbabilities, )


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_simple(simple_data, di_est, p_est):
    est = di_est(p_est())
    est.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary, simple_data.A_binary)

    assert_almost_equal(est.causally_conditioned_di[0], 0, decimal=0)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_simple_multiconditionals(simple_data, di_est, p_est):
    est = di_est(p_est())
    est.estimate(simple_data.A_binary, simple_data.B_binary, [simple_data.B_nonbinary, simple_data.A_nonbinary])

    assert_almost_equal(est.causally_conditioned_di[0], 0, decimal=0)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_simple_multiprocessing(simple_data, di_est, p_est):
    est = di_est(p_est())
    est.estimate(simple_data.A_binary, simple_data.B_binary, [simple_data.B_nonbinary, simple_data.A_nonbinary],
                 n_jobs=2)
    ccdi_multiproc = est.causally_conditioned_di

    est = di_est(p_est())
    est.estimate(simple_data.A_binary, simple_data.B_binary, [simple_data.B_nonbinary, simple_data.A_nonbinary],
                 n_jobs=1)
    ccdi_singleproc = est.causally_conditioned_di

    np.testing.assert_array_equal(ccdi_multiproc, ccdi_singleproc)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_simple_raises_disconnectedXW(di_est, p_est):
    est = di_est(p_est())

    est.estimate(np.array([0, 1, 0, 1, 1]),
                 np.array([0, 1, 0, 1, 0]),
                 [np.array([0, 0, 0, 1, 1])])

    assert_(not np.isfinite(est.causally_conditioned_di[0]))


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_proxy(simple_data, di_est, p_est):
    # TODO:
    # This is a non-Markovian system. Seems to work qualitatively. Find out why.

    # test if direct link is detected with causally cond entropy > 0
    estimator = di_est(p_est())
    estimator.estimate(simple_data.X, simple_data.Y, simple_data.Z_proxy)

    assert_array_less(0., estimator.causally_conditioned_di[0])

    # test if indirect link is detected with causally cond entropy ~ 0.
    estimator = di_est(p_est())
    estimator.estimate(simple_data.X, simple_data.Z_proxy, simple_data.Y)

    assert_almost_equal(estimator.causally_conditioned_di[0], 0., decimal=2)


@pytest.mark.parametrize('di_est', di_estimators)
@pytest.mark.parametrize('p_est', p_estimators)
def test_cascade(simple_data, di_est, p_est):
    # test if direct link is detected with causally cond entropy > 0
    estimator = di_est(p_est())
    estimator.estimate(simple_data.X, simple_data.Y, simple_data.Z_cascade)

    assert_array_less(0, estimator.causally_conditioned_di)

    # test if indirect link is detected with causally cond entropy ~ 0.
    estimator = di_est(p_est())
    estimator.estimate(simple_data.Y, simple_data.Z_cascade, simple_data.X)

    assert_almost_equal(estimator.causally_conditioned_di[0], 0., decimal=2)


def test_compare_I4_I3(simple_data):

    prob_est = informant.MSMProbabilities(msmlag=1)
    est1 = informant.CausallyConditionedDIJiaoI3(prob_est)
    est1.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary, simple_data.A_binary)

    est2 = informant.CausallyConditionedDIJiaoI4(prob_est)
    est2.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary, simple_data.A_binary)

    assert_almost_equal(est2.causally_conditioned_di[0], est1.causally_conditioned_di[0], decimal=1)
