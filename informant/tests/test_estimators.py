
import pytest
from numpy.testing import assert_, assert_array_less, assert_almost_equal

import informant
import numpy as np
import itertools

di_estimators = (informant.JiaoI4, informant.JiaoI3)
all_estimators = (informant.JiaoI4, informant.JiaoI3, informant.TransferEntropy, informant.DirectedInformation)
p_estimators = (informant.MSMProbabilities,
                informant.CTWProbabilities
                )
ccdi_estimators = (informant.CausallyConditionedDI,
                   informant.CausallyConditionedDIJiaoI3,
                   informant.CausallyConditionedDIJiaoI4,
                   informant.CausallyConditionedTE)

possible_combinations = list(itertools.product(di_estimators, p_estimators)) + \
    [(informant.TransferEntropy, informant.MSMProbabilities),
     (informant.DirectedInformation, informant.MSMProbabilities)]

class SimpleData:
    def __init__(self):
        self.A_binary = np.random.randint(0, 2, 100000)
        self.B_binary = np.random.randint(0, 2, 100000)
        self.A_nonbinary = np.random.randint(0, 3, 100000)
        self.B_nonbinary = np.random.randint(0, 4, 100000)


@pytest.fixture(scope="module")
def simple_data() -> SimpleData:
    return SimpleData()


@pytest.mark.parametrize('di_est,p_est', possible_combinations)
def test_binary(simple_data, di_est, p_est):
    estimator = di_est(p_est()).estimate(simple_data.A_binary, simple_data.B_binary)

    assert_(estimator.d > 0)
    assert_(estimator.r > 0)
    assert_(estimator.m > 0)
    if not isinstance(estimator, (informant.TransferEntropy, informant.DirectedInformation)):
        assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=1)


# TODO: should be tested with CTW, too, but seems disfunctional.
@pytest.mark.parametrize('di_est', all_estimators)
def test_multistate(simple_data, di_est):
    p_est = informant.MSMProbabilities
    estimator = di_est(p_est()).estimate(simple_data.A_nonbinary, simple_data.B_nonbinary)

    assert_array_less(0, estimator.d)
    assert_array_less(0, estimator.r)
    assert_array_less(0, estimator.m)

    if not isinstance(estimator, (informant.TransferEntropy, informant.DirectedInformation)):
        assert_almost_equal(estimator.d + estimator.r, estimator.m, decimal=1)


@pytest.mark.parametrize('di_est,p_est', possible_combinations)
def test_symmetric_estimate(simple_data, di_est, p_est):
    A = simple_data.A_binary
    B = simple_data.B_binary

    estimator_AB = di_est(p_est()).symmetrized_estimate(A, B)
    estimator_BA = di_est(p_est()).symmetrized_estimate(B, A)

    np.testing.assert_allclose(estimator_AB.r, estimator_BA.d, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(estimator_AB.d, estimator_BA.r, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(estimator_AB.m, estimator_BA.m, rtol=1e-2, atol=1e-3)


def test_MSMInfoRew():
    prob_est = informant.MSMProbabilities(msmlag=1)
    A = [np.random.randint(0, 2, 100) for _ in range(20)]
    B = [np.random.randint(0, 2, 100) for _ in range(20)]

    estimator = informant.JiaoI4(prob_est)

    estimator.estimate(A, B, traj_eq_reweighting=True)

    assert_array_less(0, estimator.d)
    assert_array_less(0, estimator.r)
    assert_array_less(0, estimator.m)

    assert_almost_equal(estimator.d + estimator.r, estimator.m)


def test_compare_I4_I3(simple_data):
    prob_est = informant.MSMProbabilities(msmlag=1)

    est1 = informant.JiaoI3(prob_est)
    est1.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary)

    est2 = informant.JiaoI4(prob_est)
    est2.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary)

    assert_almost_equal(est2.d, est1.d, decimal=1)
    assert_almost_equal(est2.r, est1.r, decimal=1)
    assert_almost_equal(est2.m, est1.m, decimal=1)


@pytest.mark.parametrize('di_est', ccdi_estimators)
def test_causally_cond_simple(simple_data, di_est):
    est = di_est(informant.MSMProbabilities())
    est.estimate(simple_data.A_nonbinary, simple_data.B_nonbinary, simple_data.A_binary)

    np.testing.assert_allclose(est.causally_conditioned_di[0], 0, atol=1e-2, rtol=1e-2)
