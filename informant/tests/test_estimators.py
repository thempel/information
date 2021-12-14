import unittest
import informant
import numpy as np
import six
import itertools
from utils import GenerateTestMatrix


class TestSimple(six.with_metaclass(GenerateTestMatrix, unittest.TestCase)):

    di_estimators = (informant.JiaoI4, informant.JiaoI3)
    all_estimators = (informant.JiaoI4, informant.JiaoI3, informant.TransferEntropy, informant.DirectedInformation)
    p_estimators = (informant.MSMProbabilities, informant.CTWProbabilities)
    ccdi_estimators = (informant.CausallyConditionedDI,
                       informant.CausallyConditionedDIJiaoI3,
                       informant.CausallyConditionedDIJiaoI4,
                       informant.CausallyConditionedTE)
    params = {
        '_test_binary': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)] +
                              [dict(di_est=informant.TransferEntropy, p_est=informant.MSMProbabilities),
                               dict(di_est=informant.DirectedInformation, p_est=informant.MSMProbabilities)],
        '_test_multistate': [dict(di_est=d, p_est=informant.MSMProbabilities) for d in all_estimators],
        # TODO: above should be tested with CTW, too, but seems disfunctional.
        '_test_compare_ctw_msm': [dict(di_est=d) for d in di_estimators],
        '_test_symmetric_estimate': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)] +
                            [dict(di_est=informant.TransferEntropy, p_est=informant.MSMProbabilities),
                             dict(di_est=informant.DirectedInformation, p_est=informant.MSMProbabilities)],
        '_test_causally_cond_simple': [dict(di_est=d) for d in ccdi_estimators]
        #'_test_polluted_state': [dict(di_est=d, p_est=informant.MSMProbabilities) for d in di_estimators],
        #'_test_congruency': [dict(di_est1=informant.JiaoI3, di_est2=informant.JiaoI4, p_est=p) for p in p_estimators],
        #'_test_symmetric_estimate': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)] +
        #[dict(di_est=informant.TransferEntropy, p_est=informant.MSMProbabilities)],
        #'_test_multitraj_support': [dict(di_est=d, p_est=p) for d, p in itertools.product(di_estimators, p_estimators)]
    }

    @classmethod
    def setUpClass(cls):
        cls.A_binary = np.random.randint(0, 2, 100000)
        cls.B_binary = np.random.randint(0, 2, 100000)

        cls.A_nonbinary = np.random.randint(0, 3, 100000)
        cls.B_nonbinary = np.random.randint(0, 4, 100000)

    def _test_binary(self, di_est, p_est):
        estimator = di_est(p_est()).estimate(self.A_binary, self.B_binary)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        if not isinstance(estimator, (informant.TransferEntropy, informant.DirectedInformation)):
            self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=1)

    def _test_multistate(self, di_est, p_est):
        estimator = di_est(p_est()).estimate(self.A_nonbinary, self.B_nonbinary)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        if not isinstance(estimator, (informant.TransferEntropy, informant.DirectedInformation)):
            self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=1)

    def _test_compare_ctw_msm(self, di_est):
        est_msm = di_est(informant.MSMProbabilities()).estimate(self.A_binary, self.B_binary)
        est_ctw= di_est(informant.CTWProbabilities()).estimate(self.A_binary, self.B_binary)

        self.assertAlmostEqual(est_msm.d, est_ctw.d, places=2)
        self.assertAlmostEqual(est_msm.r, est_ctw.r, places=2)
        self.assertAlmostEqual(est_msm.m, est_ctw.m, places=2)

    def test_CTWInfo_bad_traj(self):
        prob_est = informant.CTWProbabilities(D=3)
        A = np.ones(100, dtype=int)
        B = np.ones(100, dtype=int)

        estimator = informant.JiaoI4(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def _test_symmetric_estimate(self, di_est, p_est):
        A = self.A_binary
        B = self.B_binary

        estimator_AB = di_est(p_est()).symmetrized_estimate(A, B)
        estimator_BA = di_est(p_est()).symmetrized_estimate(B, A)

        np.testing.assert_allclose(estimator_AB.r, estimator_BA.d, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(estimator_AB.d, estimator_BA.r, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(estimator_AB.m, estimator_BA.m, rtol=1e-2, atol=1e-3)



    def test_MSMInfoRew(self):
        prob_est = informant.MSMProbabilities(msmlag=1)
        A = [np.random.randint(0, 2, 100) for _ in range(20)]
        B = [np.random.randint(0, 2, 100) for _ in range(20)]

        estimator = informant.JiaoI4(prob_est)

        estimator.estimate(A, B, traj_eq_reweighting=True)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_compare_I4_I3(self):
        prob_est = informant.MSMProbabilities(msmlag=1)

        est1 = informant.JiaoI3(prob_est)
        est1.estimate(self.A_nonbinary, self.B_nonbinary)

        est2 = informant.JiaoI4(prob_est)
        est2.estimate(self.A_nonbinary, self.B_nonbinary)

        self.assertAlmostEqual(est2.d, est1.d, places=1)
        self.assertAlmostEqual(est2.r, est1.r, places=1)
        self.assertAlmostEqual(est2.m, est1.m, places=1)

    def _test_causally_cond_simple(self, di_est):
        est = di_est(informant.MSMProbabilities())
        est.estimate(self.A_nonbinary, self.B_nonbinary, self.A_binary)

        np.testing.assert_allclose(est.causally_conditioned_di[0], 0, atol=1e-2, rtol=1e-2)
