import unittest
import information
import numpy as np


class TestSimple(unittest.TestCase):

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


    def test_CTWInfo(self):
        prob_est = information.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        estimator = information.JiaoI4(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_CTWInfo_bad_traj(self):
        prob_est = information.CTWProbabilities(D=3)
        A = np.ones(100, dtype=int)
        B = np.ones(100, dtype=int)
        prob_est.estimate(A, B)

        estimator = information.JiaoI4(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)

    def test_CTWInfoI3(self):
        prob_est = information.CTWProbabilities(D=5)
        A = np.random.randint(0, 2, 1000)
        B = np.random.randint(0, 2, 1000)
        prob_est.estimate(A, B)

        estimator = information.JiaoI3(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)

    def test_MSMInfoI3(self):
        A = np.random.randint(0, 2, size=1000)
        B = np.random.randint(0, 2, size=1000)
        prob_est = information.MSMProbabilities()
        prob_est.estimate(A, B)

        estimator = information.JiaoI3(prob_est)
        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m, places=2)

    def test_symmetric_estimate(self):
        A = np.random.randint(0, 2, size=1000)
        B = np.random.randint(0, 2, size=1000)
        prob_est_AB = information.MSMProbabilities().estimate(A, B)
        prob_est_BA = information.MSMProbabilities().estimate(B, A)

        estimator_AB = information.JiaoI4(prob_est_AB).symmetrized_estimate(A, B)
        estimator_BA = information.JiaoI4(prob_est_BA).symmetrized_estimate(B, A)

        self.assertAlmostEqual(estimator_AB.r, estimator_BA.d)
        self.assertAlmostEqual(estimator_AB.d, estimator_BA.r)
        self.assertAlmostEqual(estimator_AB.m, estimator_BA.m)

