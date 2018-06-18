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

    def test_CTWProb(self):
        prob_est = information.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        self.assertTrue(np.alltrue(prob_est.px > np.zeros_like(prob_est.px)))
        self.assertTrue(np.alltrue(prob_est.py > np.zeros_like(prob_est.py)))
        self.assertTrue(np.alltrue(prob_est.pxy > np.zeros_like(prob_est.pxy)))

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
        prob_est = information.CTWProbabilities(D=3)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        estimator = information.JiaoI3(prob_est)

        estimator.estimate(A, B)

        self.assertGreaterEqual(estimator.d, 0)
        self.assertGreaterEqual(estimator.r, 0)
        self.assertGreaterEqual(estimator.m, 0)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)


    def test_MSMInfoI3(self):
        A = np.random.randint(0, 2, size=1000)
        B = np.random.randint(0, 2, size=1000)
        prob_est = information.MSMProbabilities()
        prob_est.estimate(A, B)

        estimator4 = information.JiaoI4(prob_est)
        estimator4.estimate(A, B)

        estimator3 = information.JiaoI3(prob_est)
        estimator3.estimate(A, B)
        self.assertGreaterEqual(estimator3.d, 0)
        self.assertGreaterEqual(estimator3.r, 0)
        self.assertGreaterEqual(estimator3.m, 0)

        self.assertAlmostEqual(estimator3.d + estimator3.r, estimator3.m, places=2)

        p = 0.3
        eps = .2
        N = int(1e5)
        T = np.array([[1 - p, p], [p, 1 - p]])
        A = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        B = A.copy()
        B[_errbits] = 1 - B[_errbits]

        prob_est = information.MSMProbabilities(reversible=False)
        prob_est.estimate(A, B)

        estimator4 = information.JiaoI4(prob_est)
        estimator4.estimate(A, B)

        estimator3 = information.JiaoI3(prob_est)
        estimator3.estimate(A, B)
        self.assertGreaterEqual(estimator3.d, 0)
        self.assertGreaterEqual(estimator3.r, 0)
        self.assertGreaterEqual(estimator3.m, 0)

        self.assertAlmostEqual(estimator3.d + estimator3.r, estimator3.m, places=2)

        self.assertAlmostEqual(estimator3.d, estimator4.d, places=1)
        self.assertAlmostEqual(estimator3.r, estimator4.r, places=1)
        self.assertAlmostEqual(estimator3.m, estimator4.m, places=1)
