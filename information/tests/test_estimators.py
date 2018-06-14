import unittest
import information
import numpy as np


class TestSuperSimple(unittest.TestCase):
    def test_MSMProb(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

    def test_1(self):
        prob_est = information.MSMProbabilities(msmlag=1)
        A = np.random.randint(0, 2, 100)
        B = np.random.randint(0, 2, 100)
        prob_est.estimate(A, B)

        info_est_instance = information.JiaoI4()

        estimator = information.Information(prob_est, info_est_instance)
        estimator.estimate(A, B)
