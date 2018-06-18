import unittest
import information
import numpy as np
import msmtools


class TestCrossover(unittest.TestCase):
    def test_MSMInfo(self):
        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        # true output
        def entropy1D(x):
            # the other thing seems to be for CTW specifically, taking into account negated x (?)
            #return - x * np.log2(x)
            return - x * np.log2(x) - (1 - x) * np.log2(1 - x)

        true_value_DI = entropy1D(p) - (((1 - p) * (1 - eps) + p * eps) * entropy1D(p * eps / ((1 - p) * (1 - eps) + p * eps)) + \
                    ((p) * (1 - eps) + (1 - p) * eps) * entropy1D((1 - p) * eps / ((p) * (1 - eps) + (1 - p) * eps)))


        prob = information.MSMProbabilities().estimate(X, Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * true_value_DI)


    def test_CTWInfo(self):
        p = 0.3
        eps = .2
        N = int(1e4)
        T = np.array([[1 - p, p], [p, 1 - p]])
        X = msmtools.generation.generate_traj(T, N)
        _errbits = np.random.rand(N) < eps
        Y = X.copy()
        Y[_errbits] = 1 - Y[_errbits]

        # true output
        def entropy1D(x):
            # the other thing seems to be for CTW specifically, taking into account negated x (?)
            #return - x * np.log2(x)
            return - x * np.log2(x) - (1 - x) * np.log2(1 - x)

        true_value_DI = entropy1D(p) - (((1 - p) * (1 - eps) + p * eps) * entropy1D(p * eps / ((1 - p) * (1 - eps) + p * eps)) + \
                    ((p) * (1 - eps) + (1 - p) * eps) * entropy1D((1 - p) * eps / ((p) * (1 - eps) + (1 - p) * eps)))


        prob = information.CTWProbabilities(5).estimate(X, Y)
        estimator = information.JiaoI4(prob)
        estimator.estimate(X, Y)

        self.assertAlmostEqual(estimator.d + estimator.r, estimator.m)
        self.assertAlmostEqual(estimator.d, true_value_DI, places=1)
        self.assertLess(estimator.r, .1 * true_value_DI)
