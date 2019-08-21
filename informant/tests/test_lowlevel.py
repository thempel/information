import unittest
import informant
import numpy as np
import itertools


class TestLowlevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_multivariate_mi_independent(self):

        class P:
            def __init__(self):
                self.pi_x = [.5, .5]
                self.pi_y = [.3, .7]
                self.pi_w = [.9, .1]

                self.pi_xy = [p1 * p2 for p1, p2 in itertools.product(self.pi_y, self.pi_x)]
                self.pi_xw = [p1 * p2 for p1, p2 in itertools.product(self.pi_w, self.pi_x)]
                self.pi_yw = [p1 * p2 for p1, p2 in itertools.product(self.pi_w, self.pi_y)]

                self.pi_xyw = []
                for raveled_index in range(2**3):
                    n, m, k = np.unravel_index(raveled_index, (2, 2, 2))
                    self.pi_xyw.append(self.pi_x[n] * self.pi_y[m] * self.pi_w[k])

        est = informant.CausallyConditionedDI(P())
        est.Nx = est.Ny = est.Nw = 2  # manually set properties
        m = est._multivariate_mutual_info()

        self.assertAlmostEqual(m, 0.)

    def test_multivariate_mi_dependent(self):

        class P:
            def __init__(self):
                self.pi_x = [.5, .5]
                self.pi_y = [.3, .7]
                self.pi_w = [.9, .1]

                self.pi_xy = self.pi_xw = self.pi_yw = np.ones(4)/4
                self.pi_xyw = np.ones(8)/8
        prob = P()
        est = informant.CausallyConditionedDI(prob)
        est.Nx = est.Ny = est.Nw = 2  # manually set properties
        m = est._multivariate_mutual_info()

        def summand(p1, p2, p3):
            return 1/8 * np.log2(.25**3 / (p1 * p2 * p3 * 1/8))

        m_test = sum([summand(p1, p2, p3) for p1, p2, p3 in
                      itertools.product(prob.pi_x, prob.pi_y, prob.pi_w)])

        self.assertAlmostEqual(m, m_test)
