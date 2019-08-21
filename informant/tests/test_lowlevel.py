import unittest
import numpy as np
import itertools
from informant.utils import multivariate_mutual_info


class TestLowlevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_multivariate_mi_independent(self):


        pi_x = [.5, .5]
        pi_y = [.3, .7]
        pi_w = [.9, .1]

        pi_xy = [p1 * p2 for p1, p2 in itertools.product(pi_y, pi_x)]
        pi_xw = [p1 * p2 for p1, p2 in itertools.product(pi_w, pi_x)]
        pi_yw = [p1 * p2 for p1, p2 in itertools.product(pi_w, pi_y)]

        pi_xyw = []
        for raveled_index in range(2**3):
            n, m, k = np.unravel_index(raveled_index, (2, 2, 2))
            pi_xyw.append(pi_x[n] * pi_y[m] * pi_w[k])

        m = multivariate_mutual_info(pi_x, pi_y, pi_w, pi_xy, pi_xw, pi_yw, pi_xyw)

        self.assertAlmostEqual(m, 0.)

    def test_multivariate_mi_dependent(self):

        pi_x = [.5, .5]
        pi_y = [.3, .7]
        pi_w = [.9, .1]

        pi_xy = pi_xw = pi_yw = np.ones(4)/4
        pi_xyw = np.ones(8)/8

        m = multivariate_mutual_info(pi_x, pi_y, pi_w, pi_xy, pi_xw, pi_yw, pi_xyw)

        def summand(p1, p2, p3):
            return 1/8 * np.log2(.25**3 / (p1 * p2 * p3 * 1/8))

        m_test = sum([summand(p1, p2, p3) for p1, p2, p3 in
                      itertools.product(pi_x, pi_y, pi_w)])

        self.assertAlmostEqual(m, m_test)
