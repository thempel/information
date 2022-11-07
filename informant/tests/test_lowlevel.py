import numpy as np
import itertools

from numpy.testing import assert_almost_equal, assert_equal
from informant.utils import multivariate_mutual_info, lag_observations


def test_multivariate_mi_independent():

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

    assert_almost_equal(m, 0.)


def test_multivariate_mi_dependent():

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

    assert_almost_equal(m, m_test)


def test_lag_observations():
    t = [np.arange(0, 100).astype(int)]
    tau = 3
    t_lagged_ref = [np.arange(n, 100, tau).astype(int) for n in range(tau)]
    t_lagged = lag_observations(t, tau)
    assert_equal(len(t_lagged), len(t_lagged_ref))
    assert_equal(len(t_lagged_ref[0]), len(t_lagged[0]))
    for a, b in zip(t_lagged, t_lagged_ref):
        np.testing.assert_array_equal(a, b)
