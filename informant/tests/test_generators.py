import unittest
import informant
import numpy as np
import six
import msmtools
from utils import GenerateTestMatrix


class TestSimple(six.with_metaclass(GenerateTestMatrix, unittest.TestCase)):

    generators = (informant.generators.driven_glauber_dynamics, informant.generators.glauber_dynamics)

    params = {
        '_test_is_transition_matrix': [dict(generator=g) for g in generators],
        '_test_is_rate_matrix': [dict(generator=g) for g in generators],
        '_test_recover_tmat_from_data': [dict(generator=g) for g in generators]
    }

    @classmethod
    def setUpClass(cls):
        cls.nspins = 3

    def _test_is_transition_matrix(self, generator):
        T = generator(self.nspins)

        self.assertTrue(msmtools.analysis.is_transition_matrix(T))

    def _test_is_rate_matrix(self, generator):
        _, R = generator(self.nspins, ratematrix=True)

        self.assertTrue(msmtools.analysis.is_rate_matrix(R))

    def test_recover_tmat_from_data(self):
        nspins = 2
        t = informant.generators.gen_ising_traj(nspins, 10000, alpha=.3, gamma=.8,
                                                driven=False, show_progress=False)
        tmat_ref = informant.generators.glauber_dynamics(nspins, alpha=.3, gamma=.8)
        cmat_dat = msmtools.estimation.count_matrix(np.ravel_multi_index(t, tuple(2 for _ in range(nspins))),
                                                    1, sparse_return=False)
        tmat_dat = msmtools.estimation.transition_matrix(cmat_dat)

        np.testing.assert_allclose(tmat_dat, tmat_ref, atol=0.05)