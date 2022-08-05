import unittest
import informant
import numpy as np
import six
from deeptime.markov.tools.analysis import is_transition_matrix, is_rate_matrix
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

        self.assertTrue(is_transition_matrix(T))

    def _test_is_rate_matrix(self, generator):
        _, R = generator(self.nspins, ratematrix=True)

        self.assertTrue(is_rate_matrix(R))

    def test_recover_tmat_from_data(self):
        nspins = 2
        t = informant.generators.gen_ising_traj(nspins, 10000, alpha=.3, gamma=.8,
                                                driven=False, show_progress=False)
        tmat_ref = informant.generators.glauber_dynamics(nspins, alpha=.3, gamma=.8)


        from deeptime.markov.msm import MaximumLikelihoodMSM
        tmodel = MaximumLikelihoodMSM(reversible=True, lagtime=1).\
            fit_fetch(np.ravel_multi_index(t, tuple(2 for _ in range(nspins))))

        np.testing.assert_allclose(tmodel.transition_matrix, tmat_ref, atol=0.05)