import pytest
from numpy.testing import assert_
import informant
import numpy as np
from deeptime.markov.tools.analysis import is_transition_matrix, is_rate_matrix

generators = (informant.generators.driven_glauber_dynamics, informant.generators.glauber_dynamics)
n_spins = 3


@pytest.mark.parametrize("generator", generators)
def test_is_transition_matrix(generator):
    T = generator(nspins=n_spins)
    assert_(is_transition_matrix(T))


@pytest.mark.parametrize("generator", generators)
def test_is_rate_matrix(generator):
    _, R = generator(n_spins, ratematrix=True)
    assert_(is_rate_matrix(R))


def test_recover_tmat_from_data():
    nspins = 2
    t = informant.generators.gen_ising_traj(nspins, 10000, alpha=.3, gamma=.8,
                                            driven=False, show_progress=False)
    tmat_ref = informant.generators.glauber_dynamics(nspins, alpha=.3, gamma=.8)

    from deeptime.markov.msm import MaximumLikelihoodMSM
    tmodel = MaximumLikelihoodMSM(reversible=True, lagtime=1). \
        fit_fetch(np.ravel_multi_index(t, tuple(2 for _ in range(nspins))))

    np.testing.assert_allclose(tmodel.transition_matrix, tmat_ref, atol=0.05)
