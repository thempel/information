import numpy as np
from scipy.linalg import expm
from tqdm import tqdm_notebook


def glauber_dynamics(nspins, alpha=0.1, gamma=0.95, ratematrix=False, tau=1):
    r"""
    Implements Glauber's master equation variant of the (1D) Ising model with periodic boundary conditions
    (J Math Phys 4 294 (1963); doi: 10.1063/1.1703954)

    :param nspins: int; number of spins in model (Note: returns 2^nspins times 2^nspins matix)
    :param alpha: float; basal spin flip-rate, defines time-scale (=0.1)
    :param gamma: float; gamma is equal to tanh(\beta 2J) where J is the spin-spin coupling constant
        in a corresponding Ising model, and \beta is the inverse temperature.
    :param ratematrix: bool; return rate matrix
    :param tau: float; time discretization for transition matrix

    :returns: transition matrix, (rate matrix)
    """
    s = np.array([-1, 1])
    r = np.zeros((2 ** nspins, 2 ** nspins))

    # for each global configuration state
    for config_jointspace in range(2 ** nspins):
        config_singlespins = np.array(np.unravel_index(config_jointspace, tuple(2 for _ in range(nspins))))

        # for each spin in the chain, compute flipping probability given its neighbors
        for spin in range(nspins):
            w_i = 0.5 * alpha * (1. - 0.5 * gamma * s[config_singlespins[spin]] *
                                 (s[config_singlespins[spin - 1]] + s[config_singlespins[(spin + 1) % nspins]]))

            # compute which the number in joint space numeration would be
            new_config_singlespins = config_singlespins.copy()
            new_config_singlespins[spin] = (config_singlespins[spin] + 1) % 2
            new_config_jointspace = np.ravel_multi_index(new_config_singlespins, tuple(2 for _ in range(nspins)))

            # write to array
            r[config_jointspace, new_config_jointspace] = w_i

    # valid rate matrix
    r[np.diag_indices(2 ** nspins)] = -r.sum(axis=1)

    T = expm(tau * r)
    if ratematrix:
        return T, r
    else:
        return T


def driven_glauber_dynamics(nspins, alpha=0.1, gamma=0.95, ratematrix=False, driver=0, alpha_driver=None, tau=1):
    r"""
    Implements Glauber's master equation variant of the (1D) Ising model with periodic boundary conditions
    with a driving spin that is not affected by its neighbors.
    (J Math Phys 4 294 (1963); doi: 10.1063/1.1703954)

    :param nspins: int; number of spins in model (Note: returns 2^nspins times 2^nspins matix)
    :param alpha: float; basal spin flip-rate, defines time-scale (=0.1)
    :param gamma: float; gamma is equal to tanh(\beta 2J) where J is the spin-spin coupling constant
        in a corresponding Ising model, and \beta is the inverse temperature.
    :param ratematrix: bool; return rate matrix
    :param tau: float; time discretization for transition matrix
    :param driver: int; identity of driving spin
    :param alpha_driver: float, alpha parameter (basal flipping rate) of driving spin

    :returns: transition matrix, (rate matrix)
    """
    if alpha_driver is None:
        alpha_driver = alpha
    s = np.array([-1, 1])
    r = np.zeros((2 ** nspins, 2 ** nspins))
    # for each global configuration state
    for config_jointspace in range(2 ** nspins):
        config_singlespins = np.array(np.unravel_index(config_jointspace, tuple(2 for _ in range(nspins))))

        # for each spin in the chain, compute flipping probability given its neighbors
        for spin in range(nspins):
            if spin != driver:
                w_i = 0.5 * alpha * (1. - 0.5 * gamma * s[config_singlespins[spin]] *
                                     (s[config_singlespins[spin - 1]] + s[config_singlespins[(spin + 1) % nspins]]))
            else:
                w_i = 0.5 * alpha_driver  # rate for single spin

            # compute which the number in joint space numeration would be
            new_config_singlespins = config_singlespins.copy()
            new_config_singlespins[spin] = (config_singlespins[spin] + 1) % 2
            new_config_jointspace = np.ravel_multi_index(new_config_singlespins, tuple(2 for _ in range(nspins)))

            # write to array
            r[config_jointspace, new_config_jointspace] = w_i
    # valid rate matrix
    r[np.diag_indices(2 ** nspins)] = -r.sum(axis=1)

    T = expm(tau * r)
    if ratematrix:
        return T, r
    else:
        return T


def gen_ising_traj(nspins, nsteps, alpha=.1, gamma=0.95, driver=0, show_progress=True, alpha_driver=None, tau=1, driven=False):
    """
    Generate trajectory of driven Ising system.

    :param nspins: int; number of spins in model (Note: returns 2^nspins times 2^nspins matix)
    :param alpha: float; basal spin flip-rate, defines time-scale (=0.1)
    :param gamma: float; gamma is equal to tanh(\beta 2J) where J is the spin-spin coupling constant
        in a corresponding Ising model, and \beta is the inverse temperature.
    :param tau: float; time discretization for transition matrix
    :param driver: int; identity of driving spin
    :param alpha_driver: float, alpha parameter (basal flipping rate) of driving spin
    :param nsteps: int; length of trajectory in steps
    :param show_progress: show progressbar
    :param driven: bool; use driven Glauber dynamics (default: False)
    :return: Trajectory in shape (nspins, nsteps)
    """

    if driven:
        T = driven_glauber_dynamics(nspins, alpha=alpha, tau=tau, gamma=gamma, driver=driver, alpha_driver=alpha_driver)
    else:
        T = glauber_dynamics(nspins, alpha=alpha, tau=tau, gamma=gamma)

    config_jointspace = np.random.choice(2 ** nspins)

    def progress_reporter(iterator, total=None):
        if show_progress:
            return tqdm_notebook(iterator, total=total)
        else:
            return iterator

    # sample states
    states = np.zeros((nspins, nsteps), dtype=int)
    for n in progress_reporter(range(nsteps), total=nsteps):
        # propagate
        config_jointspace = np.random.choice(2 ** nspins, p=T[config_jointspace])
        config_singlespins = np.unravel_index(config_jointspace, tuple(2 for _ in range(nspins)))
        states[:, n] = config_singlespins
    return states
