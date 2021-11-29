import numpy as np


def lag_observations(observations, lag, stride=1):
    r""" Create new trajectories that are subsampled at lag but shifted

    Given a trajectory (s0, s1, s2, s3, s4, ...) and lag 3, this function will generate 3 trajectories
    (s0, s3, s6, ...), (s1, s4, s7, ...) and (s2, s5, s8, ...). Use this function in order to parametrize a MLE
    at lag times larger than 1 without discarding data. Do not use this function for Bayesian estimators, where
    data must be given such that subsequent transitions are uncorrelated.

    Function copied from markovmodel/bhmm, version 0.6.2

    Parameters
    ----------
    observations : list of int arrays
        observation trajectories
    lag : int
        lag time
    stride : int, default=1
        will return only one trajectory for every stride. Use this for Bayesian analysis.

    """
    obsnew = []
    for obs in observations:
        for shift in range(0, lag, stride):
            obs_lagged = (obs[shift:][::lag])
            if len(obs_lagged) > 1:
                obsnew.append(obs_lagged)
    return obsnew


def relabel_dtrajs(X):
    if np.unique(X).max() + 1 > len(set(X)):
        mapper = np.zeros(np.unique(X).max() + 1) - 1
        mapper[np.unique(X)] = list(range(np.unique(X).shape[0]))
        _x = mapper[X]
    else:
        _x = X

    return _x

def ensure_dtraj_format(A, B, W=None):

    if not isinstance(A, list): A = [A]
    if not isinstance(B, list): B = [B]
    assert isinstance(A[0], np.ndarray)
    assert isinstance(B[0], np.ndarray)

    if W is None:
        for n, (a1, a2) in enumerate(zip(A, B)):
            if a1.shape[0] != a2.shape[0]:
                raise RuntimeError('Trajectories not compatible. Lengths of {}th trajs are {} and {}, '
                                   'respectively.'.format(n, a1.shape[0], a2.shape[0]))

        return A, B

    else:
        if not isinstance(W, list): W = [W]
        assert isinstance(W[0], np.ndarray)

        for n, (a1, a2, a3) in enumerate(zip(A, B, W)):
            if a1.shape[0] != a2.shape[0] or a1.shape[0] != a3.shape[0]:
                raise RuntimeError('Trajectories not compatible. Lengths of {}th trajs are {}, {} and {}, '
                                   'respectively.'.format(n, a1.shape[0], a2.shape[0], a3.shape[0]))

        return A, B, W


def reweight_trajectories(A, B, p, size=None):
    """
    Reweights trajectories by drawing starting states according to a
    distribution of combinatorial states.
    :param A: time-series 1
    :param B: time-series 2
    :param p: combinatorial state distribution
    :param size: number of trajectories to return
    :return: A, B reweighted
    """

    from itertools import product
    if size is None:
        size = len(A)

    Nx = np.concatenate(A).max() + 1
    Ny = np.concatenate(B).max() + 1
    comb2local = np.vstack([np.tile(np.arange(Nx), Ny), np.repeat(np.arange(Ny), Nx)]).T

    states = np.arange(Nx * Ny)
    sample_comb_states = np.random.choice(states, size=size, p=p)
    sample_local_states = comb2local[sample_comb_states]
    starting_points = np.empty((Nx, Ny), dtype=object)

    for a, b in product(range(Nx), range(Ny)):
        p = np.where([np.all([_a[0], _b[0]] == [a, b]) for _a, _b in zip(A, B)])[0]
        if p.shape[0] < 1:
            raise RuntimeError('Cannot reweight to equilibrium because not all combinatorial states '
                               'are starting states.')
        else:
            starting_points[a, b] = p

    A_reweighted, B_reweighted = [], []
    for lsample in sample_local_states:
        choice = np.random.choice(starting_points[lsample[0], lsample[1]])
        A_reweighted.append(A[choice])
        B_reweighted.append(B[choice])

    return A_reweighted, B_reweighted


def multivariate_mutual_info(p_x, p_y, p_w, p_xy, p_xw, p_yw, p_xyw):
    """
    Compute multivariate mutual information I(X, Y, W) of three random
    processes given their (joint) probability distributions.

    Indices for joint probabilities are sorted as
    p(x_i, y_j) = p[i + Nx * j]
    and for p_xyw, np.unravel_index is used.

    :param p_x: np.array, unconditional probabilities of x
    :param p_y: np.array, unconditional probabilities of y
    :param p_w: np.array, unconditional probabilities of w
    :param p_xy: np.array, unconditional joint probabilities of x, y
    :param p_xw: np.array, unconditional joint probabilities of x, w
    :param p_yw: np.array, unconditional joint probabilities of y, w
    :param p_xyw: np.array, unconditional joint probabilities of x, y, w
    :return:
    """


    Nx = len(p_x)
    Ny = len(p_y)
    Nw = len(p_w)

    m = np.float32(0.)
    for n1, p1 in enumerate(p_x):
        for n2, p2 in enumerate(p_y):
            for n3, p3 in enumerate(p_w):
                i_xyw = np.ravel_multi_index(np.array([n1, n2, n3]), (Nx, Ny, Nw))
                i_xy = n1 + Nx * n2
                i_xw = n1 + Nx * n3
                i_yw = n2 + Ny * n3

                if p_xyw[i_xyw] > 0:
                    m += p_xyw[i_xyw] * np.log2(p_xy[i_xy] * p_xw[i_xw] * p_yw[i_yw] /
                                                (p1 * p2 * p3 * p_xyw[i_xyw]))
    return m

def reverse_estimate(forward_estimator, A, B):
    """
    Return backward estimator for the forward estimator from A -> B
    This function will return and fit the estimator from B -> A.
    :param forward_estimator:
    :param A:
    :param B:
    :return:
    """
    # TODO: this is ugly, however deepcopy does not reset class.
    p_estimator_args = [forward_estimator.p_estimator.__getattribute__(a) for a in
                        forward_estimator.p_estimator.__init__.__code__.co_varnames[1:]]
    reverse_p_estimator = forward_estimator.p_estimator.__class__(*p_estimator_args)

    if forward_estimator.p_estimator.is_stationary_estimate:
        if forward_estimator.p_estimator._user_tmat_x:
            reverse_p_estimator.set_transition_matrices(tmat_y=forward_estimator.p_estimator.tmat_x)
        if forward_estimator.p_estimator._user_tmat_y:
            reverse_p_estimator.set_transition_matrices(tmat_x=forward_estimator.p_estimator.tmat_y)
        if forward_estimator.p_estimator._user_tmat_xy:
            raise NotImplementedError('Transforming XY-transition matrix into YX-formulation not implemented.')
        if forward_estimator.p_estimator._dangerous_ignore_warnings_flag:
            reverse_p_estimator._dangerous_ignore_warnings_flag = True

        if forward_estimator.p_estimator.msmkwargs is not None:
            reverse_p_estimator.estimate(A, B, **forward_estimator.p_estimator.msmkwargs)

    reverse_estimator = forward_estimator.__class__(reverse_p_estimator)
    return reverse_estimator
