import numpy as np

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