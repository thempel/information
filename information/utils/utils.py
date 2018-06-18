import numpy as np

def relabel_dtrajs(X):
    if np.unique(X).max() + 1 > len(set(X)):
        mapper = np.zeros(np.unique(X).max() + 1) - 1
        mapper[np.unique(X)] = list(range(np.unique(X).shape[0]))
        _x = mapper[X]
    else:
        _x = X

    return _x

def ensure_dtraj_format(A, B):
    if not isinstance(A, list): A = [A]
    if not isinstance(B, list): B = [B]
    assert isinstance(A[0], np.ndarray)
    assert isinstance(B[0], np.ndarray)

    for n, (a1, a2) in enumerate(zip(A, B)):
        if a1.shape[0] != a2.shape[0]:
            raise RuntimeError('Trajectories not compatible. Lengths of {}th trajs are {} and {}, '
                               'respectively.'.format(n, a1.shape[0], a2.shape[0]))

    return A, B