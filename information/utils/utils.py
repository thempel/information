import numpy as np

def relabel_dtrajs(X):
    if np.unique(X).max() + 1 > len(set(X)):
        mapper = np.zeros(np.unique(X).max() + 1) - 1
        mapper[np.unique(X)] = list(range(np.unique(X).shape[0]))
        _x = mapper[X]
    else:
        _x = X

    return _x