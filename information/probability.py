import numpy as np


# returns var Px_record
def ctwalgorithm(x, Nx, D):
    # Function CTWAlgorithm outputs the universal sequential probability
    # assignments given by CTW method.
    if len(x.shape) != 1:
        raise IOError('The input vector must be a colum vector!')

    n = len(x)
    if not np.floor((Nx ** (D + 1) - 1) / (Nx - 1)) == (Nx ** (D + 1) - 1) / (Nx - 1):
        print(np.floor((Nx ** (D + 1) - 1) / (Nx - 1)), (Nx ** (D + 1) - 1) / (Nx - 1))
        print(Nx, D)
        raise UserWarning('Something did not work')

    countTree = np.zeros((Nx, int((Nx ** (D + 1) - 1) / (Nx - 1))))
    betaTree = np.ones((1, int((Nx ** (D + 1) - 1) / (Nx - 1))))
    Px_record = np.zeros((Nx, n - D))
    indexweight = Nx ** np.arange(D)
    offset = (Nx ** D - 1) / (Nx - 1) + 1

    for i in range(D, n):
        context = x[i - D:i]
        leafindex = (np.dot(context, indexweight) + offset).astype(int)
        xt = x[i]
        eta = (countTree[:Nx - 1, leafindex - 1] + 0.5) / (countTree[Nx - 1, leafindex - 1] + 0.5)

        # update the leaf
        countTree[xt, leafindex - 1] += 1
        node = np.floor((leafindex + Nx - 2) / Nx).astype(int)

        # print(node)
        while np.all(node > 0):
            countTree, betaTree, eta = ctwupdate(countTree, betaTree, eta, node, xt, 1 / 2)
            node = np.floor((node + Nx - 2) / Nx).astype(int)
        eta_sum = eta.sum() + 1

        Px_record[:, i - D] = np.hstack([eta, [1]]) / eta_sum
    return Px_record


# returns [countTree, betaTree, eta]
def ctwupdate(countTree, betaTree, eta, index, xt, alpha):
    # countTree:  countTree(a+1,:) is the tree for the count of symbol a a=0,...,M
    # betaTree:   betaTree(i(s) ) =  Pe^s / \prod_{b=0}^{M} Pw^{bs}(x^{t})
    # eta = [ p(X_t = 0|.) / p(X_t = M|.), ..., p(X_t = M-1|.) / p(X_t = M|.)

    # calculate eta and update beta a, b
    # xt is the current data

    # size of the alphbet
    Nx = eta.shape[0] + 1

    pw = np.hstack([eta, [1]])
    pw /= pw.sum();  # % pw(1) pw(2) .. pw(M+1)

    pe = (countTree[:, index] + 0.5) / (countTree[:, index].sum() + Nx / 2)

    temp = betaTree[0, index]

    if temp < 1000:
        eta = (alpha * temp * pe[:Nx - 1] + (1 - alpha) * pw[:Nx - 1]) / (
        alpha * temp * pe[Nx - 1] + (1 - alpha) * pw[Nx - 1])
    else:
        eta = (alpha * pe[:Nx - 1] + (1 - alpha) * pw[:Nx - 1] / temp) / (
        alpha * pe[Nx - 1] + (1 - alpha) * pw[Nx - 1] / temp)
    countTree[xt, index] += 1
    betaTree[0, index] = betaTree[0, index] * pe[xt] / pw[xt]

    return countTree, betaTree, eta





