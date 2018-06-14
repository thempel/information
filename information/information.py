import numpy as np
import pyemma
from bhmm import lag_observations
import itertools

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


# returns  [MI, DI, rev_DI]
def compute_DI_MI_E4(X, Y, D=-1, msmlag=-1):
    # Function `compute_DI_MI' calculates the directed information I(X^n-->
    # Y^n), mutual information I(X^n; Y^n) and reverse directed information I(Y^{n-1}-->X^n)
    # for any positive integer n smaller than the length of X and Y.

    # X and Y: two input sequences;
    # Nx:  the size of alphabet of X, assuming X and Y have the same size of
    # alphabets;
    # D:  the maximum depth of the context tree used in basic CTW algorithm,
    # for references please see F. Willems, Y. Shtarkov and T. Tjalkens, 'The
    # Context-Tree Weighting Method: Basic Properties', IEEE Transactions on
    # Information Theory, 653-664, May 1995.
    # alg:  indicates one of the four possible estimators proposed in J.
    # Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    # Estimation of Directed Information', http://arxiv.org/abs/1201.2334.
    # Users can indicate strings 'E1','E2','E3' and 'E4' for corresponding
    # estimators.
    # start_ratio: indicates how large initial proportion of input data should be ignored when displaying
    # the estimated results, for example, if start_ratio = 0.2, then the output DI
    # only contains the estimate of I(X^n \to Y^n) for n larger than
    # length(X)/5.
    if D == -1 and msmlag == -1:
        raise UserWarning('Choose between CTW or MSM probability estimation.')

    n_data = len(X)
    if len(set(X)) == 1 or  len(set(Y)) == 1:
        #print('nothing to see here')
        return np.zeros(X.shape[0] - D)
    Nx = max(X) + 1

    # mapp the data pair (X,Y) into a single variable taking value with
    # alphabet size |X||Y|
    XY = X + Nx * Y

    # Calculate the CTW probability assignment
    pxy = ctwalgorithm(XY, Nx ** 2, D)
    px = ctwalgorithm(X, Nx, D)
    py = ctwalgorithm(Y, Nx, D)

    # % px_xy is a Nx times n_data matrix, calculating p(x_i|x^{i-1},y^{i-1})
    px_xy = np.zeros((Nx, n_data - D))
    for i_x in range(Nx):
        px_xy[i_x, :] = pxy[i_x, :]
        for j in range(1, Nx):
            px_xy[i_x, :] = px_xy[i_x, :] + pxy[i_x + j * Nx, :]


            # %calculate P(y|x,X^{i-1},Y^{i-1})
    temp = np.tile(px_xy, (Nx, 1))
    py_x_xy = pxy / temp

    temp_DI = np.zeros(X.shape[0] - D)
    for iy in range(Nx):
        for ix in range(Nx):
            temp_DI = temp_DI + pxy[ix + iy * Nx, :] * np.log2(pxy[ix + iy * Nx, :] / (py[iy, :] * px_xy[ix, :]))

    return np.cumsum(temp_DI)


def compute_DI_MI_E4_imsm(X, Y, msmlag=1, reversible=True, tmat_x=None, tmat_y=None, tmat_xy=None,
                          tmat_ck_estimate=False):
    """
    Directed information computation on discrete trajectories with Markov model
    probability estimates. Convenience function that compares single state binary
    trajectories, i.e. returns directed information estimates from microstate i to j.

    Returns directed information, reverse directed information and mutual information
    as defined by Jiao et al, 2013.

    :param X: Time-series 1
    :param Y: Time-series 2
    :param msmlag: MSM lag time
    :param reversible: MSM estimator type
    :return: di, rdi, mi
    """

    if not isinstance(X, list): X = [X]
    if not isinstance(Y, list): Y = [Y]
    assert isinstance(X[0], np.ndarray)
    assert isinstance(Y[0], np.ndarray)

    Nx = np.concatenate(X).max() + 1
    Ny = np.concatenate(Y).max() + 1

    if tmat_ck_estimate: print('ATTENTION: Tmat estimated as T(1)^{}'.format(msmlag))

    for a1, a2 in zip(X, Y):
        if a1.shape[0] != a2.shape[0]:
            print(a1.shape, a2.shape)
            print('something wrong with traj lengths')
            return
    # DI is invariant under boolean inversion, so 2 state trajs don't require case differentiation
    if Nx == 2 and Ny == 2:
        assert set(np.concatenate(X)) == {0, 1}
        assert set(np.concatenate(Y)) == {0, 1}
        if tmat_x is None:
            if not tmat_ck_estimate:
                tmat_x = pyemma.msm.estimate_markov_model(X, msmlag, reversible=reversible).transition_matrix
            else:
                tmat_x = np.linalg.matrix_power(pyemma.msm.estimate_markov_model(X, 1, reversible=reversible).transition_matrix, msmlag)
        if tmat_y is None:
            if not tmat_ck_estimate:
                tmat_y = pyemma.msm.estimate_markov_model(Y, msmlag, reversible=reversible).transition_matrix
            else:
                tmat_y = np.linalg.matrix_power(pyemma.msm.estimate_markov_model(Y, 1, reversible=reversible).transition_matrix, msmlag)
        if tmat_xy is None:
            if not tmat_ck_estimate:
                tmat_xy = pyemma.msm.estimate_markov_model([_x + 2 * _y for _x, _y in zip(X, Y)],
                                                           msmlag, reversible=reversible).transition_matrix
            else:
                tmat_xy = np.linalg.matrix_power(pyemma.msm.estimate_markov_model([_x + 2 * _y for _x, _y in zip(X, Y)],
                                                           1, reversible=reversible).transition_matrix, msmlag)
        if not tmat_x.shape[0] * tmat_y.shape[0] == tmat_xy.shape[0]:
            print(tmat_x.shape, tmat_y.shape, tmat_xy.shape)
            raise NotImplementedError('Combined model is not showing all combinatorial states. Try non-reversible?')

        x_lagged = lag_observations(X, msmlag)
        y_lagged = lag_observations(Y, msmlag)
        di, rev_di, mi = _directed_information_estimator(x_lagged, y_lagged, tmat_x, tmat_y, tmat_xy, msmlag)
    else:
        if (tmat_x is not None) or (tmat_y is not None) or (tmat_xy is not None):
            print('Cannot initialize non-binary system with transition matrices. Ingnoring.')
        di = np.zeros((Nx, Ny))
        rev_di = np.zeros((Nx, Ny))
        mi = np.zeros((Nx, Ny))
        for n, (i_x, i_y) in enumerate(itertools.product(range(Nx), range(Ny))):
            if not tmat_ck_estimate:
                tmat_x = pyemma.msm.estimate_markov_model([(_x == i_x).astype(int) for _x in X],
                                                          msmlag, reversible=reversible).transition_matrix
                tmat_y = pyemma.msm.estimate_markov_model([(_y == i_y).astype(int) for _y in Y],
                                                          msmlag, reversible=reversible).transition_matrix
                tmat_xy = pyemma.msm.estimate_markov_model(
                    [(_x == i_x).astype(int) + 2 * (_y == i_y).astype(int) for _x, _y in zip(X, Y)],
                    msmlag, reversible=reversible).transition_matrix
            else:
                tmat_x = np.linalg.matrix_power(pyemma.msm.estimate_markov_model([(_x == i_x).astype(int) for _x in X],
                                                          1, reversible=reversible).transition_matrix, msmlag)
                tmat_y = np.linalg.matrix_power(pyemma.msm.estimate_markov_model([(_y == i_y).astype(int) for _y in Y],
                                                          1, reversible=reversible).transition_matrix, msmlag)
                tmat_xy = np.linalg.matrix_power(pyemma.msm.estimate_markov_model(
                    [(_x == i_x).astype(int) + 2 * (_y == i_y).astype(int) for _x, _y in zip(X, Y)],
                    1, reversible=reversible).transition_matrix, msmlag)
            if not tmat_x.shape[0] * tmat_y.shape[0] == tmat_xy.shape[0]:
                print(tmat_x.shape, tmat_y.shape, tmat_xy.shape)
                raise NotImplementedError('Combined model is not showing all combinatorial states. Try non-reversible?')

            x_lagged = [(_x == i_x).astype(int) for _x in lag_observations(X, msmlag)]
            y_lagged = [(_y == i_y).astype(int) for _y in lag_observations(Y, msmlag)]
            di[i_x, i_y], rev_di[i_x, i_y], mi[i_x, i_y] = _directed_information_estimator(x_lagged, y_lagged, tmat_x,
                                                                                           tmat_y, tmat_xy, msmlag)

    return di, rev_di, mi

def _directed_information_estimator(x_lagged, y_lagged, tmat_x, tmat_y, tmat_xy, msmlag):
    """
    Implementation of directed information estimator I4 from [1] using Markov model
    probability estimates.

    [1] Jiao et al, Universal Estimation of Directed Information, 2013.
    :param x_lagged: List of binary trajectories 1 with time step msmlag.
    :param y_lagged: List of binary trajectories 2 with time step msmlag.
    :param mmx: Markov model of binary trajectory 1
    :param mmy: Markov model of binary trajectory 2
    :param mmxy: Markov model of binary trajectory 1&2, x +2 * y state assignment
    :param msmlag: Markov model lag time
    :return: directed information, reverse directed information, mutual information
    """

    def p_xi_to_xip1_given_y1_y1p1(xi, xip1, yi):
        return tmat_xy[xi + 2 * yi, xip1 + 2 * 0] + tmat_xy[
            xi + 2 * yi, xip1 + 2 * 1]

    # iterate over time-lagged trajectory pairs
    d, r, m = 0., 0., 0.
    for indicator_state_i_x_time_tau, indicator_state_i_y_time_tau in zip(x_lagged, y_lagged):
        indicator_state_i_xy_time_tau = indicator_state_i_x_time_tau + 2 * indicator_state_i_y_time_tau

        # compute probability trajectories from state x_{i-1} to any possible state x_i
        px = tmat_x[indicator_state_i_x_time_tau, :]
        py = tmat_y[indicator_state_i_y_time_tau, :]
        pxy = tmat_xy[indicator_state_i_xy_time_tau, :]

        prob_xi_to_xip1_given_yi = np.zeros((2, 2, 2))
        for combination in itertools.product(*[range(2) for _ in range(3)]):
            prob_xi_to_xip1_given_yi[combination] = p_xi_to_xip1_given_y1_y1p1(*combination)

        px_given_y = prob_xi_to_xip1_given_yi[indicator_state_i_x_time_tau, :, indicator_state_i_y_time_tau]

        temp_mi, temp_di, temp_rev_di = np.zeros(len(indicator_state_i_x_time_tau)), np.zeros(
            len(indicator_state_i_x_time_tau)), np.zeros(len(indicator_state_i_x_time_tau))

        for iy in range(2):  # ix, iy now iterating over indicator states, not original state numbers
            for ix in range(2):
                pidx = pxy[:, ix + iy * 2] > 0  # def 0 * log(0) := 0
                temp_mi[pidx] = temp_mi[pidx] + pxy[pidx, ix + iy * 2] * np.log2(
                    pxy[pidx, ix + iy * 2] / (py[pidx, iy] * px[pidx, ix]))
                temp_di[pidx] = temp_di[pidx] + pxy[pidx, ix + iy * 2] * np.log2(
                    pxy[pidx, ix + iy * 2] / (py[pidx, iy] * px_given_y[pidx, ix]))
                temp_rev_di[pidx] = temp_rev_di[pidx] + pxy[pidx, ix + iy * 2] * np.log2(
                    px_given_y[pidx, ix] / px[pidx, ix])
        d += temp_di.mean() / msmlag
        r += temp_rev_di.mean() / msmlag
        m += temp_mi.mean() / msmlag

    return d, r, m

def dir_info(A, B, msmlag, reversible=True):
    """
    Convenience function for estimating the directed information between two
    discrete trajectories with causality conserving common time.

    This function ensures consistent results if expecting I(A->B)_rev = I(B->A),
    which is not the case for the time-lagged definition of reverse definition
    of Jiao et al.

    :param A: List of trajectories 1
    :param B: List of trajectories 2
    :param msmlag: Markov model lag time
    :param reversible: Markov estimator type
    :return: directed information, backward directed information, mutual information
    """
    di_forward, rdi_forward, mi = compute_DI_MI_E4_imsm(A, B, msmlag, reversible=reversible)
    di_backward, rdi_backward, _ = compute_DI_MI_E4_imsm(B, A, msmlag, reversible=reversible)

    return di_forward + rdi_backward, rdi_forward + di_backward, mi
