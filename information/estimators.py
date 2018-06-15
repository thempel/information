import itertools

import numpy as np
from bhmm import lag_observations

from information import ctwalgorithm


class Estimator(object):
    def __init__(self, probability_estimator):
        self.p_estimator = probability_estimator
        self.d, self.r, self.m = None, None, None

    def estimate(self, A, B):

        if self.p_estimator.is_stationary_estimate:
            self.d, self.r, self.m = self.stationary_estimate(A, B)
        else:
            self.d, self.r, self.m = self.nonstationary_estimate(A, B)

        return self

    def _stationary_estimator(self, a, b, c, d, e, f):
        raise NotImplementedError(
            'You need to overload the _stationary_estimator() method in your Estimator implementation!')

    def _nonstationary_estimator(self, a, b, D=None):
        raise NotImplementedError(
            'You need to overload the _nonstationary_estimator() method in your Estimator implementation!')

    def stationary_estimate(self, X, Y):
        """
        Directed information computation on discrete trajectories with Markov model
        probability estimates. Convenience function that returns directed information estimates
        from given estimator.

        :param X: Time-series 1
        :param Y: Time-series 2
        :param msmlag: MSM lag time
        :param reversible: MSM estimator type
        :return: di, rdi, mi
        """
        msmlag = self.p_estimator.msmlag

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        if not isinstance(X, list): X = [X]
        if not isinstance(Y, list): Y = [Y]
        assert isinstance(X[0], np.ndarray)
        assert isinstance(Y[0], np.ndarray)

        for a1, a2 in zip(X, Y):
            if a1.shape[0] != a2.shape[0]:
                print(a1.shape, a2.shape)
                print('something wrong with traj lengths')
                return

        assert np.unique(X).max() == tmat_x.shape[0] - 1 and np.unique(X).min() == 0
        assert np.unique(Y).max() == tmat_y.shape[0] - 1 and np.unique(Y).min() == 0

        x_lagged = lag_observations(X, msmlag)
        y_lagged = lag_observations(Y, msmlag)

        di, rev_di, mi = self._stationary_estimator(x_lagged, y_lagged, tmat_x, tmat_y, tmat_xy, msmlag)

        return di, rev_di, mi

    def nonstationary_estimate(self, A, B):
        return self._nonstationary_estimator(A, B, D=self.p_estimator.D)


class JiaoI4(Estimator):
    def __init__(self, probability_estimator):
        super(JiaoI4, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, X, Y, D=-1):
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

    def _stationary_estimator(self, x_lagged, y_lagged, tmat_x, tmat_y, tmat_xy, msmlag):
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
        Nx = np.unique(x_lagged).max() + 1
        Ny = np.unique(y_lagged).max() + 1


        # iterate over time-lagged trajectory pairs
        d, r, m = 0., 0., 0.
        for ix_time_tau, iy_time_tau in zip(x_lagged, y_lagged):
            ixy_time_tau = ix_time_tau + Nx * iy_time_tau

            # compute probability trajectories from state x_{i-1} to any possible state x_i
            px = tmat_x[ix_time_tau, :]
            py = tmat_y[iy_time_tau, :]
            pxy = tmat_xy[ixy_time_tau, :]

            prob_xi_to_xip1_given_yi = np.zeros((Nx, Nx, Ny))
            for xi, xip1, yi in itertools.product(*[range(Nx), range(Nx), range(Ny)]):
                prob_xi_to_xip1_given_yi[xi, xip1, yi] = np.sum([tmat_xy[xi + Nx * yi, xip1 + Nx * _y] for _y in range(Ny)])

            px_given_y = prob_xi_to_xip1_given_yi[ix_time_tau, :, iy_time_tau]

            temp_mi, temp_di, temp_rev_di = np.zeros(len(ix_time_tau)), np.zeros(
                len(ix_time_tau)), np.zeros(len(ix_time_tau))

            for iy in range(Ny):  # ix, iy now iterating over indicator states, not original state numbers
                for ix in range(Nx):
                    pidx = pxy[:, ix + iy * Nx] > 0  # def 0 * log(0) := 0
                    temp_mi[pidx] = temp_mi[pidx] + pxy[pidx, ix + iy * Nx] * np.log2(
                        pxy[pidx, ix + iy * Nx] / (py[pidx, iy] * px[pidx, ix]))
                    temp_di[pidx] = temp_di[pidx] + pxy[pidx, ix + iy * Nx] * np.log2(
                        pxy[pidx, ix + iy * Nx] / (py[pidx, iy] * px_given_y[pidx, ix]))
                    temp_rev_di[pidx] = temp_rev_di[pidx] + pxy[pidx, ix + iy * Nx] * np.log2(
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