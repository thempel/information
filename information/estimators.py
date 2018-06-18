import itertools
import numpy as np
from bhmm import lag_observations


class Estimator(object):
    """ Base class for directed information estimators

    """
    def __init__(self, probability_estimator):
        """

        :param probability_estimator: information.ProbabilityEstimator class
        """
        self.p_estimator = probability_estimator
        self.d, self.r, self.m = None, None, None
        self.Nx, self.Ny = 0, 0

    def estimate(self, A, B):
        """
        Convenience function for directed, reverse directed estimation.
        :param A: time series A
        :param B: time series B
        :return: self
        """
        if not isinstance(A, list): A = [A]
        if not isinstance(B, list): B = [B]
        assert isinstance(A[0], np.ndarray)
        assert isinstance(B[0], np.ndarray)

        for a1, a2 in zip(A, B):
            if a1.shape[0] != a2.shape[0]:
                print(a1.shape, a2.shape)
                print('something wrong with traj lengths')
                return

        self.Nx = np.unique(np.concatenate(A)).max() + 1
        self.Ny = np.unique(np.concatenate(B)).max() + 1

        if self.p_estimator.is_stationary_estimate:
            self.d, self.r, self.m = self.stationary_estimate(A, B)
        else:
            self.d, self.r, self.m = self.nonstationary_estimate(A, B)

        return self

    def _stationary_estimator(self, a, b):
        raise NotImplementedError(
            'You need to overload the _stationary_estimator() method in your Estimator implementation!')

    def _nonstationary_estimator(self, a, b):
        raise NotImplementedError(
            'You need to overload the _nonstationary_estimator() method in your Estimator implementation!')

    def stationary_estimate(self, X, Y):
        """
        Directed information estimation on discrete trajectories with Markov model
        probability estimates.

        :param X: Time-series 1
        :param Y: Time-series 2
        :return: di, rdi, mi
        """
        msmlag = self.p_estimator.msmlag

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        assert self.Nx - 1 == tmat_x.shape[0] - 1 and np.unique(np.concatenate(X)).min() == 0
        assert self.Ny - 1 == tmat_y.shape[0] - 1 and np.unique(np.concatenate(Y)).min() == 0

        x_lagged = lag_observations(X, msmlag)
        y_lagged = lag_observations(Y, msmlag)

        di, rev_di, mi = self._stationary_estimator(x_lagged, y_lagged)

        return di, rev_di, mi

    def nonstationary_estimate(self, A, B):
        """
        Directed information estimation using non-stationary probability assignments.
        :param A: Time series 1
        :param B: Time series 2
        :return:
        """
        di, rev_di, mi =  self._nonstationary_estimator(A, B)
        return di, rev_di, mi


class JiaoI4(Estimator):
    r"""Estimator for Jiao et al I4 with CTW and MSM probabilities"""
    def __init__(self, probability_estimator):
        super(JiaoI4, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, X, Y):
        """
        Original estimator I4 from Jiao et al. Original docstring:

        Function `compute_DI_MI' calculates the directed information I(X^n-->
        Y^n), mutual information I(X^n; Y^n) and reverse directed information I(Y^{n-1}-->X^n)
        for any positive integer n smaller than the length of X and Y.

        X and Y: two input sequences;
        Nx:  the size of alphabet of X, assuming X and Y have the same size of
        alphabets;
        D:  the maximum depth of the context tree used in basic CTW algorithm,
        for references please see F. Willems, Y. Shtarkov and T. Tjalkens, 'The
        Context-Tree Weighting Method: Basic Properties', IEEE Transactions on
        Information Theory, 653-664, May 1995.
        alg:  indicates one of the four possible estimators proposed in J.
        Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
        Estimation of Directed Information', http://arxiv.org/abs/1201.2334.
        Users can indicate strings 'E1','E2','E3' and 'E4' for corresponding
        estimators.
        start_ratio: indicates how large initial proportion of input data should be ignored when displaying
        the estimated results, for example, if start_ratio = 0.2, then the output DI
        only contains the estimate of I(X^n \to Y^n) for n larger than
        length(X)/5.

        :param X: time series 1
        :param Y: time series 2
        :return:
        """
        dis, rdis, mis = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
        for n, (_X, _Y) in enumerate(zip(X, Y)):
            # re-label states if necessary
            if np.unique(_X).max() + 1 > len(set(_X)):
                mapper = np.zeros(np.unique(_X).max()+1) - 1
                mapper[np.unique(_X)] = list(range(np.unique(_X).shape[0]))
                _x = mapper[_X]
            else:
                _x = _X
            if np.unique(_Y).max() + 1 > len(set(_Y)):
                mapper = np.zeros(np.unique(_Y).max()+1) - 1
                mapper[np.unique(_Y)] = list(range(np.unique(_Y).shape[0]))
                _y = mapper[_Y]
            else:
                _y = _Y


            Nx_subset = (np.unique(_x).max() + 1).astype(int)
            Ny_subset = (np.unique(_y).max() + 1).astype(int)
            n_data = len(_x)
            if len(set(_x)) == 1 or len(set(_x)) == 1:
                #print('nothing to see here')
                dis[n], rdis[n], mis[n] = 0., 0., 0.
                continue


            # mapp the data pair (X,Y) into a single variable taking value with
            # alphabet size |X||Y|
            XY = _x + Nx_subset * _y

            # Calculate the CTW probability assignment
            pxy = self.p_estimator.pxy[n]
            px = self.p_estimator.px[n]
            py = self.p_estimator.py[n]

            # % px_xy is a Nx times n_data matrix, calculating p(x_i|x^{i-1},y^{i-1})
            px_xy = np.zeros((Nx_subset, n_data - self.p_estimator.D))
            for i_x in range(Nx_subset):
                px_xy[i_x, :] = pxy[i_x, :]
                for j in range(1, Ny_subset):
                    px_xy[i_x, :] = px_xy[i_x, :] + pxy[i_x + j * Nx_subset, :]

            # %calculate P(y|x,X^{i-1},Y^{i-1})
            #temp = np.tile(px_xy, (Nx, 1))
            #py_x_xy = pxy / temp

            temp_DI = np.zeros(_x.shape[0] - self.p_estimator.D)
            temp_MI = np.zeros(_x.shape[0] - self.p_estimator.D)
            temp_rev_DI = np.zeros(_x.shape[0] - self.p_estimator.D)
            for iy in range(Ny_subset):
                for ix in range(Nx_subset):
                    temp_DI = temp_DI + pxy[ix + iy * Nx_subset] * np.log2(pxy[ix + iy * Nx_subset] / (py[iy] * px_xy[ix]))
                    # temp_DI=temp_DI + pxy(ix+(iy-1)*Nx,:).     *log2(pxy(ix+(iy-1)*Nx,:). / (py(iy,:).*  px_xy(ix,:)));
                    temp_MI = temp_MI + pxy[ix + iy * Nx_subset] * np.log2(pxy[ix + iy * Nx_subset] / (py[iy] * px[ix]))
                    # temp_MI=temp_MI+  pxy(ix+(iy-1)*Nx,:).*     log2(pxy(ix+(iy-1)*Nx,:)./(py(iy,:).*px(ix,:)));
                    temp_rev_DI = temp_rev_DI + pxy[ix + iy * Nx_subset] * np.log2(px_xy[ix] / px[ix])
                    # temp_rev_DI=temp_rev_DI+ pxy(ix+(iy-1)*Nx,:).      *log2(px_xy(ix,:)./px(ix,:));
            dis[n], rdis[n], mis[n] = np.mean(temp_DI), np.mean(temp_rev_DI), np.mean(temp_MI)

        return dis.mean(), rdis.mean(), mis.mean()

    def _stationary_estimator(self, x_lagged, y_lagged):
        """
        Implementation of directed information estimator I4 from [1] using Markov model
        probability estimates.

        [1] Jiao et al, Universal Estimation of Directed Information, 2013.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: directed information, reverse directed information, mutual information
        """

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy
        msmlag = self.p_estimator.msmlag

        # iterate over time-lagged trajectory pairs
        d, r, m = 0., 0., 0.
        for ix_time_tau, iy_time_tau in zip(x_lagged, y_lagged):
            ixy_time_tau = ix_time_tau + self.Nx * iy_time_tau

            # compute probability trajectories from state x_{i-1} to any possible state x_i
            px = tmat_x[ix_time_tau, :]
            py = tmat_y[iy_time_tau, :]
            pxy = tmat_xy[ixy_time_tau, :]

            prob_xi_to_xip1_given_yi = np.zeros((self.Nx, self.Nx, self.Ny))
            for xi, xip1, yi in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny)]):
                # TODO: check if this is true. (sum over probabilities w/o normalization)
                prob_xi_to_xip1_given_yi[xi, xip1, yi] = np.sum([tmat_xy[xi + self.Nx * yi, xip1 + self.Nx * _y] for _y in range(self.Ny)])

            px_given_y = prob_xi_to_xip1_given_yi[ix_time_tau, :, iy_time_tau]

            temp_mi, temp_di, temp_rev_di = np.zeros(len(ix_time_tau)), np.zeros(
                len(ix_time_tau)), np.zeros(len(ix_time_tau))

            for iy in range(self.Ny):  # ix, iy now iterating over indicator states, not original state numbers
                for ix in range(self.Nx):
                    pidx = pxy[:, ix + iy * self.Nx] > 0  # def 0 * log(0) := 0
                    temp_mi[pidx] = temp_mi[pidx] + pxy[pidx, ix + iy * self.Nx] * np.log2(
                        pxy[pidx, ix + iy * self.Nx] / (py[pidx, iy] * px[pidx, ix]))
                    temp_di[pidx] = temp_di[pidx] + pxy[pidx, ix + iy * self.Nx] * np.log2(
                        pxy[pidx, ix + iy * self.Nx] / (py[pidx, iy] * px_given_y[pidx, ix]))
                    temp_rev_di[pidx] = temp_rev_di[pidx] + pxy[pidx, ix + iy * self.Nx] * np.log2(
                        px_given_y[pidx, ix] / px[pidx, ix])
            d += temp_di.mean() / msmlag
            r += temp_rev_di.mean() / msmlag
            m += temp_mi.mean() / msmlag

        return d, r, m


class JiaoI4Ensemble(Estimator):
    r"""Estimator for Jiao et al I4 for MSM probabilities in ensemble average formulation"""
    def __init__(self, probability_estimator):
        super(JiaoI4Ensemble, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, a, b):
        raise RuntimeError('Not meaningful to compute nonstationary estimates the ensemble way.')

    def _stationary_estimator(self, x_lagged, y_lagged):

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        statecounts = np.bincount(np.concatenate([_x + self.Nx * _y for _x, _y in zip(x_lagged, y_lagged)]))

        prob_xi_to_xip1_given_yi = np.zeros((self.Nx, self.Nx, self.Ny))
        for xi, xip1, yi in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny)]):
            prob_xi_to_xip1_given_yi[xi, xip1, yi] = np.sum([tmat_xy[xi + self.Nx * yi, xip1 + self.Nx * _y] for _y in range(self.Ny)])

        di, rdi, mi = 0., 0., 0.
        for xi, yi in itertools.product(range(self.Nx), range(self.Ny)):

            tmat_y_at_yi_bloated = np.repeat(tmat_y[yi], self.Nx)
            tmat_x_at_xi_bloated = np.tile(tmat_x[xi], self.Ny)
            prob_xi_xip1_given_yi_at_xi_yi_bloated = np.tile(prob_xi_to_xip1_given_yi[xi, :, yi], self.Ny)
            counts_xi_yi = statecounts[xi + self.Nx * yi]

            idx = np.logical_and(tmat_xy[xi + self.Nx * yi] > 0,
                                 tmat_y_at_yi_bloated * prob_xi_xip1_given_yi_at_xi_yi_bloated > 0)

            di += counts_xi_yi * np.sum(tmat_xy[xi + self.Nx * yi][idx] * np.log2(
                tmat_xy[xi + self.Nx * yi][idx] / (tmat_y_at_yi_bloated * prob_xi_xip1_given_yi_at_xi_yi_bloated)[idx]))

            rdi += counts_xi_yi * np.sum(tmat_xy[xi + self.Nx * yi][idx] * np.log2(
                prob_xi_xip1_given_yi_at_xi_yi_bloated[idx] / tmat_x_at_xi_bloated[idx]))
            mi += counts_xi_yi * np.sum(tmat_xy[xi + self.Nx * yi][idx] * np.log2(
                tmat_xy[xi + self.Nx * yi][idx] / (tmat_y_at_yi_bloated * tmat_x_at_xi_bloated)[idx]))
        di = di / statecounts.sum()
        mi = mi / statecounts.sum()
        rdi = rdi / statecounts.sum()

        return di, rdi, mi


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