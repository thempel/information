import itertools
import numpy as np
from bhmm import lag_observations
from informant import utils


class Estimator(object):
    """ Base class for directed information estimators

    """
    def __init__(self, probability_estimator):
        """

        :param probability_estimator: informant.ProbabilityEstimator class
        """
        self.p_estimator = probability_estimator
        self.reverse_estimator = None

        self.d, self.r, self.m = None, None, None
        self.Nx, self.Ny = 0, 0

    def estimate(self, A, B, traj_eq_reweighting=False):
        """
        Convenience function for directed, reverse directed and mutual informant estimation.
        :param A: time series A
        :param B: time series B
        :traj_eq_reweighting : reweight trajectories according to stationary distribution
            only stationary estimates, experimental
        :return: self
        """
        A, B = utils.ensure_dtraj_format(A, B)

        self.Nx = np.unique(np.concatenate(A)).max() + 1
        self.Ny = np.unique(np.concatenate(B)).max() + 1

        if not self.p_estimator._estimated:
            self.p_estimator.estimate(A, B)

        if self.p_estimator.is_stationary_estimate:
            if not traj_eq_reweighting:
                self.d, self.r, self.m = self.stationary_estimate(A, B)
            else:
                from msmtools.analysis import stationary_distribution
                pi_xy = stationary_distribution(self.p_estimator.tmat_xy)
                A_re, B_re = utils.reweight_trajectories(A, B, pi_xy)
                self.d, self.r, self.m = self.stationary_estimate(A_re, B_re)
        else:
            self.d, self.r, self.m = self.nonstationary_estimate(A, B)

        return self

    def symmetrized_estimate(self, A, B, traj_eq_reweighting=False):
        """
        Ensures symmetric results for directed, reverse directed and mutual informant
        estimation, I(A->B)_rev = I(B->A). This is not the case for the original definition
        of reverse informant by Jiao et al. and the results are to be understood qualitatively
        only.
        :param A: time series A
        :param B: time series B
        :return: self
        """
        self.estimate(A, B, traj_eq_reweighting=traj_eq_reweighting)
        d_forward, r_forward, m_forward = self.d, self.r, self.m

        # for backward direction, initialize new probability / information estimators
        # TODO: this is extremely ugly code and probably not very robust.
        p_estimator_args = [self.p_estimator.__getattribute__(a) for a in self.p_estimator.__init__.__code__.co_varnames[1:]]
        reverse_p_estimator = self.p_estimator.__class__(*p_estimator_args)

        if self.p_estimator.is_stationary_estimate:
            if self.p_estimator._user_tmat_x:
                reverse_p_estimator.set_transition_matrices(tmat_y=self.p_estimator.tmat_x)
            if self.p_estimator._user_tmat_y:
                reverse_p_estimator.set_transition_matrices(tmat_x=self.p_estimator.tmat_y)
            if self.p_estimator._user_tmat_xy:
                raise NotImplementedError('Transforming XY-transition matrix into YX-formulation not implemented.')
            if self.p_estimator._dangerous_ignore_warnings_flag:
                reverse_p_estimator._dangerous_ignore_warnings_flag = True
            if self.p_estimator.msmkwargs is not None:
                reverse_p_estimator.estimate(B, A, **self.p_estimator.msmkwargs)

        self.reverse_estimator = self.__class__(reverse_p_estimator)
        self.reverse_estimator.estimate(B, A, traj_eq_reweighting=traj_eq_reweighting)
        d_backward, r_backward, m_backward = self.reverse_estimator.d, self.reverse_estimator.r, self.reverse_estimator.m

        self.d = (d_forward + r_backward)/2
        self.r = (r_forward + d_backward)/2
        self.m = (m_forward + m_backward)/2

        return self

    def _stationary_estimator(self, a, b):
        raise NotImplementedError(
            'You need to overload the _stationary_estimator() method in your Estimator implementation!')

    def _nonstationary_estimator(self, a, b):
        raise NotImplementedError(
            'You need to overload the _nonstationary_estimator() method in your Estimator implementation!')

    def stationary_estimate(self, X, Y):
        """
        Directed informant estimation on discrete trajectories with Markov model
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
        Directed informant estimation using non-stationary probability assignments.
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

        Function `compute_DI_MI' calculates the directed informant I(X^n-->
        Y^n), mutual informant I(X^n; Y^n) and reverse directed informant I(Y^{n-1}-->X^n)
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
            # map discrete time series to set {0, 1, ..., n_states_here}
            _x = utils.relabel_dtrajs(_X)
            _y = utils.relabel_dtrajs(_Y)


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
        Implementation of directed informant estimator I4 from [1] using Markov model
        probability estimates.

        [1] Jiao et al, Universal Estimation of Directed Information, 2013.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: directed informant, reverse directed informant, mutual informant
        """

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        pi_xy = self.p_estimator.pi_xy

        pi_dep = np.zeros((self.Nx  * self.Ny))
        pi_dep[self.p_estimator.active_set_xy] = pi_xy

        full2active = -1 * np.ones(self.Nx * self.Ny, dtype=int)
        full2active[self.p_estimator.active_set_xy] = np.arange(len(self.p_estimator.active_set_xy))

        prob_xi_to_xip1_given_yi = np.zeros((self.Nx, self.Nx, self.Ny))
        for xi, xip1, yi in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny)]):
            if xi + self.Nx * yi in self.p_estimator.active_set_xy:
                for _y in range(self.Ny):
                    if xip1 + self.Nx * _y in self.p_estimator.active_set_xy:
                        prob_xi_to_xip1_given_yi[xi, xip1, yi] += tmat_xy[full2active[xi + self.Nx * yi],
                                                                          full2active[xip1 + self.Nx * _y]]


        di, rdi, mi = 0., 0., 0.
        for xi, yi in itertools.product(range(self.Nx), range(self.Ny)):
            if xi + self.Nx * yi in self.p_estimator.active_set_xy:
                tmat_y_at_yi_bloated = np.repeat(tmat_y[yi], self.Nx)
                tmat_x_at_xi_bloated = np.tile(tmat_x[xi], self.Ny)
                prob_xi_xip1_given_yi_at_xi_yi_bloated = np.tile(prob_xi_to_xip1_given_yi[xi, :, yi], self.Ny)
                tmat_xy_fullset = np.zeros((self.Nx * self.Ny))
                tmat_xy_fullset[self.p_estimator.active_set_xy] = tmat_xy[full2active[xi + self.Nx * yi]]

                idx = np.logical_and(tmat_xy_fullset > 0,
                                     tmat_y_at_yi_bloated * prob_xi_xip1_given_yi_at_xi_yi_bloated > 0)

                di += pi_dep[xi + self.Nx * yi] * np.sum(tmat_xy_fullset[idx] * np.log2(
                    tmat_xy_fullset[idx] / (tmat_y_at_yi_bloated * prob_xi_xip1_given_yi_at_xi_yi_bloated)[idx]))

                rdi += pi_dep[xi + self.Nx * yi] * np.sum(tmat_xy_fullset[idx] * np.log2(
                    prob_xi_xip1_given_yi_at_xi_yi_bloated[idx] / tmat_x_at_xi_bloated[idx]))
                mi += pi_dep[xi + self.Nx * yi] * np.sum(tmat_xy_fullset[idx] * np.log2(
                    tmat_xy_fullset[idx] / (tmat_y_at_yi_bloated * tmat_x_at_xi_bloated)[idx]))

        return di, rdi, mi


class JiaoI4Ensemble(Estimator):
    r"""Estimator for Jiao et al I4 for MSM probabilities in ensemble average formulation"""
    def __init__(self, probability_estimator):
        super(JiaoI4Ensemble, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, a, b):
        raise RuntimeError('Not meaningful to compute nonstationary estimates the ensemble way.')

    def _stationary_estimator(self, x_lagged, y_lagged):
        raise DeprecationWarning('Use JiaoI4 instead!')


class JiaoI3(Estimator):
    r"""Estimator for Jiao et al I3 with CTW and MSM probabilities"""
    def __init__(self, probability_estimator):
        """
        Implementation I3 estimator by Jiao et al.
        CAUTION: ONLY IMPLEMENTED AS REFERENCE, should be thoroughly tested.
        """
        super(JiaoI3, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, X, Y):
        """
        Original estimator I3 by Jiao et al.
        CAUTION: ONLY IMPLEMENTED AS REFERENCE, should be thoroughly tested.
        Original docstring:

        Function `compute_DI_MI' calculates the directed informant I(X^n-->
        Y^n), mutual informant I(X^n; Y^n) and reverse directed informant I(Y^{n-1}-->X^n)
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
            # map discrete time series to set {0, 1, ..., n_states_here}
            _x = utils.relabel_dtrajs(_X)
            _y = utils.relabel_dtrajs(_Y)


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
            temp = np.tile(px_xy, (self.Nx, 1))
            py_x_xy = pxy / temp

            temp_DI = np.zeros(_x.shape[0] - self.p_estimator.D)
            temp_MI = np.zeros(_x.shape[0] - self.p_estimator.D)
            temp_rev_DI = np.zeros(_x.shape[0] - self.p_estimator.D)
            t = np.arange(py_x_xy.shape[1], dtype=int)
            for iy in range(Ny_subset):
                    temp_DI = temp_DI + py_x_xy[_x[self.p_estimator.D:] + Nx_subset * iy, t] * \
                                        np.log2(py_x_xy[_x[self.p_estimator.D:] + Nx_subset * iy, t] /\
                                                py[iy])
                    temp_MI = temp_MI + py_x_xy[_x[self.p_estimator.D:] + Nx_subset * iy, t] * \
                                        np.log2(pxy[_x[self.p_estimator.D:] + Nx_subset * iy, t] / \
                                                (py[iy] * px[_x[self.p_estimator.D:], t]))
                    temp_rev_DI = temp_rev_DI + px_xy[iy] * np.log2(px_xy[iy] / px[iy])

            dis[n], rdis[n], mis[n] = np.mean(temp_DI), np.mean(temp_rev_DI), np.mean(temp_MI)

        return dis.mean(), rdis.mean(), mis.mean()

    def _stationary_estimator(self, x_lagged, y_lagged):
        """
        Implementation of directed informant estimator I4 from [1] using Markov model
        probability estimates.

        [1] Jiao et al, Universal Estimation of Directed Information, 2013.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: directed informant, reverse directed informant, mutual informant
        """

        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        pi_xy = self.p_estimator.pi_xy

        pi_dep = np.zeros((self.Nx  * self.Ny))
        pi_dep[self.p_estimator.active_set_xy] = pi_xy

        full2active = -1 * np.ones(self.Nx * self.Ny, dtype=int)
        full2active[self.p_estimator.active_set_xy] = np.arange(len(self.p_estimator.active_set_xy))


        di, rdi, mi = 0., 0., 0.
        for xi, xim1, yi, yim1 in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny), range(self.Ny)]):
            p_xi_yi_given_xim1_yim1 = tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * yi]]

            # skip if transition has not been observed in the data
            if p_xi_yi_given_xim1_yim1 == 0:
                continue

            p_xi_given_xim1_yim1 = np.sum([tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * _y]] for _y in range(self.Ny)])
            p_yi_given_xi_xim1_yim1 = p_xi_yi_given_xim1_yim1 / p_xi_given_xim1_yim1

            di += pi_dep[xim1 + self.Nx * yim1] * p_xi_yi_given_xim1_yim1 * \
                  np.log2(p_yi_given_xi_xim1_yim1 / tmat_y[yim1, yi])
            rdi += pi_dep[xim1 + self.Nx * yim1] * p_xi_given_xim1_yim1 * p_xi_given_xim1_yim1 * \
                   np.log2(p_xi_given_xim1_yim1/tmat_x[xim1, xi])
            mi += pi_dep[xim1 + self.Nx * yim1] * p_xi_yi_given_xim1_yim1 * \
                 np.log2(p_xi_yi_given_xim1_yim1/(tmat_y[yim1, yi] * tmat_x[xim1, xi]))

        return di, rdi, mi


class JiaoI3Ensemble(Estimator):
    r"""Estimator for Jiao et al I3 for MSM probabilities in ensemble average formulation"""
    def __init__(self, probability_estimator):
        super(JiaoI3Ensemble, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, a, b):
        raise RuntimeError('Not meaningful to compute nonstationary estimates the ensemble way.')

    def _stationary_estimator(self, x_lagged, y_lagged):
        raise DeprecationWarning('Use JiaoI3 instead!')


class TransferEntropy(Estimator):
    r"""Estimator for Schreiber, PRL, 2000"""
    def __init__(self, probability_estimator):
        """
        Implementation transfer entropy estimator by Schreiber, PRL, 2000

        """
        super(TransferEntropy, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, a, b):
        raise NotImplementedError("Transfer entropy only implemented with MSM probabilities.")

    def _stationary_estimator(self, x_lagged, y_lagged):
        tmat_x = self.p_estimator.tmat_x
        tmat_y = self.p_estimator.tmat_y
        tmat_xy = self.p_estimator.tmat_xy

        pi_xy = self.p_estimator.pi_xy

        pi_dep = np.zeros((self.Nx  * self.Ny))
        pi_dep[self.p_estimator.active_set_xy] = pi_xy

        full2active = -1 * np.ones(self.Nx * self.Ny, dtype=int)
        full2active[self.p_estimator.active_set_xy] = np.arange(len(self.p_estimator.active_set_xy))

        if not tmat_x.shape[0] * tmat_y.shape[0] == tmat_xy.shape[0]:
            print(tmat_x.shape[0], tmat_y.shape[0], tmat_xy.shape[0])
            # return 0

        d = 0.
        for j_n in range(self.Ny):
            for i_n in range(self.Nx):
                for j_np1 in range(self.Ny):
                    if i_n + self.Nx * j_n in self.p_estimator.active_set_xy:
                        p_jnp1_given_in_jn = np.array([tmat_xy[full2active[i_n + self.Nx*j_n],
                                                               full2active[i_np1 + self.Nx * j_np1]]
                                                       for i_np1 in range(self.Nx)
                                                       if i_np1 + self.Nx * j_np1 in self.p_estimator.active_set_xy]).sum()
                        if p_jnp1_given_in_jn > 1e-16:
                            d += pi_dep[i_n + self.Nx  * j_n] * p_jnp1_given_in_jn * np.log2(p_jnp1_given_in_jn /
                                                                                             tmat_y[j_n, j_np1])
        return d, 0., 0.


class MutualInfoStationaryDistribution(Estimator):
    r"""Estimator for Schreiber, PRL, 2000"""
    def __init__(self, probability_estimator):
        """
        Implementation mutual information (as described by Schreiber, PRL, 2000)

        """
        super(MutualInfoStationaryDistribution, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, a, b):
        raise NotImplementedError("Doesn't make sense...")

    def _stationary_estimator(self, x_lagged, y_lagged):


        pi_dep = np.zeros((self.Nx * self.Ny))
        pi_dep[self.p_estimator.active_set_xy] = self.p_estimator.pi_xy

        m = 0.
        for n1, p1 in enumerate(self.p_estimator.pi_x):
            for n2, p2 in enumerate(self.p_estimator.pi_y):
                if pi_dep[n1 + self.Nx*n2] > 0:
                    m += pi_dep[n1 + self.Nx*n2] * np.log2(pi_dep[n1 + self.Nx*n2] / (p1 * p2))

        return 0., 0., m



class MultiEstimator(object):
    """ Base class for directed information estimators with multiple processes

    """
    def __init__(self, probability_estimator):
        """

        :param probability_estimator: informant.ProbabilityEstimator class
        """
        from informant import NetMSMProbabilities
        assert isinstance(probability_estimator, NetMSMProbabilities)
        self.p_estimator = probability_estimator

        self.causally_conditioned_di = None
        self.Nx, self.Ny, self.Nw = 0, 0, 0

    def estimate(self, A, B, W_, traj_eq_reweighting=False):
        """
        Convenience function for causally conditioned directed information
        :param A: np.array or list of np.arrays of dtype int. time series A
        :param B: np.array or list of np.arrays of dtype int. time series B
        :param W: np.array or list of np.arrays of dtype int. time series W, conditioned upon which DI is estimated
        Can be either supplied in same format as A, B or a list of multiple conditional time series in this format.
        :traj_eq_reweighting : reweight trajectories according to stationary distribution
            only stationary estimates, experimental (not implemented)
        :return: self
        """

        if isinstance(W_, np.ndarray):
            A, B, W_ = utils.ensure_dtraj_format(A, B, W_)
            W_ = [W_]
        elif (isinstance(W_, list) and isinstance(W_[0], np.ndarray) and isinstance(A, np.ndarray)) or (
                isinstance(W_, list) and isinstance(W_[0], list) and isinstance(W_[0][0], np.ndarray)
        ):
            formatted_W = []
            for _w in W_:
                A, B, _w = utils.ensure_dtraj_format(A, B, _w)
                formatted_W.append(_w)
            W_ = formatted_W
        else:
            raise RuntimeError('W must be a list of arrays (single condition) or a list of lists of arrays '
                               '(multiple conditions)')

        self.Nx = np.unique(np.concatenate(A)).max() + 1
        self.Ny = np.unique(np.concatenate(B)).max() + 1

        self.causally_conditioned_di = np.zeros(len(W_))
        for n_w, W in enumerate(W_):
            self.Nw = np.unique(np.concatenate(W)).max() + 1

            # new estimate for each W.
            self.p_estimator.estimate(W, A, B)

            if self.p_estimator.is_stationary_estimate:
                if not traj_eq_reweighting:
                    _causally_conditioned_di = self.stationary_estimate(W, A, B)
                else:
                    raise NotImplementedError('Equilibrium traj reweighting not yet implemented.')
            else:
                _causally_conditioned_di = self.nonstationary_estimate(A, B)

            self.causally_conditioned_di[n_w] = _causally_conditioned_di

        return self

    def _stationary_estimator(self, w, a, b):
        raise NotImplementedError(
            'You need to overload the _stationary_estimator() method in your Estimator implementation!')

    def _nonstationary_estimator(self, w, a, b):
        raise NotImplementedError(
            'You need to overload the _nonstationary_estimator() method in your Estimator implementation!')

    def stationary_estimate(self, W, X, Y):
        """
        Directed informant estimation on discrete trajectories with Markov model
        probability estimates.

        :param X: Time-series 1
        :param Y: Time-series 2
        :return: di, rdi, mi
        """
        msmlag = self.p_estimator.msmlag


        tmat_wy = self.p_estimator.tmat_wy
        tmat_wyx = self.p_estimator.tmat_wyx

        assert np.unique(np.concatenate(X)).min() == 0
        assert np.unique(np.concatenate(Y)).min() == 0
        assert np.unique(np.concatenate(W)).min() == 0

        if not self.p_estimator._dangerous_ignore_warnings_flag:
            assert self.Ny * self.Nw == tmat_wy.shape[0]
            assert self.Ny * self.Nw * self.Nx == tmat_wyx.shape[0]

        x_lagged = lag_observations(X, msmlag)
        y_lagged = lag_observations(Y, msmlag)
        w_lagged = lag_observations(W, msmlag)

        causally_cond_di = self._stationary_estimator(w_lagged, x_lagged, y_lagged)

        return causally_cond_di

    def nonstationary_estimate(self, A, B):
        raise NotImplementedError('Not implemented.')


class CausallyConditionedDI(MultiEstimator):
    r"""
    Estimator for causally condited directed information as described by Quinn et al 2011
    CAUTION: NEEDS FURTHER TESTING
    """
    def __init__(self, probability_estimator):
        super(CausallyConditionedDI, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, W, X, Y):
        raise NotImplementedError

    def _stationary_estimator(self, w_lagged, x_lagged, y_lagged):
        """
        Implementation of causally conditioned directed information from [1] using Markov model
        probability estimates.

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.
        :param w_lagged: List of binary trajectories conditioned upon which DI is conditioned. time step msmlag.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: causally conditioned directed information
        """

        tmat_wy = self.p_estimator.tmat_wy
        tmat_wyx = self.p_estimator.tmat_wyx

        pi_wy = self.p_estimator.pi_wy
        pi_wyx = self.p_estimator.pi_wyx


        full2active_wyx = -1 * np.ones(self.Nw * self.Nx * self.Ny, dtype=int)
        full2active_wyx[self.p_estimator.active_set_wyx] = np.arange(len(self.p_estimator.active_set_wyx))

        full2active_wy = -1 * np.ones(self.Nw * self.Ny, dtype=int)
        full2active_wy[self.p_estimator.active_set_wy] = np.arange(len(self.p_estimator.active_set_wy))

        # compute W, Y dependent properties
        H_Y_cond_W = 0.
        for yi, yim1, wi, wim1 in itertools.product(*[range(self.Ny), range(self.Ny),
                                                      range(self.Nw), range(self.Nw)]):

            p_wi_yi_given_wim1_yim1 = tmat_wy[full2active_wy[wim1 + self.Nw * yim1], full2active_wy[wi + self.Nw * yi]]

            # skip if transition has not been observed in the data
            if p_wi_yi_given_wim1_yim1 == 0:
                continue

            p_wi_given_wim1_yim1 = np.sum([tmat_wy[full2active_wy[wim1 + self.Nw * yim1], full2active_wy[wi + self.Nw * _y]] for _y in range(self.Ny)])
            p_yi_given_wi_wim1_yim1 = p_wi_yi_given_wim1_yim1 / p_wi_given_wim1_yim1

            # H-rate (does that make sense? it is pi-weighted twice.)
            #H_Y_cond_W -= pi_wy[full2active_wy[wim1 + self.Nw * yim1]]**2 * p_wi_yi_given_wim1_yim1 * p_wi_given_wim1_yim1 * np.log2(p_yi_given_wi_wim1_yim1)
            H_Y_cond_W -= pi_wy[full2active_wy[wim1 + self.Nw * yim1]] * p_wi_yi_given_wim1_yim1 * np.log2(
                p_yi_given_wi_wim1_yim1)

        # compute W, X, Y dependent properties
        H_Y_cond_XW = 0.
        for xi, xim1, yi, yim1, wi, wim1 in itertools.product(*[range(self.Nx), range(self.Nx),
                                                                range(self.Ny), range(self.Ny),
                                                                range(self.Nw), range(self.Nw)]):
            if wi + self.Nw * yi + (self.Nw * self.Ny) * xi in self.p_estimator.active_set_wyx and \
                    wim1 + self.Nw * yim1 + (self.Nw * self.Ny) * xim1 in self.p_estimator.active_set_wyx:

                p_wi_yi_xi_given_wim1_yim1_xim1 = tmat_wyx[
                    full2active_wyx[wim1 + self.Nw * yim1 + (self.Nw * self.Ny) * xim1],
                    full2active_wyx[wi + self.Nw * yi + (self.Nw * self.Ny) * xi]]

                # skip if transition has not been observed in the data
                if p_wi_yi_xi_given_wim1_yim1_xim1 == 0:
                    continue


                p_yi_given_wim1_yim1_xim1 = np.sum(
                    [tmat_wyx[full2active_wyx[wim1 + self.Nw * yim1 + (self.Nw * self.Ny) * xim1],
                              full2active_wyx[wi + self.Nw * _y + (self.Nw * self.Ny) * xi]]
                     for _y in range(self.Ny)
                     if wi + self.Nw * _y + (self.Nw * self.Ny) * xi in self.p_estimator.active_set_wyx])

                p_yi_given_wi_wim1_yim1_xi_xim1 = p_wi_yi_xi_given_wim1_yim1_xim1 / p_yi_given_wim1_yim1_xim1

                # below: rate
                #H_Y_cond_XW -= pi_wyx[full2active_wyx[wim1 + self.Nw * yim1 + (
                #            self.Nw * self.Ny) * xim1]] ** 2 * p_wi_yi_xi_given_wim1_yim1_xim1 * p_wi_xi_given_wim1_yim1_xim1 * np.log2(
                #    p_yi_given_wi_wim1_yim1_xi_xim1)

                H_Y_cond_XW -= pi_wyx[full2active_wyx[wim1 + self.Nw * yim1 + (
                        self.Nw * self.Ny) * xim1]] * p_wi_yi_xi_given_wim1_yim1_xim1 * np.log2(
                    p_yi_given_wi_wim1_yim1_xi_xim1)


        return H_Y_cond_W - H_Y_cond_XW
