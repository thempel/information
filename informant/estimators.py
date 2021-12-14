import itertools
import numpy as np
from informant import utils
from copy import deepcopy

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

        A_rev, B_rev = [b[:-1] for b in B], [a[1:] for a in A]
        reverse_estimator = utils.reverse_estimate(self, A_rev, B_rev)

        self.Nx = np.unique(np.concatenate(A)).max() + 1
        self.Ny = np.unique(np.concatenate(B)).max() + 1

        reverse_estimator.Nx = self.Ny
        reverse_estimator.Ny = self.Nx

        if not self.p_estimator._estimated:
            self.p_estimator.estimate(A, B)

        if not reverse_estimator.p_estimator._estimated:
            reverse_estimator.p_estimator.estimate(A_rev, B_rev)

        if self.p_estimator.is_stationary_estimate:
            if traj_eq_reweighting:
                from msmtools.analysis import stationary_distribution
                pi_xy = stationary_distribution(self.p_estimator.tmat_xy)
                A, B = utils.reweight_trajectories(A, B, pi_xy)

            d = self.stationary_estimate(A, B)
            r = reverse_estimator.stationary_estimate(A_rev, B_rev)
        else:
            d = self.nonstationary_estimate(A, B)
            r = reverse_estimator.nonstationary_estimate(A_rev, B_rev)
        self.d, self.r, self.m = d, r, d + r

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

        self.reverse_estimator = utils.reverse_estimate(self, B, A)
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

        x_lagged = utils.lag_observations(X, msmlag)
        y_lagged = utils.lag_observations(Y, msmlag)

        di = self._stationary_estimator(x_lagged, y_lagged)

        return di

    def nonstationary_estimate(self, A, B):
        """
        Directed informant estimation using non-stationary probability assignments.
        :param A: Time series 1
        :param B: Time series 2
        :return:
        """
        di =  self._nonstationary_estimator(A, B)
        return di


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
        dis = np.zeros(len(X))
        for n, (_X, _Y) in enumerate(zip(X, Y)):
            # map discrete time series to set {0, 1, ..., n_states_here}
            _x = utils.relabel_dtrajs(_X)
            _y = utils.relabel_dtrajs(_Y)


            Nx_subset = (np.unique(_x).max() + 1).astype(int)
            Ny_subset = (np.unique(_y).max() + 1).astype(int)
            n_data = len(_x)
            if len(set(_x)) == 1 or len(set(_x)) == 1:
                #print('nothing to see here')
                dis[n] = 0.
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
            dis[n] = np.mean(temp_DI)

        return dis.mean()

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


        di = 0.
        for xi, xim1, yi, yim1 in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny), range(self.Ny)]):
            p_xi_yi_given_xim1_yim1 = tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * yi]]

            # skip if transition has not been observed in the data
            if p_xi_yi_given_xim1_yim1 == 0:
                continue

            p_xi_given_xim1_yim1 = np.sum([tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * _y]] for _y in range(self.Ny)])

            di += pi_dep[xim1 + self.Nx * yim1] * p_xi_yi_given_xim1_yim1 * \
                  np.log2(p_xi_yi_given_xim1_yim1 / (tmat_y[yim1, yi] * p_xi_given_xim1_yim1))
        return di


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
            t = np.arange(py_x_xy.shape[1], dtype=int)
            for iy in range(Ny_subset):
                    temp_DI = temp_DI + py_x_xy[_x[self.p_estimator.D:] + Nx_subset * iy, t] * \
                                        np.log2(py_x_xy[_x[self.p_estimator.D:] + Nx_subset * iy, t] /\
                                                py[iy])

            dis[n] = np.mean(temp_DI)

        return dis.mean()

    def _stationary_estimator(self, x_lagged, y_lagged):
        """
        Implementation of directed informant estimator I3 from [1] using Markov model
        probability estimates.
        NOTE: This equals the direct estimation of DI.

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

        return di


class DirectedInformation(Estimator):
    r"""Estimator for definition of DI with MSM probabilities"""
    def __init__(self, probability_estimator):
        """
        Implementation of original definition of directed information
        """
        super(DirectedInformation, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, X, Y):
        raise NotImplementedError('Only DI estimators of Jiao et al implemented with non-'
                                  'stationary probabilities.')

    def _stationary_estimator(self, x_lagged, y_lagged):
        """
        Implementation of directed information formula from [1] using Markov model
        probability estimates.

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.
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

        di = 0.
        for xi, xim1, yi, yim1 in itertools.product(*[range(self.Nx), range(self.Nx), range(self.Ny), range(self.Ny)]):
            p_xi_yi_given_xim1_yim1 = tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * yi]]

            # skip if transition has not been observed in the data
            if p_xi_yi_given_xim1_yim1 == 0:
                continue

            p_xi_given_xim1_yim1 = np.sum([tmat_xy[full2active[xim1 + self.Nx * yim1], full2active[xi + self.Nx * _y]] for _y in range(self.Ny)])
            p_yi_given_xi_xim1_yim1 = p_xi_yi_given_xim1_yim1 / p_xi_given_xim1_yim1

            # di += pi_dep[xim1 + self.Nx * yim1] * p_xi_yi_given_xim1_yim1 * p_yi_given_xi_xim1_yim1 * \
            #       np.log2(p_yi_given_xi_xim1_yim1 / tmat_y[yim1, yi])
            # use Kullback entropy instead of KL divergence (as done by Schreiber)
            di += pi_dep[xim1 + self.Nx * yim1] * p_xi_yi_given_xim1_yim1 * \
                  np.log2(p_yi_given_xi_xim1_yim1 / tmat_y[yim1, yi])

        return di

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
        return d


class MutualInfoStationaryDistribution():
    r"""Estimator for Schreiber, PRL, 2000"""
    def __init__(self, probability_estimator):
        """
        Implementation mutual information (as described by Schreiber, PRL, 2000)

        """
        self.p_estimator = probability_estimator

        self.m = None
        self.Nx, self.Ny = 0, 0

    def estimate(self, A, B):
        """
        Convenience function for mutual informant estimation.
        :param A: time series A
        :param B: time series B
        :return: self
        """
        X, Y = utils.ensure_dtraj_format(A, B)

        self.Nx = np.unique(np.concatenate(X)).max() + 1
        self.Ny = np.unique(np.concatenate(Y)).max() + 1

        if not self.p_estimator._estimated:
            self.p_estimator.estimate(X, Y)

        assert self.Nx - 1 == self.p_estimator.tmat_x.shape[0] - 1 and np.unique(np.concatenate(X)).min() == 0
        assert self.Ny - 1 == self.p_estimator.tmat_y.shape[0] - 1 and np.unique(np.concatenate(Y)).min() == 0

        pi_dep = np.zeros((self.Nx * self.Ny))
        pi_dep[self.p_estimator.active_set_xy] = self.p_estimator.pi_xy

        m = 0.
        for n1, p1 in enumerate(self.p_estimator.pi_x):
            for n2, p2 in enumerate(self.p_estimator.pi_y):
                if pi_dep[n1 + self.Nx*n2] > 0:
                    m += pi_dep[n1 + self.Nx*n2] * np.log2(pi_dep[n1 + self.Nx*n2] / (p1 * p2))

        self.m = m
        return self


class MultiEstimator(object):
    """ Base class for directed information estimators with multiple processes

    """
    def __init__(self, probability_estimator):
        """

        :param probability_estimator: informant.ProbabilityEstimator class
        """
        self.p_estimator = probability_estimator

        self.causally_conditioned_di = None
        self.Nx, self.Ny, self.Nw = 0, 0, 0

    def estimate(self, A, B, W_, traj_eq_reweighting=False, n_jobs=1):
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


        if self.p_estimator.is_stationary_estimate:
            if not traj_eq_reweighting:
                if n_jobs == 1:
                    for n_w, W in enumerate(W_):
                        self.Nw = np.unique(np.concatenate(W)).max() + 1
                        self.causally_conditioned_di[n_w] = self.stationary_estimate(W, A, B)
                else:
                    from pathos.multiprocessing import Pool
                    from contextlib import closing

                    pool = Pool(processes=n_jobs)
                    args = [(W, A, B) for W in W_]

                    with closing(pool):
                        res_async = [pool.apply_async(self.stationary_estimate, a) for a in args]
                        self.causally_conditioned_di[:] = [x.get() for x in res_async]
            else:
                raise NotImplementedError('Equilibrium traj reweighting not yet implemented.')
        else:
            for n_w, W in enumerate(W_):
                self.causally_conditioned_di[n_w] = self.nonstationary_estimate(A, B)

        for n_w, W in enumerate(W_):
            self.Nw = np.unique(np.concatenate(W)).max() + 1

            if self.p_estimator.is_stationary_estimate:
                if not traj_eq_reweighting:
                    _causally_conditioned_di = self.stationary_estimate(W, A, B)
                else:
                    raise NotImplementedError('Equilibrium traj reweighting not yet implemented.')
            else:
                _causally_conditioned_di = self.nonstationary_estimate(A, B)

            self.causally_conditioned_di[n_w] = _causally_conditioned_di

        return self

    def _stationary_estimator(self, w_lagged, xw_lagged, y_lagged,
                              probability_estimator_wy, probability_estimator_xwy):
        raise NotImplementedError(
            'You need to overload the _stationary_estimator() method in your Estimator implementation!')

    def _nonstationary_estimator(self, w, a, b):
        raise NotImplementedError(
            'You need to overload the _nonstationary_estimator() method in your Estimator implementation!')

    def stationary_estimate(self, W, X, Y):
        """
        Directed informant estimation on discrete trajectories with Markov model
        probability estimates.
        :param W: Conditinal time-series
        :param X: Time-series 1
        :param Y: Time-series 2
        :return: di, rdi, mi
        """
        msmlag = self.p_estimator.msmlag

        assert np.unique(np.concatenate(X)).min() == 0
        assert np.unique(np.concatenate(Y)).min() == 0
        assert np.unique(np.concatenate(W)).min() == 0

        x_lagged = utils.lag_observations(X, msmlag)
        y_lagged = utils.lag_observations(Y, msmlag)
        w_lagged = utils.lag_observations(W, msmlag)

        Nw = np.unique(np.concatenate(W)).max() + 1

        prob_estimator_wy = deepcopy(self.p_estimator)
        prob_estimator_wy.msmlag = 1
        prob_estimator_wy.estimate(w_lagged, y_lagged)
        if not prob_estimator_wy.tmat_x.shape[0] == Nw:
            return np.NaN

        prob_estimator_xwy = deepcopy(self.p_estimator)
        prob_estimator_xwy.msmlag = 1
        xw_lagged = [_x + self.Nx * _w for _x, _w in zip(x_lagged, w_lagged)]

        prob_estimator_xwy.estimate(xw_lagged, y_lagged)
        if not prob_estimator_xwy.tmat_x.shape[0] == self.Nx * Nw:
            return np.NaN

        return self._stationary_estimator(w_lagged, xw_lagged, y_lagged,
                                          prob_estimator_wy, prob_estimator_xwy)

    def nonstationary_estimate(self, A, B):
        raise NotImplementedError('Not implemented.')

class CausallyConditionedDI(MultiEstimator):
    r"""
    Estimator for causally condited directed information as described by Quinn et al 2011
    with I3 estimator by Jiao et al
    """
    def __init__(self, probability_estimator):
        super(CausallyConditionedDI, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, W, X, Y):
        raise NotImplementedError

    def _stationary_estimator(self, w_lagged, xw_lagged, y_lagged,
                              probability_estimator_wy, probability_estimator_xwy):
        """
        Implementation of causally conditioned directed information from [1] using Markov model
        probability estimates..

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.

        :param w_lagged: List of binary trajectories conditioned upon which DI is conditioned. time step msmlag.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: causally conditioned directed information
        """

        di_w2y = DirectedInformation(probability_estimator_wy)
        di_w2y.estimate(w_lagged, y_lagged)

        di_xw2y = DirectedInformation(probability_estimator_xwy)
        di_xw2y.estimate(xw_lagged, y_lagged)

        return di_xw2y.d - di_w2y.d

class CausallyConditionedDIJiaoI3(MultiEstimator):
    r"""
    Estimator for causally condited directed information as described by Quinn et al 2011
    with I3 estimator by Jiao et al
    """
    def __init__(self, probability_estimator):
        super(CausallyConditionedDIJiaoI3, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, W, X, Y):
        raise NotImplementedError

    def _stationary_estimator(self, w_lagged, xw_lagged, y_lagged,
                              probability_estimator_wy, probability_estimator_xwy):
        """
        Implementation of causally conditioned directed information from [1] using Markov model
        probability estimates and DI estimator I3 from [2].

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.
        [2] J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
        Estimation of Directed Information', http://arxiv.org/abs/1201.2334.

        :param w_lagged: List of binary trajectories conditioned upon which DI is conditioned. time step msmlag.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: causally conditioned directed information
        """

        di_w2y = JiaoI3(probability_estimator_wy)
        di_w2y.estimate(w_lagged, y_lagged)

        di_xw2y = JiaoI3(probability_estimator_xwy)
        di_xw2y.estimate(xw_lagged, y_lagged)

        return di_xw2y.d - di_w2y.d


class CausallyConditionedDIJiaoI4(MultiEstimator):
    r"""
    Estimator for causally condited directed information as described by Quinn et al 2011
    with I4 estimator by Jiao et al
    """
    def __init__(self, probability_estimator):
        super(CausallyConditionedDIJiaoI4, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, W, X, Y):
        raise NotImplementedError

    def _stationary_estimator(self, w_lagged, xw_lagged, y_lagged,
                              probability_estimator_wy, probability_estimator_xwy):
        """
        Implementation of causally conditioned directed information from [1] using Markov model
        probability estimates and DI estimator I4 from [2].

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.
        [2] J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
        Estimation of Directed Information', http://arxiv.org/abs/1201.2334.

        :param w_lagged: List of binary trajectories conditioned upon which DI is conditioned. time step msmlag.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: causally conditioned directed information
        """

        di_w2y = JiaoI4(probability_estimator_wy)
        di_w2y.estimate(w_lagged, y_lagged)

        di_xw2y = JiaoI4(probability_estimator_xwy)
        di_xw2y.estimate(xw_lagged, y_lagged)

        return di_xw2y.d - di_w2y.d


class CausallyConditionedTE(MultiEstimator):
    r"""
    Estimator for causally condited transfer entropy analogously to Quinn et al 2011
    """
    def __init__(self, probability_estimator):
        super(CausallyConditionedTE, self).__init__(probability_estimator)

    def _nonstationary_estimator(self, W, X, Y):
        raise NotImplementedError

    def _stationary_estimator(self, w_lagged, xw_lagged, y_lagged,
                              probability_estimator_wy, probability_estimator_xwy):
        """
        Implementation of causally conditioned transfer entropy analogously to [1]
        using Markov model probability estimates and TE estimator from [2] as a
        replacement to DI.

        [1] Quinn , Coleman, Kiyavash, Hatsopoulos. J Comput Neurosci 2011.
        [2] Schreiber, PRL, 2000

        :param w_lagged: List of binary trajectories conditioned upon which DI is conditioned. time step msmlag.
        :param x_lagged: List of binary trajectories 1 with time step msmlag.
        :param y_lagged: List of binary trajectories 2 with time step msmlag.
        :return: causally conditioned directed information
        """

        di_w2y = TransferEntropy(probability_estimator_wy)
        di_w2y.estimate(w_lagged, y_lagged)

        di_xw2y = TransferEntropy(probability_estimator_xwy)
        di_xw2y.estimate(xw_lagged, y_lagged)

        return di_xw2y.d - di_w2y.d
