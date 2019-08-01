import numpy as np
import pyemma
import msmtools
from informant import utils

class MSMProbabilities:
    """
    Wrapper class for PyEMMA to estimate conditional transition probabilities.
    """
    def __init__(self, msmlag=1, reversible=False, tmat_ck_estimate=False):
        """
        Computes conditional transition probabilities from MSMs with PyEMMA.
        :param msmlag: MSM lag time (int)
        :param reversible: reversible estimate (bool)
        :param tmat_ck_estimate: Estimate higher lag time transition matrices from CK-equation
        """
        self.msmlag = msmlag
        self.reversible = reversible
        self.tmat_ck_estimate = tmat_ck_estimate

        self.is_stationary_estimate = True

        self.tmat_x, self.tmat_y, self.tmat_xy = None, None, None
        self.active_set_xy = None
        self._pi_x, self._pi_y, self._pi_xy = None, None, None

        self._user_tmat_x, self._user_tmat_y, self._user_tmat_xy = False, False, False
        self.msmkwargs = None

        self._estimated = False

        self._dangerous_ignore_warnings_flag = False

    def estimate(self, X, Y, **kwargs):
        """
        Estimates MSM probabilities from two time series separately and in combined.
        :param X: time-series 1
        :param Y: time-series 2
        :param kwargs: keyword arguments passed to pyemma.msm.estimate_markov_model()
        :return: self
        """
        self.msmkwargs = kwargs

        if not isinstance(X, list): X = [X]
        if not isinstance(Y, list): Y = [Y]

        Nx = np.unique(np.concatenate(X)).max() + 1
        Ny = np.unique(np.concatenate(Y)).max() + 1

        if not self.tmat_ck_estimate:
            if not self._user_tmat_x:
                self.tmat_x = pyemma.msm.estimate_markov_model(X, self.msmlag, reversible=self.reversible,
                                                               **kwargs).transition_matrix
            if not self._user_tmat_y:
                self.tmat_y = pyemma.msm.estimate_markov_model(Y, self.msmlag, reversible=self.reversible,
                                                               **kwargs).transition_matrix
            if not self._user_tmat_xy:
                m = pyemma.msm.estimate_markov_model([_x + Nx * _y for _x, _y in zip(X, Y)],
                                                     self.msmlag, reversible=self.reversible,
                                                     **kwargs)
                self.tmat_xy = m.transition_matrix
                self.active_set_xy = m.active_set
        else:
            if not self._user_tmat_x:
                self.tmat_x = np.linalg.matrix_power(
                    pyemma.msm.estimate_markov_model(X, 1, reversible=self.reversible,
                                                     **kwargs).transition_matrix, self.msmlag)
            if not self._user_tmat_y:
                self.tmat_y = np.linalg.matrix_power(
                    pyemma.msm.estimate_markov_model(Y, 1, reversible=self.reversible,
                                                     **kwargs).transition_matrix, self.msmlag)
            if not self._user_tmat_xy:
                m = pyemma.msm.estimate_markov_model([_x + Nx * _y for _x, _y in zip(X, Y)],
                                                     1,
                                                     reversible=self.reversible,
                                                     **kwargs)
                self.tmat_xy = np.linalg.matrix_power(m.transition_matrix, self.msmlag)
                self.active_set_xy = m.active_set

        if not self.tmat_x.shape[0] * self.tmat_y.shape[0] == self.tmat_xy.shape[0]:
            print(self.tmat_x.shape, self.tmat_y.shape, self.tmat_xy.shape)
            if not self._dangerous_ignore_warnings_flag:
                raise NotImplementedError('Combined model is not showing all combinatorial states.')

        self._estimated = True
        return self

    def set_transition_matrices(self, tmat_x=None, tmat_y=None, tmat_xy=None):
        """
        Fix transition matrices to user defined ones. Overwrites existing
        transition matrices. The ones that were not set here will be estimated with self.estimate.
        :param tmat_x: transition matrix for time series X
        :param tmat_y: transition matrix for time series Y
        :param tmat_xy: transition matrix for combinatorial time series
        :return: self
        """

        if (tmat_x is None) and (tmat_y is None) and (tmat_xy is None):
            return self

        if self.tmat_ck_estimate and self.msmlag != 1:
            print('WARNING: User-defined matrices will be matrix powered (tmat_ck_estimate=True).')

        if tmat_x is not None:
            self.tmat_x = np.linalg.matrix_power(tmat_x, self.msmlag if self.tmat_ck_estimate else 1)
            self._user_tmat_x = True
        if tmat_y is not None:
            self.tmat_y = np.linalg.matrix_power(tmat_y, self.msmlag if self.tmat_ck_estimate else 1)
            self._user_tmat_y = True
        if tmat_xy is not None:
            self.tmat_xy = np.linalg.matrix_power(tmat_xy, self.msmlag if self.tmat_ck_estimate else 1)
            self._user_tmat_xy = True

        if (tmat_x is not None) and (tmat_y is not None) and (tmat_xy is not None):
            if not self.tmat_x.shape[0] * self.tmat_y.shape[0] == self.tmat_xy.shape[0]:
                raise NotImplementedError('Combined model is not showing all combinatorial states. Auto-'
                                          'computation will save connected set.')
            self._estimated = True

        return self

    @property
    def pi_x(self):
        if not self._estimated:
            raise RuntimeError('Have to estimate before stationary distribution can be computed.')

        if self._pi_x is None:
            self._pi_x = msmtools.analysis.stationary_distribution(self.tmat_x)

        return self._pi_x

    @property
    def pi_y(self):
        if not self._estimated:
            raise RuntimeError('Have to estimate before stationary distribution can be computed.')

        if self._pi_y is None:
            self._pi_y = msmtools.analysis.stationary_distribution(self.tmat_y)

        return self._pi_y

    @property
    def pi_xy(self):
        if not self._estimated:
            raise RuntimeError('Have to estimate before stationary distribution can be computed.')

        if self._pi_xy is None:
            self._pi_xy = msmtools.analysis.stationary_distribution(self.tmat_xy)

        return self._pi_xy


class CTWProbabilities:
    def __init__(self, D=5):
        """
        Implementation of CTW algorithm as described by Jiao et al.
        :param D: context tree depth
        """
        self.D = D
        self.is_stationary_estimate = False
        self.pxy, self.px, self.py = [], [], []

        self._estimated = False

    def estimate(self, X, Y):
        if not isinstance(X, list): X = [X]
        if not isinstance(Y, list): Y = [Y]

        if np.any([len(x) <= self.D for x in X]):
            raise RuntimeError('Some trajectories are shorter than requested CT depth self.D.')

        for _X, _Y in zip(X, Y):
            # map discrete time series to set {0, 1, ..., n_states_here}
            _x = utils.relabel_dtrajs(_X)
            _y = utils.relabel_dtrajs(_Y)

            Nx_subset = (np.unique(_x).max() + 1).astype(int)
            Ny_subset = (np.unique(_y).max() + 1).astype(int)
            #if not Nx_subset == Ny_subset:
            #    print('something bad will happen...')

            self.pxy.append(self._ctwalgorithm(_x + Nx_subset * _y, Nx_subset * Ny_subset, self.D))
            self.px.append(self._ctwalgorithm(_x, Nx_subset, self.D))
            self.py.append(self._ctwalgorithm(_y, Ny_subset, self.D))
        self._estimated = True
        return self

    def _ctwalgorithm(self, x, Nx, D):
        """
        Transcribed from Matlab implementation, original to be found here:
        https://github.com/EEthinker/Universal_directed_information
        Original docstring:
        # returns var Px_record
        # Function CTWAlgorithm outputs the universal sequential probability
        # assignments given by CTW method.

        :param x: time series
        :param Nx: alphabet size
        :param D: context tree depth
        :return: probability record
        """

        if len(x.shape) != 1:
            raise IOError('The input vector must be a colum vector!')

        n = len(x)
        if Nx == 1:
            # if only one state exists, transition probability is one
            return np.ones(x.shape[0] - D)
        elif not np.floor((Nx ** (D + 1) - 1) / (Nx - 1)) == (Nx ** (D + 1) - 1) / (Nx - 1):
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
            xt = x[i].astype(int)
            eta = (countTree[:Nx - 1, leafindex - 1] + 0.5) / (countTree[Nx - 1, leafindex - 1] + 0.5)

            # update the leaf
            countTree[xt, leafindex - 1] += 1
            node = np.floor((leafindex + Nx - 2) / Nx).astype(int)

            # print(node)
            while np.all(node > 0):
                countTree, betaTree, eta = self._ctwupdate(countTree, betaTree, eta, node, xt, 1 / 2)
                node = np.floor((node + Nx - 2) / Nx).astype(int)
            eta_sum = eta.sum() + 1

            Px_record[:, i - D] = np.hstack([eta, [1]]) / eta_sum
        return Px_record

    def _ctwupdate(self, countTree, betaTree, eta, index, xt, alpha):
        """
        Transcribed from Matlab implementation, original to be found here:
        https://github.com/EEthinker/Universal_directed_information
        Original docstring:
        # returns [countTree, betaTree, eta]
        # countTree:  countTree(a+1,:) is the tree for the count of symbol a a=0,...,M
        # betaTree:   betaTree(i(s) ) =  Pe^s / \prod_{b=0}^{M} Pw^{bs}(x^{t})
        # eta = [ p(X_t = 0|.) / p(X_t = M|.), ..., p(X_t = M-1|.) / p(X_t = M|.)

        # calculate eta and update beta a, b
        # xt is the current data

        :param countTree: cf. orig. docstring
        :param betaTree:
        :param eta:
        :param index:
        :param xt:
        :param alpha:
        :return:
        """


        # size of the alphbet
        Nx = eta.shape[0] + 1

        pw = np.hstack([eta, [1]])
        pw /= pw.sum()  # % pw(1) pw(2) .. pw(M+1)

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


class NetMSMProbabilities:
    """
    Wrapper class for PyEMMA to estimate conditional transition probabilities.
    """
    def __init__(self, msmlag=1, reversible=False, tmat_ck_estimate=False):
        """
        Computes conditional transition probabilities from MSMs with PyEMMA.
        :param msmlag: MSM lag time (int)
        :param reversible: reversible estimate (bool)
        :param tmat_ck_estimate: Estimate higher lag time transition matrices from CK-equation
        """
        self.msmlag = msmlag
        self.reversible = reversible
        self.tmat_ck_estimate = tmat_ck_estimate

        self.is_stationary_estimate = True

        self.tmat_wy, self.tmat_wyx = None, None
        self.active_set_wy, self.active_set_wyx = None, None
        self._pi_wy, self._pi_wyx = None, None

        self.msmkwargs = None

        self._estimated = False
        self._user_tmat_wy, self._user_tmat_wyx = False, False

        self._dangerous_ignore_warnings_flag = False

    def estimate(self, W, X, Y, **kwargs):
        """
        Estimates MSM probabilities from two time series separately and in combined.
        :param W: time-series conditioned upon which DI is estimated
        :param X: time-series 1
        :param Y: time-series 2
        :param kwargs: keyword arguments passed to pyemma.msm.estimate_markov_model()
        :return: self
        """
        self.msmkwargs = kwargs

        if not isinstance(W, list): W = [W]
        if not isinstance(Y, list): Y = [Y]
        if not isinstance(X, list): X = [X]

        Nw = np.unique(np.concatenate(W)).max() + 1
        Ny = np.unique(np.concatenate(Y)).max() + 1
        Nx = np.unique(np.concatenate(X)).max() + 1

        #TODO: transfer all other parameters, i.e. pre-computed transition matrices
        #TODO: pairwise estimates might fail to converge to same pi_x, pi_y, pi_w!
        pairwise_xy = MSMProbabilities(self.msmlag, self.reversible, self.tmat_ck_estimate)
        pairwise_xy.estimate(X, Y, **kwargs)
        self.tmat_xy = pairwise_xy.tmat_xy
        self._pi_x = pairwise_xy.pi_x
        self._pi_y = pairwise_xy.pi_y
        self.active_set_xy = pairwise_xy.active_set_xy
        del pairwise_xy

        pairwise_xw = MSMProbabilities(self.msmlag, self.reversible, self.tmat_ck_estimate)
        pairwise_xw.estimate(X, W, **kwargs)
        self.tmat_xw = pairwise_xw.tmat_xy
        self._pi_x = pairwise_xw.pi_x
        self._pi_w = pairwise_xw.pi_y
        self.active_set_xw = pairwise_xw.active_set_xy
        del pairwise_xw

        pairwise_wy = MSMProbabilities(self.msmlag, self.reversible, self.tmat_ck_estimate)
        pairwise_wy.estimate(W, Y, **kwargs)
        self.tmat_yw = pairwise_wy.tmat_xy
        self._pi_w = pairwise_wy.pi_x
        self._pi_y = pairwise_wy.pi_y
        self.active_set_wy = pairwise_wy.active_set_xy
        del pairwise_wy

        if not self.tmat_ck_estimate:
            if not self._user_tmat_wyx:
                m = pyemma.msm.estimate_markov_model([_w + Nw * _y + (Nw * Ny) * _x for _w, _y, _x in zip(W, Y, X)],
                                                     self.msmlag, reversible=self.reversible,
                                                     **kwargs)
                self.tmat_wyx = m.transition_matrix
                self.active_set_wyx = m.active_set

        else:
            if not self._user_tmat_wyx:
                m = pyemma.msm.estimate_markov_model([_w + Nw * _y + (Nw * Ny) * _x for _w, _y, _x in zip(W, Y, X)],
                                                     1,
                                                     reversible=self.reversible,
                                                     **kwargs)
                self.tmat_wyx = np.linalg.matrix_power(m.transition_matrix, self.msmlag)
                self.active_set_wyx = m.active_set

        if not self._dangerous_ignore_warnings_flag:
            if not Nw * Ny == self.tmat_wy.shape[0]:
                    raise NotImplementedError('WY-model is not showing all combinatorial states.')
            elif not Nw * Ny * Nx == self.tmat_wyx.shape[0]:
                    raise NotImplementedError('WYX-model is not showing all combinatorial states.')

        self._estimated = True
        return self

    def set_transition_matrices(self, tmat_wy=None, tmat_wyx=None, active_set_wy=None, active_set_wyx=None):
        """
        Fix transition matrices to user defined ones. Overwrites existing
        transition matrices. The ones that were not set here will be estimated with self.estimate.
        :param tmat_wy: transition matrix for time series Y
        :param tmat_wyx: transition matrix for combinatorial time series
        :param active_set_wy: list of integers describing active set of WY time series. Ignored if tmat_wy=None.
        If not supplied, full connectivity is assumed.
        :param active_set_wyx: list of integers describing active set of WYX time series.
        Ignored if tmat_wyx=None. If not supplied, full connectivity is assumed.
        :return: self
        """

        if (tmat_wy is None) and (tmat_wyx is None):
            return self

        if self.tmat_ck_estimate and self.msmlag != 1:
            print('WARNING: User-defined matrices will be matrix powered (tmat_ck_estimate=True).')

        if tmat_wy is not None:
            self.tmat_wy = np.linalg.matrix_power(tmat_wy, self.msmlag if self.tmat_ck_estimate else 1)
            self._user_tmat_wy = True
            if active_set_wy is not None:
                self.active_set_wy = active_set_wy
            else:
                self.active_set_wy = list(range(self.tmat_wy.shape[0]))

        if tmat_wyx is not None:
            self.tmat_wyx = np.linalg.matrix_power(tmat_wyx, self.msmlag if self.tmat_ck_estimate else 1)
            self._user_tmat_wyx = True

            if active_set_wyx is not None:
                self.active_set_wyx = active_set_wyx
            else:
                self.active_set_wyx = list(range(self.tmat_wyx.shape[0]))

        # TODO: check if matrices have compatible dimensions
        if (tmat_wy is not None) and (tmat_wyx is not None):
            self._estimated = True

        return self

    def check_is_estimated(self):
        if not self._estimated:
            raise RuntimeError('Have to estimate before stationary distribution can be computed.')

    @property
    def pi_wy(self):
        self.check_is_estimated()

        self._pi_wy = msmtools.analysis.stationary_distribution(self.tmat_wy)
        return self._pi_wy

    @property
    def pi_xy(self):
        self.check_is_estimated()

        self._pi_xy = msmtools.analysis.stationary_distribution(self.tmat_xy)
        return self._pi_xy

    @property
    def pi_xw(self):
        self.check_is_estimated()

        self._pi_xw = msmtools.analysis.stationary_distribution(self.tmat_xw)
        return self._pi_xw

    @property
    def pi_wyx(self):
        self.check_is_estimated()

        self._pi_wyx = msmtools.analysis.stationary_distribution(self.tmat_wyx)
        return self._pi_wyx
