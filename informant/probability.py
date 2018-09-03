import numpy as np
import pyemma
import msmtools
from informant import utils

class MSMProbabilities:
    """
    Wrapper class for PyEMMA to estimate conditional transition probabilities.
    """
    def __init__(self, msmlag=1, reversible=True, tmat_ck_estimate=False):
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
    def __init__(self, D):
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
