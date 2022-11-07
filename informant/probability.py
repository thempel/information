import deeptime.markov.tools.estimation
import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import stationary_distribution
from informant import utils


def requires_estimated(f):
    """
    Decorator to check if an instance of probability estimator is already estimated.
    """

    def wrapper(self, *args, **kwargs):
        if not self._estimated:
            raise RuntimeError('Have to estimate before stationary distribution can be computed.')
        return f(self, *args, **kwargs)

    return wrapper


def estimate_transition_matrix(dtrajs, lag, reversible, **kwargs):
    """
    convenience function for computing transition matrix from descrete time-series

    :param dtrajs: (list of) np.ndarray, time-series of descrete data
    :param lag: int, msm lag time
    :param reversible: bool, use reversible or non-reversible msm estimator
    :param kwargs: passed to deeptime.markov.msm.MaximumLikelihoodMSM()
    :return: transition matrix, indices of largest connected set
    """
    count_model = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit_fetch(dtrajs)
    count_model_largest = count_model.submodel_largest()
    transition_probability_model = MaximumLikelihoodMSM(reversible=reversible, **kwargs).fit_fetch(count_model_largest)

    return transition_probability_model.transition_matrix, count_model.connected_sets(sort_by_population=True)[0]


class MSMProbabilitiesTransitionModel:
    """
    Wrapper for transition models
    """

    def __init__(self, data, lag, reversible, user_tmat=None, ck_estimate=False, **kw):
        """
        Wrap transition model into object.
        :param data: (list of) np.array(dtype=int); time-series data
        :param lag: int; MSM lag time
        :param reversible: bool; use a reversible estimate
        :param user_tmat: None or np.ndarray; if supplied, this is used as a transition matrix instead of a new
            estimate from the data.
        :param ck_estimate: bool; whether to compute a transition matrix from matrix exponentiation (experimental)
        :param kw: keyword arguments passed to informant.estimate_transition_matrix

        """
        if not isinstance(data, list):
            data = [data]
        if user_tmat is None:
            if not ck_estimate:
                self._tmat, self._active_set = estimate_transition_matrix(data, lag, reversible, **kw)
            else:
                _tmat, self._active_set = estimate_transition_matrix(data, 1, reversible, **kw)
                self._tmat = np.linalg.matrix_power(_tmat, lag)
        else:
            if not ck_estimate:
                self._tmat = user_tmat
            else:
                self._tmat = np.linalg.matrix_power(user_tmat, lag)
            self._active_set = np.arange(len(user_tmat))
        self._pi = None

    @property
    def pi(self):
        """
        :return: stationary probability vector
        """
        if self._pi is None:
            self._pi = stationary_distribution(self._tmat)
        return self._pi

    @property
    def tmat(self):
        """
        :return: Transition matrix
        """
        return self._tmat

    @property
    def active_set(self):
        """
        Largest connected set of transition matrix, used as active set of the MSM

        :return: active set of transition matrix
        """
        return self._active_set


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

        self._user_tmat_x, self._user_tmat_y, self._user_tmat_xy = None, None, None
        self._transition_model_x, self._transition_model_y, self._transition_model_xy = None, None, None
        self.msmkwargs = None

        self._estimated = False # TODO: _estimated -> is_estimated

        self._ignore_no_obs = True

    @staticmethod
    def _assert_shapes(tmat_x, tmat_y, tmat_xy):
        if not tmat_x.shape[0] * tmat_y.shape[0] == tmat_xy.shape[0]:
            raise NotImplementedError('Combined model is not showing all combinatorial states.' +
                                  f'x: {tmat_x.shape[0]}, y:{tmat_y.shape[0]}, ' +
                                  f'xy: {tmat_xy.shape[0]}')

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

        self._transition_model_x = MSMProbabilitiesTransitionModel(X,
                                                                   self.msmlag, self.reversible,
                                                                   ck_estimate=self.tmat_ck_estimate,
                                                                   user_tmat=self._user_tmat_x)
        self._transition_model_y = MSMProbabilitiesTransitionModel(Y,
                                                                   self.msmlag, self.reversible,
                                                                   ck_estimate=self.tmat_ck_estimate,
                                                                   user_tmat=self._user_tmat_y)
        self._transition_model_xy = MSMProbabilitiesTransitionModel([_x + Nx * _y for _x, _y in zip(X, Y)],
                                                                    self.msmlag, self.reversible,
                                                                    ck_estimate=self.tmat_ck_estimate,
                                                                    user_tmat=self._user_tmat_xy)

        if not self._ignore_no_obs:
            self._assert_shapes(self._transition_model_x.tmat, self._transition_model_y.tmat, self._transition_model_xy.tmat)

        self._estimated = True
        return self

    @property
    @requires_estimated
    def tmat_x(self):
        """
        :return: Transition matrix of time-series X
        """
        return self._user_tmat_x if self._user_tmat_x is not None else self._transition_model_x.tmat

    @property
    @requires_estimated
    def tmat_y(self):
        """
        :return: Transition matrix of time-series Y
        """
        return self._user_tmat_y if self._user_tmat_y is not None else self._transition_model_y.tmat

    @property
    @requires_estimated
    def tmat_xy(self):
        """
        :return: Transition matrix of combined time-series X, Y
        """
        return self._user_tmat_xy if self._user_tmat_xy is not None else self._transition_model_xy.tmat

    @property
    @requires_estimated
    def active_set_xy(self):
        """
        :return: Active set of transition matrix in combined X, Y space
        """
        return self._transition_model_xy.active_set

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
            self._user_tmat_x = np.linalg.matrix_power(tmat_x, self.msmlag if self.tmat_ck_estimate else 1)
        if tmat_y is not None:
            self._user_tmat_y = np.linalg.matrix_power(tmat_y, self.msmlag if self.tmat_ck_estimate else 1)
        if tmat_xy is not None:
            self._user_tmat_xy = np.linalg.matrix_power(tmat_xy, self.msmlag if self.tmat_ck_estimate else 1)
            assert deeptime.markov.tools.estimation.is_connected(self._user_tmat_xy)

        if not any((x is None for x in (tmat_x, tmat_y, tmat_xy))):
            self._assert_shapes(tmat_x, tmat_y, tmat_xy)
        self._estimated = True
        return self

    @property
    @requires_estimated
    def pi_x(self):
        """
        :return: stationary distribution of transition matrix of X
        """
        assert self._transition_model_x is not None
        return self._transition_model_x.pi

    @property
    @requires_estimated
    def pi_y(self):
        """
        :return: stationary distribution of transition matrix of Y
        """
        assert self._transition_model_y is not None
        return self._transition_model_y.pi

    @property
    @requires_estimated
    def pi_xy(self):
        """
        :return: stationary distribution of transition matrix of comined time-series X, Y
        """
        assert self._transition_model_xy is not None
        return self._transition_model_xy.pi


class CTWProbabilities:
    """
    CAUTION: ONLY IMPLEMENTED AS REFERENCE, should be thoroughly tested.

    Python implementation of CTW algorithm that was used by
    J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    Estimation of Directed Information', http://arxiv.org/abs/1201.2334.

    This code is based on the matlab code presented by the authors published at
    https://github.com/EEthinker/Universal_directed_information
    which was licensed under the MIT License:

    Copyright 2017 EEthinker

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions
    of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """
    def __init__(self, D=5):
        """
        Implementation of CTW algorithm as described by
        J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
        Estimation of Directed Information', http://arxiv.org/abs/1201.2334.
        :param D: context tree depth
        """
        self.D = D
        self.is_stationary_estimate = False
        self.pxy, self.px, self.py = [], [], []

        self._estimated = False

    def estimate(self, X, Y):
        """
        Estimate non-stationary probabilities using CTW algorithm.

        :param X: time-series X
        :param Y: time-series Y
        :return: self
        """
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

        # size of the alphabet
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
