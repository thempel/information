class Information(object):
    def __init__(self, probability_estimator, information_estimator):

        self.p_estimator = probability_estimator
        self.i_estimator = information_estimator
        self.d, self.r, self.m = None, None, None

    def estimate(self, A, B):
        self.A = A
        self.B = B

        if self.p_estimator.is_stationary_estimate:
            self.d, self.r, self.m = self.i_estimator.stationary_estimate(A, B, self.p_estimator)
        else:
            raise NotImplementedError('Not yet implemented.')

        return