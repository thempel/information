# ~ `informant` ~
## A python package for directed information estimation
### Idea
This package was intended to provide access to directed and mutual
information estimation between two time series with python. It was
developed with the idea that directed information can always be computed
if a conditional probability assignment for the time series can be made.

Since there are several ways of assigning probabilities and several
estimators for directed information, the API works like follows:
```python
# A, B are two discrete, finite-alphabet time series of the same length.
# first, compute the probabilities
p_estimator = SomeProbabilityEstimator(**kwargs)
p_estimator.estimate(A, B)

# second, estimate directed, reverse directed and mutual information
i_estimator = SomeInformationEstimator(p_estimator)
i_estimator.estimate(A, B)

# third, access directed, reverse directed and mutual information:
i_estimator.d, i_estimator.r, i_estimator.m
```

The implementation relies on the theory presented by [1], augmented by
MSM probability estimates (and a python implementation).

### Implementation details
#### Probability estimators are:
1) Continous tree weighting (CTW)-based (`CTWProbabilities`)
2) stationary MSM probabilities (`MSMProbabilities`)

#### Directed information estimators
This code relies on estimators described by [1] that have been
implemented with the CTW-algorithm in matlab [2].
1) I4 estimator (`JiaoI4`)
2) I4 ensemble average estimator from MSM probabilities (`JiaoI4Ensemble`)
3) I3 estimator for reference (`JiaoI3`) (not yet thoroughly tested)

### Literature
[1] J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    Estimation of Directed Information',
    http://arxiv.org/abs/1201.2334.

[2] https://github.com/EEthinker/Universal_directed_information
