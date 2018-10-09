# ~ `informant` ~
## A python package for estimating directed information and transfer entropy
### Idea
This package was intended to provide access to estimators for directed
information, transfer entropy and mutual information between two time
series with python 3. It was developed with the idea that the mentioned
quantities can always be computed if a conditional probability
assignment for the time series can be made.

Since there are several ways of assigning probabilities and several
estimators for directed information or transfer entropy, the API works
like follows, starting with two discrete, finite-alphabet time series
of the same length `A` and `B`.
```python
# first, define probabilities
p_estimator = SomeProbabilityEstimator(**kwargs)

# second, estimate directed information / transfer entropy
i_estimator = SomeInformationEstimator(p_estimator)
i_estimator.estimate(A, B)

# third, access directed information (or transfer entropy), reverse
directed information and mutual information:
i_estimator.d, i_estimator.r, i_estimator.m
```

The directed information estimators are implemented as described by [1],
augmented by MSM probability estimates. Transfer entropy was introduced
by Schreiber [3].

### Implementation details
#### Probability estimators are:
1) Continous tree weighting (CTW)-based (`CTWProbabilities`)
2) stationary MSM probabilities (`MSMProbabilities`, a wrapper for
   PyEMMA [4])

#### Directed information / transfer entropy estimators
This code relies on directed information estimators described by [1]
that they originally implemented with the CTW-algorithm in matlab [2].
1) Directed information I4 estimator (`JiaoI4`)
2) Directed information I3 estimator (`JiaoI3`)
3) Transfer entropy estimator (`TransferEntropy`)

### Literature
[1] J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    Estimation of Directed Information',
    http://arxiv.org/abs/1201.2334.

[2] https://github.com/EEthinker/Universal_directed_information
[3] Schreiber, 'Measuring Information Transfer', PRL, 2000
[4] Scherer et al, 'PyEMMA 2: A Software Package for Estimation,
    Validation, and Analysis of Markov Models', JCTC 2015.