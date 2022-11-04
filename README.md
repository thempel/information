# `informant`
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
# directed information and mutual information:
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
Transfer entropy and stationary distribution-based mutual information
estimator were described by [3].

1) Directed information I4 estimator (`JiaoI4`)
2) Directed information I3 estimator (`JiaoI3`)
3) Transfer entropy estimator (`TransferEntropy`)
4) Mutual information estimator (`MutualInfoStationaryDistribution`)

#### Directed network analysis
The above directed network analysis tools will also find indirect links.
To exclude them, [5] describe the concept of causally conditioned 
directed information. The directed information from `X` to `Y` is 
conditioned on a (set of) third variables `W`; it can be assessed with

1) Causally conditioned DI estimator `CausallyConditionedDIJiaoI3`
2) Causally conditioned DI estimator `CausallyConditionedDIJiaoI4`

that use the estimators described above. It implemented as
`I(X ->Y || W ) = I(X, W -> Y) - I(W -> Y)` which follows from the definitions
in [5].

### Application
#### Installation (non-invasive)
After cloning this repo and use the following lines in your
jupyter notebook at import. This is completely non-invasive. 
Dependencies should all be satisfied for pyemma users.
```python
import sys
sys.path.append('/path/to/informant/')
import informant
``` 
#### Convencience functions
It is not planned to have a full convenience API. However, 
the network of pairwise transfer entropy estimates in a protein
can be computed and plotted as follows:

```python
dtrajs = informant.md.discretize_residue_backbone_sidechains(
         topology_file,
         trajectory_files,
         tica_lag
         )
te = informant.md.compute_inter_residue_transfer_entropy(
     dtrajs,
     msmlag
     )
informant.plots.plot_directed_links(ref_trajectory, te)
```
Depending on the protein size and the amount of data, this 
can take a while. 

### Literature
[1] J. Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    Estimation of Directed Information',
    http://arxiv.org/abs/1201.2334.

[2] https://github.com/EEthinker/Universal_directed_information

[3] Schreiber, 'Measuring Information Transfer', PRL, 2000

[4] Scherer et al, 'PyEMMA 2: A Software Package for Estimation,
    Validation, and Analysis of Markov Models', JCTC 2015.
    
[5] Quinn, Coleman, Kiyavash and Hatsopoulos, 'Estimating the directed
    information to infer causal relationships in ensemble neural spike train
    recordings', J Comput Neurosci, 2011