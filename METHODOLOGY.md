# Methodology: Mathematical Methods and Approximations

This document describes the mathematical methods used in the geometric
signature system, including the embedding signature pipeline, statistical
validation procedures, and Riemannian geometry approximations.

## Embedding Signatures

### Architecture: 768-D to 20-D via Shared PCA

Each inference response is passed through nomic-embed-text-v1-5 (hosted
on MaaS) to produce a 768-dimensional embedding vector. These raw
embeddings capture the full semantic content of the response -- word
choice, sentence structure, topic coverage, and stylistic patterns.

The 768-D vectors are reduced to 20 dimensions using PCA (Principal
Component Analysis). The PCA transformation is fitted on the combined
embeddings from all enrolled agents, producing a single shared
projection matrix. Each agent's identity is represented as a centroid
(mean vector) in this shared 20-D space.

### Why Shared PCA Is Critical

The initial implementation used per-agent PCA: each agent's embeddings
were projected using a PCA fitted only on that agent's data. This
produced incomparable coordinate spaces -- dimension 1 for Agent A
might capture topic variation while dimension 1 for Agent B captures
formality variation. Distances between agents in these misaligned spaces
were meaningless.

Shared PCA fixes this by fitting a single PCA on the union of all
agents' embeddings. Every agent's centroid lives in the same coordinate
space, where each dimension has a consistent interpretation across all
agents. This is mathematically equivalent to choosing a single
orthonormal basis for the subspace that captures maximum variance across
the entire population.

The fix changed EER from 22.9% (per-agent PCA, functionally broken)
to 5.6% (shared PCA, correct), ultimately reaching 3.6% after
optimization.

### PCA Component Selection

A sweep over PCA components (5, 10, 15, 20, 25, 30, 40, 50) showed
that 20 components is optimal with an EER of 4.0%. Fewer components
lose discriminative information; more components introduce noise
dimensions that dilute the signal. The explained variance ratio at
20 components captures the majority of the inter-agent variance while
suppressing within-agent noise.

### Centroid Computation

An agent's identity centroid is the arithmetic mean of its 20-D
projected embeddings across all enrollment runs. For verification,
a new response's embedding is projected using the same shared PCA
matrix and compared to the enrolled centroid via Mahalanobis distance.

## EER Computation Methodology

The Equal Error Rate (EER) is the operating point where the False
Acceptance Rate (FAR) equals the False Rejection Rate (FRR).

### Procedure

1. Compute pairwise distances between all agent pairs (genuine pairs:
   same agent, different runs; impostor pairs: different agents).
2. Sweep a threshold from min to max distance.
3. At each threshold:
   - FAR = fraction of impostor pairs with distance below threshold
     (incorrectly accepted)
   - FRR = fraction of genuine pairs with distance above threshold
     (incorrectly rejected)
4. Find the threshold where FAR and FRR intersect. The EER is the
   value at this intersection.

For the embedding study, genuine pairs are constructed by splitting
each agent's runs into enrollment (centroid computation) and
verification sets. The distance from each verification embedding to
its own agent's centroid forms the genuine distribution; distances
to all other agents' centroids form the impostor distribution.

## Bootstrap Confidence Intervals

The bootstrap procedure estimates the uncertainty of the EER:

1. From N total agent-run observations, resample N observations with
   replacement (stratified by agent to maintain population balance).
2. Recompute the shared PCA, agent centroids, all pairwise distances,
   and the EER on this resampled dataset.
3. Repeat for B bootstrap iterations (B=20 in the reported results).
4. Report the mean and standard deviation of the B EER estimates.

The reported EER of 3.6% +/- 1.7% is the mean +/- standard deviation
across 20 bootstrap resamples. The full bootstrap recomputes PCA on
each resample, so the confidence interval accounts for uncertainty in
the projection as well as in the distance distributions.

## Ledoit-Wolf Shrinkage

When computing the covariance matrix for Mahalanobis distance, small
sample sizes (few enrollment runs per agent) can produce singular or
poorly conditioned covariance matrices. Ledoit-Wolf shrinkage
regularizes the covariance estimate by interpolating between the sample
covariance and a structured target (scaled identity matrix):

  Sigma_shrunk = (1 - alpha) * Sigma_sample + alpha * trace(Sigma_sample)/p * I

where alpha is chosen to minimize the expected squared Frobenius norm
of the estimation error. This is applied automatically when the number
of enrollment runs is small relative to the number of dimensions (the
p > n regime, which occurs when embedding dimensions exceed the number
of runs).

In practice, Ledoit-Wolf shrinkage is used for the 20-D embedding
covariance but is less critical for the 6-D Fisher-selected metric
covariance, where sample sizes typically exceed the dimensionality.

## Fisher Discriminant Ratio

The Fisher discriminant ratio measures how well a single metric
separates two groups (agents). For metric k and agents i and j:

  F_k = (mu_i_k - mu_j_k)^2 / (sigma_i_k^2 + sigma_j_k^2)

where mu is the group mean and sigma^2 is the group variance of
metric k. The ratio is the squared difference in means divided by
the sum of variances -- it quantifies separation relative to spread.

### Metric Selection

For multi-agent scenarios, the Fisher ratio is averaged across all
agent pairs for each metric, then metrics are ranked. The top-K
metrics (K=6 in the reported results) form a reduced feature vector
that concentrates discriminative signal.

### Impact

Of the 35 available metrics, 15 have Fisher ratio of zero (identical
values across agents when served by vLLM). Using all 35 metrics
produces a separation ratio of 1.42. Using only the top 6 Fisher-
selected metrics produces a separation ratio of 4.28 -- a 3x
improvement from discarding non-discriminating dimensions.

### Relation to Embedding Signatures

Fisher metric selection operates on the 35 scalar metrics. It does
not apply to the 20-D embedding signatures, which use PCA for
dimensionality reduction instead. The two approaches are complementary:
Fisher selection provides interpretable drift decomposition (which
specific metrics shifted), while embedding signatures provide the
primary identity verification signal (3.6% EER).

## Distance Metric

We use Mahalanobis distance: the square root of (difference vector)^T
times (precision matrix) times (difference vector). This weights each
dimension by the inverse covariance, so dimensions with low variance
contribute more to the distance.

This is not geodesic distance on a Riemannian manifold with curvature.
True geodesic distance would require solving the geodesic equation on
the manifold, accounting for curvature along the path. Mahalanobis
distance is equivalent to geodesic distance only on a flat manifold
(zero curvature everywhere).

## Frechet Mean

The "Frechet mean" is computed via gradient descent on a quadratic
objective: minimize the sum of squared Mahalanobis distances to all
sample points. This is the standard weighted mean in a flat space with
an anisotropic metric.

On a curved manifold, the Frechet mean requires iterative computation
using exponential and logarithmic maps (Karcher's algorithm). We do not
use exponential or logarithmic maps. Our computation is exact for flat
manifolds and a first-order approximation for mildly curved ones.

## Sectional Curvature

"Sectional curvature" is estimated as the variance of log-eigenvalues
of the covariance matrix. This captures how anisotropic the metric
space is -- high variance means the space stretches very differently
in different directions.

True sectional curvature is computed from the Riemann curvature tensor,
which requires second derivatives of the metric tensor with respect to
the coordinates. We do not compute the Riemann tensor. Our estimate is
a heuristic proxy that correlates with curvature effects in practice:
when eigenvalue spread is large, flat-space approximations degrade.

## Why These Approximations

These approximations are reasonable for the research hypothesis. The
key insight of geometric agent fingerprinting is that behavioral
metrics live in an anisotropic space where different dimensions have
different natural scales and correlations. Mahalanobis distance captures
this core property without requiring full differential geometry.

The approximations break down when:
- The metric space has strong curvature (our curvature estimate flags this)
- Sample sizes are too small to estimate covariance reliably
- The manifold has non-trivial topology (e.g., wraparound dimensions)

For the current use case (comparing behavioral signatures of LLM agents
across 35 metrics and 20-D embeddings), flat-space approximations with
anisotropic weighting perform well empirically, as validated by the
3.6% EER achieved on the embedding signatures.

## Relationship to Literature

The terminology (geodesic distance, Frechet mean, sectional curvature,
manifold) is used deliberately to connect this work to the
Riemannian-Geometric Fingerprints literature (arXiv 2506.22802). That
paper establishes the theoretical framework for treating behavioral
metrics as points on a Riemannian manifold. Our implementation uses
simplified versions of those constructs, trading mathematical exactness
for computational tractability and interpretability.

The simplifications are acknowledged, not hidden. If future work
requires true Riemannian computation (e.g., for agents with very
different behavioral profiles where curvature matters), the architecture
supports swapping in exact geodesic solvers.
