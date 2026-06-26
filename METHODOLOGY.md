# Methodology: Mathematical Approximations

This document describes the mathematical approximations used in the
geometric signature system, what they actually compute, and how they
relate to the Riemannian geometry literature they reference.

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
across 29 metrics), flat-space approximations with anisotropic weighting
perform well empirically.

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
