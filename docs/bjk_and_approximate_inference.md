# BJK pixel-likelihood and approximate inference

How the BJK iterative maximum-likelihood power-spectrum estimator relates to
the Laplace approximation, variational inference, and full posterior sampling
(Almanac HMC). Written June 2026.

---

## TL;DR

The BJK estimator is **Fisher-scoring maximum likelihood**: it iterates to the
peak of the log-likelihood with a quasi-Newton step that uses the Fisher matrix
in place of the observed Hessian, and reports the inverse Fisher at the peak as
the parameter covariance. With a flat prior (the bandpower case) the output
`(θ̂_ML, F⁻¹)` *is* a Gaussian approximation to the posterior — the **Laplace /
variational-Laplace** approximation. It coincides with the optimal Gaussian
variational posterior in the Gaussian (large-data, Bernstein–von Mises) limit,
and breaks in the same way a Gaussian variational family would when the true
posterior is non-Gaussian — most importantly near the `C_ℓ ≥ 0` boundary.

## 1. BJK as a Gaussian posterior approximation

Reduced to its structure, BJK (i) iterates to the peak of the log-likelihood and
(ii) returns the inverse Fisher at that peak as the covariance. Under a flat
prior the peak is the MAP, so

    q(θ) = N( θ̂_ML , F⁻¹(θ̂_ML) )

is a Gaussian approximation to the posterior. That is the **Laplace
approximation**: fit a Gaussian by local curvature-matching at the mode. It is
the simplest member of the deterministic approximate-inference family to which
variational inference (VI) also belongs — the zeroth-order case, obtained by
curvature-matching rather than by minimizing a divergence to the posterior.

## 2. Fisher (not Hessian) = method of scoring = natural gradient

Replacing the observed Hessian with the Fisher (expected) information is the
**method of scoring**. Two consequences:

- **Stability.** The Fisher is positive semidefinite by construction (a
  covariance of scores), whereas the observed Hessian can become indefinite away
  from the peak or under an unlucky noise realization. Fisher-Newton steps
  therefore remain ascent directions and the implied covariance stays
  positive-definite. This is the empirically observed "slightly more stable"
  behaviour, and it is the standard reason scoring is preferred over raw
  Newton–Raphson for Gaussian-field likelihoods.

- **Geometry.** The Fisher is the natural metric on the parameter manifold, so a
  Fisher-preconditioned step is a **natural-gradient** step. Natural-gradient
  Gaussian VI (Khan, Lin and collaborators) produces updates that are exactly
  Newton/Fisher-scoring-like: the mean moves by `F⁻¹·gradient` and the covariance
  tracks `F⁻¹`. BJK runs this same machinery but stops at the point estimate and
  reads off the curvature, rather than iterating a separate variational
  covariance.

So, algorithmically: **BJK ≈ a natural-gradient Gaussian-VI update evaluated at
convergence**, with the Laplace covariance as its posterior width.

## 3. Regime of validity

- **Gaussian / Bernstein–von Mises limit.** As data become constraining the
  posterior → `N(θ̂_ML, F⁻¹)`. There the BJK Gaussian, the Laplace approximation,
  and the optimal Gaussian variational posterior all coincide; BJK is a fully
  adequate approximate-inference scheme.

- **Non-Gaussian posterior.** The three diverge. Reverse-KL VI (`min KL(q‖p)`) is
  mode-seeking and variance-underestimating; Laplace/Fisher matches curvature at
  the mode; neither represents skewness or hard parameter boundaries. The
  relevant case here is the **`C_ℓ ≥ 0` positivity constraint**: near the wall
  the auto-power posterior is one-sided and skewed, so the symmetric inverse-
  Fisher Gaussian (and equally a Gaussian variational family) misrepresents it —
  under-covering and missing the positivity-induced bias. The Gaussian
  variational family is simply mis-specified against a boundary-constrained
  target.

## 4. Relation to Almanac (full HMC posterior)

Almanac samples the posterior directly with HMC, making no Gaussian assumption,
so it captures exactly the non-Gaussianity that the BJK Fisher-Gaussian omits.
Two empirical consequences, consistent with the picture above:

- Under a flat prior the **BJK MLE equals the Almanac posterior mode** (recovered
  as the half-sample mode of the bandpower samples). The peak-finder and the
  sampler agree on the peak.
- The Almanac posterior **mean is biased high** relative to the mode for the
  positive, skewed auto-power marginals — the part of the distribution the
  symmetric BJK Gaussian cannot represent. This is the same effect that produces
  the positivity-bias floor seen when running the sampler on pure noise.

So BJK and Almanac are the Gaussian-approximate and exact ends of the same
inference problem: BJK gives `(mode, inverse-Fisher)`; Almanac gives the full
posterior, which reduces to the BJK Gaussian away from constraints and departs
from it (skew, boundary, positivity bias) near `C_ℓ = 0`.

## 5. Caveat on the word "variational"

BJK optimizes the *likelihood*, not a divergence to the posterior — it is
frequentist maximum likelihood (with `F⁻¹` as the Cramér–Rao covariance), not an
ELBO/KL objective. The variational reading is via the **variational-Laplace**
correspondence (Friston et al.): a Gaussian `q` whose mean is found by Fisher
scoring and whose covariance is the inverse curvature, justified as free-energy
(ELBO) optimization under a Gaussian-`q` assumption. Strictly, the Laplace
approximation and reverse-KL VI give the *same* Gaussian only when the target is
itself Gaussian; otherwise their covariances differ. BJK is best described as the
Laplace/variational-Laplace point of the spectrum, equivalent to Gaussian VI in
the Gaussian limit.

## Pointers

- Bond, Jaffe & Knox (1998) — iterative quadratic / Newton–Raphson ML power-spectrum estimation.
- Fisher scoring / method of scoring — expected vs observed information in ML iteration.
- Opper & Archambeau — the Gaussian variational approximation.
- Khan & Lin and collaborators — natural-gradient variational inference; Newton/Fisher-scoring-like updates.
- Friston et al. — variational Laplace (free-energy formulation of the Laplace approximation).
- Bernstein–von Mises — asymptotic Gaussianity of the posterior about the MLE.

*(Bibliographic details to be filled in; author/concept pointers only.)*
