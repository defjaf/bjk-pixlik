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

## 6. Possible extensions / modifications to consider

Motivated by the picture above and by the multi-field, low-fsky regime.

### (a) A matrix band-power likelihood that handles cross-spectra

The offset-lognormal band-power form (Bond–Jaffe–Knox 2000), `z_b = ln(θ_b + N_b)`,
makes the Gaussian approximation in a coordinate where the auto-power posterior is
nearly Gaussian and the `C_ℓ ≥ 0` boundary is respected, with the offset `N_b` set
by the per-band **noise floor** (which the rotation/Gaussian noise runs measure
directly). **But it is intrinsically an auto-spectrum form** — a cross-spectrum is
signed and has no positivity boundary, so a log transform is ill-defined. Two ways
to handle a full multi-field (E/B + tomographic) matrix:

- **Simplest:** offset-lognormal for the auto-spectra, Gaussian for the
  cross-spectra. Pragmatic, but treats matrix elements inconsistently and ignores
  the joint positive-definiteness constraint linking them.

- **Robust (recommended): the Hamimeche–Lewis (2008) transform.** It works on the
  whole `n_field × n_field` band-power matrix rather than element-by-element. Per
  band, form
  ```
  X = vecp[ C_f^{1/2} · g( C^{-1/2} Ĉ C^{-1/2} ) · C_f^{1/2} ],
      g(x) = sign(x − 1) · sqrt( 2 (x − ln x − 1) )   (applied to eigenvalues)
  ```
  with `C` the model matrix, `Ĉ` the estimate, `C_f` a fiducial; then
  `−2 ln L ≈ Xᵀ M_f^{-1} X`, with `M_f` the fiducial band-power covariance. Because
  the transform acts on the matrix, **cross-spectra are handled automatically** and
  the enforced constraint is positive-definiteness of the *whole matrix* (not
  positivity of each element) — exactly what E/B + tomographic cross-spectra need.
  It is exact for a single mode (Wishart) and stays accurate into the few-modes
  regime, i.e. our low-fsky case. Cross-spectrum / low-ℓ extensions:
  Mangilli, Plaszczyński & Tristram (2015).

  *Connections:* `M_f` is just the Fisher covariance BJK already computes; the
  fiducial offset plays the role of the measured per-band noise floor; and the
  matrix structure is the analytic-Gaussian counterpart of Almanac's log-Cholesky
  positive-definite band-power matrices `Θ_b` — H&L is, in effect, the Gaussian
  approximation to what Almanac samples exactly.

  *Bandpower form and a common-bands requirement.* These approximations extend
  directly to **bandpowers**: apply the g-transform to the binned matrices `Ĉ_b`
  and use a band-power covariance `M_f` carrying the band–band (mode-coupling) and
  field–field correlations. Binning only lowers the effective modes per band `ν_b`;
  the g-function is precisely what keeps the transformed variable near-Gaussian as
  `ν_b` falls, so the binned form stays valid in the few-modes / low-fsky regime
  (this is how XFaster and the binned Planck high-ℓ likelihoods are built).
  **However, the matrix (H&L) form requires the same band partition for *all*
  fields — every tomographic bin and both E and B.** The transform acts on the
  `n_field × n_field` matrix *at a fixed band* (via `C_b^{-1/2}`, its
  eigendecomposition, and `g` on the eigenvalues), and the constraint it enforces —
  positive-definiteness — is a property of that cross-field matrix at each band. So
  a single, common ℓ-band shared by every field-pair is needed to form it; binning
  must act identically on all field-pairs. Band *widths* may vary with band index
  (variable binning in ℓ), and the in-band weighting (flat-in-`C_ℓ` vs `D_ℓ`, etc.)
  may be chosen, but both must be common across fields. Giving different bins/fields
  *different band edges* breaks the band-level matrix and the cross-spectra — it
  only "works" in the degenerate auto-only (offset-lognormal, diagonal) case, which
  discards exactly the cross-spectra and matrix structure H&L provides. If different
  effective resolution per bin is wanted, keep a common (fine) band grid and let
  `M_f` / the weighting downweight the noisy bins, rather than using per-field bands.
  (Almanac already satisfies this: one PD `Θ_b` per band with `bandedges` shared
  across all 12 fields.)

### (b) Robust (sandwich) covariance for the error bars

`F⁻¹` is the correct covariance only if the model — including the noise — is exactly
right. We have *measured* that it isn't (the ~19% gap between the rotation-noise
realization and the diagonal `1/Σw` model). A sandwich estimator
`Σ = F⁻¹ J F⁻¹` (with `J` the variance of the score) is robust to mild noise-model
misspecification. The Fisher remains the right object for the *iteration* (its
PD-by-construction stability); the sandwich is for the reported *covariance*.

### (c) Be deliberate about constraining — don't import the positivity bias

The positivity bias is a property of the *constrained Bayesian posterior* (Almanac),
not of the BJK ML estimate. The unconstrained ML band-power can go negative and is,
to leading order, **unbiased** — which is what a downstream Gaussian/`χ²`
cosmological likelihood wants as input. So keep the ML estimate unconstrained
(signed) for downstream inputs, and use the matrix-transform / offset posterior only
when a standalone positive spectrum or detection statement is wanted. "Fixing" BJK by
clamping to `θ ≥ 0` would *introduce* the bias, not remove it.

## Pointers

- Bond, Jaffe & Knox (1998) — iterative quadratic / Newton–Raphson ML power-spectrum estimation.
- Bond, Jaffe & Knox (2000) — offset-lognormal band-power likelihood ("radical compression").
- Hamimeche & Lewis (2008) — matrix-transform quasi-Gaussian likelihood for the full C_ℓ matrix (handles cross-spectra & matrix positivity).
- Mangilli, Plaszczyński & Tristram (2015) — cross-spectrum / low-ℓ extension of the H&L-type approximation.
- Fisher scoring / method of scoring — expected vs observed information in ML iteration.
- Opper & Archambeau — the Gaussian variational approximation.
- Khan & Lin and collaborators — natural-gradient variational inference; Newton/Fisher-scoring-like updates.
- Friston et al. — variational Laplace (free-energy formulation of the Laplace approximation).
- Bernstein–von Mises — asymptotic Gaussianity of the posterior about the MLE.

*(Bibliographic details/years to be verified; author/concept pointers only.)*
