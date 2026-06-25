# Flat-sky NaMaster vs BJK on Euclid RR2 — resolution note (June 2026)

## Summary

Flat-sky NaMaster (HEALPix → gnomonic projection → `NmtFieldFlat`) **works** on the
Euclid RR2v2.1 tombin-1 data and **agrees with BJK on EE to ~10–20%** in the valid
ℓ range. An earlier apparent "flat-sky is broken" conclusion was a comparison
artifact, not a real failure.

## The actual bug: double D_ℓ conversion

The comparison file
`almanac_runs/bjk_results/bjk_euclid_tombin-1_nside128_eb_n1.dat` was produced with
`band_model='Dl'`. Its column **labelled `C_l_ML` actually contains D_ℓ**
(= C_ℓ·ℓ(ℓ+1)/2π), not C_ℓ.

I treated that column as C_ℓ and multiplied by ℓ(ℓ+1)/2π **again**, inflating the
BJK "EE" curve by ~50–2000× (the factor grows with ℓ). That made flat-sky look
~100–1000× too low. Cross-checking against the clean EE-only run
(`...nside128_ee.dat`, which uses `band_model='Cl'` and really does store C_ℓ ~
3×10⁻⁸ → 2×10⁻⁹) showed the two BJK files agree once eb_n1 is read as D_ℓ.

**Lesson:** check `band_model` before converting. eb_n1.dat is already D_ℓ — plot
its columns directly.

## Decisive sanity check: map variance

The observed-pixel signal variance is a hard constraint:
`Var(Q)+Var(U) ≈ Σ_ℓ (2ℓ+1)/(4π) (C_ℓ^EE + C_ℓ^BB)`.

- Data: signal Var(Q)+Var(U) ≈ 2.7×10⁻⁵ (after subtracting 1/Ninv).
- NaMaster C_ℓ (~few×10⁻⁹) → predicted var ~10⁻⁵ ✓ consistent.
- The mis-converted eb_n1 "C_ℓ" (~4×10⁻⁶) → predicted var ~6×10⁻² ✗ (2300× too high).

This single check exposes a unit error immediately and is worth doing first in any
power-spectrum cross-comparison.

## Working comparison (NSIDE=128, matched resolution)

Script: `examples/plot_bjk_vs_flatsky_three_spectra.py`. NSIDE=128 native
resolution (~27.5′, no upsampling), gnomonic projection centred on the patch,
band edges `[2,34,66,98,130,162,194,226,257]`, 60′ apodization.

- **EE:** flat-sky / BJK ratio **0.76–1.02** for ℓ = 80–210. Lowest band (ℓ≈17,
  below the patch fundamental ℓ_min~8) is unreliable; highest band (241) runs ~0.5
  (Nyquist/edge). Mid-range is solid.
- **BB:** raw flat-sky sits on a noise floor (D_ℓ ~ 6–10×10⁻⁶). Noise-bias
  subtraction from 30 noise realizations (drawn per-pixel from 1/Ninv, projected
  through the identical pipeline) pulls it down ~3× to ~2–4×10⁻⁶. Still above BJK's
  near-zero BB — residual positivity/noise floor; BJK's ML S+N model removes noise
  more completely than a 30-sim MC bias. More sims / analytic N_ℓ would tighten it.
- **EB:** both methods null (no parity violation), scatter about zero.

## Other findings along the way

- **Matched resolution matters.** Projecting NSIDE=256 (13.7′) data onto a 2′ grid
  (≈7× upsampling) corrupts the mode-coupling deconvolution. Either project at the
  HEALPix native resolution, or project fine then **block-downsample** back to ~native
  before NaMaster.
- **Gnomonic distortion is fine here** (max area scale ~1.5× over a 46° patch).
- **The right end goal** for flat-sky is to map directly in flat-sky coordinates
  (from timestreams/CCD), avoiding HEALPix→flat resampling altogether.

## Steelman comparison: give NaMaster every advantage

Script: `examples/plot_bjk_vs_flatsky_steelman.py`. To find the *genuine* limits of
flat-sky NaMaster we removed every avoidable handicap: no B-purification (it is
unstable on this ragged 3-region patch — see below), a transfer-function correction
from signal-only sims (T_EE = T_BB ≈ 0.80), noise-bias subtraction from noise-only
sims drawn from 1/Ninv through the identical pipeline, and E→B leakage subtraction
(spurious BB per unit EE ≈ 0.09). Error bars come from a fiducial signal(EE)+noise
ensemble pushed through the full corrected pipeline.

Result (NSIDE=128, matched resolution):

- **EE:** fully corrected, NaMaster/BJK = **0.99–1.13 for ℓ = 49–210**. Excellent.
  Endpoints degrade: ℓ≈17 → 0.25 (below patch fundamental ℓ_min~8), ℓ=241 → 0.76
  (Nyquist/edge). These large/small-scale losses are **structural** and survive all
  corrections.
- **BB:** with error bars, NaMaster BB is consistent with zero — but its
  **σ(BB) is ~3.2× larger than BJK's** (≈0.63 vs 0.20 ×10⁻⁶, median ℓ≥80).
- **EB:** both null (no parity signal); NaMaster error bars similarly inflated.

## Why BJK/Almanac are more efficient (corrected noise argument)

The BB advantage is **not** that the full-likelihood methods know the specific noise
realization. Both estimators are *unbiased* given correct models. BJK/Almanac are
**minimum-variance** because they use the complete data covariance `C = S + N` with
`N` the actual **pixel-space** noise covariance — inhomogeneous (per-pixel `1/Ninv`
varies across the footprint) and in general anisotropic. NaMaster compresses noise to
an **isotropic harmonic `N_ℓ`** and applies a fixed apodized-mask weighting; that
compression is lossy (it discards where the noise is deep vs shallow and assumes
statistical isotropy), yielding a higher-variance estimator. The concrete handle is
inverse-variance-per-pixel weighting (optimal `C⁻¹`), which BJK does and
`N_ℓ`-subtraction cannot. The observed ~3.2× BB error ratio is this efficiency gap.

## The purification trilemma (sharpest argument for BJK/Almanac)

Flat-sky NaMaster on a partial sky must choose, and loses either way:

1. **Purify B** → discards ambiguous E/B modes (information loss → larger variance);
   here it is also numerically unstable, *injecting* low-ℓ BB power
   (E→B leakage at ℓ≈17 is 9.1×10⁻⁶ purified vs 2.6×10⁻⁶ unpurified).
2. **Don't purify** → E→B leakage bias, which must be *modeled and subtracted* from
   sims (extra variance and sim dependence).
3. **Either way** it needs a sim-built transfer function, noise model, and leakage
   model just to reach the agreement above.

BJK/Almanac never purify and never discard the ambiguous modes: that information is
**retained in the joint EE/BB/EB covariance** — in BJK as off-diagonal Fisher/Hessian
terms, and fully in the Almanac posterior. Whether the ambiguous-mode information
shows up as EE/BB/EB cross-correlation structure in the Almanac posterior is an open
item tracked in the Almanac summary (`euclid_rr2_results.md`, §5.4).

## Open to-dos

- More noise/ensemble sims (100+) to tighten the BB error-bar estimate.
- Investigate the high-ℓ (241) band rolloff (likely Nyquist/edge).
- Inverse-noise weighting the flat-sky map (closer to BJK's optimal weights).
- Almanac-side: map the ambiguous-mode information in the posterior (see Almanac §5.4).
