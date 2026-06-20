# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements the **Bond, Jaffe & Knox (BJK98)** pixel-space CMB power spectrum likelihood. It estimates angular power spectrum bandpowers for arbitrary combinations of spin-0 (T) and spin-2 (Q+U) tomographic bins using exact pixel-space Gaussian likelihood with Newton-Raphson maximization.

**Key reference:** Bond, Jaffe & Knox (1998), PhysRevD 57, 2117 — https://doi.org/10.1103/PhysRevD.57.2117

### Relationship to Almanac

This code was developed as a comparison method for **Almanac**, a Bayesian HMC sampler for CMB/weak-lensing power spectra located in `~/Desktop/Projects/Almanac`. While Almanac samples the full posterior `P(C_ℓ, a_ℓm | d)` via HMC, BJK provides a fast maximum-likelihood alternative with Fisher-based uncertainties. The BJK ML estimate `(θ̂_ML, F⁻¹)` is mathematically equivalent to a Laplace/variational-Laplace approximation to the posterior (see `docs/bjk_and_approximate_inference.md`).

**Typical use cases:**
- Validation: Compare Almanac posterior means to BJK ML estimates on shared simulations
- Benchmarking: BJK provides fast point estimates for method comparison (e.g., vs NaMaster pseudo-C_ℓ)
- Real data: Applied to Euclid RR2v2.1 LensMC weak-lensing data (see `~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs/run_bjk_euclid_ee.py`)

**Shared simulation data:** The `data/` directory contains Almanac-generated simulations (`sim_tombin-1_nside128_flatTT` and `flatEE`) that are also used in the Almanac project at `~/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs/`.

## Core Architecture

### Single-file implementation

The entire likelihood engine is in **`pixel_likelihood.py`** (~900 lines). The file contains:

1. **Kernel builders** (lines 1-370):
   - `_build_scalar_kernels()`: Spin-0 (TT) kernels using vectorized Legendre recurrence
   - `_compute_spin2_geometry()`: Pixel-pair position angles for spin-2 fields
   - `_build_spin2_kernels()`: Wigner d-matrix kernels (Kp, Km, Kx) for polarization
   - Block builders: `_build_te_block()`, `_build_tb_block()`, `_ee_kernel()`, `_bb_kernel()`, `_eb_kernel()`

2. **SpectraLayout class** (lines 371-470):
   - Maps `(spec_type, bin_i, bin_j, band_b)` to flat parameter index
   - Parameter groups ordered: `TT → TE → TB → EE → BB → EB`
   - Within each group: pairs first, then bands

3. **PixelLikelihood class** (lines 471-893):
   - `from_arrays()`: Construct from numpy arrays (for testing)
   - `build_signal_cov()`: Build signal covariance `C` from bandpower vector
   - `gradient_and_fisher()`: Core likelihood gradient/Fisher computation via Cholesky decomposition
   - `newton_raphson()`: ML bandpower estimation

### Mathematical conventions

- **Likelihood**: `-2 ln L(C_b) = d^T M^{-1} d + ln|M|` where `M = S(C_b) + N`
- **Spin-2 conventions**: Zaldarriaga & Seljak (1997) pixel-pair angles
- **HEALPix E-mode sign**: `a_E = -Re(...)` introduces minus sign in TE block
- **EB/TB kernels**: Only the `d^l_{2,-2}` (Km) kernel with sum angles contributes
- **Band model**: Supports `Cl` (constant C_ℓ per band) or `Dl` (constant D_ℓ per band)

### Memory scaling

**WARNING:** Memory usage is `O(N_d^2)` where `N_d = (n_T + 2*n_P) * n_obs`.

For multi-field polarization with `include_EB=True`:
- Example: `n_P=1`, 7 bands, `n_obs=4900` → ~16 GB kernel storage + ~50 GB peak RAM
- **Mitigation**: Use `NSIDE=32` (n_obs≈1000) or fewer bands if memory-limited

## Development Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

Requires: Python ≥3.9, numpy, scipy, healpy, matplotlib

### Run tests

Full test suite:
```bash
python3 tests/test_general.py     # Multi-field unit tests (gradient FD checks, recovery tests)
python3 tests/test_full_sky_tt.py # TT-only regression tests
```

Individual tests are numbered functions in `test_general.py`:
- Test 1-5: Gradient checks, symmetry, single-bin equivalence
- Test 6-8: Full-sky recovery (TT, EE+BB, joint TT+TE+EE+BB)
- Test 9-10: Multi-bin gradient checks

### Run examples

```bash
python3 examples/run_bjk_tt.py              # TT on Almanac flatTT simulation
python3 examples/run_bjk_ee.py              # EE+BB on Almanac flatEE simulation
python3 examples/test_sky_comparison.py     # BJK vs NaMaster comparison
python3 examples/run_bjk_euclid_ee.py       # Euclid RR2 tomographic bin -1
```

All examples run from repo root and output plots to `examples/`.

### Cross-method comparisons

Scripts in `comparisons/` compare BJK to NaMaster pseudo-C_ℓ estimator:
```bash
python3 comparisons/run_bjk_nmt.py           # Main BJK vs NaMaster driver
python3 comparisons/run_namaster_fsky05_midnoise.py  # NaMaster baseline
python3 comparisons/plot_overlay_nmt.py      # Overlay plot generator
```

### Derivations and verification

```bash
python3 derivations/verify_eb_tb.py   # Monte Carlo verification of EB/TB formulas
python3 derivations/find_te_formula.py # TE kernel derivation
```

## Data

`data/` contains two Almanac-generated simulations (also used in `~/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs/`):
- `sim_tombin-1_nside128_flatTT/` — NSIDE=128, flat TT spectrum (D_ℓ = 1.59×10⁻⁵)
- `sim_tombin-1_nside128_flatEE/` — NSIDE=128, flat EE spectrum

Each directory has:
- `sim_out_channel_0_data.fits` — Observed map (T or Q+U)
- `sim_out_channel_0_nInv.fits` — Noise inverse-variance map
- `flat{TT,EE}_input_cls.dat` — True input power spectrum

These simulations use the Euclid RR2v2.1 fsky≈0.01 mask (~2072 observed pixels out of 196608 total at NSIDE=128). The noise level is σ_pix ≈ 1.6×10⁻³ for temperature maps.

## Status and Known Issues

### Verified spectra
- **TT**: Verified via tests + Almanac simulation
- **EE, BB**: Verified via MC + unit tests
- **TE**: Verified via MC (~5% residual = MC noise floor)
- **TB**: Formula MC-verified; **signs were wrong in earlier versions — now fixed**
- **EB**: Verified (analytic + MC, ~7% residual = MC noise floor)

### Multi-field EB support (June 2026)
**UNTESTED**: The wiring in `run_bjk_nmt.py` for `--include-eb` is written but crashed with OOM before completing a single Newton iteration. Single-field cases (TT, EE/BB without EB) are production-ready.

### Sign conventions

The HEALPix E-mode sign convention `a_E = -Re(...)` is handled in the TE block. Earlier versions had **incorrect signs in TB kernels** — this was fixed after MC verification in `derivations/verify_eb_tb.py`.

## Typical Workflow

1. **Construct likelihood** from FITS files or arrays:
   ```python
   from pixel_likelihood import PixelLikelihood
   
   lik = PixelLikelihood.from_arrays(
       d_T_list=[T_map],
       d_Q_list=[Q_map], d_U_list=[U_map],
       obs_pix=obs_pixels,
       nside=128,
       N_T_list=[noise_T], N_Q_list=[noise_Q], N_U_list=[noise_U],
       lmin=2, lmax=256,
       band_edges=np.array([2, 34, 66, 98, 130, 162, 194, 226, 257]),
       band_model='Dl',  # or 'Cl'
       include_TB=False, include_EB=False
   )
   ```

2. **Run Newton-Raphson** to find ML bandpowers:
   ```python
   cl_init = np.full(lik.layout.n_params, 1e-4)
   cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20)
   ```

3. **Access results**:
   - `cl_ml`: ML bandpower vector (flat, ordered by `lik.layout`)
   - `sigma`: 1σ errors `sqrt(diag(F^{-1}))`
   - `F`: Fisher information matrix
   - `lik.layout.entries()`: Iterator over `(idx, spec, i, j, b)` tuples

## Parameter Vector Layout

The flat `cl_bands` parameter vector is ordered as:
```
[TT(i,j,b) | TE(i,j,b) | TB(i,j,b) | EE(i,j,b) | BB(i,j,b) | EB(i,j,b)]
```

Within each group:
- Pairs `(i,j)` are ordered with `i ≤ j` for auto-spectra (TT, EE, BB, EB)
- For cross-spectra (TE, TB), all pairs `i ∈ [0,n_T)`, `j ∈ [0,n_P)` are included
- Within each pair, bands run `b ∈ [0, nbands)`

Use `lik.layout` to decode indices.

## Additional Documentation

- `docs/bjk_and_approximate_inference.md` — Detailed note on how BJK relates to Laplace approximation, variational inference, and full posterior sampling (written June 2026)
- `~/Desktop/Projects/Almanac/Euclid-Almanac/` — Parent Almanac project with real Euclid data applications and extensive simulation runs
- `~/Desktop/Projects/Almanac/Euclid-Almanac/euclid_rr2_results.md` — Results from Euclid RR2v2.1 LensMC data analysis (June 2026)
- `~/Desktop/Projects/Almanac/Euclid-Almanac/synthesis_almanac_fsky001_june2026.md` — Technical notes on Almanac HMC sampler performance at fsky≈0.01

## Cross-Project File Locations

When working across both repositories:

**BJK code:**
- Main module: `~/Desktop/Projects/bjk-pixlik/pixel_likelihood.py`
- Tests: `~/Desktop/Projects/bjk-pixlik/tests/`

**Almanac usage of BJK:**
- Real Euclid data: `~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs/run_bjk_euclid_ee.py`
- Simulation comparisons: `~/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs/run_bjk_*.py`
- Diagnostic plots: `~/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs/diag_bjk_*.py`

**Note:** Scripts in the Almanac directory import `pixel_likelihood` by adding the BJK repo to `sys.path`, e.g.:
```python
_BJK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'sim_runs', 'bjk')
sys.path.insert(0, _BJK_DIR)
from pixel_likelihood import PixelLikelihood
```
