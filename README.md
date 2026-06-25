# bjk-pixlik

Pixel-space CMB power spectrum likelihood following Bond, Jaffe & Knox (BJK98).

Estimates bandpowers for arbitrary combinations of spin-0 (T) and spin-2 (Q+U)
tomographic bins, including TT, TE, TB, EE, BB, and EB cross-spectra, using
an exact pixel-space Gaussian likelihood with Newton-Raphson maximisation.

## Status

| Spectrum | Status |
|----------|--------|
| TT       | Verified (tests + Almanac simulation) |
| EE, BB   | Verified (MC + unit tests) |
| TE       | Verified (MC, ~5% residual = MC noise floor) |
| TB       | Formula MC-verified (`derivations/verify_eb_tb.py`); **signs were wrong in earlier versions — fixed** |
| EB       | Verified (analytic derivation + MC, ~7% residual = MC noise floor). Uses Km kernel with sum angles: QQ=−Km·sin2Σψ, QU=+Km·cos2Σψ, UU=+Km·sin2Σψ. See `derivations/verify_eb_tb.py`. |

## Background

The likelihood is

```
-2 ln L(C_b) = d^T M^{-1} d + ln|M|,   M = S(C_b) + N
```

where `S` is the pixel-space signal covariance built from Wigner d-matrix kernels
and `N` is the (diagonal) noise covariance. Gradient and Fisher matrix are computed
via Cholesky decomposition without ever forming M⁻¹ explicitly:

```
M = L L^T
g_b = ½ [ v^T K_b v − Tr(L^{-1} K_b (L^{-1})^T) ]
F_{bb'} = ½ Tr[A_b A_{b'}]  where A_b = L^{-1} K_b (L^{-T})
```

The spin-2 kernels use the Zaldarriaga & Seljak (1997) pixel-pair angle conventions.
The HEALPix E-mode sign convention (a_E = −Re(…)) introduces a minus sign in the
TE block relative to the naive Wigner theory formula.

## Installation

Editable install (recommended — one source of truth across projects):

```
pip install -e .
```

This installs the `pixel_likelihood` module and its core dependencies (numpy,
scipy, healpy). Optional extras:

```
pip install -e ".[examples]"   # matplotlib, for examples/ and comparison plots
pip install -e ".[compare]"    # pymaster, for NaMaster cross-checks
pip install -e ".[sampler]"    # blackjax/jax, for the C_l sampler prototype
pip install -e ".[dev]"        # everything above + pytest
```

Requires Python ≥ 3.9. (`requirements.txt` is retained for the example-script
dependency set; the package metadata in `pyproject.toml` is authoritative.)

## Tests

Each test file is a self-contained runner that exits non-zero on failure:

```
python3 tests/test_general.py          # multi-field unit tests (incl. gradient FD)
python3 tests/test_full_sky_tt.py      # TT regression vs reference
python3 tests/test_band_model.py       # Cl vs Dl handling (spin-0 and spin-2)
python3 tests/test_newton_boundary.py  # Newton-Raphson near the EB PD boundary
```

## Usage

### Run examples

```bash
python3 examples/run_bjk_tt.py   # TT on Almanac flatTT simulation
python3 examples/run_bjk_ee.py   # EE+BB on Almanac flatEE simulation
python3 examples/test_sky_comparison.py  # BJK vs NaMaster comparison
```

### Run tests

```bash
python3 tests/test_full_sky_tt.py   # TT regression tests
python3 tests/test_general.py       # full multi-field unit tests
```

### Construct a likelihood from arrays

```python
from pixel_likelihood import PixelLikelihood

lik = PixelLikelihood.from_arrays(
    d_T_list=[T_map],        # list of n_T temperature maps (observed pixels only)
    d_Q_list=[Q_map],        # list of n_P Q maps
    d_U_list=[U_map],        # list of n_P U maps
    obs_pix=obs_pixels,      # HEALPix pixel indices of observed pixels
    nside=128,
    N_T_list=[noise_T],      # per-pixel noise variance for each T map
    N_Q_list=[noise_Q],      # per-pixel noise variance for each Q map
    N_U_list=[noise_U],
    lmin=2, lmax=256,
    band_edges=np.array([2, 34, 66, 98, 130, 162, 194, 226, 257]),
    n_T=1, n_P=1,
)

cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20)
```

The parameter vector `cl_bands` is ordered as: TT bands, TE bands, EE bands,
BB bands (for n_T=n_P=1 without TB/EB). Use `lik.layout` for the full index map.

## Data

The `data/` directory contains two Almanac simulations:

- `sim_tombin-1_nside128_flatTT/` — NSIDE=128, flat TT spectrum, single bin
- `sim_tombin-1_nside128_flatEE/` — NSIDE=128, flat EE spectrum, single bin

These are HEALPix FITS maps covering ≈1% of the sky (Euclid RR2 tomographic bin −1).

## Key references

- Bond, Jaffe & Knox (1998), PhysRevD 57, 2117 — https://doi.org/10.1103/PhysRevD.57.2117
- Zaldarriaga & Seljak (1997): spin-2 harmonic conventions (App. A)
- HEALPix: https://healpix.sourceforge.io
