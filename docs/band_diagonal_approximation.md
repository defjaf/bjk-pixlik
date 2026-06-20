# Band-Diagonal Approximation in CMB Power Spectrum Estimation

## Overview

The **band-diagonal approximation** was a computational strategy used in 1990s implementations to reduce the `O(N^3)` cost of full covariance matrix operations. The current `pixel_likelihood.py` does **not** use this approximation — it works with the full `O(N_d^2)` covariance matrix.

This document explains what the approximation was, why it was needed historically, and why modern implementations can avoid it.

---

## The Problem: Computational Cost

### Full Covariance Matrix

For `N` data points, the likelihood involves:

```
-2 ln L = log|M| + d^T M^{-1} d
M = S(C_ℓ) + N
```

where `M` is an `N × N` covariance matrix.

**Operations:**
- Cholesky decomposition: `O(N^3)`
- Matrix-vector products: `O(N^2)`
- Total per Newton iteration: `O(n_params × N^3)`

**Memory:**
- Storing `M`: `O(N^2)`
- Storing all derivative kernels `K_b`: `O(n_params × N^2)`

### Scaling in the 1990s

**COBE DMR (1992-1999):**
- 6144 pixels (full sky, NSIDE=16)
- Multiple channels → ~1500-3000 data points
- `N^2 ≈ 9M` doubles ≈ 72 MB (manageable)
- `N^3 ≈ 27B` operations (minutes to hours on 1990s workstations)

**Ground-based experiments (1990s):**
- Saskatoon, Python V, etc.
- Hundreds to thousands of data points
- Similar scaling challenges

**Constraint:** 1990s workstations had ~100 MB - 1 GB RAM and much slower CPUs. Full matrix operations were expensive but possible for these dataset sizes.

---

## The Band-Diagonal Approximation

### Core Idea

The signal covariance matrix `S` can be approximated as **band-diagonal** in a transformed basis where correlations are localized.

### Mathematical Approach (from Fortran code comments)

From `diag_StoNmat_pspec.f:13-27` (Lloyd Knox implementation, 1996):

1. **Eigenmode decomposition** of the full signal+noise covariance:
   ```
   M = R Λ R^T
   ```
   where `R` is the rotation matrix and `Λ` is diagonal (eigenvalues).

2. **Band-limited signal covariance:**
   - Transform signal covariance to eigenmode basis
   - Keep only eigenmodes in a specific ℓ-band: `E_Bkk'`
   - Approximate other modes with a "prior noise" `σ*^2`

3. **Reduced problem:**
   ```
   C_n = 1 + σ*^2 (E_k - E_Bkk') + σ_th^2 E_Bkk'
   ```
   Only the band `B` of interest is treated exactly; other bands are regularized.

4. **Result:** The effective covariance in the rotated basis is band-diagonal, reducing storage and computational cost.

### What This Achieves

**Before (full matrix):**
- Store: `N × N` full matrix
- Compute: `O(N^3)` Cholesky decomposition

**After (band-diagonal):**
- Store: `N × bandwidth` (if bandwidth `<< N`)
- Compute: `O(N × bandwidth^2)` for banded Cholesky

**Example:**
- `N = 2000`, bandwidth `= 100`
- Storage: 2000² = 4M → 2000×100 = 200K (20× reduction)
- Compute: 8B ops → 20M ops (400× speedup)

---

## Why the Approximation Was Used

### 1. Memory Constraints (Primary)

**1990s workstations:**
- 64 MB - 512 MB RAM typical
- `N = 3000`: Full matrix = 72 MB (tight)
- Multiple derivative kernels × 72 MB = out of memory

**Band-diagonal:**
- Could fit larger problems in available RAM
- Enabled analysis of multiple datasets simultaneously

### 2. Computational Speed (Secondary)

**Full matrix:**
- COBE DMR analysis: hours per iteration on 1990s workstations
- Multiple iterations + multiple parameter sets = days of compute

**Band-diagonal:**
- Could reduce iteration time from hours to minutes
- Made parameter exploration tractable

### 3. Physical Motivation

**ℓ-space localization:**
- CMB power spectrum estimation focuses on specific ℓ-ranges
- Correlations between widely separated ℓ-modes are weak
- Band-diagonal approximation exploits this structure

---

## Why Modern Code Doesn't Use It

### 1. Hardware Improvements

**Modern workstations (2020s):**
- 32 GB - 128 GB RAM typical
- Multi-core CPUs with SIMD
- Optimized BLAS/LAPACK libraries

**Scaling:**
- Euclid RR2 fsky≈0.01, NSIDE=128: ~2000 observed pixels
- Single field: `N_d = 2000`, `N_d^2 = 4M` doubles = 32 MB ✓
- Multi-field (n_P=1): `N_d = 4000`, `N_d^2 = 16M` doubles = 128 MB ✓
- Modern machines handle this comfortably

### 2. Simplicity and Correctness

**Full matrix advantages:**
- **Exact:** No approximation errors
- **General:** Works for any sky coverage, any experiment
- **Simpler code:** No special-case band handling
- **Easier to debug:** Direct implementation of math

**Band-diagonal disadvantages:**
- **Approximation error:** Must validate per dataset
- **Complexity:** Requires eigenmode basis, band selection
- **Rigid:** Band structure may not match actual correlations
- **Code complexity:** Special linear algebra routines

### 3. When Memory Becomes an Issue

The current code **does** hit memory limits for very large problems:

**Example (from `pixel_likelihood.py:39-44`):**
```
n_P=1, include_EB=True, 7 bands, n_obs=4900:
  → 21 kernels × (2×4900)² × 8 bytes ≈ 16 GB kernel storage
  → Peak RAM ≈ 50 GB (with temporaries)
```

**Modern solution:** Reduce problem size
- Use `NSIDE=32` instead of `NSIDE=64` (4× fewer pixels)
- Fewer bands
- Or future: on-the-fly kernel construction (mentioned in code comments)

**Not used:** Band-diagonal approximation
- Would add complexity
- Approximation error unknown for modern datasets
- Better to reduce problem size explicitly

---

## Historical Implementation Details

### From Fortran Code (`diag_StoNmat_pspec.f`)

**Flag:** `idiag_band=1` activates band-diagonal mode

**Process:**
```fortran
C     read E_bkk'=R CD^-1/2 CT_band CD^-1/2 R
C          E_k, \xi_k (full, non-banded, s.t. diag(E_k)=\sum_B E_Bkk'
c          sig_*="prior noise amplitude" for complement bands
C     do second diag: 1+E_k-> C_n=1+sig*^2(E_k-E_Bkk')
C                            +C_T=sig_th^2 E_Bkk'
```

**Interpretation:**
1. Pre-compute eigenmode basis `R` for the full covariance
2. Read in band-specific covariance `E_Bkk'` in that basis
3. Regularize other bands with prior noise `σ*^2`
4. Work in the transformed, approximately-diagonal basis

### Example Usage (from `run_quadest.sh`)

```bash
case $nband in 
  10) bandlist="001 002 003 004 005 006 007 008 009 010"
      ellminmaxlist="19 66 @67 114 @115 162 ...";;
  28) bandlist="001 002 003 004 ... 028"
      ellminmaxlist="2 2 @3 3 @4 4 @5 5 ...";;
esac
```

The code would:
- Iterate over ℓ-bands
- For each band, construct approximate covariance
- Estimate bandpower for that range
- Move to next band

This is **not** the same as the band-diagonal approximation per se, but shows the band-by-band processing strategy.

---

## Comparison: Band-Diagonal vs Full Matrix

| Aspect | Band-Diagonal (1990s) | Full Matrix (Current) |
|--------|----------------------|---------------------|
| **Memory** | `O(N × bandwidth)` | `O(N^2)` |
| **Compute** | `O(N × bandwidth^2)` | `O(N^3)` |
| **Accuracy** | Approximate | Exact |
| **Complexity** | High (eigenmode basis, regularization) | Low (direct implementation) |
| **Generality** | Requires careful setup per experiment | Works for any dataset |
| **Modern use** | Obsolete for moderate N | Standard |
| **When needed** | Very large N (> 10,000?) | N < ~5000 fits in RAM |

---

## Modern Alternatives to Band-Diagonal

If memory becomes prohibitive for a specific problem, modern approaches include:

### 1. Reduced NSIDE
- Downsample HEALPix map from NSIDE=128 to NSIDE=64 or 32
- Reduces `n_obs` by 4× or 16×
- Reduces memory by 16× or 256×

### 2. On-the-Fly Kernel Construction
```python
# Instead of:
self._kernel_matrices = [build_kernel(b) for b in range(n_params)]

# Do (future):
def gradient_and_fisher(self, cl_bands):
    for b in range(n_params):
        K_b = build_kernel(b)  # Build when needed
        A_b = solve_triangular(L, solve_triangular(L, K_b, ...), ...)
        del K_b  # Discard immediately
```

**Memory:** `O(N^2)` instead of `O(n_params × N^2)`  
**Cost:** Rebuild kernels `O(n_params × 20)` times (once per Newton iteration)

### 3. Sparse Methods
- For very large, truly sparse covariance matrices
- Not applicable to CMB (signal covariance is dense)

### 4. Low-Rank Approximations
- Signal covariance `S ≈ U S_reduced U^T` where `U` is `N × k`, `k << N`
- Woodbury matrix identity: `(S + N)^{-1} = N^{-1} - N^{-1} U (...) U^T N^{-1}`
- Reduces cost to `O(k^2 N)` instead of `O(N^3)`
- Modern research direction (not in current code)

---

## Summary

**Band-diagonal approximation:**
- Computational strategy from 1990s
- Reduced `O(N^3)` → `O(N × bandwidth^2)` via eigenmode basis + regularization
- Enabled COBE DMR analysis on 1990s hardware
- **Not used in current Python implementation**

**Current approach:**
- Full `O(N^2)` memory, `O(N^3)` compute
- Feasible for modern problems (N ~ 2000-5000)
- Simpler, exact, general

**When memory is limiting:**
- Reduce NSIDE (preferred)
- Future: on-the-fly kernels
- Not: band-diagonal approximation (too complex, adds approximation error)

The shift from band-diagonal to full-matrix reflects **30 years of hardware improvement** enabling exact methods that were previously impractical.
