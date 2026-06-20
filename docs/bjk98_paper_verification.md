# Verification Against BJK98 Paper

**Reference:** Bond, J. R., Jaffe, A. H., & Knox, L. (1998). "Estimating the power spectrum of the cosmic microwave background." *Physical Review D*, 57(4), 2117-2137.  
**DOI:** [10.1103/PhysRevD.57.2117](https://doi.org/10.1103/PhysRevD.57.2117)  
**arXiv:** [astro-ph/9708203](https://arxiv.org/abs/astro-ph/9708203)

---

## Purpose

This document verifies that the current Python implementation (`pixel_likelihood.py`) correctly implements the methodology described in Bond, Jaffe & Knox (1998).

---

## Core Methodology from BJK98

### Paper Abstract (from arXiv)

The paper presents **two computational approaches** for determining the CMB power spectrum C_ℓ:
1. A direct evaluation of the likelihood function
2. An estimator that is a minimum-variance weighted quadratic function of the data

When applied iteratively, these methods converge to identical results. The quadratic approach is "much faster" than direct evaluation while maintaining rigor. Both methods accommodate "datasets with arbitrary chopping patterns and noise correlations."

### Key Equations

The implementation references **BJK98 equations 14-15** in the code comments (`pixel_likelihood.py:10-13`):

**Equation 14 (Gradient):**
```
g_b = ½ [d^T M^{-1} K_b M^{-1} d - Tr(M^{-1} K_b)]
```

**Equation 15 (Fisher Matrix):**
```
F_bb' = ½ Tr[M^{-1} K_b M^{-1} K_b']
```

where:
- `M = S(C_b) + N` is the total covariance (signal + noise)
- `K_b = ∂S/∂C_b` is the derivative kernel for bandpower `b`
- `d` is the data vector
- `g_b` is the gradient of the log-likelihood
- `F_bb'` is the Fisher information matrix

**Newton-Raphson Update:**
```
δC = F^{-1} g
C^{new} = C^{old} + δC
```

---

## Implementation Verification

### 1. Likelihood Function

**BJK98:** `-2 ln L = log|M| + d^T M^{-1} d` (Gaussian likelihood)

**Implementation** (`pixel_likelihood.py:805-814`):
```python
def log_likelihood(self, cl_bands):
    C = self.build_signal_cov(cl_bands)
    M = C + self.N_mat
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf
    logdet = 2.0 * np.log(np.diag(L)).sum()
    y = solve_triangular(L, self.d, lower=True)
    return -0.5 * (logdet + y @ y)
```

**Verification:** ✅ Correct
- Uses Cholesky `M = LL^T` → `log|M| = 2 log|L| = 2 Σ log(L_ii)`
- Computes `d^T M^{-1} d = y^T y` where `Ly = d`
- Returns `-½(log|M| + d^T M^{-1} d)` as required

### 2. Gradient Calculation (BJK98 Eq. 14)

**Implementation** (`pixel_likelihood.py:816-854`):
```python
def gradient_and_fisher(self, cl_bands):
    C = self.build_signal_cov(cl_bands)
    M = C + self.N_mat
    
    L = np.linalg.cholesky(M)
    
    # v = M^{-1} d via two triangular solves
    y = solve_triangular(L,   self.d, lower=True)   # L y = d
    v = solve_triangular(L.T, y,      lower=False)  # L^T v = y
    
    # Build A_b = L^{-1} K_b (L^{-1})^T for each kernel
    A = [None] * np_total
    for idx in range(np_total):
        K_b = self._kernel_matrices[idx]
        W   = solve_triangular(L,   K_b,   lower=True)        # L W = K_b
        A[idx] = solve_triangular(L, W.T, lower=True).T       # L A^T = W^T
    
    g = np.array([
        0.5 * (v @ (self._kernel_matrices[b] @ v) - np.trace(A[b]))
        for b in range(np_total)
    ])
```

**Verification:** ✅ Correct

The code computes:
- `v = M^{-1} d` via Cholesky solves
- `A_b = M^{-1} K_b = L^{-1} K_b (L^{-1})^T`
- `g_b = ½ [v^T K_b v - Tr(A_b)]`

which is mathematically equivalent to BJK98 Eq. 14:
- `v^T K_b v = d^T M^{-1} K_b M^{-1} d` ✓
- `Tr(A_b) = Tr(M^{-1} K_b)` ✓

### 3. Fisher Matrix (BJK98 Eq. 15)

**Implementation** (`pixel_likelihood.py:848-852`):
```python
F = np.zeros((np_total, np_total))
for b in range(np_total):
    for bp in range(b, np_total):
        val = 0.5 * (A[b] * A[bp].T).sum()
        F[b, bp] = F[bp, b] = val
```

**Verification:** ✅ Correct

This computes:
```
F_bb' = ½ Tr[A_b A_b'^T] = ½ (A_b ⊙ A_b'^T).sum()
```

where `A_b = M^{-1} K_b`, which is equivalent to BJK98 Eq. 15:
```
F_bb' = ½ Tr[M^{-1} K_b M^{-1} K_b']
```

The identity `Tr[M^{-1} K_b M^{-1} K_b'] = Tr[A_b A_b'^T]` holds because:
- `M^{-1} = (L^{-1})^T L^{-1}`
- `M^{-1} K_b = A_b`
- `M^{-1} K_b' = A_b'`
- Cyclic property of trace

### 4. Newton-Raphson Iteration

**BJK98:** Iterate `C^{i+1} = C^i + F^{-1} g` until convergence

**Implementation** (`pixel_likelihood.py:856-889`):
```python
def newton_raphson(self, cl_init, max_iter=15, tol=1e-4, damp=1.0):
    cl = np.array(cl_init, dtype=float)
    
    for it in range(max_iter):
        try:
            g, F = self.gradient_and_fisher(cl)
            delta = np.linalg.solve(F, g)
        except np.linalg.LinAlgError as e:
            break
        cl_new = cl + damp * delta
        step   = np.max(np.abs(delta) / (np.abs(cl) + 1e-40))
        logL   = self.log_likelihood(cl_new)
        cl = cl_new
        if step < tol:
            break
    
    _, F = self.gradient_and_fisher(cl)
    sigma = np.sqrt(np.maximum(np.diag(np.linalg.inv(F)), 0))
    
    return cl, sigma, F
```

**Verification:** ✅ Correct
- Implements `δ = F^{-1} g` via `np.linalg.solve(F, g)`
- Updates `C ← C + damp × δ` with optional damping
- Converges when `max|δ/C| < tol`
- Returns ML bandpowers, 1σ errors (`√diag(F^{-1})`), and Fisher matrix

### 5. Cholesky-Based Numerical Stability

**BJK98:** The paper discusses computational efficiency and numerical stability.

**Implementation:** Uses Cholesky decomposition `M = LL^T` throughout instead of explicit matrix inversion:
- `M^{-1} d` computed via triangular solves `L^T v = y, Ly = d`
- `M^{-1} K_b` computed as `A_b = L^{-1} K_b (L^{-1})^T`
- Never forms `M^{-1}` explicitly

**Verification:** ✅ Best practice
- Avoids numerical instability of direct inversion
- `O(N^3)` Cholesky + `O(N^2)` solves vs `O(N^3)` inversion
- Matches historical Fortran implementation (introduced March 1997 in `diag_StoNmat_pspec.f`)

---

## Extensions Beyond BJK98

The current implementation includes several extensions not present in the 1998 paper:

### 1. Multi-Field Tomography (n_T × n_P bins)

**BJK98:** Single temperature field

**Implementation:** Arbitrary combinations of:
- `n_T` spin-0 fields (temperature)
- `n_P` spin-2 fields (polarization Q+U)

Supports all cross-spectra: TT, TE, TB, EE, BB, EB

### 2. Spin-2 Polarization Kernels

**BJK98:** Scalar (spin-0) only

**Implementation:** Full Wigner d-matrix kernels for polarization:
- Uses Zaldarriaga & Seljak (1997) spin-2 conventions
- Handles HEALPix E-mode sign convention (`a_E = -Re(...)`)
- Verified via Monte Carlo (see `derivations/verify_eb_tb.py`)

### 3. HEALPix Integration

**BJK98:** General pixelization scheme

**Implementation:** Native HEALPix support via `healpy`
- Pixel-pair geometry via `hp.pix2vec()`
- Exploits pixelization regularity for efficiency
- Partial-sky masks via observed pixel indices

### 4. Band Model Flexibility

**BJK98:** Standard bandpower model

**Implementation:** Supports two parametrizations:
- `band_model='Cl'`: Constant C_ℓ per band
- `band_model='Dl'`: Constant D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) per band

### 5. Automated Testing

**BJK98:** Validated against COBE DMR and Saskatoon

**Implementation:** Comprehensive test suite:
- Gradient finite-difference checks
- Full-sky recovery tests (TT, EE+BB, joint)
- Multi-bin validation
- Monte Carlo verification of spin-2 kernels

---

## Consistency with Historical Implementations

The Fortran codes in `~/home/cmb/` (1995-1999) also implement BJK98:

**`quadest.f` (1997):**
```fortran
C     If \delta a_p is the correction to the input guess, a_p, 
C     and A_p \equiv {C^{-1}C_T,p} (where the ,p indicates 
C     differentiation wrt parameter a_p) then
C     \delta a_p = 0.5 F^{-1}_{pp'}Tr[X \Delta^T A_{p'} - A_{p'}]
c     with X = C^-1 Delta
C     The Fisher matrix is 1/2 Tr[A_p A_p'].
```

This matches equations 14-15 exactly, confirming the Python implementation preserves the original methodology.

---

## Computational Complexity

**BJK98:** Discusses `O(N^2)` approximations for large datasets

**Implementation:** Full `O(N_d^3)` per iteration
- Dominated by Cholesky decomposition and triangular solves
- `O(n_params × N_d^3)` total for all kernel operations
- Warns about memory scaling (`O(N_d^2)` for kernel storage)
- Suggests future on-the-fly kernel construction for memory reduction

---

## Summary

The current `pixel_likelihood.py` implementation is **faithful to BJK98**:

✅ **Equations 14-15** implemented correctly  
✅ **Cholesky-based** numerical stability  
✅ **Newton-Raphson** iteration as described  
✅ **Fisher matrix** for parameter uncertainties  
✅ **Gaussian likelihood** form preserved  

**Extensions** beyond the original paper are well-documented and tested:
- Multi-field tomography
- Spin-2 polarization (EE, BB, EB, TE, TB)
- HEALPix integration
- Modern Python/NumPy/SciPy stack

The implementation preserves the mathematical core while modernizing the code structure, generalizing to polarization, and adding comprehensive testing.

---

## References

1. Bond, J. R., Jaffe, A. H., & Knox, L. (1998). "Estimating the power spectrum of the cosmic microwave background." *Physical Review D*, 57(4), 2117-2137. doi:10.1103/PhysRevD.57.2117

2. Zaldarriaga, M., & Seljak, U. (1997). "An all-sky analysis of polarization in the microwave background." *Physical Review D*, 55(4), 1830. (Spin-2 harmonic conventions)

3. Historical Fortran implementations (1995-1999): `~/home/cmb/quadest/`, `~/home/cmb/code/`, `~/home/cmb/bandpow/`
