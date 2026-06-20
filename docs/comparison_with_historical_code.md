# Comparison: bjk-pixlik (2026) vs Historical Fortran Implementations (1995-1999)

**Author:** A. Jaffe  
**Date:** June 2026

---

## Overview

This document compares the current Python implementation (`pixel_likelihood.py`) with the original Fortran 77 implementations from the 1990s, located in:
- `~/home/cmb/quadest/quadest.f` (1997)
- `~/home/cmb/code/diag_StoNmat_pspec.f` (1995-1997)
- `~/home/cmb/code/dmr_diag_StoNmat_pspec.f` (1997-1999)
- `~/home/cmb/bandpow/bandpow_main.f` (1999)

---

## Mathematical Core (Unchanged)

Both implementations solve the same maximum-likelihood problem:

**Likelihood:**
```
-2 ln L(C_b) = d^T M^{-1} d + ln|M|
M = S(C_b) + N
```

**Newton-Raphson iteration:**
```
g_b  = ½ [d^T M^{-1} K_b M^{-1} d - Tr(M^{-1} K_b)]
F_bb' = ½ Tr[M^{-1} K_b M^{-1} K_b']
δC_b = F^{-1} g
```

**Cholesky decomposition approach:**
Both use `M = L L^T` via Cholesky instead of explicit matrix inversion:
```
g_b  = ½ [v^T K_b v - Tr(A_b)]     where v = M^{-1} d, A_b = L^{-1} K_b L^{-T}
F_bb' = ½ Tr[A_b A_b'] = ½ (A_b ⊙ A_b'^T).sum()
```

---

## Implementation Comparison

### Structure and Organization

| Aspect | Fortran (1995-1999) | Python (2026) |
|--------|---------------------|---------------|
| **Lines of code** | ~2000-4000 per program | ~900 (single file) |
| **Language** | Fortran 77 | Python 3.9+ |
| **Modularity** | Multi-file + shell scripts | Single self-contained module |
| **Parameter management** | Flat arrays + manual indexing | `SpectraLayout` class |
| **Linear algebra** | Direct LAPACK calls | `scipy.linalg` wrappers |
| **Iteration control** | Shell scripts + file I/O | Function returns |

### Fortran Code Structure (quadest.f)

```fortran
program quadest
  include 'quadest.h'
  
  ! Parameters
  parameter (npmax=40)    ! max number of parameters
  parameter (ndmax=3000)  ! max number of data points
  
  ! Core arrays
  double precision C(ndmax,ndmax), fish(npmax,npmax)
  double precision X(ndmax), delta(ndmax)
  
  ! Read iteration state from files
  read(*,*) niter
  read(*,'(a)') Apbase, CTbase, data_file, CNfile
  
  ! Iteration loop (external, via shell script)
  ! Calculate X = C^{-1} Delta via Cholesky
  ! Calculate A_p = C^{-1} CT,p for each parameter
  ! Calculate Fisher matrix
  ! Invert Fisher and get delta_ap
  ! Write new C_l to file for next iteration
```

**Key patterns:**
- Explicit file I/O at every step (`Apbasexxx`, `cl_filebasexxx`)
- Shell scripts manage iteration loop (`run_quadest.sh`)
- Data normalization factors hardcoded (`dtnorm2_1=1.0d12`)
- Manual memory management with `parameter` statements

### Python Code Structure (pixel_likelihood.py)

```python
class PixelLikelihood:
    def __init__(self, ...):
        self.layout = SpectraLayout(n_T, n_P, nbands, ...)
        self._kernel_matrices = self._build_all_kernels()
        self.N_mat = np.diag(self.N_diag)
    
    def gradient_and_fisher(self, cl_bands):
        C = self.build_signal_cov(cl_bands)
        M = C + self.N_mat
        L = np.linalg.cholesky(M)
        
        # v = M^{-1} d
        y = solve_triangular(L, self.d, lower=True)
        v = solve_triangular(L.T, y, lower=False)
        
        # A_b = L^{-1} K_b L^{-T}
        A = [solve_triangular(L, solve_triangular(L, K_b, lower=True).T, 
                              lower=True).T 
             for K_b in self._kernel_matrices]
        
        g = [0.5 * (v @ K_b @ v - np.trace(A_b)) 
             for K_b, A_b in zip(self._kernel_matrices, A)]
        F = [[0.5 * (A_b * A_bp.T).sum() 
              for A_bp in A] for A_b in A]
        
        return g, F
    
    def newton_raphson(self, cl_init, max_iter=15):
        cl = cl_init.copy()
        for it in range(max_iter):
            g, F = self.gradient_and_fisher(cl)
            delta = np.linalg.solve(F, g)
            cl += delta
            if converged: break
        return cl, sigma, F
```

**Key improvements:**
- Iteration loop internalized
- No file I/O during iteration
- Automatic parameter layout via `SpectraLayout`
- Memory management via NumPy
- Cleaner separation of concerns

---

## Kernel Construction

### Fortran: Explicit Loops

From `diag_StoNmat_pspec.f` (simplified):
```fortran
c Build signal covariance from C_l
do i=1,npts
  do j=1,npts
    cost = cos(theta_ij)
    CTmat(i,j) = 0.0d0
    do L=Lmin,Lmax
      PL = Legendre_L(cost)
      CTmat(i,j) = CTmat(i,j) + (2*L+1)/(4*pi) * CL(L) * PL
    enddo
  enddo
enddo
```

### Python: Vectorized with Unique Values

From `pixel_likelihood.py`:
```python
def _build_scalar_kernels(obs_pix, nside, lmin, lmax, band_edges, beam2):
    vec = hp.pix2vec(nside, obs_pix)
    cos_theta_full = (vec @ vec.T).ravel()
    unique_ct, inverse = np.unique(cos_theta_full, return_inverse=True)
    
    # Legendre recurrence only on unique cos θ values
    kernels = [np.zeros(len(unique_ct)) for _ in range(nbands)]
    
    Pl_prev, Pl_curr = np.ones_like(unique_ct), unique_ct.copy()
    for l in range(2, lmax + 1):
        Pl_next = ((2*l - 1) * unique_ct * Pl_curr - (l - 1) * Pl_prev) / l
        # Add to appropriate band
        b = searchsorted(band_edges, l) - 1
        if 0 <= b < nbands:
            kernels[b] += (2*l + 1) / (4*np.pi) * beam2[l] * Pl_next
        Pl_prev, Pl_curr = Pl_curr, Pl_next
    
    # Expand unique values back to full n×n matrix
    return [k[inverse].reshape(n, n) for k in kernels]
```

**Optimization:** Exploits HEALPix pixelization regularity — many pixel pairs share the same `cos θ`, so the Legendre recurrence runs only on unique values.

---

## Data Handling

### Fortran: Normalized Units

```fortran
c Normalization: dtnorm=1.0d-6
c   delta*dtnorm = DT/T
c   CNmat*dtnorm**2 = <(DT/T)^2>
c   CTmat in units of <(DT/T)^2> with sigma_8=1
parameter (dtnorm2_1=1.0d12)
```

### Python: Physical Units

```python
# Data in observed units (K_CMB or dimensionless shear)
# Noise variance in same units squared
# C_l in appropriate physical units (K^2 or rad^2)
# No internal normalization factors
```

---

## Spin-2 (Polarization) Support

### Fortran: Not Present

The original codes (`quadest.f`, `diag_StoNmat_pspec.f`) were temperature-only. Polarization support came later in specialized codes.

### Python: Full Multi-Field

```python
class PixelLikelihood:
    def __init__(self, n_T, n_P, include_TB=False, include_EB=False):
        # n_T spin-0 fields (temperature)
        # n_P spin-2 fields (polarization Q+U)
        # Supports TT, TE, TB, EE, BB, EB
        
        # Wigner d-matrix kernels for spin-2
        Kp, Km, Kx, geom = _build_spin2_kernels(...)
        
        # Position angle geometry
        cos2dpsi, sin2dpsi, cos2spsi, sin2spsi, c2j, s2j = geom
        
        # EE kernel: QQ = Kp*cos2dp + Km*cos2sp, etc.
        # BB kernel: QQ = Kp*cos2dp - Km*cos2sp, etc.
        # EB kernel: QQ = -Km*sin2sp, QU = Km*cos2sp, etc.
```

**References:**
- Zaldarriaga & Seljak (1997) for spin-2 harmonic conventions
- HEALPix E-mode sign convention handled explicitly

---

## Memory Usage

### Fortran: Compile-Time Limits

```fortran
parameter (npmax=40)    ! max 40 parameters
parameter (ndmax=3000)  ! max 3000 data points
```

Fixed at compile time. To change, must recompile.

### Python: Dynamic Allocation

```python
# Allocates based on actual problem size
N_d = (n_T + 2*n_P) * n_obs
```

**Memory scaling:** Both implementations are `O(N_d^2)`, but Python version explicitly warns about memory limits and suggests mitigations.

---

## Testing and Validation

### Fortran: External Validation

- Comparison with published results (COBE DMR, ground-based experiments)
- Manual verification via separate analysis scripts
- No unit tests

### Python: Comprehensive Test Suite

```bash
python3 tests/test_general.py     # 10 unit tests
python3 tests/test_full_sky_tt.py # Regression tests
```

Tests include:
- Gradient finite-difference checks
- Signal covariance symmetry
- Full-sky recovery tests
- Multi-bin validation
- Monte Carlo verification of EB/TB formulas

---

## Performance Considerations

### Fortran (1995-1999):
- **Pros:** Compiled, optimized BLAS/LAPACK, minimal overhead
- **Cons:** Manual memory management, fixed array sizes, shell script coordination
- **Typical runtime:** Minutes to hours on 1990s workstations

### Python (2026):
- **Pros:** Vectorized NumPy (calls same BLAS/LAPACK), cleaner code, easier debugging
- **Cons:** Python interpreter overhead (minor for matrix operations)
- **Typical runtime:** Seconds to minutes on modern hardware
- **Bottleneck:** Kernel construction and Cholesky decomposition (same as Fortran)

---

## Migration Path and Lessons Learned

### What was preserved:
1. **Mathematical core** — Newton-Raphson with Cholesky-based gradient/Fisher
2. **Algorithmic structure** — Iterate on bandpowers, not individual C_ℓ
3. **Numerical stability** — Use of Cholesky over explicit matrix inversion

### What was modernized:
1. **Language and ecosystem** — Fortran 77 → Python 3 with NumPy/SciPy/healpy
2. **Data structures** — Fixed arrays → dynamic allocation, manual indexing → `SpectraLayout`
3. **Generalization** — Single-field TT → arbitrary n_T × n_P tomographic bins
4. **Testing** — Manual validation → automated unit tests + MC verification
5. **Documentation** — Inline comments → structured docstrings + separate markdown docs

### Design decisions:
- **Single-file implementation** — Prioritizes readability and self-containment over modularity
- **Pre-build all kernels** — Trades memory for speed (alternative: build on-the-fly)
- **No explicit iteration files** — Return values instead of file I/O
- **HEALPix native** — Direct integration vs custom pixelization schemes

---

## Code Archaeology Notes

### Fortran Code Comments (Preserved History)

From `diag_StoNmat_pspec.f`:
```fortran
c AHJ 01/97 all real*4->real, real*8->double precision
c AHJ 11/20/96 for LK Ebeta case, need to transform data by L_Bk matrix
c AHJ 1-2/97 New get_theory1_diag uses LAPACK for eigenstystem
c AHJ 3/97 New get_theory1_diag_ch uses Cholesky instead of CD^-1/2
```

**Translation:** The Cholesky-based approach (currently used in both implementations) was introduced in March 1997 as an improvement over eigenvalue decomposition.

### Shell Script Coordination (quadest)

```bash
for niter in $niterlist; do
  # Calculate CT matrix
  ${CT_prog} < ${data_file} > ${CTbase}${niter}
  
  # Calculate derivative kernels (first iteration only)
  if [ $niter -eq 1 ]; then
    for iband in $bandlist; do
      ${CT_prog} < ${Cderivbase}${iband} > ${Apbase}${iband}
    done
  fi
  
  # Run quadratic estimator
  quadest < input_params > output_${niter}
done
```

**Modern equivalent:** Entire loop internalized in `PixelLikelihood.newton_raphson()`.

---

## Conclusion

The current `pixel_likelihood.py` is a faithful reimplementation of the 1990s Fortran codes with modern best practices:

- **Mathematical equivalence** maintained
- **Numerical stability** preserved (Cholesky-based approach)
- **Generalization** to multi-field tomography
- **Usability** vastly improved (single file, no shell scripts, automatic iteration)
- **Testing** formalized (unit tests, MC verification)
- **Integration** with modern tools (HEALPix, Almanac, NaMaster comparisons)

The Fortran implementations remain in `~/home/cmb/` as archival reference and historical context for the BJK98 method.
