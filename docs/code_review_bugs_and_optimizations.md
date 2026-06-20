# Code Review: Bugs and Optimization Opportunities

**Date:** June 2026  
**Reviewer:** Claude Sonnet 4.5  
**Context:** Euclid fsky≈0.01 multi-field runs were hitting memory limits

---

## Executive Summary

**Bugs Found:** 0 critical, 1 potential numerical issue  
**Memory Bottleneck:** Kernel pre-allocation dominates (16+ GB for multi-field)  
**Primary Recommendation:** Implement on-the-fly kernel construction  
**Secondary Optimizations:** 5 identified, ranging from easy to moderate difficulty

---

## Part 1: Bug Analysis

### 1.1 Critical Bugs

**✅ None found.** The code correctly implements BJK98 equations 14-15.

### 1.2 Potential Numerical Issue

**Location:** `pixel_likelihood.py:848-852`

```python
F = np.zeros((np_total, np_total))
for b in range(np_total):
    for bp in range(b, np_total):
        val = 0.5 * (A[b] * A[bp].T).sum()
        F[b, bp] = F[bp, b] = val
```

**Issue:** Element-wise multiplication `A[b] * A[bp].T` creates full N_d × N_d temporary array.

**Memory Impact:** Temporary `N_d² × 8` bytes per Fisher element  
- Example: N_d=4000 → 128 MB temporary per element
- Total Fisher: ~200 elements → repeated allocation/deallocation

**Severity:** Low (doesn't cause OOM, but adds GC pressure)

**Fix:**
```python
# Option 1: Use einsum (cleaner, same performance)
val = 0.5 * np.einsum('ij,ji->', A[b], A[bp])

# Option 2: Trace formulation (equivalent)
val = 0.5 * np.trace(A[b] @ A[bp].T)

# Option 3: Vectorize (faster for small matrices)
val = 0.5 * np.sum(A[b] * A[bp].T)  # same as current
```

**Recommendation:** Switch to `einsum` for clarity, or leave as-is (working correctly).

---

## Part 2: Memory Analysis

### 2.1 Memory Breakdown (Euclid fsky≈0.01, n_obs≈2000)

**Single-field TT (n_T=1, n_P=0):**
```
n_obs     = 2000
N_d       = 2000
nbands    = 8
n_params  = 8 (TT only)

Data vectors:
  d              : 2000 × 8 bytes        = 16 KB
  N_diag         : 2000 × 8 bytes        = 16 KB
  
Kernels (pre-allocated):
  _TT_kernels    : 8 × 2000² × 8 bytes   = 256 MB
  _kernel_matrices: 8 × 2000² × 8 bytes  = 256 MB  (redundant!)
  
Working arrays:
  M, C           : 2000² × 8 bytes       = 32 MB
  L (Cholesky)   : 2000² × 8 bytes       = 32 MB
  A list         : 8 × 2000² × 8 bytes   = 256 MB
  
TOTAL: ~800 MB (comfortable)
```

**Multi-field EE+BB (n_T=0, n_P=1, no EB):**
```
n_obs     = 2000
N_d       = 4000 (Q + U)
nbands    = 8
n_params  = 16 (8 EE + 8 BB)

Spin-2 kernels (pre-allocated):
  _Kp, _Km, _Kx  : 3 × 8 × 2000² × 8    = 768 MB
  geom (6 arrays): 6 × 2000² × 8        = 384 MB
  
Derivative kernels:
  _kernel_matrices: 16 × 4000² × 8      = 2048 MB  (2 GB!)
  
Working arrays:
  M, C           : 4000² × 8            = 128 MB
  L              : 4000² × 8            = 128 MB
  A list         : 16 × 4000² × 8       = 2048 MB
  
TOTAL: ~5.5 GB (manageable)
```

**Multi-field EE+BB+EB (n_T=0, n_P=1, include_EB=True):**
```
n_obs     = 2000
N_d       = 4000
nbands    = 8
n_params  = 24 (8 EE + 8 BB + 8 EB)

Spin-2 kernels: same as above         = 1152 MB

Derivative kernels:
  _kernel_matrices: 24 × 4000² × 8     = 3072 MB  (3 GB!)
  
Working arrays:
  M, C           : 4000² × 8           = 128 MB
  L              : 4000² × 8           = 128 MB
  A list         : 24 × 4000² × 8      = 3072 MB
  
TOTAL: ~7.5 GB (tight on 8 GB machines)
```

**Multi-field TT+TE+EE+BB (n_T=1, n_P=1, no TB/EB):**
```
n_obs     = 2000  
N_d       = 6000 (T + Q + U)
nbands    = 8
n_params  = 32 (8 TT + 8 TE + 8 EE + 8 BB)

All kernels:
  _TT_kernels    : 8 × 2000² × 8       = 256 MB
  _Kp, _Km, _Kx  : 3 × 8 × 2000² × 8   = 768 MB
  geom           : 6 × 2000² × 8       = 384 MB
  _kernel_matrices: 32 × 6000² × 8     = 9216 MB  (9.2 GB!)
  
Working arrays:
  M, C           : 6000² × 8           = 288 MB
  L              : 6000² × 8           = 288 MB
  A list         : 32 × 6000² × 8      = 9216 MB
  
TOTAL: ~20 GB (OOM on 16 GB machines!)
```

**Multi-field with EB (n_T=1, n_P=1, include_EB=True, 7 bands):**
```
n_obs     = 4900 (NSIDE=64, fsky=0.01)
N_d       = 9800 (T + Q + U)
nbands    = 7
n_params  = 42 (7 TT + 7 TE + 7 EE + 7 BB + 7 EB)

Derivative kernels:
  _kernel_matrices: 42 × 9800² × 8     = 32,000 MB  (32 GB!)
  
Working arrays:
  A list         : 42 × 9800² × 8      = 32,000 MB
  
TOTAL: ~70+ GB (OOM on 64 GB machines as reported!)
```

### 2.2 The Bottleneck

**Primary issue:** `_kernel_matrices` pre-allocation

**Why it exists:**
```python
# In __init__ (line 613):
self._kernel_matrices = self._build_all_kernels()

# In _build_all_kernels (line 721-784):
kernels = [None] * self.layout.n_params
for idx, spec, i, j, b in self.layout.entries():
    K = np.zeros((Nd, Nd))  # Full N_d × N_d matrix
    # ... fill K ...
    kernels[idx] = K
return kernels
```

**Each kernel is N_d × N_d, stored permanently.**

**Used in:**
1. `build_signal_cov()`: Needs all kernels to build C
2. `gradient_and_fisher()`: Needs all kernels for v^T K_b v and A_b calculations

**Problem:** Most of the time, most kernels sit idle in memory.

---

## Part 3: Optimization Strategies

### Strategy 1: On-the-Fly Kernel Construction ⭐⭐⭐

**Impact:** 50% memory reduction  
**Complexity:** Moderate  
**Breaks:** Nothing (backward compatible)

**Current:**
```python
# __init__ builds all kernels once
self._kernel_matrices = self._build_all_kernels()  # Stored forever

# gradient_and_fisher uses pre-built kernels
for idx in range(np_total):
    K_b = self._kernel_matrices[idx]  # Just fetch
    W = solve_triangular(L, K_b, lower=True)
```

**Memory:** `n_params × N_d²` permanently

**Proposed:**
```python
# __init__ does NOT build full kernels
self._kernel_matrices = None  # Or remove entirely

# gradient_and_fisher builds on-the-fly
for idx in range(np_total):
    K_b = self._build_kernel(idx)  # Build when needed
    W = solve_triangular(L, K_b, lower=True)
    # K_b goes out of scope and is garbage collected
```

**Memory:** `1 × N_d²` temporary (reused)

**Implementation:**
```python
def _build_kernel(self, idx):
    """Build a single derivative kernel matrix on-the-fly."""
    Nd = len(self.d)
    n = self.n_obs
    K = np.zeros((Nd, Nd))
    
    _, spec, i, j, b = list(self.layout.entries())[idx]
    
    if spec == 'TT':
        si = self._T_slice(i)
        sj = self._T_slice(j)
        K_tt = self._TT_kernels[b]
        K[si, sj] = K_tt
        if i != j:
            K[sj, si] = K_tt.T
    
    elif spec == 'TE':
        si = self._T_slice(i)
        sj = self._P_slice(j)
        Kx_b = self._Kx[b]
        _, _, _, _, c2j, s2j = self._geom
        blk = _build_te_block(Kx_b, c2j, s2j)
        K[si, sj] = blk
        K[sj, si] = blk.T
    
    # ... etc for other spectra
    
    return K
```

**Cost:** Rebuild each kernel `~20 times per Newton run` (once per iteration)  
- Kernel construction: `O(N_d²)` array operations (cheap)  
- Total overhead: `20 × n_params × O(N_d²)` vs `1 × n_params × O(N_d²)` (20× rebuild cost)
- But triangular solves are `O(N_d³)`, so rebuild is ~`N_d/n_params` of the total cost

**Example:** N_d=6000, n_params=32  
- Kernel rebuild per iteration: `32 × 6000² ops` ≈ 1B ops
- Triangular solves per iteration: `32 × 6000³ ops` ≈ 7000B ops
- Overhead: ~0.01% (negligible!)

**Memory saved:** 9 GB → 4.5 GB for TT+TE+EE+BB case

---

### Strategy 2: Sparse Kernel Storage for Diagonal Spectra ⭐⭐

**Impact:** Small (only helps for TT, EE, BB, not TE)  
**Complexity:** Moderate  
**Applicability:** Limited

**Observation:** For same-field spectra (TT, EE, BB), derivative kernels are block-diagonal:
```
TT: K[T_i, T_j] is zero unless i=j
EE: K[P_i, P_j] is zero unless i=j
```

**Current:** Store full N_d × N_d even though most is zero

**Proposed:** Use block-diagonal storage
```python
# Instead of:
K_TT = np.zeros((n_T * n_obs, n_T * n_obs))
for i in range(n_T):
    si = slice(i*n_obs, (i+1)*n_obs)
    K_TT[si, si] = self._TT_kernels[b]  # n_obs × n_obs block

# Do:
K_TT_blocks = [self._TT_kernels[b]] * n_T  # Just n_T pointers!
```

**Memory saved:** `n_T × (N_d² - n_obs²)` per TT band  
- Example: n_T=1 → no saving (already minimal)
- Example: n_T=3, N_d=6000, n_obs=2000 → saves ~250 MB per band

**Complexity:** Requires custom matrix-vector and solve routines for block-diagonal

**Recommendation:** **Not worth it** — Strategy 1 is simpler and more effective

---

### Strategy 3: Reduce Geometry Storage ⭐

**Impact:** Minor (~400 MB for n_obs=2000)  
**Complexity:** Easy  
**Safe:** Yes

**Current:** `_compute_spin2_geometry()` returns 6 full arrays:
```python
cos2dpsi, sin2dpsi, cos2spsi, sin2spsi, c2j, s2j = geom
# Each: n_obs × n_obs × 8 bytes
# Total: 6 × 2000² × 8 = 384 MB
```

**Used in:** Kernel block builders (`_ee_kernel`, `_bb_kernel`, etc.)

**Observation:** These are **only used during kernel construction**, not during gradient/Fisher.

**Proposed:** Don't store `geom` permanently
```python
# Current:
def __init__(...):
    self._geom = _compute_spin2_geometry(...)  # Stored
    self._kernel_matrices = self._build_all_kernels()  # Uses geom

# Proposed:
def __init__(...):
    geom = _compute_spin2_geometry(...)  # Temporary
    self._kernel_matrices = self._build_all_kernels(geom)  # Pass it in
    # geom goes out of scope

def _build_all_kernels(self, geom):  # Takes geom as argument
    # ... use geom ...
```

**Memory saved:** 384 MB

**Combined with Strategy 1:** If building kernels on-the-fly, need to compute geom on-the-fly too OR store it. Trade-off:
- Store geom (384 MB) + build kernels on-the-fly (save 9 GB) = net save 8.6 GB ✓
- Recompute geom every time (slow) = not recommended

**Recommendation:** **Keep geom stored** if using Strategy 1, or discard if using pre-built kernels

---

### Strategy 4: Exploit Symmetry in Fisher Matrix ⭐

**Impact:** 2× speedup for Fisher calculation  
**Complexity:** Easy  
**Safe:** Yes

**Current:** `pixel_likelihood.py:848-852`
```python
F = np.zeros((np_total, np_total))
for b in range(np_total):
    for bp in range(b, np_total):  # Only upper triangle
        val = 0.5 * (A[b] * A[bp].T).sum()
        F[b, bp] = F[bp, b] = val  # Fill both
```

**Already exploits symmetry!** ✅

**Potential improvement:** Vectorize outer loop
```python
# Current: nested Python loops
for b in range(np_total):
    for bp in range(b, np_total):
        val = 0.5 * (A[b] * A[bp].T).sum()

# Proposed: vectorized
A_stack = np.array(A)  # (n_params, Nd, Nd)
# Use einsum or broadcasting for batch computation
F_flat = 0.5 * np.einsum('ijk,ljk->il', A_stack, A_stack)
F = (F_flat + F_flat.T) / 2  # Symmetrize
```

**Issue:** `A_stack` would be `n_params × N_d² × 8` bytes = same as `_kernel_matrices`!

**Recommendation:** **Keep as-is** — the loop is fast enough and avoids extra allocation

---

### Strategy 5: Lazy Cholesky for build_signal_cov ⭐

**Impact:** Minimal (only affects log_likelihood calls outside Newton)  
**Complexity:** Trivial  
**Safe:** Yes

**Current:** `build_signal_cov()` is called independently in `log_likelihood()` (line 806)
```python
def log_likelihood(self, cl_bands):
    C = self.build_signal_cov(cl_bands)  # Build C
    M = C + self.N_mat
    L = np.linalg.cholesky(M)
```

And also in `gradient_and_fisher()` (line 825):
```python
def gradient_and_fisher(self, cl_bands):
    C = self.build_signal_cov(cl_bands)  # Duplicate work!
    M = C + self.N_mat
    L = np.linalg.cholesky(M)
```

**In Newton-Raphson:** Both are called with the same `cl_bands` at each iteration

**Proposed:** Cache C or merge methods
```python
def gradient_fisher_and_likelihood(self, cl_bands):
    """Combined to avoid duplicate signal covariance construction."""
    C = self.build_signal_cov(cl_bands)
    M = C + self.N_mat
    L = np.linalg.cholesky(M)
    
    # Likelihood
    logdet = 2.0 * np.log(np.diag(L)).sum()
    y = solve_triangular(L, self.d, lower=True)
    logL = -0.5 * (logdet + y @ y)
    
    # Gradient and Fisher
    v = solve_triangular(L.T, y, lower=False)
    # ... rest of gradient_and_fisher ...
    
    return logL, g, F
```

**Memory impact:** None  
**Speed impact:** Eliminates one `build_signal_cov()` call per Newton iteration

**Issue:** Changes API (backward incompatible)

**Recommendation:** **Low priority** — `build_signal_cov()` is cheap compared to Cholesky

---

## Part 4: Recommended Implementation Plan

### Phase 1: Immediate (Easy, Safe)

1. **Add profiling to identify actual bottleneck:**
   ```python
   import time
   import tracemalloc
   
   tracemalloc.start()
   # ... run code ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   for stat in top_stats[:10]:
       print(stat)
   ```

2. **Document current memory usage in README:**
   - Add table showing N_d, n_params, memory for typical cases
   - Warn about n_T=1, n_P=1 hitting 20 GB

### Phase 2: High Impact (Moderate effort)

3. **Implement on-the-fly kernel construction (Strategy 1):**
   ```python
   class PixelLikelihood:
       def __init__(self, ..., lazy_kernels=True):
           # ...
           if lazy_kernels:
               self._kernel_matrices = None
           else:
               self._kernel_matrices = self._build_all_kernels()
       
       def _get_kernel(self, idx):
           """Get kernel (either from cache or build on-the-fly)."""
           if self._kernel_matrices is not None:
               return self._kernel_matrices[idx]
           else:
               return self._build_kernel(idx)
       
       def gradient_and_fisher(self, cl_bands):
           # ...
           for idx in range(np_total):
               K_b = self._get_kernel(idx)  # Changed
               # ... rest unchanged ...
   ```
   
   **Testing:** Verify results identical with `lazy_kernels=False`  
   **Memory saving:** 50% for multi-field  
   **Speed cost:** <1% (kernel rebuild dominated by triangular solves)

### Phase 3: Refinements (Lower priority)

4. **Consider discard geom after init** if not using Strategy 1

5. **Add memory monitoring to newton_raphson:**
   ```python
   def newton_raphson(self, cl_init, max_iter=15, tol=1e-4, damp=1.0):
       import psutil
       process = psutil.Process()
       
       for it in range(max_iter):
           mem_gb = process.memory_info().rss / 1e9
           print(f"  iter {it+1}: mem={mem_gb:.1f} GB ...")
   ```

---

## Part 5: Alternative: Reduce Problem Size

If implementation time is limited, **reducing N_d is more effective:**

**Option A: Reduce NSIDE**
- NSIDE=128 → NSIDE=64: n_obs reduced 4×, memory reduced 16×
- NSIDE=64 → NSIDE=32: n_obs reduced 4×, memory reduced 16×

**Example:**
- Current: NSIDE=64, n_obs=4900, N_d=9800, mem=70 GB
- Reduced: NSIDE=32, n_obs=1225, N_d=2450, mem=4.5 GB

**Trade-off:** Lose high-ℓ information (l_max = 2×NSIDE-1)

**Option B: Fewer bands**
- 7 bands → 5 bands: n_params reduced 30%, memory reduced 30%

**Trade-off:** Coarser ℓ-resolution

**Option C: Analyze TT, EE+BB separately**
- Don't run TT+TE+EE+BB jointly
- Run TT alone (256 MB) + EE+BB alone (2 GB) sequentially

**Trade-off:** Cannot estimate TE simultaneously (TE couples T and P fields)

---

## Part 6: Summary of Findings

### Bugs
- ✅ No critical bugs
- ⚠️ Minor: Fisher calculation uses temporary arrays (low impact)

### Memory Bottleneck
- **Root cause:** Pre-allocated `_kernel_matrices` = `n_params × N_d²`
- **Scaling:** Multi-field cases hit 20-70 GB
- **Location:** `PixelLikelihood.__init__()` → `_build_all_kernels()`

### Recommended Fix
1. **Primary:** On-the-fly kernel construction (Strategy 1)
   - 50% memory reduction
   - <1% speed cost
   - Moderate implementation (100 lines)
   - **Enables Euclid multi-field runs on 32 GB machines**

2. **Alternative:** Reduce NSIDE (immediate workaround)
   - NSIDE=64 → 32: 16× memory reduction
   - Loses high-ℓ (ℓ_max = 63 instead of 127)

3. **Optional:** Minor optimizations (Strategies 3-5)
   - Marginal gains (~5-10%)
   - Lower priority

### Next Steps
1. Add memory profiling to confirm diagnosis
2. Implement `lazy_kernels=True` option
3. Test on Euclid fsky=0.01 multi-field case
4. Update memory warnings in documentation

---

## Appendix: Code Snippets

### A1: Memory Profiling Snippet

Add to `tests/`:
```python
"""Memory profiling for pixel likelihood."""
import numpy as np
import tracemalloc
from pixel_likelihood import PixelLikelihood

def profile_memory():
    tracemalloc.start()
    
    # Euclid-like setup
    nside = 64
    n_obs = 4900
    obs_pix = np.arange(n_obs)
    
    # Multi-field
    T = np.random.randn(n_obs)
    Q = np.random.randn(n_obs)
    U = np.random.randn(n_obs)
    N_T = np.full(n_obs, 1e-6)
    N_P = np.full(n_obs, 1.5e-6)
    
    print("Building likelihood...")
    lik = PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=nside,
        N_T_list=[N_T], N_Q_list=[N_P], N_U_list=[N_P],
        lmin=2, lmax=127,
        band_edges=np.array([2, 20, 40, 60, 80, 100, 120, 128]),
        include_TB=False, include_EB=True
    )
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\n=== Top 10 memory allocations ===")
    for stat in top_stats[:10]:
        print(f"{stat.size / 1e6:>8.1f} MB  {stat}")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory: {current / 1e9:.2f} GB")
    print(f"Peak memory: {peak / 1e9:.2f} GB")
    
    tracemalloc.stop()

if __name__ == '__main__':
    profile_memory()
```

### A2: On-the-Fly Kernel Implementation

```python
def _build_kernel(self, idx):
    """Build a single derivative kernel matrix on demand."""
    entry = list(self.layout.entries())[idx]
    _, spec, i, j, b = entry
    
    Nd = len(self.d)
    n = self.n_obs
    K = np.zeros((Nd, Nd), dtype=np.float64)
    
    if spec == 'TT':
        si, sj = self._T_slice(i), self._T_slice(j)
        K_tt = self._TT_kernels[b]
        K[si, sj] = K_tt
        if i != j:
            K[sj, si] = K_tt.T
    
    elif spec == 'TE':
        si, sj = self._T_slice(i), self._P_slice(j)
        Kx_b = self._Kx[b]
        _, _, _, _, c2j, s2j = self._geom
        blk = _build_te_block(Kx_b, c2j, s2j)
        K[si, sj] = blk
        K[sj, si] = blk.T
    
    elif spec == 'TB':
        si, sj = self._T_slice(i), self._P_slice(j)
        Kx_b = self._Kx[b]
        _, _, _, _, c2j, s2j = self._geom
        blk = _build_tb_block(Kx_b, c2j, s2j)
        K[si, sj] = blk
        K[sj, si] = blk.T
    
    elif spec == 'EE':
        sj, sk = self._P_slice(i), self._P_slice(j)
        Kp_b, Km_b = self._Kp[b], self._Km[b]
        blk = _ee_kernel(Kp_b, Km_b, self._geom)
        K[sj, sk] = blk
        if i != j:
            K[sk, sj] = blk.T
    
    elif spec == 'BB':
        sj, sk = self._P_slice(i), self._P_slice(j)
        Kp_b, Km_b = self._Kp[b], self._Km[b]
        blk = _bb_kernel(Kp_b, Km_b, self._geom)
        K[sj, sk] = blk
        if i != j:
            K[sk, sj] = blk.T
    
    elif spec == 'EB':
        sj, sk = self._P_slice(i), self._P_slice(j)
        blk = _eb_kernel(self._Km[b], self._geom)
        K[sj, sk] = blk
        if i != j:
            K[sk, sj] = blk.T
    
    return K
```

---

**End of Review**
