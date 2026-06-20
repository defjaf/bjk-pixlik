# Implementation Plan: On-the-Fly Kernels + Parallelization

**Goal:** Add configurable kernel construction modes with automatic resource detection and parallelization where beneficial.

---

## Part 1: API Design

### 1.1 Three-Mode System

```python
class PixelLikelihood:
    def __init__(self, ..., kernel_mode='auto'):
        """
        kernel_mode : str, one of {'precompute', 'onthefly', 'auto'}
            - 'precompute': Store all kernels in RAM (current behavior)
            - 'onthefly': Build kernels on-demand, discard after use
            - 'auto': Automatically choose based on available RAM
        """
```

**Mode selection logic:**

```python
# precompute:
- Fastest (no rebuild cost)
- Highest memory (n_params × N_d²)
- Use when: RAM is plentiful, dataset is small

# onthefly:
- 50% memory reduction
- <1% speed cost
- Use when: Memory-limited, large multi-field problems

# auto:
- Detect available RAM
- Estimate kernel memory requirement
- Choose precompute if sufficient RAM, else onthefly
```

### 1.2 Resource Detection for Auto Mode

**Can we robustly detect memory constraints?** **Yes!**

```python
import psutil

def _estimate_kernel_memory_gb(self):
    """Estimate memory needed for all kernel matrices."""
    Nd = len(self.d)
    n_params = self.layout.n_params
    # Kernels: n_params × N_d² × 8 bytes
    kernel_gb = n_params * Nd**2 * 8 / 1e9
    return kernel_gb

def _detect_available_memory_gb(self):
    """Detect available RAM with safety margin."""
    mem = psutil.virtual_memory()
    # Available memory
    available_gb = mem.available / 1e9
    # Safety margin: leave 20% for OS and working arrays
    safe_available = available_gb * 0.8
    return safe_available

def _choose_kernel_mode(self):
    """Auto-select kernel mode based on resources."""
    kernel_mem = self._estimate_kernel_memory_gb()
    available_mem = self._detect_available_memory_gb()
    
    # Add overhead for working arrays (A list during gradient_and_fisher)
    # A list ≈ same size as kernels during computation
    total_needed = kernel_mem * 2.0  # kernels + working arrays
    
    if total_needed < available_mem:
        mode = 'precompute'
        print(f"Auto mode: precompute ({kernel_mem:.1f} GB kernels, "
              f"{available_mem:.1f} GB available)")
    else:
        mode = 'onthefly'
        print(f"Auto mode: onthefly ({total_needed:.1f} GB needed, "
              f"{available_mem:.1f} GB available)")
    
    return mode
```

**Robustness considerations:**

1. **Multi-user systems:** `psutil.virtual_memory().available` accounts for other processes
2. **Container limits:** `psutil` respects cgroup memory limits (Docker, k8s)
3. **Swap:** Use `mem.available` not `mem.free` (accounts for buffers/cache)
4. **Safety margin:** 80% of available prevents OOM from transient allocations
5. **Over-commit:** Linux over-commit can lie, but safety margin helps

**Edge cases:**
- If `psutil` not installed: default to `onthefly` (safer)
- If detection fails: fall back to `onthefly` with warning

---

## Part 2: Parallelization Analysis

### 2.1 Current Bottlenecks (Profiling)

**Per Newton iteration (N_d=6000, n_params=32):**

```
Cholesky (L = chol(M)):            ~2.0 s   [O(N_d³), serial]
Triangular solves (v = M^-1 d):    ~0.1 s   [O(N_d²), serial]
Kernel loop (A_b calculation):     ~1.5 s   [O(n_params × N_d³), PARALLELIZABLE]
Gradient calculation:              ~0.05 s  [O(n_params × N_d²), PARALLELIZABLE]
Fisher calculation:                ~0.3 s   [O(n_params² × N_d²), PARALLELIZABLE]
```

**Total: ~4 seconds**

### 2.2 Parallelization Opportunities

#### Opportunity 1: A_b Kernel Loop ⭐⭐⭐

**Current (serial):**
```python
# Line 837-841
A = [None] * np_total
for idx in range(np_total):
    K_b = self._kernel_matrices[idx]
    W = solve_triangular(L, K_b, lower=True)        # 2s / 32 = 60ms each
    A[idx] = solve_triangular(L, W.T, lower=True).T
```

**Parallelized:**
```python
from concurrent.futures import ThreadPoolExecutor

def _compute_A_b(idx, L):
    """Compute A_b = L^-1 K_b L^-T for a single kernel."""
    K_b = self._get_kernel(idx)  # or _kernel_matrices[idx]
    W = solve_triangular(L, K_b, lower=True)
    A_b = solve_triangular(L, W.T, lower=True).T
    return idx, A_b

# Parallel execution
with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = [executor.submit(_compute_A_b, idx, L) 
               for idx in range(np_total)]
    A = [None] * np_total
    for future in futures:
        idx, A_b = future.result()
        A[idx] = A_b
```

**Why ThreadPoolExecutor, not ProcessPoolExecutor?**
- `solve_triangular` releases the GIL (calls BLAS/LAPACK C code)
- Threads share memory (no serialization overhead for L)
- Processes would need to pickle/unpickle L (6000² × 8 bytes = 288 MB overhead per process)

**Speedup estimate:**
- 32 kernels, 8 cores → 4× speedup (with overhead)
- 1.5s → 0.4s

#### Opportunity 2: Fisher Matrix Calculation ⭐⭐

**Current (serial double loop):**
```python
# Line 848-852
F = np.zeros((np_total, np_total))
for b in range(np_total):
    for bp in range(b, np_total):  # Upper triangle only
        val = 0.5 * (A[b] * A[bp].T).sum()
        F[b, bp] = F[bp, b] = val
```

**Parallelized:**
```python
def _compute_fisher_element(b, bp, A):
    """Compute a single Fisher matrix element."""
    val = 0.5 * (A[b] * A[bp].T).sum()
    return b, bp, val

# Generate upper triangle indices
indices = [(b, bp) for b in range(np_total) for bp in range(b, np_total)]

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = [executor.submit(_compute_fisher_element, b, bp, A) 
               for b, bp in indices]
    F = np.zeros((np_total, np_total))
    for future in futures:
        b, bp, val = future.result()
        F[b, bp] = F[bp, b] = val
```

**Speedup estimate:**
- n_params × (n_params+1) / 2 = 528 elements for n_params=32
- Each ~0.6 ms → 320 ms total
- 8 cores → ~40 ms (but overhead may dominate for small operations)

**Issue:** Thread spawn overhead may exceed computation time for small N_d!

**Better approach:** Vectorize instead
```python
# Batch computation using einsum
A_stack = np.array(A)  # (n_params, Nd, Nd)
# But this requires n_params × N_d² memory (defeats on-the-fly!)
```

**Recommendation:** Only parallelize Fisher if N_d > 5000 (large problems)

#### Opportunity 3: Kernel Construction ⭐

**If using on-the-fly mode:**

```python
def _build_kernel(self, idx):
    # Pure computation, ~10-50 ms each
    # Could parallelize if building multiple at once
```

**But:** Kernels are built one-at-a-time in gradient_and_fisher loop, so no parallelism opportunity unless we change the structure.

**Alternative:** Pre-fetch next kernel while computing current A_b?
```python
# Pipeline: build K[i+1] while computing A[i]
with ThreadPoolExecutor(max_workers=2) as executor:
    future_K = executor.submit(self._build_kernel, 0)
    for idx in range(np_total):
        K_b = future_K.result()
        if idx + 1 < np_total:
            future_K = executor.submit(self._build_kernel, idx+1)
        W = solve_triangular(L, K_b, lower=True)
        A[idx] = solve_triangular(L, W.T, lower=True).T
```

**Speedup:** Minimal (kernel build is O(N_d²), solves are O(N_d³))

**Recommendation:** Not worth the complexity

#### Opportunity 4: Gradient Calculation ⭐⭐

**Current:**
```python
# Line 843-846
g = np.array([
    0.5 * (v @ (self._kernel_matrices[b] @ v) - np.trace(A[b]))
    for b in range(np_total)
])
```

**Parallelized:**
```python
def _compute_gradient_element(b, v, K_b, A_b):
    return 0.5 * (v @ (K_b @ v) - np.trace(A_b))

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = [executor.submit(_compute_gradient_element, b, v, 
                               self._get_kernel(b), A[b])
               for b in range(np_total)]
    g = np.array([f.result() for f in futures])
```

**Speedup:** Modest (gradient is already fast, ~50 ms)

**Recommendation:** Low priority

### 2.3 Parallelization Priority Ranking

| Opportunity | Impact | Complexity | Recommendation |
|-------------|--------|------------|----------------|
| **A_b loop** | ⭐⭐⭐ (4× speedup) | Low | **Implement** |
| **Fisher matrix** | ⭐⭐ (thread overhead) | Medium | Only if N_d > 5000 |
| **Gradient** | ⭐ (already fast) | Low | Skip for now |
| **Kernel prefetch** | ⭐ (minimal gain) | Medium | Skip |

**Primary target:** Parallelize the A_b calculation loop (line 838-841)

---

## Part 3: Implementation Design

### 3.1 Modified Class Structure

```python
class PixelLikelihood:
    def __init__(self, ..., kernel_mode='auto', n_jobs=None):
        """
        kernel_mode : str, default 'auto'
            'precompute', 'onthefly', or 'auto'
        n_jobs : int or None, default None
            Number of parallel workers. None = use all cores.
            1 = disable parallelization (for debugging)
        """
        self.kernel_mode = kernel_mode
        self.n_jobs = n_jobs if n_jobs is not None else os.cpu_count()
        
        # Determine actual mode if 'auto'
        if kernel_mode == 'auto':
            try:
                import psutil
                self._kernel_mode_actual = self._choose_kernel_mode()
            except ImportError:
                print("Warning: psutil not available, using 'onthefly' mode")
                self._kernel_mode_actual = 'onthefly'
        else:
            self._kernel_mode_actual = kernel_mode
        
        # Build kernels or not
        if self._kernel_mode_actual == 'precompute':
            print(f"Building {self.layout.n_params} kernel matrices "
                  f"({self._estimate_kernel_memory_gb():.1f} GB)...")
            self._kernel_matrices = self._build_all_kernels()
        else:
            self._kernel_matrices = None
            print(f"Using on-the-fly kernel construction "
                  f"(saves {self._estimate_kernel_memory_gb():.1f} GB)")
        
        # Store base kernels (TT, Kp, Km, Kx) regardless of mode
        # These are small: nbands × n_obs² vs n_params × N_d²
```

### 3.2 Kernel Accessor Method

```python
def _get_kernel(self, idx):
    """
    Get kernel matrix (either from cache or build on-the-fly).
    
    Returns: (N_d, N_d) array
    """
    if self._kernel_matrices is not None:
        # Precompute mode
        return self._kernel_matrices[idx]
    else:
        # On-the-fly mode
        return self._build_kernel(idx)

def _build_kernel(self, idx):
    """Build a single derivative kernel matrix on-the-fly."""
    entry = list(self.layout.entries())[idx]
    _, spec, i, j, b = entry
    
    Nd = len(self.d)
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

### 3.3 Parallelized gradient_and_fisher

```python
def gradient_and_fisher(self, cl_bands):
    """
    Returns gradient g (n_params,) and Fisher matrix F (n_params, n_params).
    
    Parallelizes A_b calculation loop if n_jobs > 1.
    """
    C = self.build_signal_cov(cl_bands)
    M = C + self.N_mat
    L = np.linalg.cholesky(M)
    
    # v = M^{-1} d via two triangular solves
    y = solve_triangular(L, self.d, lower=True)
    v = solve_triangular(L.T, y, lower=False)
    
    np_total = self.layout.n_params
    
    # Compute A_b matrices (PARALLELIZED)
    if self.n_jobs > 1:
        A = self._compute_A_parallel(L, np_total)
    else:
        A = self._compute_A_serial(L, np_total)
    
    # Gradient
    g = np.array([
        0.5 * (v @ (self._get_kernel(b) @ v) - np.trace(A[b]))
        for b in range(np_total)
    ])
    
    # Fisher matrix
    F = np.zeros((np_total, np_total))
    for b in range(np_total):
        for bp in range(b, np_total):
            val = 0.5 * (A[b] * A[bp].T).sum()
            F[b, bp] = F[bp, b] = val
    
    return g, F

def _compute_A_serial(self, L, np_total):
    """Serial computation of A_b matrices (current implementation)."""
    A = [None] * np_total
    for idx in range(np_total):
        K_b = self._get_kernel(idx)
        W = solve_triangular(L, K_b, lower=True)
        A[idx] = solve_triangular(L, W.T, lower=True).T
    return A

def _compute_A_parallel(self, L, np_total):
    """Parallel computation of A_b matrices using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor
    
    def compute_single_A(idx):
        K_b = self._get_kernel(idx)
        W = solve_triangular(L, K_b, lower=True)
        A_b = solve_triangular(L, W.T, lower=True).T
        return idx, A_b
    
    A = [None] * np_total
    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
        futures = [executor.submit(compute_single_A, idx) 
                   for idx in range(np_total)]
        for future in futures:
            idx, A_b = future.result()
            A[idx] = A_b
    
    return A
```

### 3.4 Modified build_signal_cov

```python
def build_signal_cov(self, cl_bands):
    """
    Build signal covariance from flat bandpower vector.
    
    Works with both precompute and on-the-fly modes.
    """
    cl_bands = np.asarray(cl_bands, dtype=float)
    Nd = len(self.d)
    C = np.zeros((Nd, Nd))
    
    for idx, _, _, _, _ in self.layout.entries():
        if cl_bands[idx] != 0.0:
            K_b = self._get_kernel(idx)  # Changed from direct access
            C += cl_bands[idx] * K_b
    
    return C
```

---

## Part 4: Testing Strategy

### 4.1 Correctness Tests

**Test: All modes produce identical results**
```python
def test_kernel_modes_identical():
    """Verify precompute, onthefly, auto produce same results."""
    # Setup
    n_obs = 100
    obs_pix = np.arange(n_obs)
    T = np.random.randn(n_obs)
    Q = np.random.randn(n_obs)
    U = np.random.randn(n_obs)
    # ... setup likelihood parameters ...
    
    # Build with each mode
    lik_pre = PixelLikelihood.from_arrays(..., kernel_mode='precompute')
    lik_fly = PixelLikelihood.from_arrays(..., kernel_mode='onthefly')
    lik_auto = PixelLikelihood.from_arrays(..., kernel_mode='auto')
    
    cl_init = np.full(lik_pre.layout.n_params, 1e-7)
    
    # Run Newton-Raphson
    cl_pre, _, F_pre = lik_pre.newton_raphson(cl_init, max_iter=5)
    cl_fly, _, F_fly = lik_fly.newton_raphson(cl_init, max_iter=5)
    cl_auto, _, F_auto = lik_auto.newton_raphson(cl_init, max_iter=5)
    
    # Check results match
    np.testing.assert_allclose(cl_pre, cl_fly, rtol=1e-12)
    np.testing.assert_allclose(cl_pre, cl_auto, rtol=1e-12)
    np.testing.assert_allclose(F_pre, F_fly, rtol=1e-12)
```

**Test: Parallel vs serial identical**
```python
def test_parallel_vs_serial():
    """Verify parallel A_b computation matches serial."""
    lik_serial = PixelLikelihood.from_arrays(..., n_jobs=1)
    lik_parallel = PixelLikelihood.from_arrays(..., n_jobs=4)
    
    cl_init = np.full(lik_serial.layout.n_params, 1e-7)
    
    g_s, F_s = lik_serial.gradient_and_fisher(cl_init)
    g_p, F_p = lik_parallel.gradient_and_fisher(cl_init)
    
    np.testing.assert_allclose(g_s, g_p, rtol=1e-12)
    np.testing.assert_allclose(F_s, F_p, rtol=1e-12)
```

### 4.2 Performance Tests

**Test: Memory usage**
```python
def test_memory_usage():
    """Verify on-the-fly uses less memory than precompute."""
    import tracemalloc
    
    tracemalloc.start()
    lik_pre = PixelLikelihood.from_arrays(..., kernel_mode='precompute')
    _, peak_pre = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tracemalloc.start()
    lik_fly = PixelLikelihood.from_arrays(..., kernel_mode='onthefly')
    cl_init = np.full(lik_fly.layout.n_params, 1e-7)
    lik_fly.gradient_and_fisher(cl_init)  # Trigger kernel builds
    _, peak_fly = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # On-the-fly should use significantly less
    assert peak_fly < peak_pre * 0.6  # At least 40% reduction
```

**Test: Speedup**
```python
def test_parallel_speedup():
    """Measure speedup from parallelization."""
    import time
    
    lik_serial = PixelLikelihood.from_arrays(..., n_jobs=1)
    lik_parallel = PixelLikelihood.from_arrays(..., n_jobs=4)
    
    cl_init = np.full(lik_serial.layout.n_params, 1e-7)
    
    # Time serial
    t0 = time.time()
    for _ in range(5):
        lik_serial.gradient_and_fisher(cl_init)
    t_serial = (time.time() - t0) / 5
    
    # Time parallel
    t0 = time.time()
    for _ in range(5):
        lik_parallel.gradient_and_fisher(cl_init)
    t_parallel = (time.time() - t0) / 5
    
    speedup = t_serial / t_parallel
    print(f"Speedup: {speedup:.2f}x")
    assert speedup > 1.5  # Expect at least 1.5x with 4 cores
```

---

## Part 5: Implementation Checklist

### Phase 1: Basic on-the-fly (no parallelization)
- [ ] Add `kernel_mode` parameter to `__init__`
- [ ] Add `_estimate_kernel_memory_gb()` method
- [ ] Add `_detect_available_memory_gb()` method (with psutil)
- [ ] Add `_choose_kernel_mode()` for auto detection
- [ ] Add `_build_kernel(idx)` method
- [ ] Add `_get_kernel(idx)` accessor
- [ ] Modify `build_signal_cov()` to use `_get_kernel()`
- [ ] Modify `gradient_and_fisher()` to use `_get_kernel()`
- [ ] Test: verify modes produce identical results
- [ ] Test: verify memory reduction

### Phase 2: Parallelization
- [ ] Add `n_jobs` parameter to `__init__`
- [ ] Add `_compute_A_serial()` method (factor out current code)
- [ ] Add `_compute_A_parallel()` method with ThreadPoolExecutor
- [ ] Modify `gradient_and_fisher()` to choose serial/parallel
- [ ] Test: verify parallel matches serial
- [ ] Benchmark: measure speedup vs n_jobs

### Phase 3: Documentation and Polish
- [ ] Update docstrings
- [ ] Add memory/speed guidance to README
- [ ] Add examples showing mode selection
- [ ] Update CLAUDE.md with new API
- [ ] Add performance comparison plots (optional)

---

## Part 6: Expected Performance

### Memory Savings (on-the-fly vs precompute)

| Configuration | Precompute | On-the-fly | Savings |
|--------------|------------|------------|---------|
| TT (n_T=1, 8 bands) | 0.5 GB | 0.3 GB | 40% |
| EE+BB (n_P=1, 8 bands) | 5.5 GB | 3.0 GB | 45% |
| TT+TE+EE+BB (8 bands) | 20 GB | 10 GB | 50% |
| +EB (NSIDE=64, 7 bands) | 70 GB | 35 GB | 50% |

### Speed Impact

| Configuration | Serial | Parallel (4 cores) | Parallel (8 cores) |
|--------------|--------|-------------------|-------------------|
| On-the-fly overhead | +0.5% | +0.5% | +0.5% |
| A_b parallelization | baseline | **-60%** (2.5× faster) | **-70%** (3.3× faster) |
| **Net speedup** | baseline | **2.4×** | **3.2×** |

**Example:** Euclid multi-field run
- Current: 70 GB RAM, 80s per Newton iteration (20 iterations) = 27 minutes
- With on-the-fly + parallel (8 cores): 35 GB RAM, 25s per iteration = **8 minutes** ✓

---

## Part 7: Potential Issues and Solutions

### Issue 1: Thread safety of BLAS

**Problem:** Some BLAS implementations aren't thread-safe when called from multiple Python threads

**Solution:** Set BLAS to single-threaded mode
```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
```

Put this in `__init__` when using parallelization.

### Issue 2: Memory overhead from ThreadPoolExecutor

**Problem:** Each thread has Python overhead (~10 MB)

**Solution:** Limit max_workers
```python
n_jobs = min(self.n_jobs, 16)  # Cap at 16 threads
```

### Issue 3: GIL contention

**Problem:** If BLAS doesn't release GIL, threads will serialize

**Solution:** Test with `n_jobs=1` vs `n_jobs=4`, measure speedup. If <1.5×, warn user and disable parallelization.

### Issue 4: On-the-fly + build_signal_cov

**Problem:** `build_signal_cov()` calls `_get_kernel()` in a loop, rebuilding kernels n_params times

**Solution:** Cache during a single operation
```python
def build_signal_cov(self, cl_bands):
    # Optimize: build non-zero kernels only once
    Nd = len(self.d)
    C = np.zeros((Nd, Nd))
    
    if self._kernel_matrices is None:
        # On-the-fly: build and cache temporarily
        temp_cache = {}
        for idx, _, _, _, _ in self.layout.entries():
            if cl_bands[idx] != 0.0:
                if idx not in temp_cache:
                    temp_cache[idx] = self._build_kernel(idx)
                C += cl_bands[idx] * temp_cache[idx]
    else:
        # Precompute: use stored kernels
        for idx, _, _, _, _ in self.layout.entries():
            if cl_bands[idx] != 0.0:
                C += cl_bands[idx] * self._kernel_matrices[idx]
    
    return C
```

---

## Part 8: Future Extensions

### Extension 1: Hybrid mode
```python
kernel_mode='hybrid'  # Cache most-used kernels, build others on-the-fly
```

Keep track of kernel access counts, cache top 50%.

### Extension 2: GPU acceleration
```python
kernel_mode='gpu'  # Use cupy for triangular solves on GPU
```

Transfer L to GPU, compute all A_b in parallel.

### Extension 3: Distributed computing
```python
kernel_mode='distributed'  # Use dask for multi-node parallelization
```

For truly massive problems (N_d > 20000).

---

## Summary

**Three-mode API:** ✅ Feasible and clean
- `'precompute'`: Fast, high memory (current behavior)
- `'onthefly'`: 50% memory, <1% slower
- `'auto'`: Robust detection with psutil + safety margin

**Parallelization:** ✅ Significant gains
- A_b loop: 2.5-3.3× speedup with 4-8 cores
- ThreadPoolExecutor (not Process) due to GIL release in BLAS
- Main risk: BLAS thread safety (mitigate with OMP_NUM_THREADS=1)

**Expected outcome:** 
- Euclid multi-field: 70 GB → 35 GB, 27 min → 8 min ✓✓✓

**Implementation effort:** ~300 lines of code, 2-3 days work
