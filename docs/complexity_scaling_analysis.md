# Complexity Scaling Analysis: Time and Space

**Goal:** Derive time and space complexity as a function of problem parameters for each mode.

---

## Notation and Parameters

### Input Parameters

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| `n_T` | Number of temperature fields | 0-6 |
| `n_P` | Number of polarization fields (each has Q+U) | 0-6 |
| `n_obs` | Observed pixels per field | 1000-5000 |
| `n_bands` | Number of ℓ-bands | 5-10 |
| `l_max` | Maximum multipole | 50-500 |

**Assumption:** All fields use the same pixelization (same `n_obs` and `obs_pix`)

### Derived Quantities

```
N_d = (n_T + 2*n_P) × n_obs              [Total data vector length]

n_pairs_TT = n_T(n_T+1)/2                [TT spectrum pairs]
n_pairs_TE = n_T × n_P                   [TE cross-spectra]
n_pairs_EE = n_P(n_P+1)/2                [EE pairs]
n_pairs_BB = n_P(n_P+1)/2                [BB pairs]
n_pairs_EB = n_P(n_P+1)/2                [EB pairs, if included]
n_pairs_TB = n_T × n_P                   [TB pairs, if included]

Total pairs (without TB/EB):
n_pairs = n_T(n_T+1)/2 + n_T×n_P + n_P(n_P+1)/2 + n_P(n_P+1)/2
        = n_T(n_T+1)/2 + n_T×n_P + n_P(n_P+1)
        ≈ (n_T² + 2n_T×n_P + 2n_P²) / 2
        ≈ O(n_T² + n_P²)
        
n_params = n_pairs × n_bands             [Total bandpower parameters]
         ≈ O((n_T² + n_P²) × n_bands)
```

**Simplified notation:**
- Let `n_fields = n_T + n_P` (total number of field types)
- Then `n_params ≈ O(n_fields² × n_bands)`
- And `N_d ≈ O(n_fields × n_obs)` (assuming n_T ≈ n_P or one dominates)

---

## Part 1: One-Time Initialization Costs

### 1.1 Base Kernel Construction

**Scalar (TT) kernels:**
```
Operation: Legendre recurrence on unique cos(θ) values
- Unique values: O(n_obs²) at most (typically ~10-30% of n_obs²)
- Recurrence: l = 2..l_max for each unique value
- Per band: accumulate contributions

Time: O(n_obs² × l_max)
Space: O(n_bands × n_obs²)  [Store n_bands TT kernels]
```

**Spin-2 (polarization) kernels:**
```
Operations:
1. Wigner d-matrix recurrence: O(n_obs² × l_max)
2. Pixel-pair geometry: O(n_obs²) [6 arrays: cos2dp, sin2dp, etc.]

Time: O(n_obs² × l_max)
Space: O(n_bands × n_obs² × 3)  [Kp, Km, Kx per band]
       + O(n_obs² × 6)          [Geometry arrays]
       = O(n_bands × n_obs²)
```

**Total base kernel construction:**
```
Time: O(n_obs² × l_max)                    [Dominated by recurrence]
Space: O(n_bands × n_obs²)                 [Base kernels only]
```

### 1.2 Full Derivative Kernel Assembly

**Per derivative kernel K_b:**
```
Operation: Assemble N_d × N_d matrix from base kernels
- For TT(i,j,b): Copy n_obs² block into N_d² matrix
- For EE(i,j,b): Build 2n_obs × 2n_obs block using Kp, Km, geometry
- For TE(i,j,b): Build n_obs × 2n_obs block using Kx, geometry

Time per kernel: O(N_d²) = O(n_fields² × n_obs²)
Total for all kernels: n_params × O(N_d²) 
                     = O(n_fields² × n_bands × n_fields² × n_obs²)
                     = O(n_fields⁴ × n_bands × n_obs²)

Space (if storing all): n_params × N_d² × 8 bytes
                      = O(n_fields⁴ × n_bands × n_obs²)  [MEMORY BOTTLENECK!]
```

**Key observation:** Space complexity for full kernels scales as **O(n_fields⁴)**!

---

## Part 2: Per Newton Iteration Costs

### 2.1 Build Signal Covariance C

**Operation:** `C = Σ cl_b × K_b` for non-zero cl_b

```
Precompute mode:
  - Fetch K_b from memory: O(1) per kernel
  - Accumulate: O(N_d²) per kernel
  - Total: O(n_params × N_d²) = O(n_fields⁴ × n_bands × n_obs²)

On-the-fly mode:
  - Build K_b: O(N_d²) per kernel
  - Accumulate: O(N_d²) per kernel
  - Total: Same O(n_fields⁴ × n_bands × n_obs²)
  - But: Only build non-zero kernels (typically all in ML estimation)
```

### 2.2 Cholesky Decomposition

**Operation:** `L = chol(M)` where `M = C + N`

```
Time: O(N_d³) = O(n_fields³ × n_obs³)
Space: O(N_d²) = O(n_fields² × n_obs²)

Note: This is INDEPENDENT of n_bands!
```

### 2.3 Compute v = M^{-1} d

**Operation:** Solve `Ly = d` then `L^T v = y`

```
Time: 2 × O(N_d²) = O(n_fields² × n_obs²)
Space: O(N_d)
```

### 2.4 Compute A_b = M^{-1} K_b for All Kernels

**Per kernel:**
```
Operations:
1. Get K_b: 
   - Precompute: O(1) fetch
   - On-the-fly: O(N_d²) build
2. Solve LW = K_b: O(N_d³) [Triangular solve with N_d RHS vectors]
3. Solve LA^T = W^T: O(N_d³)

Time per kernel: O(N_d³) = O(n_fields³ × n_obs³)  [Dominated by solves]
```

**All n_params kernels:**
```
Serial mode:
  Time: n_params × O(N_d³) 
      = O(n_fields² × n_bands × n_fields³ × n_obs³)
      = O(n_fields⁵ × n_bands × n_obs³)
  
  Space: n_params × N_d² [Store all A_b]
       = O(n_fields⁴ × n_bands × n_obs²)

Parallel mode (p cores):
  Time: (n_params / p) × O(N_d³)
      = O(n_fields⁵ × n_bands × n_obs³ / p)
  
  Space: Same O(n_fields⁴ × n_bands × n_obs²) [Must store all A_b]

On-the-fly (kernel building):
  Time: n_params × [O(N_d²) build + O(N_d³) solve]
      = O(n_fields⁵ × n_bands × n_obs³)  [Dominated by O(N_d³)]
      ≈ same as serial
  
  Space: O(N_d²) for working arrays [Only 1 kernel at a time!]
        + n_params × N_d² for A_b storage
        = O(n_fields⁴ × n_bands × n_obs²)
```

### 2.5 Gradient Calculation

**Operation:** `g_b = 0.5 × [v^T K_b v - Tr(A_b)]`

```
Per element:
  - Get K_b: precompute O(1), on-the-fly O(N_d²)
  - Compute K_b v: O(N_d²)
  - Compute v^T (K_b v): O(N_d)
  - Trace(A_b): O(N_d)

Time: n_params × O(N_d²) = O(n_fields⁴ × n_bands × n_obs²)
Space: O(n_params) for gradient vector
```

### 2.6 Fisher Matrix Calculation

**Operation:** `F_bb' = 0.5 × Tr[A_b A_b'^T] = 0.5 × (A_b ⊙ A_b'^T).sum()`

```
Per element:
  - Element-wise multiply: O(N_d²)
  - Sum: O(N_d²)

Upper triangle: n_params(n_params+1)/2 elements

Time: n_params² × O(N_d²) 
    = O(n_fields⁴ × n_bands² × n_fields² × n_obs²)
    = O(n_fields⁶ × n_bands² × n_obs²)

Space: O(n_params²) = O(n_fields⁴ × n_bands²)
```

---

## Part 3: Summary Tables

### 3.1 Initialization (One-Time)

| Operation | Time | Space | Scales with |
|-----------|------|-------|-------------|
| **Base kernels** | O(n_obs² × l_max) | O(n_bands × n_obs²) | n_bands, n_obs² |
| **Full kernels (precompute)** | O(n_fields⁴ × n_bands × n_obs²) | O(n_fields⁴ × n_bands × n_obs²) | **n_fields⁴**, n_bands |
| **Full kernels (on-the-fly)** | 0 (deferred) | O(n_bands × n_obs²) | n_bands only |

**Key:** Precompute mode pays O(n_fields⁴ × n_bands × n_obs²) space upfront!

### 3.2 Per Newton Iteration

| Operation | Time | Space | Scales with |
|-----------|------|-------|-------------|
| **Build C** | O(n_fields⁴ × n_bands × n_obs²) | O(n_fields² × n_obs²) | n_fields⁴, n_bands |
| **Cholesky** | **O(n_fields³ × n_obs³)** | O(n_fields² × n_obs²) | **n_obs³** only! |
| **Solve for v** | O(n_fields² × n_obs²) | O(n_fields × n_obs) | n_obs² |
| **A_b loop (serial)** | **O(n_fields⁵ × n_bands × n_obs³)** | O(n_fields⁴ × n_bands × n_obs²) | **n_fields⁵**, n_bands, **n_obs³** |
| **A_b loop (parallel)** | **O(n_fields⁵ × n_bands × n_obs³ / p)** | O(n_fields⁴ × n_bands × n_obs²) | Same / p |
| **Gradient** | O(n_fields⁴ × n_bands × n_obs²) | O(n_fields² × n_bands) | n_fields⁴ |
| **Fisher** | **O(n_fields⁶ × n_bands² × n_obs²)** | O(n_fields⁴ × n_bands²) | **n_fields⁶**, **n_bands²** |

**Dominant terms per iteration:**
1. **A_b loop:** O(n_fields⁵ × n_bands × n_obs³) — parallelizable
2. **Fisher:** O(n_fields⁶ × n_bands² × n_obs²) — becomes dominant for large n_fields!
3. **Cholesky:** O(n_fields³ × n_obs³) — serial, unavoidable

### 3.3 Mode Comparison

**Precompute mode:**
```
Init time:  O(n_fields⁴ × n_bands × n_obs²)
Init space: O(n_fields⁴ × n_bands × n_obs²)  [BOTTLENECK]

Per iteration:
  Time:  O(n_fields⁵ × n_bands × n_obs³)     [A_b loop]
       + O(n_fields⁶ × n_bands² × n_obs²)    [Fisher]
  Space: O(n_fields⁴ × n_bands × n_obs²)     [Stored kernels + A_b]
```

**On-the-fly mode:**
```
Init time:  O(n_obs² × l_max)              [Base kernels only]
Init space: O(n_bands × n_obs²)            [50% reduction!]

Per iteration:
  Time:  O(n_fields⁵ × n_bands × n_obs³)  [Same, build dominated by solve]
       + O(n_fields⁶ × n_bands² × n_obs²)
  Space: O(n_fields⁴ × n_bands × n_obs²)  [A_b storage during computation]
```

**Parallel mode (p cores):**
```
Same as above, but:
  A_b loop time: O(n_fields⁵ × n_bands × n_obs³ / p)
  
Speedup limited by Amdahl's law:
  If A_b loop is fraction f of total time, speedup ≤ 1/(1-f + f/p)
```

---

## Part 4: Scaling Behavior Analysis

### 4.1 How Does Complexity Scale?

**With n_fields (n_T + n_P):**
```
Space (precompute): O(n_fields⁴)  [QUARTIC!]
Time per iteration: O(n_fields⁵) for A_b, O(n_fields⁶) for Fisher

Example: 1 field → 6 fields
- Space: 1 → 6⁴ = 1296× increase!
- Time (A_b): 1 → 6⁵ = 7776× increase!
- Time (Fisher): 1 → 6⁶ = 46656× increase!

THIS IS WHY MULTI-FIELD IS SO EXPENSIVE!
```

**With n_bands:**
```
Space (precompute): O(n_bands)     [Linear]
Time (A_b): O(n_bands)             [Linear]
Time (Fisher): O(n_bands²)         [Quadratic]

Example: 5 bands → 10 bands
- Space: 2× increase
- Time (A_b): 2× increase
- Time (Fisher): 4× increase
```

**With n_obs:**
```
Space (precompute): O(n_obs²)      [Quadratic]
Time (A_b): O(n_obs³)              [Cubic]
Time (Fisher): O(n_obs²)           [Quadratic]

Example: 2000 pix → 4000 pix
- Space: 4× increase
- Time (A_b): 8× increase
- Time (Fisher): 4× increase
```

### 4.2 Which Parameter Dominates?

**For typical Euclid fsky=0.01:**
- n_fields ≈ 2-3 (e.g., n_T=1, n_P=1)
- n_obs ≈ 2000-5000
- n_bands ≈ 5-10

**Scaling comparison:**
```
Doubling n_fields: 2⁴ = 16× space, 2⁵ = 32× time (A_b)
Doubling n_obs:    2² = 4× space,  2³ = 8× time (A_b)
Doubling n_bands:  2× space,       2× time (A_b)

Conclusion: n_fields scaling is CATASTROPHIC!
```

**This explains why:**
- TT only (n_T=1, n_P=0): Fast and manageable
- TT+EE+BB (n_T=1, n_P=1): 16× slower, 16× more memory
- Multi-tomographic (n_T=6, n_P=6): Essentially impossible!

### 4.3 Crossover Points

**When does Fisher dominate over A_b?**

```
A_b time:    O(n_fields⁵ × n_bands × n_obs³)
Fisher time: O(n_fields⁶ × n_bands² × n_obs²)

Fisher > A_b when:
  n_fields⁶ × n_bands² × n_obs² > n_fields⁵ × n_bands × n_obs³
  n_fields × n_bands > n_obs

Example: n_obs = 2000
  If n_fields = 3, n_bands = 8: 3 × 8 = 24 < 2000 → A_b dominates
  If n_fields = 10, n_bands = 20: 10 × 20 = 200 < 2000 → A_b still dominates
  
Conclusion: For realistic problems, A_b loop is the bottleneck.
```

**When is parallelization worth it?**

```
Overhead ≈ 10-20% (thread creation, GIL, etc.)
Need speedup > 1.2 to be worthwhile

With p cores: time → time/p + overhead
Worthwhile if: 1/p + 0.2 < 0.8
  → p > 1.25 (i.e., any parallelization helps)

But: diminishing returns beyond ~8 cores due to memory bandwidth
```

---

## Part 5: Concrete Examples

### Example 1: Single-Field TT (n_T=1, n_P=0)

```
Parameters:
  n_T = 1, n_P = 0, n_obs = 2000, n_bands = 8, l_max = 256
  N_d = 2000
  n_params = 8 (8 TT bands)

Space:
  Base kernels: 8 × 2000² × 8 bytes = 256 MB
  Full kernels (precompute): 8 × 2000² × 8 bytes = 256 MB
  On-the-fly saves: 0 (kernels are the same!)

Time per iteration:
  Cholesky: O(2000³) ≈ 8B ops ≈ 1s
  A_b loop: 8 × O(2000³) ≈ 64B ops ≈ 8s (serial)
           → 2s (with 4 cores)
  Fisher: 36 × O(2000²) ≈ 144M ops ≈ 0.01s
  Total: ≈ 10s (serial), ≈ 3s (parallel)
```

### Example 2: Single-Field EE+BB (n_T=0, n_P=1)

```
Parameters:
  n_T = 0, n_P = 1, n_obs = 2000, n_bands = 8
  N_d = 4000 (Q + U)
  n_params = 16 (8 EE + 8 BB)

Space:
  Base kernels: 8 × 2000² × 8 × 3 = 768 MB (Kp, Km, Kx)
  Full kernels (precompute): 16 × 4000² × 8 = 2048 MB (2 GB)
  On-the-fly saves: 2048 - 768 = 1280 MB (60% reduction)

Time per iteration:
  Cholesky: O(4000³) ≈ 64B ops ≈ 8s
  A_b loop: 16 × O(4000³) ≈ 1T ops ≈ 128s (serial)
           → 32s (with 4 cores)
  Fisher: 136 × O(4000²) ≈ 2.2B ops ≈ 0.2s
  Total: ≈ 136s (serial), ≈ 40s (parallel)
```

### Example 3: Multi-Field TT+TE+EE+BB (n_T=1, n_P=1)

```
Parameters:
  n_T = 1, n_P = 1, n_obs = 2000, n_bands = 8
  N_d = 6000 (T + Q + U)
  n_params = 32 (8 TT + 8 TE + 8 EE + 8 BB)

Space:
  Base kernels: (8 × 2000² × 8) + (8 × 2000² × 8 × 3) = 1024 MB
  Full kernels (precompute): 32 × 6000² × 8 = 9216 MB (9.2 GB)
  On-the-fly saves: 9216 - 1024 = 8192 MB (89% reduction!)

Time per iteration:
  Cholesky: O(6000³) ≈ 216B ops ≈ 27s
  A_b loop: 32 × O(6000³) ≈ 6.9T ops ≈ 864s (serial)
           → 216s (with 4 cores)
  Fisher: 528 × O(6000²) ≈ 19B ops ≈ 1.9s
  Total: ≈ 893s ≈ 15 min (serial), ≈ 4 min (parallel)
```

### Example 4: Multi-Field with EB (n_T=1, n_P=1, NSIDE=64)

```
Parameters:
  n_T = 1, n_P = 1, n_obs = 4900, n_bands = 7
  N_d = 9800
  n_params = 42 (7 × 6 spectra)

Space:
  Base kernels: 7 × 4900² × 8 × (1 + 3) = 2688 MB
  Full kernels (precompute): 42 × 9800² × 8 = 32256 MB (32 GB)
  On-the-fly saves: 32256 - 2688 = 29568 MB (92% reduction!)

Time per iteration:
  Cholesky: O(9800³) ≈ 941B ops ≈ 118s
  A_b loop: 42 × O(9800³) ≈ 39T ops ≈ 4956s (serial)
           → 1239s (with 4 cores) ≈ 20 min
  Fisher: 903 × O(9800²) ≈ 87B ops ≈ 8.7s
  Total: ≈ 5083s ≈ 85 min (serial), ≈ 22 min (parallel)
```

---

## Part 6: Optimization Impact

### 6.1 On-the-Fly vs Precompute

**Space savings:**
```
Savings = (Full kernels - Base kernels) / Full kernels
        = 1 - (n_bands × n_obs²) / (n_params × N_d²)
        = 1 - (n_bands × n_obs²) / (n_fields² × n_bands × n_fields² × n_obs²)
        = 1 - 1/n_fields⁴

For n_fields = 1: 0% savings (kernels = base kernels)
For n_fields = 2: 93.75% savings
For n_fields = 3: 98.8% savings
For n_fields = 6: 99.92% savings

Conclusion: On-the-fly is ESSENTIAL for multi-field!
```

**Time cost:**
```
Overhead = (n_params × O(N_d²)) / (n_params × O(N_d³))
         = 1 / N_d
         = 1 / (n_fields × n_obs)

For n_obs = 2000, n_fields = 2: 1/4000 ≈ 0.025% (negligible!)

Conclusion: On-the-fly is essentially free.
```

### 6.2 Parallelization

**Speedup for A_b loop:**
```
Ideal speedup: p cores → p× faster
Real speedup: ~0.75p due to overhead

For p = 4: 3× faster (67% time reduction)
For p = 8: 6× faster (83% time reduction)
```

**Overall speedup (Amdahl's law):**
```
If A_b is fraction f of total time:
  Speedup = 1 / (1 - f + f/p)

Example 3 (TT+TE+EE+BB):
  A_b: 864s, Total: 893s → f = 0.97
  With p=4: Speedup = 1 / (0.03 + 0.97/4) = 3.6×
  Total time: 893s / 3.6 = 248s ≈ 4 min

Example 4 (with EB):
  A_b: 4956s, Total: 5083s → f = 0.975
  With p=4: Speedup = 1 / (0.025 + 0.975/4) = 3.7×
  Total time: 5083s / 3.7 = 1374s ≈ 23 min
```

---

## Part 7: Key Insights

### 7.1 Scaling Hierarchy (Worst to Best)

1. **n_fields:** O(n_fields⁴) space, O(n_fields⁵) time → **CATASTROPHIC**
2. **n_obs:** O(n_obs²) space, O(n_obs³) time → **SEVERE**
3. **n_bands:** O(n_bands) space, O(n_bands) time (A_b), O(n_bands²) time (Fisher) → **MANAGEABLE**

### 7.2 Why Multi-Field is Hard

```
Single field → Multi-field (2 fields):
  - Space: 16× increase
  - Time: 32× increase (A_b)

This is because:
  - n_params ∝ n_fields²
  - N_d ∝ n_fields
  - Storage: n_params × N_d² ∝ n_fields⁴
  - Compute: n_params × N_d³ ∝ n_fields⁵
```

### 7.3 Optimization Priorities

**For memory:**
1. **Use on-the-fly** (92%+ savings for multi-field)
2. Reduce n_obs (NSIDE 64→32: 4× reduction)
3. Reduce n_bands (moderate savings)

**For speed:**
1. **Parallelize A_b loop** (3-6× speedup)
2. Reduce n_obs (8× speedup for 2× reduction)
3. Reduce n_fields if scientifically acceptable

### 7.4 Practical Limits

**On a 32 GB machine:**
```
Available for kernels: ~20 GB (leaving 12 GB for OS + working arrays)
Max n_params × N_d²: 20 GB / 8 bytes = 2.5B elements

Examples that fit:
  - n_T=1, n_P=1, n_obs=2000, n_bands=8: 32 × 6000² = 1.15B ✓
  - n_T=1, n_P=1, n_obs=2000, n_bands=16: 64 × 6000² = 2.3B ✓
  - n_T=1, n_P=1, n_obs=3000, n_bands=8: 32 × 9000² = 2.6B ✗
  
With on-the-fly: All fit! (Only need base kernels)
```

---

## Summary Table: Complexity Scaling

| Mode | Init Space | Init Time | Iter Space | Iter Time |
|------|------------|-----------|------------|-----------|
| **Precompute** | O(n_fields⁴ n_bands n_obs²) | O(n_fields⁴ n_bands n_obs²) | O(n_fields⁴ n_bands n_obs²) | O(n_fields⁵ n_bands n_obs³) |
| **On-the-fly** | O(n_bands n_obs²) | O(n_obs² l_max) | O(n_fields⁴ n_bands n_obs²) | O(n_fields⁵ n_bands n_obs³) |
| **Parallel (p)** | Same as mode | Same | Same | O(n_fields⁵ n_bands n_obs³ / p) |

**Dominant scaling:**
- **Space:** n_fields⁴ (quartic!) for precompute, n_bands (linear) for on-the-fly
- **Time:** n_fields⁵ (quintic!) and n_obs³ (cubic)
- **n_bands:** Linear in time and space (except Fisher: quadratic in time)
