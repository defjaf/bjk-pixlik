# Disk-Based Kernel Storage: 1990s vs Modern Trade-offs

**Context:** The 1990s Fortran implementation stored derivative kernels (`A_p = C^{-1} K_p`) on disk rather than in RAM. Is this worth considering for the modern Python implementation?

---

## Historical Implementation (quadest.f)

### How It Worked

**Strategy:** Pre-compute all derivative kernels, write to disk, read only 2 at a time during Fisher calculation.

**File structure:**
```fortran
! Shell script provides base filenames:
Apbase = "./Apdat/dmr_95_Ap."

! For each parameter p:
!   Write A_p to: Apbase001, Apbase002, ..., Apbase028
```

**Fisher matrix calculation (quadest.f:219-229):**
```fortran
do i = 1, np
    call read_Ap(nd, Apbase, i, B1)     ! Read A_i from disk
    do j = 1, i
        call read_Ap(nd, Apbase, j, B2) ! Read A_j from disk
        call get_fish(nd, B1, B2, fish_element)
        Fish(i,j) = 0.5 * Tr[B1 B2]     ! Compute F_ij
        Fish(j,i) = fish_element
    enddo
enddo
```

**Key observation (line 208):**
```fortran
! only need two A_p matrices because only have two in memory at a time.
double precision B1(ndmax,ndmax), B2(ndmax,ndmax)
```

**Memory footprint:**
- Kernels on disk: `n_params × N_d² × 4 bytes` (single precision)
- In RAM at once: `2 × N_d² × 8 bytes` (double precision)

**Example (COBE DMR):**
- N_d = 1500, n_params = 28
- Disk storage: 28 × 1500² × 4 = 252 MB
- RAM at once: 2 × 1500² × 8 = 36 MB ✓ (fits in 64 MB machine!)

---

## Why It Worked in the 1990s

### 1. RAM was the Limiting Resource

**Typical 1990s workstation:**
- RAM: 64-512 MB
- Disk: 2-9 GB (cheap, plentiful)
- Disk I/O: ~5-10 MB/s

**Trade-off:**
- **RAM constraint:** Cannot hold 28 × 36 MB = 1 GB in RAM
- **Disk solution:** Store 252 MB on disk, use 36 MB in RAM
- **I/O cost:** Read 252 MB total during Fisher calculation ≈ 25-50 seconds

### 2. Kernels Were Pre-Computed

**Workflow (run_quadest.sh):**
```bash
# Iteration 0: Build all derivative kernels (done once)
for iband in $bandlist; do
    ${CT_prog} < ${Cderivbase}${iband} > ${Apbase}${iband}
done

# Iterations 1-10: Read kernels from disk, compute Fisher, iterate
for niter in $niterlist; do
    quadest < input_params  # Reads Ap files from disk
done
```

**Key:** Kernels built once, reused across iterations (amortized cost).

### 3. Sequential Access Pattern

**Disk reads:** Sequential, large blocks
- Read entire N_d × N_d matrix in one `read(16)` call
- 1990s disks optimized for sequential access (5-10 MB/s)
- Random access penalty small (kernels are separate files)

---

## Modern Context (2026)

### Changed Hardware Landscape

| Resource | 1990s | 2020s | Factor |
|----------|-------|-------|--------|
| **RAM** | 64-512 MB | 32-128 GB | **250×** |
| **Disk speed (seq)** | 5-10 MB/s | 3000-7000 MB/s (NVMe SSD) | **500×** |
| **Disk speed (rand)** | 0.1-1 MB/s | 500-1500 MB/s (NVMe SSD) | **1500×** |
| **CPU cache** | 256 KB-1 MB | 8-64 MB | **50×** |

**Key change:** RAM is now abundant, disk is fast.

### Modern Trade-off Analysis

**Option A: In-RAM (current)**
```python
self._kernel_matrices = self._build_all_kernels()  # 32 GB
```
- **Pro:** Instant access (0 latency)
- **Con:** 32 GB RAM for multi-field + EB

**Option B: On-the-fly (proposed)**
```python
K_b = self._build_kernel(idx)  # Rebuild when needed
```
- **Pro:** 16 GB RAM (50% reduction)
- **Con:** Rebuild cost ~0.01% of total compute

**Option C: Disk-based (1990s strategy)**
```python
K_b = np.load(f'kernels/kernel_{idx:03d}.npy')  # Read from disk
```
- **Pro:** Minimal RAM (~256 MB for 2 kernels)
- **Con:** Disk I/O overhead

---

## Disk I/O Cost Analysis

### Read Pattern During gradient_and_fisher

**Current code structure:**
```python
for idx in range(np_total):
    K_b = self._kernel_matrices[idx]  # In-RAM: 0 ns
    W = solve_triangular(L, K_b, lower=True)  # O(N_d^3)
```

**With disk storage:**
```python
for idx in range(np_total):
    K_b = np.load(f'kernels/kernel_{idx:03d}.npy')  # Disk: ? ms
    W = solve_triangular(L, K_b, lower=True)
```

### I/O Time Calculation

**Example: Multi-field (n_T=1, n_P=1, EB), NSIDE=64**
- N_d = 9800
- n_params = 42
- Kernel size: 9800² × 8 bytes = 768 MB per kernel

**Per Newton iteration:**
- Read 42 kernels × 768 MB = 32 GB total I/O

**Modern NVMe SSD (7 GB/s sequential):**
- Time: 32 GB / 7 GB/s = **4.6 seconds** per iteration

**Triangular solves (CPU):**
- 42 × O(9800³) ≈ 42 × 1B ops ≈ **4 seconds** per iteration (rough estimate)

**Ratio:** I/O time ≈ 50% of compute time (significant overhead!)

### For Smaller Problems

**Single-field EE+BB (n_T=0, n_P=1), n_obs=2000**
- N_d = 4000
- n_params = 16
- Kernel size: 4000² × 8 = 128 MB per kernel

**Per Newton iteration:**
- Read 16 × 128 MB = 2 GB total I/O
- NVMe time: 2 GB / 7 GB/s = **0.3 seconds**
- Triangular solves: ~1 second
- Ratio: I/O ≈ 30% of compute (moderate overhead)

---

## Implementation Complexity

### Disk-Based Implementation

**Challenges:**
1. **File management:**
   ```python
   def _save_kernels_to_disk(self, cache_dir='./kernel_cache'):
       os.makedirs(cache_dir, exist_ok=True)
       for idx in range(self.layout.n_params):
           K = self._build_kernel(idx)
           np.save(f'{cache_dir}/kernel_{idx:03d}.npy', K)
   
   def _load_kernel(self, idx, cache_dir='./kernel_cache'):
       return np.load(f'{cache_dir}/kernel_{idx:03d}.npy')
   ```

2. **Cache invalidation:** What if parameters change?
   - Need hash of (obs_pix, nside, lmin, lmax, band_edges, beam, n_T, n_P)
   - Store metadata to verify cache validity

3. **Disk space:** 32 GB for multi-field case (but cheap)

4. **Parallel I/O:** Can't easily parallelize kernel reads

### On-the-Fly Implementation

**Much simpler:**
```python
def _build_kernel(self, idx):
    # ~30 lines of code, pure computation
    # No file I/O, no cache management
```

---

## Recommendation: **No, Disk Storage Not Worth It**

### Why Not?

**1. RAM is cheap and abundant**
- 64 GB RAM costs ~$200 in 2026
- Multi-field problem fits in 32 GB with on-the-fly kernels
- Even 32 GB RAM costs ~$100

**2. I/O overhead is significant**
- NVMe SSD: 4.6s I/O vs 4s compute = 50% overhead
- Worse with SATA SSD or HDD
- On-the-fly rebuild: <0.01% overhead

**3. Implementation complexity**
- Disk: need file management, cache validation, cleanup
- On-the-fly: pure computation, no state

**4. No amortization in modern workflow**
- 1990s: Pre-compute kernels once, reuse across many iterations
- Modern: Typically run once per dataset (single Newton run)
- Cannot amortize disk write cost

**5. Modern SSDs aren't that much faster than RAM**
- NVMe: 7 GB/s sequential
- RAM: 50-100 GB/s bandwidth
- Latency: SSD ~100 µs vs RAM ~100 ns (1000× difference)

### When Disk Storage Might Make Sense

**Only if ALL of:**
1. RAM truly limited (<16 GB available)
2. Cannot reduce NSIDE
3. Cannot use on-the-fly kernels (why not?)
4. Fast NVMe SSD available
5. Running many Newton iterations (amortize write cost)

**More realistic scenario:** Use on-the-fly kernels (50% RAM reduction, <1% overhead)

---

## Hybrid Strategy: Disk Cache for Pre-Computation

**Alternative use case:** Not for gradient/Fisher, but for **exploratory analysis**.

**Scenario:**
```python
# Expensive: Build likelihood multiple times with same geometry
for beam_fwhm in [5, 7, 10, 15]:
    lik = PixelLikelihood(..., beam=gaussian_beam(beam_fwhm))
    # Geometry and base kernels are identical, only beam changes
```

**Optimization:**
```python
# Cache geometry and base kernels
cache_key = hash((obs_pix, nside, lmin, lmax))
if os.path.exists(f'cache/{cache_key}_geometry.npz'):
    geom = np.load(f'cache/{cache_key}_geometry.npz')
    Kp_base = np.load(f'cache/{cache_key}_Kp.npy')
    # ...
else:
    geom = _compute_spin2_geometry(...)
    Kp_base = _build_spin2_kernels(...)  # Without beam
    # Save to cache
```

**Benefit:** Avoid recomputing geometry (expensive) for parameter scans

**Use case:** Limited (most analyses use fixed geometry)

---

## Modern Best Practice Hierarchy

**For memory-limited multi-field runs:**

1. **Reduce NSIDE** (16× memory reduction, lose high-ℓ)
   - NSIDE 64 → 32: Immediate, no code changes

2. **On-the-fly kernels** (50% memory reduction, <1% overhead)
   - ~100 lines of code, backward compatible

3. **Fewer bands** (proportional reduction, lose ℓ-resolution)
   - Change `band_edges` parameter

4. **Separate runs** (avoid TT+TE+EE+BB joint)
   - Run TT alone, then EE+BB alone

5. **Buy more RAM** ($100-200 for 32 GB)
   - Most cost-effective if running frequently

**Not recommended:**
- ❌ Disk-based kernel storage (complex, 50% I/O overhead)
- ❌ Band-diagonal approximation (complex, approximation error)

---

## Summary

**Historical context:**
- 1990s Fortran stored kernels on disk because RAM was limited (64-512 MB)
- I/O was acceptable (25-50s) compared to total runtime (hours)
- Kernels pre-computed once, amortized across many iterations

**Modern context:**
- RAM is 250× larger (32-128 GB)
- Disk is 500× faster (NVMe SSD)
- **But**: RAM is still 100× faster than SSD for repeated access
- **And**: RAM has grown more than kernel memory requirements

**Conclusion:**
- Disk storage solved a 1990s RAM limitation
- Modern RAM abundance makes this unnecessary
- On-the-fly kernel construction is simpler and faster
- Disk storage would add 50% I/O overhead with complex implementation

**Recommendation:** Implement on-the-fly kernels, not disk storage.
