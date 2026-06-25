#!/usr/bin/env python3
"""
Benchmark memory usage and performance for different kernel modes.
"""
import numpy as np
import healpy as hp
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

def format_bytes(nbytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"

def test_case(name, n_T, n_P, n_obs, lmax, nbands, kernel_mode):
    """Run a single test case and report stats."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Config: n_T={n_T}, n_P={n_P}, n_obs={n_obs}, lmax={lmax}, nbands={nbands}")
    print(f"Kernel mode: {kernel_mode}")

    nside = 16
    npix = hp.nside2npix(nside)

    # Generate dummy data
    np.random.seed(42)
    obs_pix = np.arange(n_obs)

    d_T_list = [np.random.randn(n_obs) * 1e-3 for _ in range(n_T)]
    d_Q_list = [np.random.randn(n_obs) * 1e-3 for _ in range(n_P)]
    d_U_list = [np.random.randn(n_obs) * 1e-3 for _ in range(n_P)]

    N_T_list = [np.full(n_obs, 1e-6) for _ in range(n_T)]
    N_Q_list = [np.full(n_obs, 1e-6) for _ in range(n_P)]
    N_U_list = [np.full(n_obs, 1e-6) for _ in range(n_P)]

    band_edges = np.linspace(2, lmax + 1, nbands + 1, dtype=int)

    # Build likelihood
    t0 = time.time()
    lik = PixelLikelihood.from_arrays(
        d_T_list=d_T_list,
        d_Q_list=d_Q_list,
        d_U_list=d_U_list,
        obs_pix=obs_pix,
        nside=nside,
        N_T_list=N_T_list,
        N_Q_list=N_Q_list,
        N_U_list=N_U_list,
        lmin=2,
        lmax=lmax,
        band_edges=band_edges,
        band_model='Cl',
        kernel_mode=kernel_mode
    )
    t_init = time.time() - t0

    # Estimate memory
    Nd = len(lik.d)
    n_params = lik.layout.n_params

    base_kernel_mem = 0
    if lik._TT_kernels is not None:
        base_kernel_mem += nbands * n_obs**2 * 8
    if lik._Kp is not None:
        base_kernel_mem += 3 * nbands * n_obs**2 * 8  # Kp, Km, Kx

    full_kernel_mem = n_params * Nd**2 * 8

    actual_kernel_mem = full_kernel_mem if lik._kernel_matrices is not None else base_kernel_mem

    print(f"\nData vector: N_d = {Nd}")
    print(f"Parameters: n_params = {n_params}")
    print(f"Base kernels: {format_bytes(base_kernel_mem)}")
    print(f"Full kernels: {format_bytes(full_kernel_mem)}")
    print(f"Actual storage: {format_bytes(actual_kernel_mem)}")

    if lik._kernel_matrices is not None:
        savings_pct = 0
    else:
        savings_pct = 100 * (1 - base_kernel_mem / full_kernel_mem)
    print(f"Memory savings: {savings_pct:.1f}%")
    print(f"Init time: {t_init:.3f} s")

    # Run gradient/Fisher
    cl_init = np.full(n_params, 1e-5)
    t0 = time.time()
    g, F = lik.gradient_and_fisher(cl_init)
    t_grad = time.time() - t0

    print(f"Gradient+Fisher time: {t_grad:.3f} s")

    return {
        'Nd': Nd,
        'n_params': n_params,
        'base_mem': base_kernel_mem,
        'full_mem': full_kernel_mem,
        'actual_mem': actual_kernel_mem,
        'savings_pct': savings_pct,
        't_init': t_init,
        't_grad': t_grad,
        'mode': lik._resolved_mode
    }

# Test cases
print("KERNEL MODE COMPARISON BENCHMARK")
print("="*70)

cases = [
    ("Small: Single-field TT", 1, 0, 500, 32, 4),
    ("Medium: Single-field EE+BB", 0, 1, 1000, 64, 6),
    ("Large: Multi-field TT+TE+EE+BB", 1, 1, 1500, 64, 8),
]

results = []
for name, n_T, n_P, n_obs, lmax, nbands in cases:
    for mode in ['precompute', 'onthefly']:
        result = test_case(name, n_T, n_P, n_obs, lmax, nbands, mode)
        result['case'] = name
        results.append(result)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for case_name, n_T, n_P, n_obs, lmax, nbands in cases:
    pre = [r for r in results if r['case'] == case_name and r['mode'] == 'precompute'][0]
    fly = [r for r in results if r['case'] == case_name and r['mode'] == 'onthefly'][0]

    print(f"\n{case_name}:")
    print(f"  Memory: {format_bytes(pre['full_mem'])} → {format_bytes(fly['actual_mem'])} "
          f"({fly['savings_pct']:.1f}% savings)")

    overhead_pct = 100 * (fly['t_grad'] - pre['t_grad']) / pre['t_grad']
    print(f"  Grad/Fisher time: {pre['t_grad']:.3f}s → {fly['t_grad']:.3f}s "
          f"({overhead_pct:+.2f}% overhead)")

print("\n" + "="*70)
print("Test auto mode on large case (should choose onthefly if memory-limited)")
print("="*70)

result_auto = test_case(
    "Large Multi-field (auto mode)",
    n_T=1, n_P=1, n_obs=2000, lmax=64, nbands=8,
    kernel_mode='auto'
)

print(f"\nAuto mode chose: {result_auto['mode']}")
