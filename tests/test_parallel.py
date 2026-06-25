#!/usr/bin/env python3
"""
Test parallelization speedup.
"""
import numpy as np
import healpy as hp
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# Generate test data
nside = 16
lmax = 48
npix = hp.nside2npix(nside)

np.random.seed(42)
T = np.random.randn(npix) * 1e-3
Q = np.random.randn(npix) * 1e-3
U = np.random.randn(npix) * 1e-3

obs_pix = np.arange(0, npix, 4)
n_obs = len(obs_pix)

N_T = np.full(n_obs, 1e-6)
N_Q = np.full(n_obs, 1e-6)
N_U = np.full(n_obs, 1e-6)

band_edges = np.array([2, 16, 32, 49])

print("="*70)
print("PARALLELIZATION TEST")
print("="*70)

for n_threads in [1, 2, 4, 8]:
    print(f"\n--- n_threads={n_threads} ---")

    lik = PixelLikelihood.from_arrays(
        d_T_list=[T[obs_pix]],
        d_Q_list=[Q[obs_pix]],
        d_U_list=[U[obs_pix]],
        obs_pix=obs_pix,
        nside=nside,
        N_T_list=[N_T],
        N_Q_list=[N_Q],
        N_U_list=[N_U],
        lmin=2,
        lmax=lmax,
        band_edges=band_edges,
        band_model='Cl',
        kernel_mode='onthefly',
        n_threads=n_threads,
        include_EB=False
    )

    cl_init = np.full(lik.layout.n_params, 1e-5)

    # Warm up
    _ = lik.gradient_and_fisher(cl_init)

    # Time it
    t0 = time.time()
    g, F = lik.gradient_and_fisher(cl_init)
    t_elapsed = time.time() - t0

    print(f"Time: {t_elapsed:.3f}s")
    if n_threads == 1:
        t_serial = t_elapsed
    else:
        speedup = t_serial / t_elapsed
        efficiency = speedup / n_threads * 100
        print(f"Speedup: {speedup:.2f}× (efficiency: {efficiency:.0f}%)")

print("\n" + "="*70)
print("Verify results are identical across thread counts")
print("="*70)

results = []
for n_threads in [1, 8]:
    lik = PixelLikelihood.from_arrays(
        d_T_list=[T[obs_pix]],
        d_Q_list=[Q[obs_pix]],
        d_U_list=[U[obs_pix]],
        obs_pix=obs_pix,
        nside=nside,
        N_T_list=[N_T],
        N_Q_list=[N_Q],
        N_U_list=[N_U],
        lmin=2,
        lmax=lmax,
        band_edges=band_edges,
        band_model='Cl',
        kernel_mode='onthefly',
        n_threads=n_threads,
        include_EB=False
    )

    g, F = lik.gradient_and_fisher(cl_init)
    results.append((g, F))

g1, F1 = results[0]
g8, F8 = results[1]

g_diff = np.abs(g1 - g8).max()
F_diff = np.abs(F1 - F8).max()

print(f"Max gradient difference (1 vs 8 threads): {g_diff:.2e}")
print(f"Max Fisher difference (1 vs 8 threads): {F_diff:.2e}")

if g_diff < 1e-12 and F_diff < 1e-12:
    print("\n✓ PASS: Results are identical")
else:
    print(f"\n✗ FAIL: Results differ!")
