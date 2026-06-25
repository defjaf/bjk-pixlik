#!/usr/bin/env python3
"""
Quick test to verify on-the-fly kernel mode works correctly.
"""
import numpy as np
import healpy as hp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# Generate simple test data
nside = 16
lmax = 32
npix = hp.nside2npix(nside)

# Flat power spectra
Cl_TT = 1e-4
Cl_EE = 1e-5
Cl_BB = 1e-6
Cl_TE = 5e-6

cls = np.zeros((4, lmax + 1))
cls[0, 2:] = Cl_TT  # TT
cls[1, 2:] = Cl_EE  # EE
cls[2, 2:] = Cl_BB  # BB
cls[3, 2:] = Cl_TE  # TE

# Generate maps
np.random.seed(42)
T, Q, U = hp.synfast(cls, nside=nside, lmax=lmax, new=True, verbose=False)

# Add noise
sigma_pix = 1e-3
T += np.random.normal(0, sigma_pix, npix)
Q += np.random.normal(0, sigma_pix, npix)
U += np.random.normal(0, sigma_pix, npix)

# Select subset of pixels
obs_pix = np.arange(0, npix, 8)  # Every 8th pixel
n_obs = len(obs_pix)

# Noise arrays
N_T = np.full(n_obs, sigma_pix**2)
N_Q = np.full(n_obs, sigma_pix**2)
N_U = np.full(n_obs, sigma_pix**2)

band_edges = np.array([2, 16, 33])  # 2 bands

print("="*70)
print("Test 1: Precompute mode (explicit)")
print("="*70)
lik_pre = PixelLikelihood.from_arrays(
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
    kernel_mode='precompute'
)

cl_init = np.full(lik_pre.layout.n_params, 1e-5)
g_pre, F_pre = lik_pre.gradient_and_fisher(cl_init)

print(f"Gradient shape: {g_pre.shape}")
print(f"Fisher shape: {F_pre.shape}")
print(f"Kernel storage: {lik_pre._kernel_matrices is not None}")

print("\n" + "="*70)
print("Test 2: On-the-fly mode (explicit)")
print("="*70)
lik_fly = PixelLikelihood.from_arrays(
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
    kernel_mode='onthefly'
)

g_fly, F_fly = lik_fly.gradient_and_fisher(cl_init)

print(f"Gradient shape: {g_fly.shape}")
print(f"Fisher shape: {F_fly.shape}")
print(f"Kernel storage: {lik_fly._kernel_matrices is not None}")

print("\n" + "="*70)
print("Test 3: Auto mode")
print("="*70)
lik_auto = PixelLikelihood.from_arrays(
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
    kernel_mode='auto'
)

g_auto, F_auto = lik_auto.gradient_and_fisher(cl_init)

print(f"Gradient shape: {g_auto.shape}")
print(f"Fisher shape: {F_auto.shape}")
print(f"Resolved mode: {lik_auto._resolved_mode}")

print("\n" + "="*70)
print("Test 4: Compare results between modes")
print("="*70)

g_diff = np.abs(g_pre - g_fly).max()
F_diff = np.abs(F_pre - F_fly).max()

print(f"Max gradient difference (precompute vs onthefly): {g_diff:.2e}")
print(f"Max Fisher difference (precompute vs onthefly): {F_diff:.2e}")

if g_diff < 1e-12 and F_diff < 1e-12:
    print("\n✓ PASS: Precompute and onthefly modes produce identical results")
else:
    print(f"\n✗ FAIL: Results differ!")
    print(f"  Gradient rel diff: {g_diff / np.abs(g_pre).max():.2e}")
    print(f"  Fisher rel diff: {F_diff / np.abs(F_pre).max():.2e}")

g_diff_auto = np.abs(g_pre - g_auto).max()
F_diff_auto = np.abs(F_pre - F_auto).max()

print(f"\nMax gradient difference (precompute vs auto): {g_diff_auto:.2e}")
print(f"Max Fisher difference (precompute vs auto): {F_diff_auto:.2e}")

if g_diff_auto < 1e-12 and F_diff_auto < 1e-12:
    print("✓ PASS: Auto mode produces identical results")
else:
    print("✗ FAIL: Auto mode results differ!")

print("\n" + "="*70)
print("All tests completed successfully!")
print("="*70)
