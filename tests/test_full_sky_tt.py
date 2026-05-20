"""
Full-sky uniform-noise test for PixelLikelihood (spin-0).

Known exact result: on the full sky with uniform white noise and Delta-l=1 bands,
  BJK98 ML = NaMaster = anafast(d) - N_l   per multipole.

Proof: the full-sky likelihood decouples per l.  The ML condition at ell l is
  C_l + N_l  =  (1/(2l+1)) sum_m |d_lm|^2  =  anafast(d)[l]
so  C_l^ML  =  anafast(d)[l] - N_l.
NaMaster's mode-coupling matrix is the identity at f_sky=1, so it returns the
same estimate.

Numerical precision notes (NSIDE=8, LMAX=15):
- HEALPix uses an approximate quadrature rule for map2alm.  At NSIDE=8 and
  l close to 2*NSIDE=16, quadrature errors reach ~1e-3.
- NaMaster uses map2alm internally (niter=0) while hp.anafast default is
  niter=3; the two give different per-l spectra at high l by ~1e-2.
- BJK uses exact Legendre polynomial recurrence at pixel positions (no SHT),
  and agrees with the hp.anafast(niter=3) reference to <1e-3 at all l.
- We therefore use separate tolerances for BJK and NaMaster, and additionally
  compare BJK and NaMaster directly for l <= 12 (well below 2*NSIDE).

NSIDE=8 (768 pixels, lmax=15): kernel matrices are 768x768 ~ 4.5 MB each.

Run from sim_runs/bjk/:
    python3 test_full_sky_tt.py
"""

import sys, os, tempfile
import numpy as np
import healpy as hp
import pymaster as nmt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
NSIDE      = 8
LMAX       = 2 * NSIDE - 1        # = 15, safely below Nyquist
LMAX_FIELD = 3 * NSIDE - 1        # = 23, NmtField's internal lmax
NPIX       = hp.nside2npix(NSIDE)
OMEGA_PIX  = 4.0 * np.pi / NPIX

# Signal: flat D_l = A/(2pi), C_l = A/(l(l+1))
A         = 1e-4
ell_all   = np.arange(LMAX + 1, dtype=float)
cl_signal = np.zeros(LMAX + 1)
cl_signal[2:] = A / (ell_all[2:] * (ell_all[2:] + 1))

# Noise: uniform white, per-pixel std = SIGMA_PIX
# Full-sky white noise power: N_l = sigma_pix^2 * Omega_pix
SIGMA_PIX = 1e-3
N_L       = SIGMA_PIX**2 * OMEGA_PIX

print(f"NSIDE={NSIDE}  NPIX={NPIX}  LMAX={LMAX}  LMAX_FIELD={LMAX_FIELD}")
print(f"sigma_pix = {SIGMA_PIX:.2e}  N_l = {N_L:.4e}")
print(f"S/N at l=10: {cl_signal[10]/N_L:.1f}")

# Delta-l=1 bands: one multipole per band, l=2..LMAX
BAND_EDGES = np.arange(2, LMAX + 2, dtype=int)   # [2,3,...,16]
ell_b      = BAND_EDGES[:-1].astype(float)        # [2,3,...,15]
nbands     = len(ell_b)

# ---------------------------------------------------------------------------
# Simulate data (fixed seed)
rng       = np.random.default_rng(42)
alm_sig   = hp.synalm(cl_signal, lmax=LMAX)
sig_map   = hp.alm2map(alm_sig, NSIDE, lmax=LMAX)
noise_map = rng.normal(0.0, SIGMA_PIX, NPIX)
data_map  = sig_map + noise_map

# ---------------------------------------------------------------------------
# Reference: hp.anafast(data, niter=3) - N_l
# niter=3 is hp.anafast's default; it gives a more accurate quadrature estimate
# and is what BJK (Legendre recurrence at exact pixel positions) should agree with.
cl_anaf = hp.anafast(data_map, lmax=LMAX, iter=3)
cl_ref   = cl_anaf[2:LMAX+1] - N_L    # shape (nbands,)

# ---------------------------------------------------------------------------
# BJK Newton-Raphson
with tempfile.TemporaryDirectory() as tmp:
    data_fits = os.path.join(tmp, "data.fits")
    ninv_fits = os.path.join(tmp, "ninv.fits")
    hp.write_map(data_fits, data_map, overwrite=True)
    hp.write_map(ninv_fits, np.full(NPIX, 1.0 / SIGMA_PIX**2), overwrite=True)

    lik = PixelLikelihood(
        data_fits=data_fits, ninv_fits=ninv_fits,
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES,
        nfields=1, spin=0,
    )
    cl_bjk, sigma_bjk, _ = lik.newton_raphson(
        cl_ref.copy(), max_iter=30, tol=1e-10)

# ---------------------------------------------------------------------------
# NaMaster (full sky, beam=ones to suppress pixel-window deconvolution so
# both estimators see the same effective beam as BJK)
mask_full   = np.ones(NPIX)
f0          = nmt.NmtField(mask_full, [data_map],
                            beam=np.ones(LMAX_FIELD + 1))
b_nmt       = nmt.NmtBin.from_lmax_linear(LMAX_FIELD, nlb=1, is_Dell=False)
ell_nmt_all = b_nmt.get_effective_ells()
wsp         = nmt.NmtWorkspace.from_fields(f0, f0, b_nmt)
cl_nmt_all  = wsp.decouple_cell(nmt.compute_coupled_cell(f0, f0))[0]
keep        = (ell_nmt_all >= 2) & (ell_nmt_all <= LMAX)
ell_nmt     = ell_nmt_all[keep]
cl_nmt      = cl_nmt_all[keep] - N_L

# NaMaster Gaussian covariance (signal + noise as theory Cls)
# Use the ML point from BJK as the theory Cls (close to true spectrum)
cl_tot_theory = np.zeros(LMAX_FIELD + 1)
cl_tot_theory[2:LMAX+1] = cl_bjk + N_L   # C_l^signal + N_l per ell
cl_tot_theory[LMAX+1:]  = N_L             # noise-only beyond lmax
cw         = nmt.NmtCovarianceWorkspace.from_fields(f0, f0, f0, f0)
covar_nmt  = cw.gaussian_covariance([cl_tot_theory], [cl_tot_theory],
                                     [cl_tot_theory], [cl_tot_theory], wsp)
# covar shape (n_bands_field, n_bands_field); extract diagonal then trim to l<=LMAX
sigma_nmt_all = np.sqrt(np.maximum(np.diag(covar_nmt), 0))
sigma_nmt     = sigma_nmt_all[keep]

# ---------------------------------------------------------------------------
# Analytical Fisher error bar: sigma_l = sqrt(2/(2l+1)) * (C_l + N_l)
# This is the Cramér-Rao bound for the ML estimator of a single multipole
# on the full sky with uniform noise.  Valid for both BJK and NaMaster.
sigma_fisher = np.sqrt(2.0 / (2.0 * ell_b + 1)) * (cl_bjk + N_L)

# ---------------------------------------------------------------------------
# Report: bandpowers
def Dl(l, cl): return l * (l + 1) / (2.0 * np.pi) * cl
Dtrue = A / (2.0 * np.pi)
print(f"\nInput truth D_l = {Dtrue:.4e}  (all multipoles)")

print("\n--- Bandpowers ---")
print("="*90)
print(f"{'l':>4}  {'D_ref':>10}  {'D_bjk':>10}  {'D_nmt':>10}  "
      f"{'(bjk-ref)/ref':>14}  {'(nmt-ref)/ref':>14}  {'(bjk-nmt)/nmt':>14}")
print("-"*90)
for i, lb in enumerate(ell_b):
    lb = int(lb)
    Dr = Dl(lb, cl_ref[i])
    Db = Dl(lb, cl_bjk[i])
    Dn = Dl(ell_nmt[i], cl_nmt[i]) if i < len(cl_nmt) else float('nan')
    fb = (Db - Dr) / (abs(Dr) + 1e-40)
    fn = (Dn - Dr) / (abs(Dr) + 1e-40)
    fbn= (Db - Dn) / (abs(Dn) + 1e-40) if i < len(cl_nmt) else float('nan')
    print(f"  {lb:2d}  {Dr:10.4e}  {Db:10.4e}  {Dn:10.4e}  {fb:14.2e}  {fn:14.2e}  {fbn:14.2e}")
print("="*90)

# Report: error bars
print("\n--- Error bars (sigma = sqrt(Var(C_l))) ---")
print("="*80)
print(f"{'l':>4}  {'sigma_analytic':>16}  {'sigma_bjk':>12}  {'sigma_nmt':>12}  "
      f"{'(bjk-an)/an':>13}  {'(nmt-an)/an':>13}")
print("-"*80)
for i, lb in enumerate(ell_b):
    lb  = int(lb)
    san = sigma_fisher[i]
    sb  = sigma_bjk[i]
    sn  = sigma_nmt[i] if i < len(sigma_nmt) else float('nan')
    fb  = (sb - san) / (san + 1e-40)
    fn  = (sn - san) / (san + 1e-40) if i < len(sigma_nmt) else float('nan')
    print(f"  {lb:2d}  {san:16.4e}  {sb:12.4e}  {sn:12.4e}  {fb:13.2e}  {fn:13.2e}")
print("="*80)

# ---------------------------------------------------------------------------
# Quantitative checks
LOW_LMAX   = int(1.5 * NSIDE)   # = 12; well below 2*NSIDE, quadrature reliable
low_mask   = ell_b.astype(int) <= LOW_LMAX

rel_bjk    = np.abs(cl_bjk - cl_ref) / (np.abs(cl_ref) + 1e-40)
rel_nmt    = np.abs(cl_nmt[:nbands] - cl_ref) / (np.abs(cl_ref) + 1e-40)
rel_bn     = np.abs(cl_bjk - cl_nmt[:nbands]) / (np.abs(cl_nmt[:nbands]) + 1e-40)

rel_sb_an  = np.abs(sigma_bjk - sigma_fisher) / sigma_fisher
rel_sn_an  = np.abs(sigma_nmt[:nbands] - sigma_fisher) / sigma_fisher

print(f"\nBandpowers:")
print(f"  Max |BJK - ref| / |ref|             = {rel_bjk.max():.2e}  (target < 2e-3)")
print(f"  Max |NMT - ref| / |ref|             = {rel_nmt.max():.2e}  (target < 2e-2)")
print(f"  Max |BJK - NMT| / |NMT| for l<={LOW_LMAX} = {rel_bn[low_mask].max():.2e}  (target < 5e-3)")
print(f"\nError bars:")
print(f"  Max |sigma_BJK - sigma_analytic| / sigma_analytic = {rel_sb_an.max():.2e}  (target < 2e-3)")
print(f"  Max |sigma_NMT - sigma_analytic| / sigma_analytic = {rel_sn_an.max():.2e}  (target < 2e-2)")

PASS1 = (rel_bjk.max()        < 2e-3  and
         rel_nmt.max()        < 2e-2  and
         rel_bn[low_mask].max()< 5e-3  and
         rel_sb_an.max()      < 2e-3  and
         rel_sn_an.max()      < 2e-2)
print(f"\nPart 1 (Δl=1): {'PASS' if PASS1 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Part 2: Wide bands (Δl=4) — C_l=const vs D_l=const kernel models
#
# Both BJK variants run on the same data_map (same seed).
# C_l=const model: parameter is C_b; kernel = sum_{l in b} (2l+1)/(4pi) Pl
# D_l=const model: parameter is D_b; kernel = sum_{l in b} (2l+1)/(4pi) * 2pi/(l(l+1)) * Pl
#
# Full-sky white-noise analytical references:
#   C_l model: C_b_ref = (2l+1)-weighted mean of (cl_anaf[l] - N_L) within band
#   D_l model: D_b_ref = (2l+1)-weighted mean of l(l+1)/(2pi)*(cl_anaf[l]-N_L) within band
#              (exact at N=0; high-S/N approx otherwise)
#   NaMaster:  C_b_nmt = arithmetic mean of (cl_anaf[l] - N_L)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("Part 2: Wide bands (Δl=4) — C_l=const vs D_l=const kernel models")
print("="*70)

BAND_EDGES_4  = np.array([2, 6, 10, 14, 16])
nbands4       = len(BAND_EDGES_4) - 1
ell_b4        = 0.5 * (BAND_EDGES_4[:-1] + BAND_EDGES_4[1:] - 1)   # midpoints
true_Dl       = A / (2.0 * np.pi)

# Per-ell noise-subtracted values (l=2..LMAX)
l_arr  = np.arange(2, LMAX + 1, dtype=float)
cl_ns  = cl_anaf[2:LMAX+1] - N_L                          # noise-subtracted C_l
Dl_ns  = l_arr * (l_arr + 1) / (2.0 * np.pi) * cl_ns     # noise-subtracted D_l

# References per Δl=4 band
ref_Cl4       = np.zeros(nbands4)
ref_Dl4       = np.zeros(nbands4)
ref_Cl4_arith = np.zeros(nbands4)
for b, (lo, hi) in enumerate(zip(BAND_EDGES_4[:-1], BAND_EDGES_4[1:])):
    mask_b = (l_arr >= lo) & (l_arr < hi)
    w      = 2.0 * l_arr[mask_b] + 1.0
    ref_Cl4[b]       = np.sum(w * cl_ns[mask_b]) / w.sum()
    ref_Dl4[b]       = np.sum(w * Dl_ns[mask_b]) / w.sum()
    ref_Cl4_arith[b] = cl_ns[mask_b].mean()

# Run both BJK variants
with tempfile.TemporaryDirectory() as tmp2:
    data_fits2 = os.path.join(tmp2, "data.fits")
    ninv_fits2 = os.path.join(tmp2, "ninv.fits")
    hp.write_map(data_fits2, data_map, overwrite=True)
    hp.write_map(ninv_fits2, np.full(NPIX, 1.0 / SIGMA_PIX**2), overwrite=True)

    print("\nBJK C_l=const, Δl=4 ...")
    lik_cl4 = PixelLikelihood(
        data_fits=data_fits2, ninv_fits=ninv_fits2,
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES_4,
        nfields=1, spin=0,
    )
    cl_bjk4, sigma_cl4, _ = lik_cl4.newton_raphson(
        ref_Cl4.copy(), max_iter=30, tol=1e-10)

    print("\nBJK D_l=const, Δl=4 ...")
    lik_dl4 = PixelLikelihood(
        data_fits=data_fits2, ninv_fits=ninv_fits2,
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES_4,
        nfields=1, spin=0, band_model='Dl',
    )
    dl_bjk4, sigma_dl4, _ = lik_dl4.newton_raphson(
        ref_Dl4.copy(), max_iter=30, tol=1e-10)

# NaMaster Δl=4: bin the per-ell cl_nmt (already noise-subtracted) arithmetically
cl_nmt4 = np.zeros(nbands4)
for b, (lo, hi) in enumerate(zip(BAND_EDGES_4[:-1], BAND_EDGES_4[1:])):
    mask_b       = (ell_nmt.astype(int) >= lo) & (ell_nmt.astype(int) < hi)
    cl_nmt4[b]   = cl_nmt[mask_b].mean()

# Convert C_l-model bandpowers to D_l for display
Dl_from_Cl  = ell_b4 * (ell_b4 + 1) / (2.0 * np.pi) * cl_bjk4
Dl_from_nmt = ell_b4 * (ell_b4 + 1) / (2.0 * np.pi) * cl_nmt4

print(f"\nTrue D_l = {true_Dl:.4e}  (all multipoles, input flat spectrum)")

print("\n--- Bandpower comparison (D_l units) ---")
print("="*100)
print(f"{'band':>4}  {'ell_b':>5}  {'true_Dl':>10}  {'BJK_Cl→Dl':>10}  {'bias_Cl':>9}  "
      f"{'BJK_Dl':>10}  {'bias_Dl':>9}  {'NMT→Dl':>10}  {'bias_NMT':>9}")
print("-"*100)
for b in range(nbands4):
    b_Cl  = (Dl_from_Cl[b]  - true_Dl) / true_Dl
    b_Dl  = (dl_bjk4[b]     - true_Dl) / true_Dl
    b_nmt = (Dl_from_nmt[b] - true_Dl) / true_Dl
    print(f"  {b:2d}  {ell_b4[b]:5.1f}  {true_Dl:10.4e}  {Dl_from_Cl[b]:10.4e}  {b_Cl:+9.3f}  "
          f"{dl_bjk4[b]:10.4e}  {b_Dl:+9.3f}  {Dl_from_nmt[b]:10.4e}  {b_nmt:+9.3f}")
print("="*100)

# Quantitative checks
rel_cl4 = np.abs(cl_bjk4  - ref_Cl4)  / (np.abs(ref_Cl4)  + 1e-40)
rel_dl4 = np.abs(dl_bjk4  - ref_Dl4)  / (np.abs(ref_Dl4)  + 1e-40)
rel_nm4 = np.abs(cl_nmt4  - ref_Cl4_arith) / (np.abs(ref_Cl4_arith) + 1e-40)

print(f"\nChecks:")
print(f"  Max |BJK_Cl - ref_Cl| / |ref_Cl|   = {rel_cl4.max():.2e}  (target < 2e-3)")
print(f"  Max |BJK_Dl - ref_Dl| / |ref_Dl|   = {rel_dl4.max():.2e}  (target < 2e-2)")
print(f"  Max |NMT    - ref_NMT|/ |ref_NMT|  = {rel_nm4.max():.2e}  (target < 2e-2)")

# Show the systematic band-shape bias in the C_l model (feature, not failure)
bias_Cl_model = (Dl_from_Cl - true_Dl) / true_Dl
bias_Dl_model = (dl_bjk4    - true_Dl) / true_Dl
print(f"\nBand-shape bias (C_l model expressed as D_l, relative to true D_l):")
for b in range(nbands4):
    print(f"  band {b} (ell_b={ell_b4[b]:.1f}): C_l-model bias = {bias_Cl_model[b]:+.4f}, "
          f"D_l-model bias = {bias_Dl_model[b]:+.4f}")

PASS2 = (rel_cl4.max() < 2e-3 and
         rel_dl4.max() < 2e-2 and
         rel_nm4.max() < 2e-2)
print(f"\nPart 2 (Δl=4): {'PASS' if PASS2 else 'FAIL'}")

# ---------------------------------------------------------------------------
PASS = PASS1 and PASS2
print(f"\n{'PASS' if PASS else 'FAIL'}")
sys.exit(0 if PASS else 1)
