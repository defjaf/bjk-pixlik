"""
Determine the correct TE cross-covariance formula empirically.

Generates N realizations with pure TE spectrum (TT=EE=BB=0), measures
sample covariance <T(p) Q(p')> and <T(p) U(p')>, then compares
against four candidate formulas to find the right sign combination.
"""

import numpy as np
import healpy as hp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import _build_spin2_kernels

NSIDE  = 4
LMAX   = 7
NPIX   = hp.nside2npix(NSIDE)
OBS    = np.arange(NPIX)
BEAM2  = np.ones(LMAX + 1)

# Build kernels once
print("Building spin-2 kernels...")
Kp_b, Km_b, Kx_b, geom = _build_spin2_kernels(
    OBS, NSIDE, 2, LMAX, np.array([2, LMAX + 1]), BEAM2)
_, _, _, _, c2j, s2j = geom
# Single-band total Kx
Kx_total = Kx_b[0]    # shape (NPIX, NPIX)
K_TE = 2.0 * Kx_total  # factor of 2: TE uses (2l+1)/4pi, Kx uses (2l+1)/8pi

# Monte Carlo: use correlated T+P maps with TE / sqrt(TT*EE) = 0.9
# The cross-covariance <T(p) Q(p')> is entirely due to TE (TT and EE contribute zero)
C_TT = 2e-6
C_EE = 1e-6
r    = 0.9
C_TE = r * np.sqrt(C_TT * C_EE)
cl_TT = np.full(LMAX + 1, C_TT);  cl_TT[:2] = 0
cl_EE = np.full(LMAX + 1, C_EE);  cl_EE[:2] = 0
cl_BB = np.zeros(LMAX + 1)
cl_TE = np.full(LMAX + 1, C_TE);  cl_TE[:2] = 0
cls = np.array([cl_TT, cl_EE, cl_BB, cl_TE])

rng = np.random.default_rng(0)
N = 30000
TQ_sum = np.zeros((NPIX, NPIX))
TU_sum = np.zeros((NPIX, NPIX))

print(f"Running {N} Monte Carlo realizations...")
for k in range(N):
    T, Q, U = hp.synfast(cls, nside=NSIDE, lmax=LMAX, new=True,
                          verbose=False, pol=True)
    TQ_sum += np.outer(T, Q)
    TU_sum += np.outer(T, U)

TQ_exact = TQ_sum / N  # (NPIX, NPIX): TQ_exact[p, p'] = <T(p) Q(p')>
TU_exact = TU_sum / N

# Candidates for the TE block:
# C^{TQ}[p,p'] = sgn_Q * K_TE * geom_Q
# C^{TU}[p,p'] = sgn_U * K_TE * geom_U
# geom options: c2j or s2j
# c2j[p,p'] = cos(2*psi at p' toward p)
# s2j[p,p'] = sin(2*psi at p' toward p)

def relative_err(pred, exact):
    scale = np.sqrt(np.mean(exact**2))
    return np.sqrt(np.mean((pred - exact)**2)) / (scale + 1e-30)

# Scale theory by the actual C_TE value used
# C^{TQ}[p,p'] = C_TE * 2 * Kx_total * geom_factor
K_scaled = C_TE * K_TE  # (NPIX, NPIX): C_TE * 2 * Kx_total

print("\nCandidates for C^{TQ}[p,p'] (rows=T pixel, cols=Q pixel):")
candidates_TQ = [
    ("+C_TE*K*c2j",  K_scaled * c2j),
    ("-C_TE*K*c2j", -K_scaled * c2j),
    ("+C_TE*K*s2j",  K_scaled * s2j),
    ("-C_TE*K*s2j", -K_scaled * s2j),
]
for name, pred in candidates_TQ:
    err = relative_err(pred, TQ_exact)
    print(f"  {name:16s}  rms_rel_err = {err:.4f}")

print("\nCandidates for C^{TU}[p,p'] (rows=T pixel, cols=U pixel):")
candidates_TU = [
    ("+C_TE*K*c2j",  K_scaled * c2j),
    ("-C_TE*K*c2j", -K_scaled * c2j),
    ("+C_TE*K*s2j",  K_scaled * s2j),
    ("-C_TE*K*s2j", -K_scaled * s2j),
]
for name, pred in candidates_TU:
    err = relative_err(pred, TU_exact)
    print(f"  {name:16s}  rms_rel_err = {err:.4f}")

# Also check diagonal flip: C^{TQ}[p,p'] vs c2i (angle at T pixel toward P pixel)
_, _, _, _, c2j_arr, s2j_arr = geom
c2i = 2*(np.cos(hp.pix2ang(NSIDE, OBS)[0][:,None] - 0))**2 - 1  # wrong, just placeholder

# Real c2i: angle at p (T pixel) toward p'
# c2i[p,p'] = cos(2*psi at p toward p')
# In _compute_spin2_geometry: cos_psi_i[i,j] is at pixel i, so c2i is 2*cos_psi_i^2 - 1
# but c2j[i,j] is at pixel j (toward i)
# In geom: c2j = 2*cos_psi_j**2 - 1 where cos_psi_j[i,j] = angle at j toward i
# So c2j[i,j] = cos(2*psi at p_j pointing toward p_i) -- this is what theory says

# Let's check the transpose: c2j.T[p,p'] = c2j[p',p] = cos(2*psi at p toward p')
print("\nAlso checking transposed versions:")
for name, pred in [
    ("+K*c2j.T",  K_scaled * c2j.T),
    ("-K*c2j.T", -K_scaled * c2j.T),
    ("+K*s2j.T",  K_scaled * s2j.T),
    ("-K*s2j.T", -K_scaled * s2j.T),
]:
    err_TQ = relative_err(pred, TQ_exact)
    err_TU = relative_err(pred, TU_exact)
    print(f"  {name:14s}  rms_TQ = {err_TQ:.4f}  rms_TU = {err_TU:.4f}")

print("\nBest candidate summary:")
best_tq = min(candidates_TQ, key=lambda x: relative_err(x[1], TQ_exact))
best_tu = min(candidates_TU, key=lambda x: relative_err(x[1], TU_exact))
print(f"  TQ: {best_tq[0]}  (rms_rel={relative_err(best_tq[1], TQ_exact):.4f})")
print(f"  TU: {best_tu[0]}  (rms_rel={relative_err(best_tu[1], TU_exact):.4f})")

# Show scale ratios to detect factor-of-2 or sign issues
print(f"\nScale checks (rms of exact vs rms of candidates):")
print(f"  rms(TQ_exact) = {np.sqrt(np.mean(TQ_exact**2)):.4e}")
print(f"  rms(TU_exact) = {np.sqrt(np.mean(TU_exact**2)):.4e}")
print(f"  rms(K_scaled*c2j) = {np.sqrt(np.mean((K_scaled*c2j)**2)):.4e}")
print(f"  rms(K_scaled*s2j) = {np.sqrt(np.mean((K_scaled*s2j)**2)):.4e}")
