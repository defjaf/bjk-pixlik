"""
Empirical verification of the EB and TB pixel-space covariance formulas.

For each cross-spectrum we:
  1. Generate N realizations using custom correlated alm (hp.alm2map handles E/B→Q/U)
  2. Estimate the sample covariance matrix
  3. Fit candidate prefactors and report rms_rel_err

Key conventions:
  - c2j[i,j] = cos(2*psi at T/E pixel i toward j)  [misleading name — see geometry]
  - c2j.T[i,j] = cos(2*psi at P pixel j toward i)   [used for TE block]
  - sin2sp[i,j] = sin(2*(psi_i + psi_j))            [sum angle, used for EB]
  - cos2sp[i,j] = cos(2*(psi_i + psi_j))
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
N_REAL = 30000

print("Building spin-2 kernels...")
Kp_b, Km_b, Kx_b, geom = _build_spin2_kernels(
    OBS, NSIDE, 2, LMAX, np.array([2, LMAX + 1]), BEAM2)
cos2dp, sin2dp, cos2sp, sin2sp, c2j, s2j = geom
Kx = Kx_b[0]   # single band total


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def rel_err(pred, exact):
    scale = np.sqrt(np.mean(exact**2))
    return np.sqrt(np.mean((pred - exact)**2)) / (scale + 1e-30)


def gen_alm_2x2(lmax, cl_AA, cl_AB, cl_BB, seed):
    """Generate two sets of correlated alm from 2×2 spectrum at each l."""
    rng = np.random.default_rng(seed)
    sz = hp.Alm.getsize(lmax)
    alm_A = np.zeros(sz, dtype=complex)
    alm_B = np.zeros(sz, dtype=complex)
    for l in range(2, lmax + 1):
        C = np.array([[cl_AA[l], cl_AB[l]], [cl_AB[l], cl_BB[l]]])
        # Add tiny regulariser so l=2 degenerate matrix doesn't fail
        eigvals = np.linalg.eigvalsh(C)
        if eigvals.min() < 0:
            continue
        L = np.linalg.cholesky(C + 1e-40 * np.eye(2))
        for m in range(l + 1):
            idx = hp.Alm.getidx(lmax, l, m)
            if m == 0:
                x = rng.normal(0, 1, 2)
            else:
                x = (rng.normal(0, 1, 2) + 1j * rng.normal(0, 1, 2)) / np.sqrt(2)
            a = L @ x
            alm_A[idx] = a[0]
            alm_B[idx] = a[1]
    return alm_A, alm_B


# ─────────────────────────────────────────────────────────────────────────────
# EB: covariance of Q and U maps for a pure EB spectrum
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("EB cross-spectrum verification")
print(f"{'='*60}")

C_EE = 1e-6
C_BB = 1e-6
r_EB = 0.7
C_EB = r_EB * np.sqrt(C_EE * C_BB)   # = 0.7e-6

cl_EE = np.full(LMAX + 1, C_EE);  cl_EE[:2] = 0
cl_BB = np.full(LMAX + 1, C_BB);  cl_BB[:2] = 0
cl_EB = np.full(LMAX + 1, C_EB);  cl_EB[:2] = 0

# Clean approach: generate correlated (alm_E, alm_B), then compute Q/U from
# each mode separately.  The cross-product <Q_E(p) * Q_B(p')> directly
# measures the EB covariance with no subtraction needed.
QQ_EB_sum = np.zeros((NPIX, NPIX))
QU_EB_sum = np.zeros((NPIX, NPIX))   # Q from E × U from B
UQ_EB_sum = np.zeros((NPIX, NPIX))   # U from E × Q from B
UU_EB_sum = np.zeros((NPIX, NPIX))

zeros = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)

print(f"Running {N_REAL} realizations for EB (cross-product method)...")
for k in range(N_REAL):
    alm_E, alm_B = gen_alm_2x2(LMAX, cl_EE, cl_EB, cl_BB, seed=k)
    _, QE, UE = hp.alm2map([zeros, alm_E, zeros], nside=NSIDE, lmax=LMAX)
    _, QB, UB = hp.alm2map([zeros, zeros, alm_B], nside=NSIDE, lmax=LMAX)
    QQ_EB_sum += np.outer(QE, QB) + np.outer(QB, QE)
    QU_EB_sum += np.outer(QE, UB) + np.outer(QB, UE)
    UU_EB_sum += np.outer(UE, UB) + np.outer(UB, UE)

# <Q_E Q_B + Q_B Q_E> = 2 * C^{EB} * K_EB; divide by 2 to get C^{EB} * K_EB
QQ_EB = QQ_EB_sum / (2 * N_REAL)
QU_EB = QU_EB_sum / (2 * N_REAL)
UU_EB = UU_EB_sum / (2 * N_REAL)

Kp = Kp_b[0];  Km = Km_b[0]

# Extended basis: each Wigner kernel × each trig factor,
# plus scaled versions (2*Km etc.) from the analytic derivation
basis = {}
for kname, K in [("Kx", Kx), ("Kp", Kp), ("Km", Km),
                 ("2*Km", 2*Km)]:
    for gname, G in [("cos2dp", cos2dp), ("sin2dp", sin2dp),
                     ("cos2sp", cos2sp), ("sin2sp", sin2sp),
                     ("c2j.T",  c2j.T),  ("s2j.T",  s2j.T)]:
        basis[f"{kname}*{gname}"] = K * G

def best_candidates(exact, n_top=4):
    """Return (name, scale, rel_err) for the n_top best single-term candidates."""
    results = []
    for name, b in basis.items():
        # Fit optimal scalar scale: a = (b · exact) / ||b||^2
        a = np.dot(b.ravel(), exact.ravel()) / (np.dot(b.ravel(), b.ravel()) + 1e-100)
        err = rel_err(a * b, exact)
        results.append((name, a, err))
    results.sort(key=lambda x: x[2])
    return results[:n_top]

print(f"\nBest single-term fits for C^{{QQ}} EB (fitted scale shown):")
for name, a, err in best_candidates(QQ_EB):
    print(f"  {name:20s}  scale={a/C_EB:+.3f}*C_EB  rms_rel={err:.4f}")

print(f"\nBest single-term fits for C^{{QU}} EB:")
for name, a, err in best_candidates(QU_EB):
    print(f"  {name:20s}  scale={a/C_EB:+.3f}*C_EB  rms_rel={err:.4f}")

print(f"\nBest single-term fits for C^{{UU}} EB:")
for name, a, err in best_candidates(UU_EB):
    print(f"  {name:20s}  scale={a/C_EB:+.3f}*C_EB  rms_rel={err:.4f}")

print(f"\nDerived formula (analytic + MC): QQ=-Km*sin2sp, QU=+Km*cos2sp, UU=+Km*sin2sp")
print(f"  (Km = d^l_{{2,-2}} Wigner kernel; no Kx, no factor of 2)")
print(f"  Sign of QQ: Q_B = -U_E from HEALPix convention (Q+iU)=-sum(a_E+ia_B)_{{+2}}Y")
derived = {"QQ": -Km*sin2sp, "QU": Km*cos2sp, "UU": Km*sin2sp}
for comp, exact_mat in [("QQ", QQ_EB), ("QU", QU_EB), ("UU", UU_EB)]:
    rms = rel_err(C_EB * derived[comp], exact_mat)
    print(f"  {comp:2s}: rms_rel={rms:.4f}  {'<-- matches MC noise floor' if rms < 0.15 else 'FAIL'}")

print(f"\nScale check:")
print(f"  rms(QQ_EB_exact)     = {np.sqrt(np.mean(QQ_EB**2)):.4e}")
print(f"  rms(C_EB*Km*sin2sp)  = {np.sqrt(np.mean((C_EB*Km*sin2sp)**2)):.4e}")
print(f"  rms(C_EB*Kx*sin2sp)  = {np.sqrt(np.mean((C_EB*Kx*sin2sp)**2)):.4e}  (old wrong formula)")


# ─────────────────────────────────────────────────────────────────────────────
# TB: covariance of T and Q/U maps for a pure TB spectrum
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("TB cross-spectrum verification")
print(f"{'='*60}")

C_TT = 2e-6
C_BB2 = 1e-6
r_TB = 0.7
C_TB = r_TB * np.sqrt(C_TT * C_BB2)   # ~0.99e-6

cl_TT = np.full(LMAX + 1, C_TT);  cl_TT[:2] = 0
cl_BB2 = np.full(LMAX + 1, C_BB2); cl_BB2[:2] = 0
cl_TB = np.full(LMAX + 1, C_TB);  cl_TB[:2] = 0

TQ_sum = np.zeros((NPIX, NPIX))
TU_sum = np.zeros((NPIX, NPIX))

print(f"Running {N_REAL} realizations for TB...")
for k in range(N_REAL):
    alm_T, alm_B = gen_alm_2x2(LMAX, cl_TT, cl_TB, cl_BB2, seed=k + 200000)
    alm_E_zero = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    T = hp.alm2map(alm_T, nside=NSIDE, lmax=LMAX)
    _, Q, U = hp.alm2map([np.zeros(hp.Alm.getsize(LMAX), dtype=complex),
                          alm_E_zero, alm_B], nside=NSIDE, lmax=LMAX)
    TQ_sum += np.outer(T, Q)
    TU_sum += np.outer(T, U)

TQ_exact = TQ_sum / N_REAL
TU_exact = TU_sum / N_REAL

K_TB = 2.0 * Kx   # same kernel as TE

print(f"\nCandidates for C^{{TQ}} TB (angle at P pixel toward T pixel = c2j.T):")
for name, cand in [
    ("+K*c2j.T",  C_TB * K_TB * c2j.T),
    ("-K*c2j.T", -C_TB * K_TB * c2j.T),
    ("+K*s2j.T",  C_TB * K_TB * s2j.T),
    ("-K*s2j.T", -C_TB * K_TB * s2j.T),
]:
    print(f"  {name:12s}  rms_rel_err = {rel_err(cand, TQ_exact):.4f}")

print(f"\nCandidates for C^{{TU}} TB:")
for name, cand in [
    ("+K*c2j.T",  C_TB * K_TB * c2j.T),
    ("-K*c2j.T", -C_TB * K_TB * c2j.T),
    ("+K*s2j.T",  C_TB * K_TB * s2j.T),
    ("-K*s2j.T", -C_TB * K_TB * s2j.T),
]:
    print(f"  {name:12s}  rms_rel_err = {rel_err(cand, TU_exact):.4f}")

print(f"\nScale check:")
print(f"  rms(TQ_exact)    = {np.sqrt(np.mean(TQ_exact**2)):.4e}")
print(f"  rms(C_TB*K*c2j.T) = {np.sqrt(np.mean((C_TB*K_TB*c2j.T)**2)):.4e}")
print(f"  rms(C_TB*K*s2j.T) = {np.sqrt(np.mean((C_TB*K_TB*s2j.T)**2)):.4e}")
