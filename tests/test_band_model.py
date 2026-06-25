"""
Regression tests for band_model ('Cl' vs 'Dl') handling.

Guards the bug fixed in c80e56b: the ell_weights implied by band_model='Dl'
(w_l = 2*pi/(l*(l+1))) were applied to spin-0 (T) kernels but NOT to spin-2
(Q/U polarization) kernels, so for n_P>0 the 'Dl' option silently had no effect
and bandpowers came out ~l(l+1)/(2*pi) too small.

Tests:
  1. Exact Dl/Cl equivalence at the signal-covariance level for spin-2 (EE/BB),
     using single-l bands. This FAILS on the pre-c80e56b code.
  2. Same exact equivalence for spin-0 (TT) — cross-check the already-correct path.
  3. End-to-end: recover a constant-D_l input spectrum with band_model='Dl'.

Run from repo root or tests/:
    python3 tests/test_band_model.py
"""

import sys, os
import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _FAILURES.append(name)


NSIDE = 4
LMAX = 2 * NSIDE - 1            # = 7
NPIX = hp.nside2npix(NSIDE)
OMEGA = 4 * np.pi / NPIX
RNG = np.random.default_rng(42)

# Single-l bands: each band spans exactly one multipole l = 2..7
BAND_EDGES_SINGLE = np.array([2, 3, 4, 5, 6, 7, 8], dtype=int)
# l value carried by each single-l band b is BAND_EDGES_SINGLE[b]
ELL_B = BAND_EDGES_SINGLE[:-1].astype(float)
W_DL = 2.0 * np.pi / (ELL_B * (ELL_B + 1.0))    # weight relating Dl <-> Cl


def _pol_lik(band_model):
    """Pol-only (n_P=1) likelihood; data irrelevant (kernels are data-independent)."""
    Q = RNG.standard_normal(NPIX) * 1e-3
    U = RNG.standard_normal(NPIX) * 1e-3
    NP = np.full(NPIX, (1e-3) ** 2 * OMEGA)
    return PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[], N_Q_list=[NP], N_U_list=[NP],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES_SINGLE,
        band_model=band_model)


def _temp_lik(band_model):
    """Temperature-only (n_T=1) likelihood."""
    T = RNG.standard_normal(NPIX) * 1e-3
    NT = np.full(NPIX, (1e-3) ** 2 * OMEGA)
    return PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[], d_U_list=[],
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[NT], N_Q_list=[], N_U_list=[],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES_SINGLE,
        band_model=band_model)


def test_dl_cl_equivalence_spin2():
    print("\nTest 1: spin-2 Dl/Cl equivalence (guards c80e56b)")
    lik_cl = _pol_lik('Cl')
    lik_dl = _pol_lik('Dl')
    layout = lik_cl.layout

    # Pick D_l per (spec, band); equivalent C_l is D_l * 2pi/(l(l+1)) for single-l bands
    D = np.zeros(layout.n_params)
    C = np.zeros(layout.n_params)
    for spec in ('EE', 'BB'):
        for b in range(lik_cl.nbands):
            idx = layout.index(spec, 0, 0, b)
            D[idx] = (b + 1) * 1e-6
            C[idx] = D[idx] * W_DL[b]

    cov_dl = lik_dl.build_signal_cov(D)
    cov_cl = lik_cl.build_signal_cov(C)
    err = np.abs(cov_dl - cov_cl).max()
    scale = np.abs(cov_cl).max()
    check("cov(Dl, D_l) == cov(Cl, D_l*2pi/l(l+1))  [EE+BB]",
          np.allclose(cov_dl, cov_cl, rtol=1e-10, atol=1e-12 * scale),
          f"max|Δ|={err:.2e} (scale {scale:.2e})")

    # And confirm the buggy interpretation (weights ignored) would NOT match,
    # i.e. the weight genuinely changed the covariance.
    cov_dl_unweighted = lik_cl.build_signal_cov(D)   # Cl-mode treats D as C_l
    differs = not np.allclose(cov_dl, cov_dl_unweighted, rtol=1e-6)
    check("Dl weighting actually changes the spin-2 covariance", differs,
          "weights had no effect — regression to pre-c80e56b behaviour")


def test_dl_cl_equivalence_spin0():
    print("\nTest 2: spin-0 Dl/Cl equivalence (cross-check TT path)")
    lik_cl = _temp_lik('Cl')
    lik_dl = _temp_lik('Dl')
    layout = lik_cl.layout

    D = np.zeros(layout.n_params)
    C = np.zeros(layout.n_params)
    for b in range(lik_cl.nbands):
        idx = layout.index('TT', 0, 0, b)
        D[idx] = (b + 1) * 1e-6
        C[idx] = D[idx] * W_DL[b]

    cov_dl = lik_dl.build_signal_cov(D)
    cov_cl = lik_cl.build_signal_cov(C)
    err = np.abs(cov_dl - cov_cl).max()
    scale = np.abs(cov_cl).max()
    check("cov(Dl, D_l) == cov(Cl, D_l*2pi/l(l+1))  [TT]",
          np.allclose(cov_dl, cov_cl, rtol=1e-10, atol=1e-12 * scale),
          f"max|Δ|={err:.2e} (scale {scale:.2e})")


def test_dl_recovery_spin2():
    print("\nTest 3: end-to-end recovery of a constant-D_l spectrum (band_model='Dl')")
    band_edges = np.array([2, 5, 8], dtype=int)     # 2 bands: l=2-4, l=5-7
    D_EE, D_BB = 2e-6, 0.5e-6                        # constant D_l per multipole

    ll = np.arange(LMAX + 1)
    fac = np.zeros(LMAX + 1)
    fac[2:] = 2.0 * np.pi / (ll[2:] * (ll[2:] + 1.0))
    cls = np.zeros((4, LMAX + 1))
    cls[1] = D_EE * fac      # EE
    cls[2] = D_BB * fac      # BB
    # TE/TB/EB = 0

    seed = 7
    np.random.seed(seed)
    T, Q, U = hp.synfast(cls, nside=NSIDE, lmax=LMAX, new=True)
    sigma_p = 5e-4
    Q = Q + np.random.normal(0, sigma_p, NPIX)
    U = U + np.random.normal(0, sigma_p, NPIX)
    NP = np.full(NPIX, sigma_p ** 2)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[], N_Q_list=[NP], N_U_list=[NP],
        lmin=2, lmax=LMAX, band_edges=band_edges, band_model='Dl')

    layout = lik.layout
    cl_true = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        cl_true[layout.index('EE', 0, 0, b)] = D_EE   # constant D_l => bandpower = D_EE
        cl_true[layout.index('BB', 0, 0, b)] = D_BB

    cl_ml, sigma, _ = lik.newton_raphson(cl_true * 1.5, max_iter=30, tol=1e-6)

    for b in range(lik.nbands):
        for spec in ('EE', 'BB'):
            idx = layout.index(spec, 0, 0, b)
            dev = abs(cl_ml[idx] - cl_true[idx])
            check(f"  {spec} band {b}: |D_ML - D_true| < 3 sigma",
                  dev < 3 * sigma[idx] + 1e-10,
                  f"D_ML={cl_ml[idx]:.3e} D_true={cl_true[idx]:.3e} sigma={sigma[idx]:.3e}")


if __name__ == '__main__':
    test_dl_cl_equivalence_spin2()
    test_dl_cl_equivalence_spin0()
    test_dl_recovery_spin2()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} test(s): {_FAILURES}")
        sys.exit(1)
    print("All band_model tests passed.")
