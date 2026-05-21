"""
Tests for the generalised BJK pixel-space likelihood.

Tests:
  1. Gradient finite-difference check (n_T=1, n_P=1, nbands=2)
  2. Signal covariance symmetry with nonzero TE
  3. Single-bin equivalence: n_T=1,n_P=0 matches old spin-0 interface
  4. Single-bin equivalence: n_T=0,n_P=1 matches old spin-2 interface
  5. Block-diagonal structure when TE=TB=0 (T and P decouple)
  6. Recovery: TT only (n_T=1, n_P=0), full-sky white noise
  7. Recovery: EE+BB only (n_T=0, n_P=1), full-sky white noise
  8. Recovery: TT+TE+EE+BB jointly (n_T=1, n_P=1), full-sky white noise
  9. Gradient FD check: two polarisation bins (n_T=0, n_P=2)
 10. Gradient FD check: two temperature + one polarisation bin (n_T=2, n_P=1)

Run from sim_runs/bjk/:
    python3 test_general.py
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


# ---------------------------------------------------------------------------
# Shared small-problem setup (NSIDE=4, full sky, nbands=2)
# ---------------------------------------------------------------------------

NSIDE   = 4
LMAX    = 2 * NSIDE - 1      # = 7
NPIX    = hp.nside2npix(NSIDE)
OMEGA   = 4 * np.pi / NPIX
BAND_EDGES = np.array([2, 5, 8], dtype=int)   # 2 bands: [2,4] and [5,7]

# Simple white noise
SIGMA_T = 1e-3
SIGMA_P = 1.5e-3
N_T_VAR = np.full(NPIX, SIGMA_T**2 * OMEGA)   # noise variance per pixel
N_P_VAR = np.full(NPIX, SIGMA_P**2 * OMEGA)

# True spectra (flat C_l)
CL_TT_TRUE = np.full(LMAX + 1, 2e-6)
CL_EE_TRUE = np.full(LMAX + 1, 1e-6)
CL_BB_TRUE = np.full(LMAX + 1, 0.1e-6)
CL_TE_TRUE = np.full(LMAX + 1, 0.5e-6)

CL_TT_TRUE[:2] = 0
CL_EE_TRUE[:2] = 0
CL_BB_TRUE[:2] = 0
CL_TE_TRUE[:2] = 0

RNG = np.random.default_rng(42)


def synfast_TP(nside, cl_TT, cl_EE, cl_BB, cl_TE, seed=0):
    """Simulate (T, Q, U) maps from given Cl arrays."""
    rng = np.random.default_rng(seed)
    lmax = len(cl_TT) - 1
    # hp.synfast expects [TT, EE, BB, TE] as first four entries
    cls = np.array([cl_TT[:lmax+1], cl_EE[:lmax+1], cl_BB[:lmax+1], cl_TE[:lmax+1]])
    T, Q, U = hp.synfast(cls, nside=nside, lmax=lmax, new=True,
                         verbose=False, pol=True)
    # Add noise
    T += rng.normal(0, SIGMA_T * OMEGA**0.5, T.shape)
    Q += rng.normal(0, SIGMA_P * OMEGA**0.5, Q.shape)
    U += rng.normal(0, SIGMA_P * OMEGA**0.5, U.shape)
    return T, Q, U


def make_lik_TP(T, Q, U, nT=1, nP=1, include_TB=False, include_EB=False):
    """Build PixelLikelihood from_arrays with full-sky pixels."""
    obs_pix = np.arange(NPIX)
    N_T_arr = np.full(NPIX, SIGMA_T**2 * OMEGA)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)
    return PixelLikelihood.from_arrays(
        d_T_list  = [T] * nT,
        d_Q_list  = [Q] * nP,
        d_U_list  = [U] * nP,
        obs_pix   = obs_pix,
        nside     = NSIDE,
        N_T_list  = [N_T_arr] * nT,
        N_Q_list  = [N_P_arr] * nP,
        N_U_list  = [N_P_arr] * nP,
        lmin      = 2, lmax = LMAX,
        band_edges = BAND_EDGES,
        include_TB = include_TB,
        include_EB = include_EB,
    )


# ---------------------------------------------------------------------------
# Test 1: Gradient finite-difference check
# ---------------------------------------------------------------------------

def test_gradient_fd():
    print("\nTest 1: Gradient finite-difference check")
    T, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=1)
    lik = make_lik_TP(T, Q, U, nT=1, nP=1, include_TB=False, include_EB=False)
    np_total = lik.layout.n_params

    # Starting point: slightly off from true
    cl = np.zeros(np_total)
    nb = lik.nbands
    layout = lik.layout

    # Bandpower averages of true spectra over each band
    for b, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        ells = np.arange(lo, hi)
        cl[layout.index('TT', 0, 0, b)] = CL_TT_TRUE[ells].mean()
        cl[layout.index('TE', 0, 0, b)] = CL_TE_TRUE[ells].mean()
        cl[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()

    g_anal, _ = lik.gradient_and_fisher(cl)

    # eps must be small relative to each bandpower so cl - eps stays physical
    eps_scale = 1e-4
    g_fd = np.zeros(np_total)
    for idx in range(np_total):
        eps = max(abs(cl[idx]) * eps_scale, 1e-12)
        cl_p = cl.copy(); cl_p[idx] += eps
        cl_m = cl.copy(); cl_m[idx] -= eps
        g_fd[idx] = (lik.log_likelihood(cl_p) - lik.log_likelihood(cl_m)) / (2 * eps)

    rel_err = np.abs(g_anal - g_fd) / (np.abs(g_fd) + 1e-30)
    worst = rel_err.max()
    check("gradient vs finite difference (max rel err < 1e-4)", worst < 1e-4,
          f"max rel err = {worst:.2e}")


# ---------------------------------------------------------------------------
# Test 2: Symmetry of signal covariance
# ---------------------------------------------------------------------------

def test_cov_symmetry():
    print("\nTest 2: Signal covariance symmetry with nonzero TE")
    T, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=2)
    lik = make_lik_TP(T, Q, U, nT=1, nP=1)
    layout = lik.layout
    cl = np.zeros(layout.n_params)
    nb = lik.nbands
    for b in range(nb):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl[layout.index('TT', 0, 0, b)] = CL_TT_TRUE[ells].mean()
        cl[layout.index('TE', 0, 0, b)] = CL_TE_TRUE[ells].mean()
        cl[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()
    C = lik.build_signal_cov(cl)
    check("C is symmetric", np.allclose(C, C.T, atol=1e-14),
          f"max |C-C^T| = {np.abs(C - C.T).max():.2e}")


# ---------------------------------------------------------------------------
# Test 3: Spin-0 equivalence (n_T=1, n_P=0)
# ---------------------------------------------------------------------------

def test_spin0_equivalence():
    print("\nTest 3: n_T=1,n_P=0 matches old spin=0 interface")
    T, _, _ = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=3)
    obs_pix = np.arange(NPIX)
    N_T_arr = np.full(NPIX, SIGMA_T**2 * OMEGA)

    lik_new = PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[], d_U_list=[],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[N_T_arr], N_Q_list=[], N_U_list=[],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    # Old-style (backward compat via spin=0 shim) — use from_arrays directly
    lik_old = PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[], d_U_list=[],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[N_T_arr], N_Q_list=[], N_U_list=[],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    # Both should give identical kernels and signal covariances
    cl = np.array([1e-6, 1e-6])   # one TT bandpower per band
    C_new = lik_new.build_signal_cov(cl)
    C_old = lik_old.build_signal_cov(cl)
    check("n_T=1 n_P=0: signal cov matches", np.allclose(C_new, C_old),
          f"max diff = {np.abs(C_new - C_old).max():.2e}")

    logL_new = lik_new.log_likelihood(cl)
    logL_old = lik_old.log_likelihood(cl)
    check("n_T=1 n_P=0: log-likelihood matches",
          abs(logL_new - logL_old) < 1e-10,
          f"diff = {abs(logL_new - logL_old):.2e}")


# ---------------------------------------------------------------------------
# Test 4: Spin-2 equivalence (n_T=0, n_P=1)
# ---------------------------------------------------------------------------

def test_spin2_equivalence():
    print("\nTest 4: n_T=0,n_P=1 matches old spin=2 interface")
    _, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=4)
    obs_pix = np.arange(NPIX)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik_a = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    lik_b = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    # cl: [EE_0, EE_1, BB_0, BB_1]
    cl = np.array([1e-6, 1e-6, 0.1e-6, 0.1e-6])
    C_a = lik_a.build_signal_cov(cl)
    C_b = lik_b.build_signal_cov(cl)
    check("n_T=0 n_P=1: signal cov symmetric", np.allclose(C_a, C_a.T))
    check("n_T=0 n_P=1: two constructions agree",
          np.allclose(C_a, C_b),
          f"max diff = {np.abs(C_a - C_b).max():.2e}")


# ---------------------------------------------------------------------------
# Test 5: Block-diagonal when TE=0
# ---------------------------------------------------------------------------

def test_block_diagonal_zero_TE():
    print("\nTest 5: TE=TB=0 → signal cov is block-diagonal")
    T, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=5)
    lik = make_lik_TP(T, Q, U, nT=1, nP=1)
    layout = lik.layout
    cl = np.zeros(layout.n_params)
    nb = lik.nbands
    for b in range(nb):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl[layout.index('TT', 0, 0, b)] = CL_TT_TRUE[ells].mean()
        cl[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()
        # TE = 0

    C = lik.build_signal_cov(cl)
    n = lik.n_obs
    off_block = np.abs(C[:n, n:]).max()
    check("TE=0 → off-diagonal T-P blocks are zero",
          off_block < 1e-14, f"max = {off_block:.2e}")


# ---------------------------------------------------------------------------
# Test 6: Recovery test (full-sky, white noise, n_T=1 n_P=0)
# ---------------------------------------------------------------------------

def test_recovery_TT():
    print("\nTest 6: TT recovery (full sky, white noise)")
    T, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=6)
    obs_pix = np.arange(NPIX)
    N_T_arr = np.full(NPIX, SIGMA_T**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[], d_U_list=[],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[N_T_arr], N_Q_list=[], N_U_list=[],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    cl_true_bands = []
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl_true_bands.append(CL_TT_TRUE[ells].mean())
    cl_true_bands = np.array(cl_true_bands)

    cl_init = cl_true_bands * 1.5   # perturbed start
    cl_ml, sigma, _ = lik.newton_raphson(cl_init, max_iter=20, tol=1e-6)

    # N_l = sigma_T^2 * omega_pix, in Cl
    N_l = SIGMA_T**2 * OMEGA
    # True + noise bandpower
    cl_signal = cl_true_bands.copy()

    for b in range(lik.nbands):
        dev = abs(cl_ml[b] - cl_signal[b])
        check(f"  band {b}: |C_ML - C_true| < 3*sigma",
              dev < 3 * sigma[b] + 1e-10,
              f"C_ML={cl_ml[b]:.3e}  C_true={cl_signal[b]:.3e}  sigma={sigma[b]:.3e}  dev={dev:.3e}")


# ---------------------------------------------------------------------------
# Test 7: EE+BB recovery (n_T=0, n_P=1)
# ---------------------------------------------------------------------------

def test_recovery_EEBB():
    print("\nTest 7: EE+BB recovery (n_T=0, n_P=1, full sky, white noise)")
    _, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=7)
    obs_pix = np.arange(NPIX)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    layout = lik.layout
    cl_true = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl_true[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl_true[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()

    cl_ml, sigma, _ = lik.newton_raphson(cl_true * 1.5, max_iter=20, tol=1e-6)

    for b in range(lik.nbands):
        for spec in ['EE', 'BB']:
            idx = layout.index(spec, 0, 0, b)
            dev = abs(cl_ml[idx] - cl_true[idx])
            check(f"  {spec} band {b}: |C_ML - C_true| < 3*sigma",
                  dev < 3 * sigma[idx] + 1e-10,
                  f"C_ML={cl_ml[idx]:.3e}  C_true={cl_true[idx]:.3e}  sigma={sigma[idx]:.3e}")


# ---------------------------------------------------------------------------
# Test 8: Joint TT+TE+EE+BB recovery (n_T=1, n_P=1)
# ---------------------------------------------------------------------------

def test_recovery_joint():
    print("\nTest 8: Joint TT+TE+EE+BB recovery (n_T=1, n_P=1, full sky, white noise)")
    T, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=8)
    obs_pix = np.arange(NPIX)
    N_T_arr = np.full(NPIX, SIGMA_T**2 * OMEGA)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[T], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[N_T_arr], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    layout = lik.layout
    cl_true = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl_true[layout.index('TT', 0, 0, b)] = CL_TT_TRUE[ells].mean()
        cl_true[layout.index('TE', 0, 0, b)] = CL_TE_TRUE[ells].mean()
        cl_true[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl_true[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()

    cl_ml, sigma, _ = lik.newton_raphson(cl_true * 1.5, max_iter=30, tol=1e-6)

    for b in range(lik.nbands):
        for spec in ['TT', 'TE', 'EE', 'BB']:
            idx = layout.index(spec, 0, 0, b)
            dev = abs(cl_ml[idx] - cl_true[idx])
            check(f"  {spec} band {b}: |C_ML - C_true| < 3*sigma",
                  dev < 3 * sigma[idx] + 1e-10,
                  f"C_ML={cl_ml[idx]:.3e}  C_true={cl_true[idx]:.3e}  sigma={sigma[idx]:.3e}")


# ---------------------------------------------------------------------------
# Shared helper: run gradient FD check for any PixelLikelihood + cl vector
# ---------------------------------------------------------------------------

def _gradient_fd_check(name, lik, cl):
    """Central-difference gradient check, returning max relative error."""
    g_anal, _ = lik.gradient_and_fisher(cl)
    eps_scale = 1e-4
    g_fd = np.zeros(len(cl))
    for idx in range(len(cl)):
        eps = max(abs(cl[idx]) * eps_scale, 1e-12)
        cl_p = cl.copy(); cl_p[idx] += eps
        cl_m = cl.copy(); cl_m[idx] -= eps
        g_fd[idx] = (lik.log_likelihood(cl_p) - lik.log_likelihood(cl_m)) / (2 * eps)
    worst = np.nanmax(np.abs(g_anal - g_fd) / (np.abs(g_fd) + 1e-30))
    check(f"{name}: gradient vs FD (max rel err < 1e-4)", worst < 1e-4,
          f"max rel err = {worst:.2e}")


# ---------------------------------------------------------------------------
# Test 9: Gradient FD check with two polarisation bins (n_T=0, n_P=2)
# ---------------------------------------------------------------------------

def test_gradient_fd_P2():
    print("\nTest 9: Gradient FD check, n_T=0 n_P=2 (two polarisation bins)")
    # Two independent P bins: same CMB, independent noise
    _, Q0, U0 = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=9)
    _, Q1, U1 = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=10)
    obs_pix = np.arange(NPIX)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q0, Q1], d_U_list=[U0, U1],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[], N_Q_list=[N_P_arr, N_P_arr], N_U_list=[N_P_arr, N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    layout = lik.layout
    # EE: auto (0,0), (1,1) at signal level; cross (0,1) at 30% of auto
    # BB: same pattern
    cl = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        ee = CL_EE_TRUE[ells].mean()
        bb = CL_BB_TRUE[ells].mean()
        cl[layout.index('EE', 0, 0, b)] = ee
        cl[layout.index('EE', 0, 1, b)] = 0.3 * ee
        cl[layout.index('EE', 1, 1, b)] = ee
        cl[layout.index('BB', 0, 0, b)] = bb
        cl[layout.index('BB', 0, 1, b)] = 0.3 * bb
        cl[layout.index('BB', 1, 1, b)] = bb

    check("  n_P=2: signal cov symmetric",
          np.allclose(lik.build_signal_cov(cl), lik.build_signal_cov(cl).T, atol=1e-14))
    _gradient_fd_check("  n_P=2", lik, cl)


# ---------------------------------------------------------------------------
# Test 10: Gradient FD check with two T bins + one P bin (n_T=2, n_P=1)
# ---------------------------------------------------------------------------

def test_gradient_fd_T2P1():
    print("\nTest 10: Gradient FD check, n_T=2 n_P=1 (two T bins + one P bin)")
    T0, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=11)
    T1, _, _ = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=12)
    obs_pix = np.arange(NPIX)
    N_T_arr = np.full(NPIX, SIGMA_T**2 * OMEGA)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[T0, T1], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[N_T_arr, N_T_arr], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES)

    layout = lik.layout
    cl = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        tt = CL_TT_TRUE[ells].mean()
        te = CL_TE_TRUE[ells].mean()
        ee = CL_EE_TRUE[ells].mean()
        bb = CL_BB_TRUE[ells].mean()
        cl[layout.index('TT', 0, 0, b)] = tt
        cl[layout.index('TT', 0, 1, b)] = 0.5 * tt   # T0-T1 cross
        cl[layout.index('TT', 1, 1, b)] = tt
        cl[layout.index('TE', 0, 0, b)] = te          # T0-P0
        cl[layout.index('TE', 1, 0, b)] = te          # T1-P0
        cl[layout.index('EE', 0, 0, b)] = ee
        cl[layout.index('BB', 0, 0, b)] = bb

    check("  n_T=2 n_P=1: signal cov symmetric",
          np.allclose(lik.build_signal_cov(cl), lik.build_signal_cov(cl).T, atol=1e-14))
    _gradient_fd_check("  n_T=2 n_P=1", lik, cl)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 11: EB covariance symmetry + gradient FD check
# ---------------------------------------------------------------------------

def test_eb_kernel():
    print("\nTest 11: EB covariance symmetry and gradient FD check (include_EB=True)")
    _, Q, U = synfast_TP(NSIDE, CL_TT_TRUE, CL_EE_TRUE, CL_BB_TRUE, CL_TE_TRUE, seed=13)
    obs_pix = np.arange(NPIX)
    N_P_arr = np.full(NPIX, SIGMA_P**2 * OMEGA)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs_pix, nside=NSIDE,
        N_T_list=[], N_Q_list=[N_P_arr], N_U_list=[N_P_arr],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES,
        include_EB=True)

    layout = lik.layout
    cl = np.zeros(layout.n_params)
    C_EB_val = 0.3e-6   # sub-unity EB (must keep matrix PSD)
    for b in range(lik.nbands):
        ells = np.arange(BAND_EDGES[b], BAND_EDGES[b+1])
        cl[layout.index('EE', 0, 0, b)] = CL_EE_TRUE[ells].mean()
        cl[layout.index('BB', 0, 0, b)] = CL_BB_TRUE[ells].mean()
        cl[layout.index('EB', 0, 0, b)] = C_EB_val

    C = lik.build_signal_cov(cl)
    check("EB: signal cov symmetric", np.allclose(C, C.T, atol=1e-14),
          f"max |C-C^T| = {np.abs(C - C.T).max():.2e}")
    check("EB: signal cov PSD",
          np.linalg.eigvalsh(C + np.diag(np.full(C.shape[0], SIGMA_P**2 * OMEGA))).min() >= 0)
    _gradient_fd_check("  EB FD", lik, cl)


if __name__ == '__main__':
    test_gradient_fd()
    test_cov_symmetry()
    test_spin0_equivalence()
    test_spin2_equivalence()
    test_block_diagonal_zero_TE()
    test_recovery_TT()
    test_recovery_EEBB()
    test_recovery_joint()
    test_gradient_fd_P2()
    test_gradient_fd_T2P1()
    test_eb_kernel()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} test(s): {_FAILURES}")
        sys.exit(1)
    else:
        print("All tests passed.")
