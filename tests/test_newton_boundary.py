"""
Regression tests for Newton-Raphson robustness near the positive-definite
boundary (backtracking line search added in c80e56b).

When fitting EE+BB+EB on noise-dominated data with true EB ~ 0, an unconstrained
Newton step can drive the signal model across the PD constraint
S_EE * S_BB >= S_EB^2, making M = S + N non-positive-definite. The Cholesky then
fails (log-likelihood = -inf) and, pre-fix, the iteration could stall or return
NaN. The backtracking line search must keep every accepted iterate PD and return
a finite ML point with a valid (invertible) Fisher matrix.

Tests:
  1. NR from an EB initial guess pushed up against the boundary still returns a
     finite, PD solution with finite positive 1-sigma errors, improves the
     log-likelihood, and recovers EB ~ 0 within 3 sigma.
  2. Line-search contract: NR never returns a point worse than its start, and the
     returned point is always PD — across a range of (including adversarial) starts.

NOTE on scope: on small, well-conditioned problems (full-sky NSIDE=4) the raw
Newton step never actually leaves the PD region, so the line-search *rejection*
branch is not hit here — verified by construction while writing these tests. The
backtracking matters for ill-conditioned real-data fits (partial sky, many
noise-dominated EB bands), which we do not reproduce at unit-test scale. These
tests therefore guard the observable *contract* (finite, PD, non-decreasing logL),
which is what protects against a regression that returns NaN / non-PD / a worse
point near the boundary.

Run from repo root or tests/:
    python3 tests/test_newton_boundary.py
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
LMAX = 2 * NSIDE - 1
NPIX = hp.nside2npix(NSIDE)
BAND_EDGES = np.array([2, 5, 8], dtype=int)     # 2 bands


def _make_eb_lik(seed=11):
    """EE+BB+EB pol likelihood; data has true EB=0 (independent E,B), noisy."""
    D_EE, D_BB = 2e-6, 1e-6
    cls = np.zeros((4, LMAX + 1))
    cls[1, 2:] = D_EE        # EE (flat C_l)
    cls[2, 2:] = D_BB        # BB
    np.random.seed(seed)
    T, Q, U = hp.synfast(cls, nside=NSIDE, lmax=LMAX, new=True)
    sigma_p = 1.2e-3         # noise comparable to signal -> EB noise-dominated
    Q = Q + np.random.normal(0, sigma_p, NPIX)
    U = U + np.random.normal(0, sigma_p, NPIX)
    NP = np.full(NPIX, sigma_p ** 2)
    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[], N_Q_list=[NP], N_U_list=[NP],
        lmin=2, lmax=LMAX, band_edges=BAND_EDGES,
        include_EB=True, band_model='Cl')
    return lik, D_EE, D_BB


def _is_pd(lik, cl):
    M = lik.build_signal_cov(cl) + lik.N_mat
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def _boundary_start(lik, D_EE, D_BB):
    """Initial guess with EB pushed to ~0.95*sqrt(EE*BB): just inside the PD wall."""
    layout = lik.layout
    cl0 = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ee = D_EE
        bb = D_BB
        cl0[layout.index('EE', 0, 0, b)] = ee
        cl0[layout.index('BB', 0, 0, b)] = bb
        cl0[layout.index('EB', 0, 0, b)] = 0.95 * np.sqrt(ee * bb)
    return cl0


def test_nr_converges_near_boundary():
    print("\nTest 1: NR with backtracking returns a valid solution near EB boundary")
    lik, D_EE, D_BB = _make_eb_lik()
    layout = lik.layout
    cl0 = _boundary_start(lik, D_EE, D_BB)

    check("initial guess is PD (S+N)", _is_pd(lik, cl0))
    logL0 = lik.log_likelihood(cl0)

    cl_ml, sigma, F = lik.newton_raphson(cl0, max_iter=40, tol=1e-6)

    check("ML point finite", np.all(np.isfinite(cl_ml)),
          f"cl_ml={cl_ml}")
    check("sigma finite and positive", np.all(np.isfinite(sigma)) and np.all(sigma > 0),
          f"sigma={sigma}")
    check("Fisher finite", np.all(np.isfinite(F)))
    check("ML solution is PD (S+N)", _is_pd(lik, cl_ml))
    logL_ml = lik.log_likelihood(cl_ml)
    check("log-likelihood improved (or held)", logL_ml >= logL0 - 1e-8,
          f"logL0={logL0:.4f} logL_ml={logL_ml:.4f}")

    # EB is consistent with zero within 3 sigma in every band
    for b in range(lik.nbands):
        idx = layout.index('EB', 0, 0, b)
        check(f"  EB band {b}: consistent with 0 within 3 sigma",
              abs(cl_ml[idx]) < 3 * sigma[idx] + 1e-10,
              f"EB={cl_ml[idx]:.3e} sigma={sigma[idx]:.3e}")


def test_linesearch_contract():
    print("\nTest 2: line-search contract (non-decreasing logL, PD result) over starts")
    lik, D_EE, D_BB = _make_eb_lik()
    layout = lik.layout

    starts = {}
    starts['near-wall EB'] = _boundary_start(lik, D_EE, D_BB)
    # inflated EE/BB with EB at the (inflated) wall
    cl_hi = np.zeros(layout.n_params)
    for b in range(lik.nbands):
        ee, bb = 10 * D_EE, 10 * D_BB
        cl_hi[layout.index('EE', 0, 0, b)] = ee
        cl_hi[layout.index('BB', 0, 0, b)] = bb
        cl_hi[layout.index('EB', 0, 0, b)] = 0.99 * np.sqrt(ee * bb)
    starts['inflated EE/BB, EB at wall'] = cl_hi

    for label, cl0 in starts.items():
        if not (np.isfinite(lik.log_likelihood(cl0)) and _is_pd(lik, cl0)):
            continue
        logL0 = lik.log_likelihood(cl0)
        cl_ml, sigma, F = lik.newton_raphson(cl0, max_iter=40, tol=1e-6)
        logL_ml = lik.log_likelihood(cl_ml)
        check(f"  [{label}] logL non-decreasing", logL_ml >= logL0 - 1e-8,
              f"logL0={logL0:.4f} logL_ml={logL_ml:.4f}")
        check(f"  [{label}] ML point PD and finite",
              _is_pd(lik, cl_ml) and np.all(np.isfinite(cl_ml)) and np.all(np.isfinite(sigma)))


if __name__ == '__main__':
    test_nr_converges_near_boundary()
    test_linesearch_contract()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} test(s): {_FAILURES}")
        sys.exit(1)
    print("All Newton-boundary tests passed.")
