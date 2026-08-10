"""
Regression tests for the absolute normalization of the spin-2 signal covariance,
with emphasis on EB.

Guards the bug found 10 Aug 2026: `_eb_kernel` was missing a factor of 2, so
every fitted EB bandpower -- and, because the Fisher matrix is built from the
same kernel, its error bar -- came out exactly 2x the standard C_l^EB.  Writing
Q = Q_E + Q_B, the EB part of <Q_i Q_j> is <Q^E_i Q^B_j> + <Q^B_i Q^E_j>: BOTH
orderings.  The kernel counted one.

The error was invisible to every check performed before: EB had only ever been
run on real data as a null test, where value and error scale together and so
pulls, "consistent with zero at N sigma", chi^2 and detection significance are
all EXACTLY invariant under a common rescaling.  It shows up only against a
known nonzero truth.

Test 1 is the decisive one and carries no Monte-Carlo noise.  The (Q,U) map is a
LINEAR function of the alm real degrees of freedom,

    x = sum_dof a_dof * v_dof,      v_dof = alm2map(unit alm),

so the exact pixel covariance is sum_dof,dof' <a_dof a_dof'> v_dof v_dof'^T,
computable to machine precision.  Comparing that against build_signal_cov()
tests the production path with no fitted prefactors and no MC noise floor.

Tests:
  1. Exact alm->map covariance vs build_signal_cov for EE, BB and EB. Machine
     precision.  EB FAILS by exactly 2x on the pre-fix code.
  2. End-to-end: recover a known nonzero C_l^EB from a full-sky realization.
     A factor 2 shows up here at >10 sigma.

Run from repo root or tests/:
    python3 tests/test_eb_normalization.py
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


def rel_rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))


# ===========================================================================
# Test 1: exact alm->map covariance vs build_signal_cov
# ===========================================================================

def test_exact_covariance_normalization():
    """Machine-precision check of the EE, BB and EB covariance normalization."""
    print("\nTest 1: exact alm->map covariance vs build_signal_cov")

    nside, lmin, lmax = 4, 2, 7
    npix = hp.nside2npix(nside)
    obs = np.arange(npix)
    band_edges = np.array([lmin, lmax + 1])       # one band spanning l=2..7

    C_EE, C_BB = 1.0e-6, 1.0e-6
    C_EB = 0.7 * np.sqrt(C_EE * C_BB)

    # ---- exact covariance from the alm -> map linear operator ----
    sz = hp.Alm.getsize(lmax)
    zero = np.zeros(sz, dtype=complex)

    def basis(l, m, field, part):
        a = np.zeros(sz, dtype=complex)
        a[hp.Alm.getidx(lmax, l, m)] = 1.0 if part == 're' else 1.0j
        alm_E, alm_B = (a, zero) if field == 'E' else (zero, a)
        _, Q, U = hp.alm2map([zero, alm_E, alm_B], nside=nside, lmax=lmax)
        return np.concatenate([Q, U])

    n2 = 2 * npix
    cov_ee = np.zeros((n2, n2))
    cov_bb = np.zeros((n2, n2))
    cov_eb = np.zeros((n2, n2))
    for l in range(lmin, lmax + 1):
        for m in range(l + 1):
            # a_l0 is real; for m>0 the Re and Im parts each carry variance C_l/2
            parts = ['re'] if m == 0 else ['re', 'im']
            w = 1.0 if m == 0 else 0.5
            for part in parts:
                vE = basis(l, m, 'E', part)
                vB = basis(l, m, 'B', part)
                cov_ee += w * np.outer(vE, vE)
                cov_bb += w * np.outer(vB, vB)
                cov_eb += w * (np.outer(vE, vB) + np.outer(vB, vE))

    # ---- BJK's covariance, via the production path ----
    rng = np.random.default_rng(0)
    sigma = 1e-3
    Np = np.full(npix, sigma ** 2 * 4 * np.pi / npix)
    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[rng.standard_normal(npix) * sigma],
        d_U_list=[rng.standard_normal(npix) * sigma],
        obs_pix=obs, nside=nside,
        N_T_list=[], N_Q_list=[Np], N_U_list=[Np],
        lmin=lmin, lmax=lmax, band_edges=band_edges,
        band_model='Cl', include_EB=True)

    idx = {spec: i for i, spec, _, _, _ in lik.layout.entries()}

    def signal(cee, cbb, ceb):
        p = np.zeros(lik.layout.n_params)
        p[idx['EE']], p[idx['BB']], p[idx['EB']] = cee, cbb, ceb
        return lik.build_signal_cov(p)

    r_ee = rel_rms(signal(C_EE, 0, 0), C_EE * cov_ee)
    r_bb = rel_rms(signal(0, C_BB, 0), C_BB * cov_bb)
    r_eb = rel_rms(signal(0, 0, C_EB), C_EB * cov_eb)
    r_all = rel_rms(signal(C_EE, C_BB, C_EB),
                    C_EE * cov_ee + C_BB * cov_bb + C_EB * cov_eb)

    print(f"    EE     rel rms residual = {r_ee:.3e}")
    print(f"    BB     rel rms residual = {r_bb:.3e}")
    print(f"    EB     rel rms residual = {r_eb:.3e}")
    print(f"    EE+BB+EB               = {r_all:.3e}")

    # Amplitude of the EB kernel relative to truth: 1.0 correct, 0.5 pre-fix.
    K_eb = lik._get_kernel(idx['EB'])
    amp = (np.dot(K_eb.ravel(), cov_eb.ravel())
           / np.dot(cov_eb.ravel(), cov_eb.ravel()))
    print(f"    EB kernel / exact d(Cov)/d(C_EB) = {amp:.6f}"
          f"   (1.0 correct, 0.5 = the pre-Aug-2026 bug)")

    tol = 1e-10
    check("EE covariance exact", r_ee < tol, f"{r_ee:.3e}")
    check("BB covariance exact", r_bb < tol, f"{r_bb:.3e}")
    check("EB covariance exact", r_eb < tol,
          f"{r_eb:.3e} (kernel is {amp:.4f}x the truth)")
    check("EE+BB+EB covariance exact", r_all < tol, f"{r_all:.3e}")
    check("EB kernel amplitude == 1", abs(amp - 1.0) < 1e-10, f"{amp:.6f}")


# ===========================================================================
# Test 2: end-to-end recovery of a known nonzero EB
# ===========================================================================

def test_nonzero_eb_recovery():
    """Full-sky ML recovery of a known nonzero C_l^EB, single band."""
    print("\nTest 2: end-to-end recovery of a known nonzero EB")

    nside, lmin, lmax = 8, 2, 16
    npix = hp.nside2npix(nside)
    obs = np.arange(npix)
    band_edges = np.array([lmin, lmax + 1])

    C_EE, C_BB = 4.0e-6, 1.0e-6
    r_EB = 0.8
    C_EB = r_EB * np.sqrt(C_EE * C_BB)

    # Correlated (a_E, a_B) with a flat 2x2 spectrum over l = 2..lmax
    rng = np.random.default_rng(20260810)
    sz = hp.Alm.getsize(lmax)
    alm_E = np.zeros(sz, dtype=complex)
    alm_B = np.zeros(sz, dtype=complex)
    L = np.linalg.cholesky(np.array([[C_EE, C_EB], [C_EB, C_BB]]))
    for l in range(lmin, lmax + 1):
        for m in range(l + 1):
            i = hp.Alm.getidx(lmax, l, m)
            if m == 0:
                x = rng.standard_normal(2)
            else:
                x = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) / np.sqrt(2)
            a = L @ x
            alm_E[i], alm_B[i] = a[0], a[1]

    zero = np.zeros(sz, dtype=complex)
    _, Q, U = hp.alm2map([zero, alm_E, alm_B], nside=nside, lmax=lmax)

    sigma = 3e-4                              # well below the signal
    Q = Q + rng.standard_normal(npix) * sigma
    U = U + rng.standard_normal(npix) * sigma
    Np = np.full(npix, sigma ** 2)

    lik = PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=obs, nside=nside,
        N_T_list=[], N_Q_list=[Np], N_U_list=[Np],
        lmin=lmin, lmax=lmax, band_edges=band_edges,
        band_model='Cl', include_EB=True)

    idx = {spec: i for i, spec, _, _, _ in lik.layout.entries()}
    start = np.zeros(lik.layout.n_params)
    start[idx['EE']], start[idx['BB']] = C_EE, C_BB
    cl_ml, sig, _ = lik.newton_raphson(start, max_iter=30)

    truth = {'EE': C_EE, 'BB': C_BB, 'EB': C_EB}
    pulls = {}
    for spec in ('EE', 'BB', 'EB'):
        i = idx[spec]
        pulls[spec] = (cl_ml[i] - truth[spec]) / sig[i]
        print(f"    {spec}: ML = {cl_ml[i]:.4e} +/- {sig[i]:.3e}   "
              f"truth = {truth[spec]:.4e}   ratio = {cl_ml[i]/truth[spec]:.3f}   "
              f"pull = {pulls[spec]:+.2f}")

    # A factor-2 EB error lands at pull ~ +1/sigma_rel, many sigma away.
    pull_if_2x = (2 * C_EB - C_EB) / sig[idx['EB']]
    print(f"    (a 2x EB bug would give pull = {pull_if_2x:+.1f})")

    for spec in ('EE', 'BB', 'EB'):
        check(f"{spec} recovered within 3 sigma", abs(pulls[spec]) < 3.0,
              f"pull = {pulls[spec]:+.2f}")


if __name__ == '__main__':
    print("=" * 70)
    print("EB normalization regression tests")
    print("=" * 70)
    test_exact_covariance_normalization()
    test_nonzero_eb_recovery()
    print()
    if _FAILURES:
        print(f"{FAIL}: {len(_FAILURES)} check(s) failed: {', '.join(_FAILURES)}")
        sys.exit(1)
    print(f"{PASS}: all checks passed")
