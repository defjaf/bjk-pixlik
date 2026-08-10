"""
Regression tests for Newton-Raphson failure reporting.

Guards the behaviour flagged in the Aug 2026 EB handoff: with a poor starting
point the routine printed "no finite improvement, stopping" and then returned
the STARTING VECTOR together with a plausible-looking sigma, rather than
signalling failure.  A downstream pull table built from that output looked
entirely reasonable while containing no fit at all.

Tests:
  1. A hopeless start raises NewtonRaphsonError (strict=True, the default).
  2. strict=False downgrades it to a RuntimeWarning and reports the failure in
     lik.nr_info, with n_accepted == 0 and the return equal to the input.
  3. A good run reports converged with n_accepted > 0.
  4. Starting AT the optimum is reported as converged, not as failure -- the
     line search finds no improvement there either, so the two must be told
     apart by how much logL could still move, not by whether a step was taken.

Run from repo root or tests/:
    python3 tests/test_newton_failure.py
"""

import sys, os
import warnings
import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood, NewtonRaphsonError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _FAILURES.append(name)


NSIDE, LMIN, LMAX = 4, 2, 7
NPIX = hp.nside2npix(NSIDE)
BAND_EDGES = np.array([LMIN, LMAX + 1])
D_EE, D_BB = 2e-6, 1e-6
SIGMA_P = 4e-4


def _make_lik(seed=3):
    cls = np.zeros((4, LMAX + 1))
    cls[1, LMIN:] = D_EE
    cls[2, LMIN:] = D_BB
    np.random.seed(seed)
    _, Q, U = hp.synfast(cls, nside=NSIDE, lmax=LMAX, new=True)
    rng = np.random.default_rng(seed)
    Q = Q + rng.normal(0, SIGMA_P, NPIX)
    U = U + rng.normal(0, SIGMA_P, NPIX)
    NP = np.full(NPIX, SIGMA_P ** 2)
    return PixelLikelihood.from_arrays(
        d_T_list=[], d_Q_list=[Q], d_U_list=[U],
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[], N_Q_list=[NP], N_U_list=[NP],
        lmin=LMIN, lmax=LMAX, band_edges=BAND_EDGES, band_model='Cl')


# A start that makes M = S + N non-positive-definite: a large negative EE.
BAD_START = np.array([-1.0, -1.0])          # (EE, BB) for the single band
GOOD_START = np.array([D_EE, D_BB])


def test_strict_raises():
    print("\nTest 1: hopeless start raises under strict=True (default)")
    lik = _make_lik()
    try:
        lik.newton_raphson(BAD_START, max_iter=10)
    except NewtonRaphsonError as e:
        print(f"    raised NewtonRaphsonError: {str(e)[:60]}...")
        check("raises NewtonRaphsonError", True)
        check("nr_info records the failure",
              lik.nr_info['n_accepted'] == 0 and not lik.nr_info['converged'],
              f"{lik.nr_info}")
        return
    check("raises NewtonRaphsonError", False, "no exception raised")


def test_nonstrict_warns():
    print("\nTest 2: strict=False warns and reports via nr_info")
    lik = _make_lik()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl, sigma, F = lik.newton_raphson(BAD_START, max_iter=10, strict=False)
    got = [x for x in w if issubclass(x.category, RuntimeWarning)]
    print(f"    status = {lik.nr_info['status']!r}, "
          f"n_accepted = {lik.nr_info['n_accepted']}, warnings = {len(got)}")
    check("emits a RuntimeWarning", len(got) >= 1, f"{len(got)} warnings")
    check("status flags failure",
          lik.nr_info['status'] in ('no_progress', 'singular'),
          f"{lik.nr_info['status']!r}")
    check("n_accepted == 0", lik.nr_info['n_accepted'] == 0)
    check("returns the input vector unchanged", np.allclose(cl, BAD_START),
          f"{cl}")


def test_good_run_converges():
    print("\nTest 3: a good start converges and says so")
    lik = _make_lik()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl, sigma, F = lik.newton_raphson(GOOD_START, max_iter=30)
    got = [x for x in w if issubclass(x.category, RuntimeWarning)]
    print(f"    status = {lik.nr_info['status']!r}, "
          f"n_accepted = {lik.nr_info['n_accepted']}, "
          f"cl = {cl}, sigma = {sigma}")
    check("converged", lik.nr_info['converged'], f"{lik.nr_info}")
    check("took at least one step", lik.nr_info['n_accepted'] >= 1)
    check("no warning on a clean run", len(got) == 0, f"{[str(x.message) for x in got]}")
    check("sigma finite and positive", np.all(np.isfinite(sigma)) and np.all(sigma > 0))


def test_start_at_optimum():
    print("\nTest 4: starting at the optimum is convergence, not failure")
    lik = _make_lik()
    cl_ml, _, _ = lik.newton_raphson(GOOD_START, max_iter=30)
    # Re-run starting exactly at the ML point: no step can improve logL, which
    # is the same line-search outcome as a hopeless start.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl2, sigma2, _ = lik.newton_raphson(cl_ml, max_iter=30)
    got = [x for x in w if issubclass(x.category, RuntimeWarning)]
    print(f"    status = {lik.nr_info['status']!r}, "
          f"n_accepted = {lik.nr_info['n_accepted']}, warnings = {len(got)}")
    check("reported as converged", lik.nr_info['converged'], f"{lik.nr_info}")
    check("no spurious warning", len(got) == 0,
          f"{[str(x.message) for x in got]}")
    check("stays at the optimum", np.allclose(cl2, cl_ml, rtol=1e-3),
          f"{cl2} vs {cl_ml}")


def test_exhausted_line_search_at_optimum():
    """A line search that fails AT the optimum must report convergence.

    Exercises the logL_eps branch directly: with tol=0 the step criterion can
    never fire, so the iteration runs until no halving improves logL at all --
    the same code path a genuinely stuck fit takes.  The two are separated by
    how much logL could still move, not by whether a step was accepted.
    """
    print("\nTest 5: exhausted line search at the optimum reports convergence")
    lik = _make_lik()
    cl_ml, _, _ = lik.newton_raphson(GOOD_START, max_iter=30)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lik.newton_raphson(cl_ml, max_iter=10, tol=0.0)
    got = [x for x in w if issubclass(x.category, RuntimeWarning)]
    print(f"    status = {lik.nr_info['status']!r}, "
          f"n_accepted = {lik.nr_info['n_accepted']}, warnings = {len(got)}")
    check("logL_eps branch reports converged", lik.nr_info['converged'],
          f"{lik.nr_info}")
    check("no spurious warning from the stalled line search", len(got) == 0,
          f"{[str(x.message) for x in got]}")


if __name__ == '__main__':
    print("=" * 70)
    print("Newton-Raphson failure-reporting tests")
    print("=" * 70)
    test_strict_raises()
    test_nonstrict_warns()
    test_good_run_converges()
    test_start_at_optimum()
    test_exhausted_line_search_at_optimum()
    print()
    if _FAILURES:
        print(f"{FAIL}: {len(_FAILURES)} check(s) failed: {', '.join(_FAILURES)}")
        sys.exit(1)
    print(f"{PASS}: all checks passed")
