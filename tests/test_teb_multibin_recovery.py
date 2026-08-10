"""
End-to-end closed-loop recovery for the FULL parameter set: T, E and B with
multiple tomographic bins, all six spectrum types, all bin pairs.

This is the end-to-end validation deferred on 10 Aug 2026.  The kernels were
already known to be exact against the true covariance
(tests/test_covariance_exact.py), but that is a statement about
build_signal_cov, not about whether the FIT recovers known inputs: layout
wiring, the Fisher matrix, Newton-Raphson and the parameter<->spectrum mapping
are all untested by a covariance check.

Configuration: n_T=2, n_P=2 with include_TB and include_EB.  That is deliberate
-- BJK's 21 pair-parameters are then EXACTLY the 21 independent entries of the
6x6 field covariance over (T0, T1, E0, B0, E1, B1), so the model is a complete
parameterisation of the simulation and there is nothing the truth can contain
that the fit cannot represent.  In particular it exercises:

  TT(i,j) auto and cross-bin            EE(i,j), BB(i,j) auto and cross-bin
  TE(i,j), TB(i,j) for all 4 ordered (i,j)
  EB(i,j) for all 4 ORDERED pairs, with C^{E_i B_j} != C^{E_j B_i} in the
    truth -- the case that unordered pairs could not represent (fixed 8c28aec)

Why an ensemble: at NSIDE=4 there are only 60 modes for 21 parameters, so a
single realisation gives ~50% errors per parameter and cannot resolve a
factor-2-scale bias.  Averaging pulls over N_SIM realisations reduces the error
on each family's mean pull by sqrt(N_SIM), which turns a 2x normalisation error
from marginal into overwhelming, while staying fast (seconds per fit).

Run from repo root or tests/:
    python3 tests/test_teb_multibin_recovery.py
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


NSIDE, LMIN, LMAX = 4, 2, 7
NPIX = hp.nside2npix(NSIDE)
BAND_EDGES = np.array([LMIN, LMAX + 1])          # single band, C_l flat in it
N_T, N_P = 2, 2
N_SIM = 24
SIGMA_T, SIGMA_P = 2.0e-4, 2.0e-4

# field order must match the (spec, i, j) -> matrix entry mapping below
FIELDS = ['T0', 'T1', 'E0', 'B0', 'E1', 'B1']
FIDX = {f: k for k, f in enumerate(FIELDS)}

# amplitudes: T brightest, then E, then B -- but kept within ~1 decade so the
# Fisher matrix stays well conditioned (the 5-decade case is a separate, known
# hard problem: see CLAUDE.md on the lowbb sim)
AMP = {'T0': 2.0e-5, 'T1': 1.2e-5, 'E0': 6.0e-6,
       'B0': 2.0e-6, 'E1': 4.0e-6, 'B1': 1.5e-6}


def truth_matrix():
    """A positive-definite 6x6 spectral matrix with asymmetric EB cross-terms."""
    n = len(FIELDS)
    # correlation matrix: start from a random PD matrix, shrink to keep
    # correlations moderate, then force the two EB cross-bin entries apart
    rng = np.random.default_rng(4242)
    M = rng.standard_normal((n, n))
    R = M @ M.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    R = 0.45 * R + 0.55 * np.eye(n)              # shrink toward identity
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)

    # make the EB cross-bin pair explicitly asymmetric: C^{E0B1} != C^{E1B0}
    def setcorr(a, b, v):
        R[FIDX[a], FIDX[b]] = R[FIDX[b], FIDX[a]] = v
    setcorr('E0', 'B1', +0.30)
    setcorr('E1', 'B0', -0.12)
    setcorr('E0', 'B0', +0.20)                   # auto-bin EB
    setcorr('E1', 'B1', -0.18)

    amp = np.array([AMP[f] for f in FIELDS])
    S = R * np.sqrt(np.outer(amp, amp))
    w = np.linalg.eigvalsh(S)
    assert w.min() > 0, f"truth matrix not PD (min eig {w.min():.3e})"
    return S


S_TRUE = truth_matrix()

# BJK parameter (spec, i, j) -> entry of the 6x6 matrix
SPEC_FIELDS = {'TT': ('T', 'T'), 'TE': ('T', 'E'), 'TB': ('T', 'B'),
               'EE': ('E', 'E'), 'BB': ('B', 'B'), 'EB': ('E', 'B')}


def truth_for(spec, i, j):
    fa, fb = SPEC_FIELDS[spec]
    return S_TRUE[FIDX[f"{fa}{i}"], FIDX[f"{fb}{j}"]]


def simulate(seed):
    """Draw correlated alms for all 6 fields, return (T maps, Q maps, U maps)."""
    rng = np.random.default_rng(seed)
    sz = hp.Alm.getsize(LMAX)
    n = len(FIELDS)
    alm = np.zeros((n, sz), dtype=complex)
    L = np.linalg.cholesky(S_TRUE)
    for l in range(LMIN, LMAX + 1):
        for m in range(l + 1):
            k = hp.Alm.getidx(LMAX, l, m)
            if m == 0:
                x = rng.standard_normal(n)
            else:
                x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
            alm[:, k] = L @ x

    zero = np.zeros(sz, dtype=complex)
    T_list, Q_list, U_list = [], [], []
    for i in range(N_T):
        t = hp.alm2map(alm[FIDX[f'T{i}']].copy(), nside=NSIDE, lmax=LMAX)
        T_list.append(t + rng.normal(0, SIGMA_T, NPIX))
    for j in range(N_P):
        _, q, u = hp.alm2map([zero, alm[FIDX[f'E{j}']].copy(),
                              alm[FIDX[f'B{j}']].copy()], nside=NSIDE, lmax=LMAX)
        Q_list.append(q + rng.normal(0, SIGMA_P, NPIX))
        U_list.append(u + rng.normal(0, SIGMA_P, NPIX))
    return T_list, Q_list, U_list


def build_lik(seed):
    T_list, Q_list, U_list = simulate(seed)
    NT = np.full(NPIX, SIGMA_T ** 2)
    NP_ = np.full(NPIX, SIGMA_P ** 2)
    return PixelLikelihood.from_arrays(
        d_T_list=T_list, d_Q_list=Q_list, d_U_list=U_list,
        obs_pix=np.arange(NPIX), nside=NSIDE,
        N_T_list=[NT] * N_T, N_Q_list=[NP_] * N_P, N_U_list=[NP_] * N_P,
        lmin=LMIN, lmax=LMAX, band_edges=BAND_EDGES,
        band_model='Cl', include_TB=True, include_EB=True)


def truth_vector(layout):
    return np.array([truth_for(spec, i, j)
                     for _, spec, i, j, _ in layout.entries()])


def run_one(seed, sigma_ref):
    """Fit one realisation.  Pulls use sigma_ref (the Fisher error AT THE
    TRUTH), not the per-fit sigma.

    Using the per-fit sigma biases every closed-loop statistic low: sigma is
    the Fisher error evaluated at the ML point and scales with the fitted
    amplitude, so downward fluctuations get a smaller sigma and therefore a
    larger pull and more weight.  That is a property of the estimator, not of
    the kernels -- it showed up coherently in ALL six families, including
    TT/EE/BB, whose covariance is exact to 1e-15.  The Fisher at the truth is
    data-independent, so it is the same for every realisation and removes the
    correlation.
    """
    lik = build_lik(seed)
    start = truth_vector(lik.layout)
    cl_ml, sigma_fit, _ = lik.newton_raphson(start, max_iter=40, tol=1e-6)

    rows = []
    for idx, spec, i, j, b in lik.layout.entries():
        t = truth_for(spec, i, j)
        rows.append((spec, i, j, cl_ml[idx], sigma_ref[idx], t,
                     (cl_ml[idx] - t) / sigma_ref[idx], sigma_fit[idx]))
    return lik, rows


if __name__ == '__main__':
    print("=" * 72)
    print(f"Full TEB multi-bin recovery: n_T={N_T}, n_P={N_P}, TB+EB, "
          f"NSIDE={NSIDE}, l={LMIN}..{LMAX}")
    print("=" * 72)

    # Fisher error at the TRUTH: depends on the model, not the data, so it is
    # identical for every realisation and is computed once.
    _lik0 = build_lik(999)
    _, F_true = _lik0.gradient_and_fisher(truth_vector(_lik0.layout))
    SIGMA_REF = np.sqrt(np.diag(np.linalg.inv(F_true)))
    print(f"\nFisher errors evaluated at the truth "
          f"({len(SIGMA_REF)} parameters, data-independent)\n")

    all_rows, n_conv = [], 0
    for s in range(N_SIM):
        lik, rows = run_one(1000 + s, SIGMA_REF)
        n_conv += int(lik.nr_info['converged'])
        all_rows.extend(rows)
        print(f"  sim {s}: {lik.nr_info['status']:>10s}  "
              f"({lik.nr_info['n_accepted']} steps)")

    npar = len(all_rows) // N_SIM
    print(f"\n{npar} parameters x {N_SIM} sims = {len(all_rows)} pulls")

    # Per-family fitted NORMALISATION alpha (recovered = alpha x truth).
    #
    # This is the statistic that would have caught the EB factor-2 error, and
    # the one nobody computed: the handoff noted the signature "lived only in
    # the SLOPE of BJK EB vs truth across bands -- 2 instead of 1".  Mean pull
    # is the wrong tool for a MULTIPLICATIVE error when the truths have mixed
    # signs (here EB truths do), because the induced shifts +t_k/sigma_k
    # partially cancel in the mean.  The weighted slope does not cancel.
    #
    #   alpha = sum(t_k v_k / s_k^2) / sum(t_k^2 / s_k^2)
    #   sigma_alpha = 1 / sqrt(sum(t_k^2 / s_k^2))
    #
    # alpha = 1 is correct; the pre-Aug-2026 EB kernel would give alpha = 2.
    print(f"\n{'spec':<5} {'n':>3} {'mean pull':>10} {'rms pull':>9} "
          f"{'alpha':>8} {'+-':>7} {'(alpha-1)/sig':>14} {'2x would be':>12}")
    ok = True
    for spec in ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']:
        sel = [r for r in all_rows if r[0] == spec]
        p = np.array([r[6] for r in sel])
        v = np.array([r[3] for r in sel])
        s = np.array([r[4] for r in sel])
        t = np.array([r[5] for r in sel])
        w = (t / s) ** 2
        alpha = np.sum(t * v / s ** 2) / np.sum(t ** 2 / s ** 2)
        sig_a = 1.0 / np.sqrt(np.sum(t ** 2 / s ** 2))
        print(f"{spec:<5} {len(p):>3} {p.mean():>+10.3f} "
              f"{np.sqrt((p**2).mean()):>9.3f} {alpha:>8.3f} {sig_a:>7.3f} "
              f"{(alpha-1)/sig_a:>+14.2f} {1.0/sig_a:>+12.1f}")
        # alpha must be 1 within 4 sigma; the last column shows how many sigma
        # a factor-2 error would be, i.e. the test's actual sensitivity
        if abs(alpha - 1.0) / sig_a > 4.0:
            ok = False

    allp = np.array([r[6] for r in all_rows])
    print(f"\n{'ALL':<5} {len(allp):>3} {allp.mean():>+10.3f} "
          f"{np.sqrt((allp**2).mean()):>9.3f}")

    # a representative single realisation, for eyeballing
    print("\nFirst realisation, EB block (the ordered-pair case):")
    print(f"  {'spec':<5} {'i':>2} {'j':>2} {'ML':>12} {'sigma':>11} "
          f"{'truth':>12} {'pull':>7}")
    for r in all_rows[:npar]:
        if r[0] == 'EB':
            print(f"  {r[0]:<5} {r[1]:>2} {r[2]:>2} {r[3]:>12.4e} "
                  f"{r[4]:>11.3e} {r[5]:>12.4e} {r[6]:>+7.2f}")

    # ------------------------------------------------------------------
    # Mis-indexing sensitivity: are the 21 spectra actually DIFFERENT enough
    # that a wiring/indexing error would show?  A recovery test proves nothing
    # if the truths are so similar that swapping two of them costs nothing.
    # So score the fitted values against deliberately WRONG index mappings and
    # require the correct one to win decisively.
    # (Diagonal chi^2: the parameters are correlated, so this understates the
    # true discrimination -- it is a lower bound on sensitivity.)
    # ------------------------------------------------------------------
    print("\nMis-indexing sensitivity (chi^2 of the SAME fits vs a wrong truth map)")

    def swapT(i):
        return 1 - i if N_T == 2 else i

    def swapP(i):
        return 1 - i if N_P == 2 else i

    WRONG = {
        'correct                     ': lambda sp, i, j: truth_for(sp, i, j),
        'EB transposed (i<->j)       ': lambda sp, i, j: truth_for(sp, j, i) if sp == 'EB' else truth_for(sp, i, j),
        'TE transposed (i<->j)       ': lambda sp, i, j: truth_for(sp, j, i) if sp == 'TE' else truth_for(sp, i, j),
        'TB transposed (i<->j)       ': lambda sp, i, j: truth_for(sp, j, i) if sp == 'TB' else truth_for(sp, i, j),
        'P bins swapped (0<->1)      ': lambda sp, i, j: truth_for(sp, i if sp in ('TT',) else (swapP(i) if sp not in ('TE', 'TB') else i), swapP(j) if sp != 'TT' else j),
        'T bins swapped (0<->1)      ': lambda sp, i, j: truth_for(sp, swapT(i) if sp in ('TT', 'TE', 'TB') else i, swapT(j) if sp == 'TT' else j),
        'EE <-> BB                   ': lambda sp, i, j: truth_for({'EE': 'BB', 'BB': 'EE'}.get(sp, sp), i, j),
    }
    chi2 = {}
    for label, f in WRONG.items():
        c = 0.0
        for r in all_rows:
            sp, i, j, v, sref = r[0], r[1], r[2], r[3], r[4]
            c += ((v - f(sp, i, j)) / sref) ** 2
        chi2[label] = c
    base = chi2['correct                     ']
    for label, c in chi2.items():
        tag = '' if label.startswith('correct') else f'   Dchi2 = {c - base:+9.1f}'
        print(f"    {label} chi2 = {c:9.1f}  ({c/len(all_rows):.2f} per dof){tag}")
    worst_wrong = min(v for k, v in chi2.items() if not k.startswith('correct'))
    print(f"\n    correct mapping is preferred by Dchi2 = {worst_wrong - base:.1f} "
          f"over the closest wrong one")

    print()
    check("every wrong index mapping is rejected (Dchi2 > 50)",
          worst_wrong - base > 50.0, f"closest wrong is +{worst_wrong-base:.1f}")
    check("all fits converged", n_conv == N_SIM, f"{n_conv}/{N_SIM}")
    check("no spectrum family biased at >4 SE", ok)
    check("overall rms pull ~ 1", 0.7 < np.sqrt((allp**2).mean()) < 1.4,
          f"{np.sqrt((allp**2).mean()):.3f}")
    print()
    if _FAILURES:
        print(f"{FAIL}: {len(_FAILURES)} check(s) failed: {', '.join(_FAILURES)}")
        sys.exit(1)
    print(f"{PASS}: all checks passed")
