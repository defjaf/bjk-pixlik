# BJK EB bandpower is a factor 2 high — handoff  [RESOLVED 10 Aug 2026]

> **RESOLVED.** Everything below was written before the fix and is kept as the
> record of how the error was found. Outcome:
>
> - The factor of 2 is **confirmed exactly** — the EB kernel was 0.500000x the
>   true derivative, verified to ~1e-15 against the exact alm->map covariance
>   (no Monte Carlo). Fixed in `2379951`.
> - The "prime suspect" below was **correct**: `derivations/verify_eb_tb.py`
>   divided by an extra 2, fitting the prefactor against half the true
>   covariance. That script is corrected so it cannot re-seed the error.
> - The repair belonged in the **kernel normalisation**, not in documenting the
>   parameter as 2 C_l^EB.
> - A **second, independent** defect surfaced: EB used unordered bin pairs, so
>   for n_P>1 one parameter stood for the independent spectra C^{E_i B_j} and
>   C^{E_j B_i}. Fixed in `8c28aec` (n_P^2 ordered pairs). n_P=1 unaffected.
> - **TE and TB are clean** — all six spectrum types now audited exactly in
>   `tests/test_covariance_exact.py` (`382918c`).
> - Side note 1 (Newton-Raphson returning its input) fixed in `fdf469d`: it now
>   raises `NewtonRaphsonError` rather than returning the starting vector with
>   plausible-looking sigmas.
> - **Downstream sweep done.** No re-running was needed: the old kernel was
>   exactly K_new/2, so the fit built the identical covariance and the
>   correction is an exact reparameterisation (EB and sigma_EB / 2, EE and BB
>   bit-identical). Five stored `.dat` files corrected with `.PREFIX_EB2X.bak`
>   originals kept, and five plots regenerated. The eb03 closed-loop pulls went
>   from mean +1.96 / rms 2.16 to **+0.12 / 0.96**, matching the prediction
>   below exactly. See
>   `~/Desktop/Projects/Almanac/Euclid-Almanac/EB_FACTOR2_CORRECTION.md`.
> - Still open: `bjk_eb_lowbb_dell10.dat` is unreliable for an unrelated reason
>   (BB pull -31 sigma; the silent-NR case) and should be re-run; and the
>   n_P>1 EB path has no end-to-end validation on real data.

---


Found 10 Aug 2026 while completing item 3 of the walnutpie study (BJK+EB
end-to-end at fsky=0.1, NSIDE=64). Written for a fresh conversation working in
the bjk-pixlik tree. **Nothing in bjk-pixlik has been modified**; that repo is
at `bf0d8a9` with a clean tree apart from the pre-existing untracked
`examples/run_bjk_TR1_dl10_raw.py`.

## The claim

**BJK's fitted EB bandpower is exactly 2x the standard C_l^EB. EE and BB are
unbiased. BJK's sigma on EB is also 2x.**

Closed-loop test against a simulation with known, nonzero EB:

| spec | mean pull | rms pull | max abs pull |
|------|-----------|----------|--------------|
| EE   | +0.14 | 0.97 | 1.58 |
| BB   | +0.06 | 0.67 | 1.07 |
| EB   | **+1.96** | **2.16** | **3.63** |

All 7 EB bands pull positive. Per-band recovered/truth = 2.61, 2.37, 1.75, 2.31,
2.61, 1.29, 2.09 (mean 2.15). Rescaling the truth by exactly 2 gives mean pull
+0.12, rms 0.96, max 1.83 — statistically perfect. So it is a clean factor of 2,
not a bias that happens to be positive.

## Why the factor is BJK's, established by elimination

1. **The sim is faithful to its own input.** AlmaSim's realised spectra vs input:
   EE 1.024, EB 1.007, BB 0.995; realised r_EB = 0.298 vs input 0.300.
2. **AlmaSim's C_l^EB is the standard HEALPix one.** `healpy.anafast` on the
   (full-sky) signal map, an independent implementation, returns EE/input
   1.0240, BB/input 0.9946, **EB/input 1.0071**, r_EB 0.298.
3. **BJK is internally consistent**, so this is a normalisation error rather
   than an inconsistency: `_eb_kernel` is exactly d/d(cEB) of the covariance
   assembly in `pixel_likelihood.py`, and both match the EB formula fixed
   earlier (QQ = -Km sin2Xpsi, QU = +Km cos2Xpsi, UU = +Km sin2Xpsi).

Almanac and HEALPix agree with each other; BJK disagrees with both by 2.

## Prime suspect (NOT confirmed)

The fitted prefactor in `derivations/verify_eb_tb.py`. Its MC estimator
symmetrises correctly —

```python
QQ_EB_sum += np.outer(QE, QB) + np.outer(QB, QE)
```

— i.e. it measures the full EB contribution, both cross terms. The script then
"fits candidate prefactors". If the prefactor was fitted against a model that
counts the EB block only ONCE, the resulting kernel is half the true
contribution, and the ML fit must set cEB = 2 C_l^EB to compensate. That is
exactly the observed signature.

**First thing to do: rerun that script and check the fitted prefactor against a
model that counts both (E,B) and (B,E).** That determines whether the repair
belongs in the kernel normalisation or in documenting BJK's parameter as
2 C_l^EB.

## Why every previous EB check passed

This is the important part, and it is not "EB was zero so there was nothing to
see" — on an EB=0 sim the per-band realised EB scatters and is not zero.

**The error bar is inflated by the same factor 2**, because sigma is built from
the same half-sized kernel via the Fisher matrix. Measured against the Gaussian
expectation sqrt((EE BB + EB^2)/nu), with nu taken from the EE and BB errors:

```
 band   nu_eff   sigma_EB(BJK)   expected      ratio
    0     28.3     4.637e-06     2.209e-06      2.10
    1    109.5     2.293e-06     1.123e-06      2.04
    2    179.6     1.753e-06     8.767e-07      2.00
    3    231.5     1.575e-06     7.722e-07      2.04
    4    253.4     1.494e-06     7.381e-07      2.02
    5    307.9     1.306e-06     6.697e-07      1.95
    6    149.6     1.944e-06     9.606e-07      2.02
                                     mean       2.03
```

So BJK's EB is a cleanly rescaled parameter, `theta +/- sigma_theta =
2 (C +/- sigma_C)`, and every normalised diagnostic is EXACTLY invariant:

- pull against zero: (2d)/(2s) = d/s
- "consistent with zero at N sigma"
- chi^2 against a null model
- detection significance theta/sigma_theta

In a BJK-vs-Almanac plot BJK sits at 2d +/- 2s and Almanac at d +/- s; since d
is a noise fluctuation with |d| <~ s, the intervals overlap comfortably. The
checks performed were per-band interval overlap, which is blind to a common
scaling. The signature lived only in the SLOPE of BJK EB vs Almanac EB across
bands — 2 instead of 1 — which nobody fitted.

Also relevant: all three EB-enabled scripts (`run_bjk_euclid_eb_n1.py`,
`run_bjk_TR1_dl10.py`, `run_bjk_TR1_dl10_raw.py`) run on REAL data, where EB is
a systematics null test. BJK's closed-loop sim validations were TT and EE only.
No sim with known nonzero EB had ever been pushed through BJK.

**General lesson worth keeping: a multiplicative error in a parameter whose
error bar is derived from the same wrong kernel is invisible to every null test
and to every "agrees within errors" comparison.** It surfaces only against a
known nonzero truth, or as a slope between two estimators.

## Consequences

- Any **significance** ever quoted from BJK EB is still correct (value and error
  scale together).
- Any statement of the EB **amplitude** in physical units has been 2x high with
  2x error bars.
- BJK is the reference estimator used to validate Almanac, so every
  BJK-vs-Almanac EB comparison is off by 2, including the committed TR1
  EE+BB+EB script. EE and BB comparisons are unaffected.

## Reproducing / regression test

Already committed in Euclid-Almanac (`0725480` and later):

```
cd ~/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs
python3 run_bjk_eb_nside64_fsky01.py --sim eb03
```

Sim `sim_eb_nside64_fsky01` (EB/sqrt(EE BB) = 0.3, BB above noise, lmax=128).
Runs in ~10 min, Delta_ell=20, 7 bands, 21 params, ~22.5 GB peak in `onthefly`.
**After a fix, EB pulls should land at rms ~1 against the UNSCALED truth.**
Worth wiring a nonzero-EB sim into bjk-pixlik's own tests permanently.

Deliberately chosen: EE, BB and EB in this sim are all proportional to
1/(l(l+1)), so all three have constant D_l and `band_model='Dl'` is exact —
there is no within-band model mismatch to muddy the comparison.

## Two side notes for the BJK conversation

1. **Newton-Raphson returns its input on failure.** With a poor start it printed
   `iter 1: no finite improvement, stopping` and returned the initial vector
   together with plausible-looking sigmas, rather than raising. A downstream
   pull table then looked superficially reasonable. Consider raising, or
   returning a convergence flag. (Encountered with a flat 1e-8 start on a sim
   where EE and BB differ by ~6 decades; worked around in the driver with a
   pseudo-C_l start, no bjk-pixlik change.)
2. **Memory correction, now fixed:** an earlier note claimed the EB formula fix
   was applied "in both pixel_likelihood.py copies". There is only ONE copy.
   `sim_runs/bjk/` was an untracked symlink into bjk-pixlik; see Euclid-Almanac
   `23de735`, which removed it in favour of `sim_runs/bjk_path.py`.

## Memory entry

`~/.claude/projects/-Users-jaffe-Desktop-Projects-Almanac/memory/project_bjk_eb_factor2.md`
(indexed in `MEMORY.md`). Full write-up also in
`~/Developer/Almanac/InternalPapers/walnutpie_status_aug2026.md`.
