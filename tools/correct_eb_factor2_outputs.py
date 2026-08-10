"""
ONE-TIME MIGRATION (10 Aug 2026), kept for provenance and reproducibility.

Correct stored BJK EB outputs for the factor-2 kernel error (bjk-pixlik
2379951).  Regenerates from the .PREFIX_EB2X.bak originals every time, so it is
idempotent and safe to re-run.

EB value and sigma are divided by 2; EE and BB are untouched.  This is exact:
the old kernel was K_new/2, so the old fit returned theta = 2*C_EB and built
the IDENTICAL covariance.  The EE/BB estimates are bit-identical and the EB
axis maps by theta -> theta/2, with the Fisher error taking the same Jacobian.

Already applied. Safe to re-run: it regenerates from the .bak originals, so
it is idempotent and will not double-halve anything.

IMPORTANT: the column-name header must stay on line 1 -- consumers read these
with np.genfromtxt(names=True), which takes the first line as field names.
The explanatory banner therefore goes immediately AFTER it.
"""
import os
import shutil
import numpy as np

BASE = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac')

FILES = [
    ('almanac_runs/bjk_results/bjk_euclid_tombin-1_nside128_eb_n1.dat', 'A'),
    ('TR1/runs/bjk_results/bjk_TR1_nside128_dl10_raw_eb.dat', 'A'),
    ('TR1/runs/bjk_results/bjk_TR1_nside128_dl10_eb.dat', 'A'),
    ('sim_runs/sim_lowbb_nside64_fsky01/bjk_eb_lowbb_dell10.dat', 'B'),
    ('sim_runs/sim_eb_nside64_fsky01/bjk_eb_eb03_dell20.dat', 'B'),
]

BANNER = """\
# =====================================================================
# CORRECTED 2026-08-10: EB value and sigma DIVIDED BY 2.
#
# The BJK EB kernel was missing a factor of 2 (fixed in bjk-pixlik
# commit 2379951, 10 Aug 2026), so every fitted EB bandpower AND its
# Fisher error came out exactly 2 x C_l^EB.
#
# EE and BB rows are UNCHANGED and were never affected. The old and new
# models build the identical covariance, so this is an exact
# reparameterisation of the EB axis, not a re-fit -- no re-run needed.
#
# SNR is invariant under the rescaling (value and sigma scale together),
# which is exactly why the error went unnoticed in null tests. The
# 'pull' column IS affected wherever the EB truth is nonzero.
#
# Original preserved at: {bak}
# ====================================================================="""

LOWBB_WARNING = """\
#
# ---------------------------------------------------------------------
# WARNING -- THIS FIT LOOKS UNRELIABLE, INDEPENDENTLY OF THE ABOVE.
# BB truth 6.03e-10 vs fit -3.47e-09 with sigma 1.30e-10: a -31 sigma
# pull, and the EB pulls against a ZERO truth reach 6 sigma. This is the
# sim where EE and BB differ by ~5 decades -- the case in which
# Newton-Raphson was found to fail silently and return its starting
# vector with plausible-looking sigmas (fixed in bjk-pixlik fdf469d,
# which now raises instead). RE-RUN THIS BEFORE USING IT.
# Note EB truth is 0 here, so pull = value/sigma is invariant under the
# rescaling: the pull column is unchanged, and that is expected.
# ---------------------------------------------------------------------"""


def fix(path, fmt):
    full = os.path.join(BASE, path)
    bak = full + '.PREFIX_EB2X.bak'
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    with open(bak) as fh:
        lines = fh.read().splitlines()

    out, before, after = [], [], []
    header_done = False
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            out.append(line)
            if not header_done:                     # column names: line 1
                out.append(BANNER.format(bak=os.path.basename(bak)))
                if 'lowbb' in path:
                    out.append(LOWBB_WARNING)
                header_done = True
            continue
        tok = s.split()
        if fmt == 'A':
            ell, spec, i, j, val, sig, _snr = tok
            if spec != 'EB':
                out.append(line)
                continue
            v, g = float(val) / 2.0, float(sig) / 2.0
            before.append(float(val)); after.append(v)
            out.append(f"{float(ell):.2f}  {spec}  {i}  {j}  "
                       f"{v:.6e}  {g:.6e}  {v / g:.4f}")
        else:
            spec, band, ell, truth, val, sig, pull = tok
            if spec != 'EB':
                out.append(line)
                continue
            v, g, t = float(val) / 2.0, float(sig) / 2.0, float(truth)
            before.append(float(pull)); after.append((v - t) / g)
            out.append(f"{spec} {band} {float(ell):.2f} {t:.6e} "
                       f"{v:.6e} {g:.6e} {(v - t) / g:.4f}")

    with open(full, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    return fmt, before, after


print("Correcting stored BJK EB outputs (regenerated from .bak)\n" + "=" * 62)
for path, fmt in FILES:
    f, b, a = fix(path, fmt)
    print(f"\n{path}\n  {len(b)} EB rows corrected")
    if f == 'B':
        b, a = np.array(b), np.array(a)
        print(f"  pull before : mean {b.mean():+.2f}  rms {np.sqrt((b**2).mean()):.2f}"
              f"  max|{np.abs(b).max():.2f}|")
        print(f"  pull after  : mean {a.mean():+.2f}  rms {np.sqrt((a**2).mean()):.2f}"
              f"  max|{np.abs(a).max():.2f}|")

# verify every file still parses the way its consumers read it
print("\n" + "=" * 62)
print("Re-parse check (np.genfromtxt(names=True), as the plot scripts do)")
for path, _ in FILES:
    d = np.genfromtxt(os.path.join(BASE, path), names=True, dtype=None,
                      encoding='utf-8')
    print(f"  OK  {len(d):3d} rows, fields {d.dtype.names}  {os.path.basename(path)}")
