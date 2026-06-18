"""
Gelman-Rubin R-hat and ESS diagnostics for nmt1-nmt4 chains.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

sys.path.insert(0, '/Users/jaffe/Developer/Almanac/Almanac/Plotting/PostProcessAnalysis')
from utils.generic_utils import read_npy_memmap

BASE   = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
BURNIN = 50000
THIN   = 100
LMIN, LMAX, NFIELDS = 2, 256, 2

EXTRACT_FILES = [
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt1/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt2/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt3/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt4/almanac_out.extract.npy',
]

ells = np.arange(LMIN, LMAX + 1)
ncls = NFIELDS * (NFIELDS + 1) // 2
idx  = np.arange(len(ells))
cols = {
    'C_EE': idx * ncls + 0,
    'C_EB': idx * ncls + 1,
    'C_BB': idx * ncls + 2,
}

print("Loading chains (thinning by %d)..." % THIN)
chains = []
for f in EXTRACT_FILES:
    s = read_npy_memmap(f)[BURNIN::THIN]
    chains.append(s)
    print(f"  {f.split('/')[-2]}: {s.shape[0]} draws after burnin+thin")

min_len = min(c.shape[0] for c in chains)
chains  = [c[:min_len] for c in chains]
print(f"Using {min_len} draws per chain")

idata = az.from_dict({'posterior': {
    spec: np.stack([c[:, col_idx] for c in chains], axis=0)
    for spec, col_idx in cols.items()
}})

print("\nGelman-Rubin R-hat (median over ells):")
for spec in cols:
    rhat_vals = az.rhat(idata, var_names=[spec])[spec].values
    print(f"  {spec}: median={np.median(rhat_vals):.4f}  max={np.max(rhat_vals):.4f}  "
          f"frac>1.01={np.mean(rhat_vals>1.01):.3f}  frac>1.1={np.mean(rhat_vals>1.1):.3f}")

print("\nESS bulk (median over ells):")
for spec in cols:
    ess_vals = az.ess(idata, var_names=[spec], method='bulk')[spec].values
    print(f"  {spec}: median={np.median(ess_vals):.1f}  min={np.min(ess_vals):.1f}")

# Plot R-hat vs ell
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, spec in zip(axes, cols):
    rhat_vals = az.rhat(idata, var_names=[spec])[spec].values
    ax.plot(ells, rhat_vals, lw=0.8)
    ax.axhline(1.01, color='orange', ls='--', lw=1, label='1.01')
    ax.axhline(1.1,  color='red',    ls='--', lw=1, label='1.1')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('R-hat')
    ax.set_title(f'{spec}  (median={np.median(rhat_vals):.3f})')
    ax.legend(fontsize=8)
fig.suptitle('Gelman-Rubin R-hat — nmt1–nmt4', fontsize=11)
fig.tight_layout()
out = f'{BASE}/gelman_rubin_nmt.png'
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.close()
