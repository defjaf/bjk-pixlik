"""
All four nmt chains overlaid on a single binned C_l plot vs NaMaster.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jaffe/Developer/Almanac/Almanac/Plotting/PostProcessAnalysis')
from utils.generic_utils import read_npy_memmap

BASE     = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR  = f'{BASE}/sim_tombin-1_nside128_flatEE'
NMT_FILE = f'{SIM_DIR}/namaster_out/namaster_bandpowers.dat'
OUT_FILE = f'{BASE}/plot_overlay_nmt.png'

CHAINS = [
    (f'{BASE}/sim_tombin-1_nside128_flatEE_nmt1/almanac_out.extract.npy', 'nmt1'),
    (f'{BASE}/sim_tombin-1_nside128_flatEE_nmt2/almanac_out.extract.npy', 'nmt2'),
    (f'{BASE}/sim_tombin-1_nside128_flatEE_nmt3/almanac_out.extract.npy', 'nmt3'),
    (f'{BASE}/sim_tombin-1_nside128_flatEE_nmt4/almanac_out.extract.npy', 'nmt4'),
]
BURNIN  = 50000
LMIN, LMAX, NFIELDS = 2, 256, 2

nmt      = np.loadtxt(NMT_FILE)
ell_b    = nmt[:, 0]
nmt_cls  = [nmt[:,1], nmt[:,3], nmt[:,5]]
nmt_errs = [nmt[:,2], nmt[:,4], nmt[:,6]]

ells  = np.arange(LMIN, LMAX + 1)
ncls  = NFIELDS * (NFIELDS + 1) // 2
idx   = np.arange(len(ells))
cols  = [idx*ncls+0, idx*ncls+1, idx*ncls+2]

mids      = np.round(0.5*(ell_b[:-1]+ell_b[1:])).astype(int)
bin_edges = np.concatenate([[LMIN], mids, [LMAX+1]])
dell_w    = ells*(ells+1)/(2*np.pi)

def bin_samples(samples, col_idx):
    Dl = samples[:, col_idx] * dell_w[np.newaxis, :]
    out = np.zeros((samples.shape[0], len(ell_b)))
    for b, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        m = (ells >= lo) & (ells < hi)
        out[:, b] = Dl[:, m].mean(axis=1)
    return out

def ci(x):
    return np.median(x, 0), np.percentile(x, 16, 0), np.percentile(x, 84, 0)

def dell(l, cl):
    return l*(l+1)/(2*np.pi)*cl

chain_colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
labels_spec  = ['EE', 'EB', 'BB']
D_truth = 1e-4 / (2*np.pi)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, label, col_idx, nmt_cl, nmt_err in zip(axes, labels_spec, cols, nmt_cls, nmt_errs):
    nmt_D    = dell(ell_b, nmt_cl)
    nmt_Derr = dell(ell_b, nmt_err)

    for (path, name), color in zip(CHAINS, chain_colors):
        samp   = read_npy_memmap(path)[BURNIN:]
        binned = bin_samples(samp, col_idx)
        med, lo, hi = ci(binned)
        ax.fill_between(ell_b, lo, hi, alpha=0.20, color=color)
        ax.plot(ell_b, med, 'o-', color=color, ms=3, lw=1.0, label=name)

    ax.errorbar(ell_b + 3, nmt_D, yerr=nmt_Derr, fmt='s', ms=4, color='tomato',
                capsize=3, zorder=5, label='NaMaster')
    if label == 'EE':
        ax.axhline(D_truth, color='k', lw=1.5, ls='-', label='Input truth')

    all_y = np.concatenate([nmt_D - nmt_Derr, nmt_D + nmt_Derr])
    pad = 0.15*(all_y.max() - all_y.min()) if all_y.max() != all_y.min() else 1e-7
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\langle D_\ell^{%s}\rangle_{\rm bin}$' % label)
    ax.set_title(f'{label}  (Δℓ=32 bins)')
    ax.legend(fontsize=7)

fig.suptitle('sim_tombin-1_nside128_flatEE — nmt1–nmt4 overlaid vs NaMaster', fontsize=10)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
fig.savefig(OUT_FILE.replace('.png', '.svg'))
print(f"Saved: {OUT_FILE} + .svg")
plt.close()
