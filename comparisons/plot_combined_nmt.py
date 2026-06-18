"""
Combined posterior from all four nmt chains vs NaMaster.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jaffe/Developer/Almanac/Almanac/Plotting/PostProcessAnalysis')
from utils.generic_utils import read_npy_memmap

BASE       = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR    = f'{BASE}/sim_tombin-1_nside128_flatEE'
NMT_FILE   = f'{SIM_DIR}/namaster_out/namaster_bandpowers.dat'
TRUTH_FILE = f'{SIM_DIR}/flatEE_input_cls.dat'
OUT_FILE   = f'{BASE}/plot_combined_nmt.png'

EXTRACT_FILES = [
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt1/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt2/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt3/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside128_flatEE_nmt4/almanac_out.extract.npy',
]
BURNIN  = 50000
LMIN, LMAX, NFIELDS = 2, 256, 2

parts   = [read_npy_memmap(f)[BURNIN:] for f in EXTRACT_FILES]
samples = np.concatenate(parts, axis=0)
print(f"Combined: {samples.shape[0]} samples from {len(parts)} chains (each after burnin={BURNIN})")

ells = np.arange(LMIN, LMAX + 1)
ncls = NFIELDS * (NFIELDS + 1) // 2
idx  = np.arange(len(ells))
col_EE = idx*ncls+0
col_EB = idx*ncls+1
col_BB = idx*ncls+2

def ci(samp_col):
    return np.median(samp_col, 0), np.percentile(samp_col, 16, 0), np.percentile(samp_col, 84, 0)

def dell(l, cl):
    return l*(l+1)/(2*np.pi)*cl

nmt      = np.loadtxt(NMT_FILE)
ell_b    = nmt[:, 0]
nmt_EE, err_EE = nmt[:,1], nmt[:,2]
nmt_EB, err_EB = nmt[:,3], nmt[:,4]
nmt_BB, err_BB = nmt[:,5], nmt[:,6]

truth    = np.loadtxt(TRUTH_FILE, delimiter=',')
ell_in   = truth[:, 0]
cl_EE_in = truth[:, 1]

mids      = np.round(0.5*(ell_b[:-1]+ell_b[1:])).astype(int)
bin_edges = np.concatenate([[LMIN], mids, [LMAX+1]])
dell_w    = ells*(ells+1)/(2*np.pi)

def bin_almanac(cols):
    Dl  = samples[:, cols] * dell_w[np.newaxis, :]
    out = np.zeros((samples.shape[0], len(ell_b)))
    for b, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        m = (ells >= lo) & (ells < hi)
        out[:, b] = Dl[:, m].mean(axis=1)
    return out

alm_EE_med, alm_EE_lo, alm_EE_hi = ci(samples[:, col_EE])
alm_EB_med, alm_EB_lo, alm_EB_hi = ci(samples[:, col_EB])
alm_BB_med, alm_BB_lo, alm_BB_hi = ci(samples[:, col_BB])

alm_EE_bin = bin_almanac(col_EE)
alm_EB_bin = bin_almanac(col_EB)
alm_BB_bin = bin_almanac(col_BB)

D_truth = 1e-4 / (2*np.pi)

fig, axes  = plt.subplots(2, 3, figsize=(15, 9))
labels     = ['EE', 'EB', 'BB']
alm_meds   = [alm_EE_med, alm_EB_med, alm_BB_med]
alm_los    = [alm_EE_lo,  alm_EB_lo,  alm_BB_lo]
alm_his    = [alm_EE_hi,  alm_EB_hi,  alm_BB_hi]
nmt_cls    = [nmt_EE,     nmt_EB,     nmt_BB]
nmt_errs   = [err_EE,     err_EB,     err_BB]
alm_bins   = [alm_EE_bin, alm_EB_bin, alm_BB_bin]

for col_i, (label, med, lo, hi, nmt_cl, nmt_err, alm_bin) in enumerate(
        zip(labels, alm_meds, alm_los, alm_his, nmt_cls, nmt_errs, alm_bins)):

    ax = axes[0, col_i]
    ax.fill_between(ells, dell(ells, lo), dell(ells, hi),
                    alpha=0.35, color='steelblue', label='Almanac 68% CI', rasterized=True)
    ax.plot(ells, dell(ells, med), color='steelblue', lw=1.0, label='Almanac median', rasterized=True)
    if label == 'EE':
        ax.plot(ell_in, dell(ell_in, cl_EE_in), 'k-', lw=1.5, label='Input truth')
    mask30 = ells > 30
    all_y  = np.concatenate([dell(ells[mask30], lo[mask30]), dell(ells[mask30], hi[mask30])])
    pad    = 0.15*(all_y.max()-all_y.min())
    ymax   = 0.0002 if label == 'EE' else all_y.max()+pad
    ax.set_ylim(all_y.min()-pad, ymax)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell^{%s}$' % label)
    ax.set_title(f'{label} — per-ell (combined)'); ax.legend(fontsize=7)

    ax = axes[1, col_i]
    bmed, blo, bhi = ci(alm_bin)
    nmt_D    = dell(ell_b, nmt_cl)
    nmt_Derr = dell(ell_b, nmt_err)
    ax.fill_between(ell_b, blo, bhi, alpha=0.35, color='steelblue', label='Almanac 68% CI')
    ax.plot(ell_b, bmed, 'o-', color='steelblue', ms=4, lw=1.2, label='Almanac median')
    ax.errorbar(ell_b+3, nmt_D, yerr=nmt_Derr, fmt='s', ms=4, color='tomato',
                capsize=3, zorder=5, label='NaMaster')
    if label == 'EE':
        ax.axhline(D_truth, color='k', lw=1.5, ls='-', label='Input truth')
    all_y2 = np.concatenate([blo, bhi, nmt_D-nmt_Derr, nmt_D+nmt_Derr])
    pad2   = 0.15*(all_y2.max()-all_y2.min())
    ax.set_ylim(all_y2.min()-pad2, all_y2.max()+pad2)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$\langle D_\ell^{%s}\rangle_{\rm bin}$' % label)
    ax.set_title(f'{label} — binned (Δℓ=32)'); ax.legend(fontsize=7)

fig.suptitle(
    f'sim_tombin-1_nside128_flatEE — nmt1–nmt4 combined ({samples.shape[0]} samples) vs NaMaster',
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
fig.savefig(OUT_FILE.replace('.png', '.svg'))
print(f"Saved: {OUT_FILE} + .svg")
plt.close()
