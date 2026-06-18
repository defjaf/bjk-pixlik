"""
Diagnostic: compare nmt starting Cl (theory-based) to NaMaster bandpowers
and the fix-chain posterior medians.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jaffe/Developer/Almanac/Almanac/Plotting/PostProcessAnalysis')
from utils.generic_utils import read_npy_memmap

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside128_flatEE'
NMT_FILE   = f'{SIM_DIR}/namaster_out/namaster_bandpowers.dat'
TRUTH_FILE = f'{SIM_DIR}/flatEE_input_cls.dat'
OUT_FILE   = f'{BASE}/diagnostic_nmt_starts_fixed.png'

LMIN, LMAX, NFIELDS = 2, 256, 2
BURNIN = 50000

# NaMaster
nmt    = np.loadtxt(NMT_FILE)
ell_b  = nmt[:, 0]
nmt_EE, err_EE = nmt[:, 1], nmt[:, 2]
nmt_EB, err_EB = nmt[:, 3], nmt[:, 4]
nmt_BB, err_BB = nmt[:, 5], nmt[:, 6]

# Truth
truth    = np.loadtxt(TRUTH_FILE, delimiter=',')
ell_in   = truth[:, 0]
cl_EE_in = truth[:, 1]

# Starting Cl files for each nmt chain
chain_colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
start_cls = {}
for n in range(1, 5):
    d = np.loadtxt(f'{BASE}/sim_tombin-1_nside128_flatEE_nmt{n}/start_cl.dat', delimiter=',')
    start_cls[n] = {'ell': d[:, 0], 'EE': d[:, 1], 'EB': d[:, 2], 'BB': d[:, 3]}

# Fix-chain posterior medians (after burnin)
ells  = np.arange(LMIN, LMAX + 1)
ncls  = NFIELDS * (NFIELDS + 1) // 2
idx   = np.arange(len(ells))
col_EE = idx * ncls + 0
col_EB = idx * ncls + 1
col_BB = idx * ncls + 2

fix_meds = {}
for n in range(1, 5):
    fpath = f'{BASE}/sim_tombin-1_nside128_flatEE_fix{n}/almanac_out.extract.npy'
    try:
        s = read_npy_memmap(fpath)[BURNIN:]
        fix_meds[n] = {
            'EE': np.median(s[:, col_EE], 0),
            'EB': np.median(s[:, col_EB], 0),
            'BB': np.median(s[:, col_BB], 0),
        }
        print(f"fix{n}: {s.shape[0]} samples after burnin")
    except Exception as e:
        print(f"fix{n}: could not load ({e})")

def dell(l, cl):
    return l * (l + 1) / (2 * np.pi) * cl

labels_spec = ['EE', 'EB', 'BB']
nmt_cls_list = [nmt_EE, nmt_EB, nmt_BB]
nmt_err_list = [err_EE, err_EB, err_BB]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, label, nmt_cl, nmt_err in zip(axes, labels_spec, nmt_cls_list, nmt_err_list):
    # NaMaster
    nmt_D    = dell(ell_b, nmt_cl)
    nmt_Derr = dell(ell_b, nmt_err)
    ax.errorbar(ell_b + 2, nmt_D, yerr=nmt_Derr, fmt='s', ms=4, color='tomato',
                capsize=3, zorder=5, label='NaMaster', alpha=0.8)

    # Input truth for EE
    if label == 'EE':
        ax.plot(ell_in, dell(ell_in, cl_EE_in), 'k-', lw=1.5, label='Input truth', zorder=10)

    # Fix-chain medians
    if fix_meds:
        for n, color in zip(range(1, 5), chain_colors):
            if n in fix_meds:
                ax.plot(ells, dell(ells, fix_meds[n][label]), color=color,
                        lw=0.7, alpha=0.5, label=f'fix{n} median' if n == 1 else None)

    # nmt starting points
    for n, color in zip(range(1, 5), chain_colors):
        d = start_cls[n]
        ls = start_cls[n]
        ax.plot(d['ell'], dell(d['ell'], d[label]), color=color,
                lw=1.2, ls='--', label=f'nmt{n} start')

    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{%s}$' % label)
    ax.set_title(f'{label}  — starting points (dashed) vs NaMaster vs fix medians')
    ax.legend(fontsize=6, ncol=2)

fig.suptitle('nmt starting Cl (theory-based, fixed) vs NaMaster & fix-chain medians', fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
print(f"Saved: {OUT_FILE}")
plt.close()
