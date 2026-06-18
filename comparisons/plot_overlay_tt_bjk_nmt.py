"""
Three-way comparison: Almanac posterior vs BJK ML vs NaMaster.
All on the same sim_tombin-1_nside128_flatTT data (Case B, Δℓ=32).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt

sys.path.insert(0, '/Users/jaffe/Developer/Almanac/Almanac/Plotting/PostProcessAnalysis')
from utils.generic_utils import read_npy_memmap

BASE     = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR  = f'{BASE}/sim_tombin-1_nside128_flatTT'
BJK_DAT  = f'{BASE}/bjk_tt_bandpowers.dat'
OUT_FILE = f'{BASE}/plot_overlay_tt_bjk_nmt.png'

CHAINS = [
    (f'{BASE}/sim_tombin-1_nside128_flatTT_chol1/almanac_out2.extract.npy',    'chol1',    'steelblue'),
    (f'{BASE}/sim_tombin-1_nside128_flatTT_chol2/almanac_out2.extract.npy',    'chol2',    'cornflowerblue'),
    (f'{BASE}/sim_tombin-1_nside128_flatTT_classic1/almanac_out2.extract.npy', 'classic1', 'darkorange'),
    (f'{BASE}/sim_tombin-1_nside128_flatTT_classic2/almanac_out2.extract.npy', 'classic2', 'sandybrown'),
]
BURNIN = 10000
LMIN, LMAX = 2, 256
NSIDE = 128
LMAX_FIELD = 3 * NSIDE - 1

bin_edges = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
ells   = np.arange(LMIN, LMAX + 1)
ell_b  = 0.5 * (bin_edges[:-1] + bin_edges[1:] - 1)
dell_w = ells * (ells + 1) / (2 * np.pi)
D_TRUE = 1e-4 / (2 * np.pi)

# ---- NaMaster on actual data ----
print("Running NaMaster on actual data...")
data_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_data.fits', verbose=False)
ninv_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_nInv.fits', verbose=False)
mask = (ninv_map > 0).astype(float)
sigma_pix = 1.0 / np.sqrt(ninv_map[ninv_map > 0].mean())
N_l = sigma_pix**2 * (4 * np.pi / hp.nside2npix(NSIDE))

_ini = np.append(bin_edges[:-1], bin_edges[-1])
_end = np.append(bin_edges[1:],  LMAX_FIELD + 1)
b_nmt = nmt.NmtBin.from_edges(_ini, _end, is_Dell=True)
ell_nmt = b_nmt.get_effective_ells()

mask_apo = nmt.mask_apodization(mask, 5.0, apotype='C2')
f0  = nmt.NmtField(mask_apo, [data_map], beam=np.ones(LMAX_FIELD + 1))
wsp = nmt.NmtWorkspace.from_fields(f0, f0, b_nmt)
cl_d = wsp.decouple_cell(nmt.compute_coupled_cell(f0, f0))[0]
keep = ell_nmt < bin_edges[-1]
dl_nmt = cl_d[keep] - N_l * ell_nmt[keep] * (ell_nmt[keep] + 1) / (2 * np.pi)
ell_nmt = ell_nmt[keep]
print(f"  NaMaster done. N_l={N_l:.4e}")

# ---- BJK ML ----
bjk = np.loadtxt(BJK_DAT)
ell_bjk, dl_bjk, sig_bjk = bjk[:, 0], bjk[:, 1], bjk[:, 2]

# ---- Almanac posteriors ----
def bin_samples(samples):
    Dl  = samples * dell_w[np.newaxis, :]
    out = np.zeros((samples.shape[0], len(ell_b)))
    for b, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        m = (ells >= lo) & (ells < hi)
        out[:, b] = Dl[:, m].mean(axis=1)
    return out

def ci(x):
    return np.median(x, 0), np.percentile(x, 16, 0), np.percentile(x, 84, 0)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6))

# Almanac chains
for path, name, color in CHAINS:
    samp   = read_npy_memmap(path)[BURNIN:]
    binned = bin_samples(samp)
    med, lo, hi = ci(binned)
    ax.fill_between(ell_b, lo, hi, alpha=0.15, color=color)
    ax.plot(ell_b, med, 'o-', color=color, ms=3, lw=1.0, label=f'Almanac {name}')

# BJK
ax.errorbar(ell_bjk, dl_bjk, yerr=sig_bjk, fmt='s', color='black',
            capsize=4, ms=5, lw=1.5, zorder=5, label='BJK ML ± 1σ')

# NaMaster
ax.plot(ell_nmt, dl_nmt, '^', color='green', ms=6, zorder=4, label='NaMaster')

# Truth
ax.axhline(D_TRUE, color='red', lw=1.5, ls='--', label=f'Input truth ({D_TRUE:.2e})')
ax.axhline(0, color='k', lw=0.5, ls=':')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{TT}$')
ax.set_title('TT: Almanac (16σ CI) vs BJK ML vs NaMaster\n'
             f'sim_tombin-1_nside128_flatTT, Δℓ=32, f_sky≈0.01')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
fig.savefig(OUT_FILE.replace('.png', '.svg'))
print(f"Saved: {OUT_FILE}")
plt.close()
