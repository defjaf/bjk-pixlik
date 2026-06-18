"""
NaMaster pseudo-Cl power spectrum for NSIDE=32 flatTT fsky≈0.5 midnoise.

Uses the MASTER algorithm to deconvolve the mode-coupling matrix induced
by the Galactic cut mask, giving an approximately unbiased C_ell estimate.
Output: namaster_fsky05_midnoise_dell1.dat  (ell, Cl_NMT, sigma_NMT)
"""
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise'
LMIN, LMAX = 2, 64
NSIDE  = 32
BURNIN = 5000

data_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_data.fits', verbose=False)
ninv_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_nInv.fits', verbose=False)
mask     = (ninv_map > 0).astype(float)
fsky     = mask.mean()
N_l      = (1.0 / ninv_map[ninv_map > 0].mean()) * (4 * np.pi / hp.nside2npix(NSIDE))
C_truth  = lambda ells: 1e-4 / (ells * (ells + 1))

print(f"fsky = {fsky:.4f}   N_l = {N_l:.3e}   NSIDE = {NSIDE}")

# ── NaMaster Δℓ=1 bandpowers ──────────────────────────────────────────────────
# Define one bin per ell
ells_arr = np.arange(0, LMAX + 2)
bpws     = np.zeros(LMAX + 2, dtype=int)
for i, ell in enumerate(range(LMIN, LMAX + 1)):
    bpws[ell] = i  # bin i = ell - LMIN

b = nmt.NmtBin.from_edges(
    ells_arr[LMIN:LMAX + 1],
    ells_arr[LMIN + 1:LMAX + 2],
)
print(f"NmtBin: {b.get_n_bands()} bands, ell={b.get_effective_ells()[:3]}...{b.get_effective_ells()[-3:]}")

# Fields — data_map already has zeros outside mask (AlmaSim output),
# so tell NaMaster it's pre-masked
f0 = nmt.NmtField(mask, [data_map], lmax=LMAX, masked_on_input=True)

# NaMaster MASTER deconvolution for mode-coupling, then simple noise subtraction.
# For auto-spectra, NaMaster's cl_noise convention is confusing; use direct approach:
# PCL → MASTER deconvolve → subtract N_l.
print("Computing MASTER estimate...")
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f0, f0, b)
pcl_masked = hp.anafast(data_map * mask, lmax=LMAX)
pcl_coupled = np.array([pcl_masked])  # shape (1, nell)
cl_deconvolved_full = w.decouple_cell(pcl_coupled)[0]  # M^{-1} · PCL, shape (nbands,)
cl_signal = cl_deconvolved_full - N_l  # noise debias (both are length nbands=63)

ell_eff = b.get_effective_ells()
print(f"Done. C_ell shape: {cl_signal.shape}")
print(f"C_ell(ell=2) = {cl_signal[0]:.3e}  (truth {C_truth(np.array([2]))[0]:.3e})")

# Approximate uncertainty (on signal): Gaussian approximation
# Var(C_l) ≈ 2/(2l+1)/fsky * (C_l_signal + N_l)^2
sigma_nmt = np.array([
    np.sqrt(2.0 / ((2 * ell + 1) * fsky)) * (max(cl_signal[i], 0) + N_l)
    for i, ell in enumerate(ell_eff.astype(int))
])

# Save
out_dat = f'{BASE}/namaster_fsky05_midnoise_dell1.dat'
np.savetxt(out_dat,
           np.column_stack([ell_eff, cl_signal, sigma_nmt]),
           header='ell_eff  Cl_NMT_debiased  sigma_NMT',
           fmt='%.6e')
print(f"\nSaved: {out_dat}")

# ── Quick comparison plot ─────────────────────────────────────────────────────
CHAINS = {
    'classic_new': (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_classic_new/almanac_out.extract.npy', 'darkorange', 'new classic'),
    'chol_new':    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_chol_new/almanac_out.extract.npy',    'steelblue',  'new chol'),
    'classic_old': (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_classic_old/almanac_out.extract.npy', 'peru',       'old classic'),
}

ells = np.arange(LMIN, LMAX + 1, dtype=float)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.08)

def Dl(ell, cl):
    return ell * (ell + 1) / (2 * np.pi) * cl

ax = axes[0]
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.axvline(20, color='purple', ls=':', lw=1, alpha=0.5, label='S/N=1 (ℓ=20)')

# NaMaster
dl_nmt   = Dl(ell_eff, cl_signal)
dl_sigma = Dl(ell_eff, sigma_nmt)
ax.errorbar(ell_eff, dl_nmt, yerr=dl_sigma,
            fmt='s', ms=4, color='black', elinewidth=1.5, capsize=3,
            label='NaMaster ± 1σ', zorder=6)
ax.plot(ells, Dl(ells, C_truth(ells)), 'g:', lw=1.5, label='Truth')

offsets = np.linspace(-0.25, 0.25, len(CHAINS))
for k, (name, (path, color, label)) in enumerate(CHAINS.items()):
    try:
        s = np.load(path)[BURNIN:]
        med  = np.median(s, axis=0)
        lo68 = np.percentile(s, 16, axis=0)
        hi68 = np.percentile(s, 84, axis=0)
        x = ells + offsets[k]
        ax.errorbar(x, Dl(x, med), yerr=[Dl(x, med-lo68), Dl(x, hi68-med)],
                    fmt='o', ms=3, color=color, elinewidth=1.2, capsize=0,
                    label=label, zorder=5)
    except FileNotFoundError:
        pass

ax.set_ylabel('D_ℓ = ℓ(ℓ+1)/(2π) C_ℓ')
ax.set_title('NaMaster (black, noise-debiased) vs Almanac chains — NSIDE=32 fsky≈0.5 midnoise', fontsize=10)
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.axhline(1.0, color='black', lw=1.0, ls='--')
# Only plot ratio where NaMaster is positive
pos = cl_signal > 0
ax.fill_between(ell_eff[pos],
                (cl_signal[pos] - sigma_nmt[pos]) / cl_signal[pos],
                (cl_signal[pos] + sigma_nmt[pos]) / cl_signal[pos],
                color='gray', alpha=0.25, label='NaMaster ±1σ')
for k, (name, (path, color, label)) in enumerate(CHAINS.items()):
    try:
        s = np.load(path)[BURNIN:]
        med = np.median(s, axis=0)
        ax.plot(ells[pos] + offsets[k], med[pos] / cl_signal[pos],
                'o-', ms=3, lw=0.8, color=color, label=label)
    except FileNotFoundError:
        pass
ax.axvline(20, color='purple', ls=':', lw=1, alpha=0.5)
ax.set_ylabel('chain median / NaMaster (positive ells only)')
ax.set_xlabel('ℓ')
ax.set_ylim(0.5, 1.8)
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)
ax.set_xlim(LMIN - 0.5, LMAX + 0.5)

fig.suptitle('NSIDE=32 flatTT fsky≈0.5 midnoise: NaMaster vs Almanac', fontsize=11)
out_png = f'{BASE}/diag_namaster_fsky05_midnoise.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_png}")
plt.close()
