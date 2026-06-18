"""
NSIDE=32 flatTT fsky=1  delta_ell=8 bandpower comparison.

For full-sky uniform noise TT the bandpower marginal posterior is exact:
  z = C_b + N  ~  InvGamma(alpha = n_b/2 - 1,  beta = n_b/2 * (S_b + N))
where n_b = sum_{ell in b} (2*ell+1)  and  S_b = dof-weighted mean S_ell.

No BJK pixel-likelihood calls needed.  The analytic curve IS the exact posterior
for the constrained flat-band model (= BJK profile = marginal for full sky).

The Almanac histogram is the empirical distribution of dof-weighted projected
bandpowers from the free-spectrum chain — a weighted sum of independent
InvGamma(ell-0.5, ...) variates, which is broader than the flat-band posterior
when Delta_ell > 1.
"""
import numpy as np
import healpy as hp
import scipy.stats as ss
import matplotlib.pyplot as plt

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1_highnoise'
LMIN, LMAX = 2, 64
NSIDE  = 32
NPIX   = hp.nside2npix(NSIDE)
BURNIN = 5000

CHAIN_PATHS = [
    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1_highnoise_chol3_lin/almanac_out.extract.npy',    'chol3',    'steelblue'),
    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1_highnoise_classic3_lin/almanac_out.extract.npy', 'classic3', 'darkorange'),
    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1_highnoise_chol4_lin/almanac_out.extract.npy',    'chol4',    'cornflowerblue'),
    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1_highnoise_classic4_lin/almanac_out.extract.npy', 'classic4', 'tomato'),
]

band_edges = np.array([2, 10, 18, 26, 34, 42, 50, 58, 65])
nbands     = len(band_edges) - 1
ells_all   = np.arange(LMIN, LMAX + 1)

data_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_data.fits', verbose=False)
ninv_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_nInv.fits', verbose=False)
S_ell    = hp.anafast(data_map, lmax=LMAX)[LMIN:]
N_l      = (1.0 / ninv_map.mean()) * (4 * np.pi / NPIX)

print(f"N_l = {N_l:.3e}")

# Load Almanac chains (use only files that exist)
chains = []
for path, name, color in CHAIN_PATHS:
    try:
        s = np.load(path)[BURNIN:]
        chains.append((s, name, color))
        print(f"Loaded {name}: {s.shape[0]} post-burnin samples")
    except FileNotFoundError:
        print(f"Skipping {name}: file not found")

print(f"\nN_l = {N_l:.3e}")
print(f"\n{'band':>5}  {'ells':>10}  {'n_b':>6}  {'S_b':>12}  {'alpha':>8}  {'beta':>12}")
for b in range(nbands):
    mask      = (ells_all >= band_edges[b]) & (ells_all < band_edges[b+1])
    band_ells = ells_all[mask]
    dof       = 2 * band_ells + 1
    n_b       = dof.sum()
    S_b       = (dof * S_ell[mask]).sum() / n_b
    alpha     = n_b / 2.0 - 1.0
    beta      = n_b / 2.0 * (S_b + N_l)
    lo, hi    = band_edges[b], band_edges[b+1] - 1
    print(f"{b:>5}  {lo:>4}-{hi:<4}   {n_b:>6}  {S_b:>12.3e}  {alpha:>8.1f}  {beta:>12.3e}")


def analytic_pdf_logx(cs, n_b, S_b, N):
    """c * P(c) normalised on log-x axis for InvGamma bandpower posterior."""
    alpha = n_b / 2.0 - 1.0
    beta  = n_b / 2.0 * (S_b + N)
    z     = cs + N
    lp    = ss.invgamma.logpdf(z, a=alpha, scale=beta)
    p     = np.exp(lp - lp.max())
    norm  = np.trapezoid(p * cs, np.log(cs))
    return p / norm * cs


fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes_flat = axes.flatten()

for b in range(nbands):
    ax   = axes_flat[b]
    lo, hi = band_edges[b], band_edges[b+1] - 1
    mask      = (ells_all >= band_edges[b]) & (ells_all < band_edges[b+1])
    band_ells = ells_all[mask]
    dof       = 2 * band_ells + 1
    n_b       = dof.sum()
    S_b       = (dof * S_ell[mask]).sum() / n_b

    c_lo = max(1e-12, S_b * 5e-3)
    c_hi = S_b * 30
    cs   = np.logspace(np.log10(c_lo), np.log10(c_hi), 300)

    # Analytic flat-band posterior (exact for full-sky, = BJK profile)
    cPdf = analytic_pdf_logx(cs, n_b, S_b, N_l)
    ax.semilogx(cs, cPdf, 'k-', lw=2, label='Exact (flat-band)', zorder=5)
    ax.axvline(S_b, color='black', ls='--', lw=1, alpha=0.5)

    # Almanac projected bandpower (free-spectrum, dof-weighted mean per band)
    log_bins = np.logspace(np.log10(c_lo), np.log10(c_hi), 40)
    dlnc     = np.diff(np.log(log_bins))
    bin_ctrs = np.exp(0.5 * (np.log(log_bins[:-1]) + np.log(log_bins[1:])))
    for s, name, color in chains:
        vals      = (dof[:, None] * s[:, mask].T).sum(axis=0) / n_b
        counts, _ = np.histogram(vals, bins=log_bins)
        logdens   = counts / (len(vals) * dlnc)
        ax.step(bin_ctrs, logdens, where='mid',
                color=color, alpha=0.7, lw=1.2, label=name)

    ax.set_title(f'ℓ={lo}–{hi}  (n_b={n_b})', fontsize=9)
    ax.set_xlabel('C_b', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(bottom=0)
    if b == 0:
        ax.legend(fontsize=7, loc='upper left')

fig.suptitle(
    'NSIDE=32 flatTT fsky=1  Δℓ=8  —  analytic InvGamma vs Almanac free-spectrum projection\n'
    'Black = exact flat-band posterior (= BJK profile for full sky);  colour = Almanac dof-weighted bandpower',
    fontsize=10)
fig.tight_layout()
out = f'{BASE}/diag_bjk_nside32_dell8.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()
