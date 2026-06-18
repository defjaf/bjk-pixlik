"""
BJK vs Almanac vs exact posterior comparison for NSIDE=8 flatTT fsky=1.

For each delta_ell in {1, 2, 4}:
  - Run BJK Newton-Raphson to find MAP bandpowers
  - Profile the BJK log-likelihood per band around MAP
  - Compute analytical InvGamma posterior per band
  - Project Almanac C_l samples onto bandpowers  (dof-weighted mean)
  - Plot all three on log-x axis as c*P(c)

Analytical bandpower posterior (full sky, uniform noise N_l = N):
  n_b   = sum_{l in b} (2l+1)
  S_b   = sum_{l in b} (2l+1)*S_l / n_b     (dof-weighted mean of anafast S_l)
  z = C_b + N  ~  InvGamma(alpha = n_b/2 - 1, beta = n_b/2 * (S_b + N))
  mode[C_b] = S_b
"""
import sys
import numpy as np
import healpy as hp
import scipy.stats as ss
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jaffe/Desktop/Projects/bjk-pixlik')
from pixel_likelihood import PixelLikelihood

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside8_flatTT_fsky1'
LMIN, LMAX = 2, 16
NSIDE  = 8
NPIX   = hp.nside2npix(NSIDE)
BURNIN = 5000

# Good classic chains only
CHAIN_PATHS = [
    f'{BASE}/sim_tombin-1_nside8_flatTT_fsky1_chol1_lin/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside8_flatTT_fsky1_chol3_lin/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside8_flatTT_fsky1_classic1_lin/almanac_out.extract.npy',
    f'{BASE}/sim_tombin-1_nside8_flatTT_fsky1_classic2_lin/almanac_out.extract.npy',
]
CHAIN_COLORS = ['steelblue', 'cornflowerblue', 'darkorange', 'tomato']
CHAIN_NAMES  = ['chol1', 'chol3', 'classic1', 'classic2']

data_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_data.fits', verbose=False)
ninv_map = hp.read_map(f'{SIM_DIR}/sim_out_channel_0_nInv.fits', verbose=False)
S_ell    = hp.anafast(data_map, lmax=LMAX)[LMIN:]   # ell=2..16
N_l      = (1.0 / ninv_map.mean()) * (4 * np.pi / NPIX)
ells_all = np.arange(LMIN, LMAX + 1)

# Load Almanac chains
chains = []
for path, name, color in zip(CHAIN_PATHS, CHAIN_NAMES, CHAIN_COLORS):
    s = np.load(path)[BURNIN:]
    chains.append((s, name, color))

# Band edge definitions
DELTA_ELL_CONFIGS = {
    1: np.arange(LMIN, LMAX + 2),                        # [2,3,...,17]
    2: np.array([2, 4, 6, 8, 10, 12, 14, 16, 17]),       # 8 bands, last width-1
    4: np.array([2, 6, 10, 14, 17]),                      # 4 bands
}


def band_analytical_posterior(band_ells, S_ell_arr, N):
    """InvGamma params for a multi-ell band on full sky."""
    dof    = 2 * band_ells + 1
    n_b    = dof.sum()
    S_b    = (dof * S_ell_arr).sum() / n_b   # dof-weighted S_ell
    alpha  = n_b / 2.0 - 1.0
    beta   = n_b / 2.0 * (S_b + N)
    return alpha, beta, S_b


def normalized_pdf_logx(cs, alpha, beta, N):
    """c * P(c) on log-x axis, normalised so integral over d(ln c) = 1."""
    z  = cs + N
    lp = ss.invgamma.logpdf(z, a=alpha, scale=beta)
    p  = np.exp(lp - lp.max())
    norm = np.trapezoid(p * cs, np.log(cs))
    return p / norm * cs


def profile_bjk(lik, band_idx, cl_map, n_pts=80):
    """Profile log-likelihood along band_idx, others fixed at cl_map."""
    S_b = cl_map[band_idx]
    c_lo = max(1e-12, S_b * 1e-2)
    c_hi = S_b * 50
    cs   = np.logspace(np.log10(c_lo), np.log10(c_hi), n_pts)
    lls  = np.empty(n_pts)
    cl   = cl_map.copy()
    for i, c in enumerate(cs):
        cl[band_idx] = c
        lls[i]       = lik.log_likelihood(cl)
    cl[band_idx] = cl_map[band_idx]
    # Normalise as c*P(c) on log-x axis
    lls -= lls.max()
    p    = np.exp(lls)
    norm = np.trapezoid(p * cs, np.log(cs))
    return cs, p / norm * cs


# ── Main loop over delta_ell ───────────────────────────────────────────────────
for dell, band_edges in DELTA_ELL_CONFIGS.items():
    nbands = len(band_edges) - 1
    print(f"\n{'='*60}")
    print(f"delta_ell = {dell}  ({nbands} bands)")

    lik = PixelLikelihood(
        data_fits = f'{SIM_DIR}/sim_out_channel_0_data.fits',
        ninv_fits = f'{SIM_DIR}/sim_out_channel_0_nInv.fits',
        lmin=LMIN, lmax=LMAX,
        band_edges=band_edges,
        n_T=1, n_P=0,
    )

    # Initial guess: dof-weighted S_ell per band
    cl_init = np.zeros(nbands)
    for b in range(nbands):
        mask = (ells_all >= band_edges[b]) & (ells_all < band_edges[b + 1])
        dof  = 2 * ells_all[mask] + 1
        cl_init[b] = (dof * S_ell[mask]).sum() / dof.sum()

    cl_map, sigma_map, F_map = lik.newton_raphson(cl_init, max_iter=20)

    print(f"\n{'band':>6}  {'ells':>10}  {'S_b(init)':>12}  {'C_b(MAP)':>12}  {'sigma':>10}")
    for b in range(nbands):
        lo, hi = band_edges[b], band_edges[b+1] - 1
        print(f"{b:>6}  {lo:>4}-{hi:<4}  {cl_init[b]:>12.3e}  {cl_map[b]:>12.3e}  {sigma_map[b]:>10.3e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ncols = min(nbands, 5)
    nrows = (nbands + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for b in range(nbands):
        ax = axes_flat[b]
        lo, hi = band_edges[b], band_edges[b+1] - 1
        mask = (ells_all >= band_edges[b]) & (ells_all < band_edges[b+1])
        band_ells = ells_all[mask]

        alpha, beta, S_b = band_analytical_posterior(band_ells, S_ell[mask], N_l)

        c_lo = max(1e-12, S_b * 5e-3)
        c_hi = S_b * 30
        cs   = np.logspace(np.log10(c_lo), np.log10(c_hi), 300)

        # Analytical
        cPdf = normalized_pdf_logx(cs, alpha, beta, N_l)
        ax.semilogx(cs, cPdf, 'k-', lw=2, label='Exact', zorder=4)
        ax.axvline(S_b, color='black', ls='--', lw=1, alpha=0.6)

        # BJK profile
        cs_bjk, bjk_pdf = profile_bjk(lik, b, cl_map)
        ax.semilogx(cs_bjk, bjk_pdf, 'b-', lw=1.5, alpha=0.8, label='BJK profile')

        # Almanac projected bandpower
        log_bins = np.logspace(np.log10(c_lo), np.log10(c_hi), 45)
        dlnc     = np.diff(np.log(log_bins))
        bin_ctrs = np.exp(0.5 * (np.log(log_bins[:-1]) + np.log(log_bins[1:])))
        for s, name, color in chains:
            dof  = 2 * band_ells + 1
            # dof-weighted mean of C_l samples within band
            vals = (dof[:, None] * s[:, mask].T).sum(axis=0) / dof.sum()
            counts, _ = np.histogram(vals, bins=log_bins)
            logdens   = counts / (len(vals) * dlnc)
            ax.step(bin_ctrs, logdens, where='mid',
                    color=color, alpha=0.6, lw=1.2, label=name)

        ax.set_title(f'ℓ={lo}–{hi}', fontsize=9)
        ax.set_xlabel('C_b', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)
        if b == 0:
            ax.legend(fontsize=6, loc='upper left')

    for ax in axes_flat[nbands:]:
        ax.set_visible(False)

    fig.suptitle(f'NSIDE=8 flatTT fsky=1:  Δℓ={dell}  —  BJK profile vs exact posterior vs Almanac\n'
                 f'Black=exact InvGamma; Blue=BJK profile; colour=Almanac (dof-weighted per band)',
                 fontsize=9)
    fig.tight_layout()
    out = f'{BASE}/diag_bjk_vs_almanac_dell{dell}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()

print("\nAll done.")
