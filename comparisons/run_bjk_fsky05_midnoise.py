"""
BJK pixel-space ML bandpowers for NSIDE=32 flatTT fsky=0.5 midnoise.

Uses PixelLikelihood from bjk-pixlik with Delta_ell=1 (one band per ell).
Newton-Raphson gives the ML C_ell and Fisher-curvature sigma.

For partial sky the analytic full-sky InvGamma is no longer exact; BJK gives
a close-to-exact reference (profile likelihood = marginal for narrow bands
when likelihood is approximately Gaussian in each C_ell direction).

Memory: ~20 GB for 63 kernel matrices at NSIDE=32 fsky=0.5.
Time:   ~10-20 minutes.

Outputs:
  bjk_fsky05_midnoise_dell1.dat    -- ell, C_ell_ML, sigma_ML
  diag_bjk_fsky05_midnoise.png     -- comparison with Almanac chains
"""
import sys, os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
BJK_ROOT = '/Users/jaffe/Desktop/Projects/bjk-pixlik'
BASE     = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR  = f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise'

sys.path.insert(0, BJK_ROOT)
from pixel_likelihood import PixelLikelihood

LMIN, LMAX = 2, 64
BURNIN     = 5000
NSIDE      = 32
C_truth    = lambda ells: 1e-4 / (ells * (ells + 1))

DATA_FITS = f'{SIM_DIR}/sim_out_channel_0_data.fits'
NINV_FITS = f'{SIM_DIR}/sim_out_channel_0_nInv.fits'

# Δℓ=1 band edges: [2,3,4,...,65]
band_edges = np.arange(LMIN, LMAX + 2)
nbands     = len(band_edges) - 1
ells       = np.arange(LMIN, LMAX + 1, dtype=float)

print(f"NSIDE={NSIDE}  fsky≈0.51  lmax={LMAX}")
print(f"Bands: {nbands} (Δℓ=1, ℓ={LMIN}..{LMAX})")
print(f"Data:  {DATA_FITS}")
print()

# ── Build likelihood ──────────────────────────────────────────────────────────
print("Building PixelLikelihood (precomputing kernel matrices — ~1-5 min)...")
lik = PixelLikelihood(
    data_fits  = DATA_FITS,
    ninv_fits  = NINV_FITS,
    lmin       = LMIN,
    lmax       = LMAX,
    band_edges = band_edges,
    nfields    = 1,
    spin       = 0,
    band_model = np.ones(LMAX + 1),  # C_ell = const within band (trivial for Δℓ=1)
)
print(f"n_obs = {lik.n_obs} pixels  (fsky = {lik.n_obs / hp.nside2npix(NSIDE):.4f})")
print()

# ── Initial guess: PCL / fsky ─────────────────────────────────────────────────
data_map = hp.read_map(DATA_FITS, verbose=False)
mask     = hp.read_map(NINV_FITS, verbose=False) > 0
fsky     = mask.mean()
S_ell    = hp.anafast(data_map * mask, lmax=LMAX)[LMIN:]
cl_init  = S_ell / fsky  # PCL corrected for fsky as initial guess
print(f"PCL/fsky initial guess: C_2={cl_init[0]:.3e}  C_10={cl_init[8]:.3e}")

# ── Newton-Raphson ────────────────────────────────────────────────────────────
# Custom Newton-Raphson with backtracking line search.
# The Fisher matrix is a poor Hessian approximation for partial-sky (mode coupling),
# so fixed damping oscillates; backtracking guarantees monotone logL improvement.
def newton_raphson_linesearch(lik, cl_init, max_iter=100, tol=1e-6, max_ls=20):
    import numpy as np
    cl = np.array(cl_init, dtype=float)
    print(f"\nNewton-Raphson + backtracking line search ({len(cl)} params, max_iter={max_iter})")
    for it in range(max_iter):
        try:
            g, F = lik.gradient_and_fisher(cl)
            delta = np.linalg.solve(F, g)
        except np.linalg.LinAlgError as e:
            print(f"  iter {it+1}: stopping ({e})")
            break
        logL_cur = lik.log_likelihood(cl)
        # Backtracking: halve step until logL improves
        step = 1.0
        for _ in range(max_ls):
            cl_try = cl + step * delta
            logL_try = lik.log_likelihood(cl_try)
            if np.isfinite(logL_try) and logL_try > logL_cur:
                break
            step *= 0.5
        else:
            print(f"  iter {it+1}: line search failed, stopping")
            break
        cl = cl_try
        rel = np.max(np.abs(step * delta) / (np.abs(cl) + 1e-40))
        print(f"  iter {it+1}: logL={logL_try:.4f}  step={step:.4f}  max|step*δ/C|={rel:.2e}")
        if rel < tol:
            print("  Converged.")
            break
    # Final Fisher for uncertainties
    try:
        _, F_final = lik.gradient_and_fisher(cl)
        sigma = np.sqrt(np.maximum(np.diag(np.linalg.inv(F_final)), 0))
    except np.linalg.LinAlgError:
        sigma = np.full(len(cl), np.nan)
        F_final = None
    return cl, sigma, F_final

cl_ml, sigma_ml, F = newton_raphson_linesearch(lik, cl_init, max_iter=100, tol=1e-6)

# ── Save results ──────────────────────────────────────────────────────────────
out_dat = f'{BASE}/bjk_fsky05_midnoise_dell1.dat'
np.savetxt(out_dat,
           np.column_stack([ells, cl_ml, sigma_ml]),
           header='ell  C_ell_ML  sigma_ML',
           fmt='%.6e')
print(f"\nSaved: {out_dat}")

# ── Diagnostic plot: BJK vs Almanac chains ────────────────────────────────────
CHAINS = {
    'classic_new': (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_classic_new/almanac_out.extract.npy', 'darkorange'),
    'chol_new':    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_chol_new/almanac_out.extract.npy',    'steelblue'),
    'classic_old': (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_classic_old/almanac_out.extract.npy', 'peru'),
    'chol_old':    (f'{BASE}/sim_tombin-1_nside32_flatTT_fsky05_midnoise_chol_old/almanac_out.extract.npy',    'cornflowerblue'),
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.08)

ax = axes[0]
# BJK ML ± 1σ (Fisher)
ax.errorbar(ells, cl_ml, yerr=sigma_ml,
            fmt='s', ms=4, color='black', elinewidth=1.5, capsize=3,
            label='BJK ML ± 1σ (Fisher)', zorder=6)
ax.plot(ells, C_truth(ells), 'g:', lw=1.5, label='Truth')
ax.plot(ells, S_ell,         'k--', lw=1.0, alpha=0.4, label='PCL (masked)')

offsets = np.linspace(-0.25, 0.25, len(CHAINS))
for k, (name, (path, color)) in enumerate(CHAINS.items()):
    try:
        s = np.load(path)[BURNIN:]
        lo68 = np.percentile(s, 16,  axis=0)
        med  = np.percentile(s, 50,  axis=0)
        hi68 = np.percentile(s, 84,  axis=0)
        lo95 = np.percentile(s, 2.5, axis=0)
        hi95 = np.percentile(s, 97.5, axis=0)
        x = ells + offsets[k]
        ax.errorbar(x, med, yerr=[med-lo68, hi68-med],
                    fmt='o', ms=3, color=color, elinewidth=1.2, capsize=0,
                    label=name, zorder=5)
        ax.vlines(x, lo95, hi95, color=color, lw=0.5, alpha=0.3)
    except FileNotFoundError:
        print(f"Skipping {name} (not found)")

ax.set_ylabel('C_ℓ')
ax.set_yscale('log')
ax.set_title('BJK ML (black squares) vs Almanac chains — NSIDE=32 fsky≈0.5 midnoise  Δℓ=1', fontsize=10)
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# Ratio to BJK ML
ax = axes[1]
ax.axhline(1.0, color='black', lw=1.0, ls='--')
ax.fill_between(ells,
                (cl_ml - sigma_ml) / cl_ml,
                (cl_ml + sigma_ml) / cl_ml,
                color='gray', alpha=0.25, label='BJK ±1σ')
for k, (name, (path, color)) in enumerate(CHAINS.items()):
    try:
        s = np.load(path)[BURNIN:]
        med = np.median(s, axis=0)
        ax.plot(ells + offsets[k], med / cl_ml,
                'o-', ms=3, lw=0.8, color=color, label=name)
    except FileNotFoundError:
        pass
ax.set_ylabel('chain median / BJK ML')
ax.set_xlabel('ℓ')
ax.set_ylim(0.5, 1.8)
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)
ax.set_xlim(LMIN - 0.5, LMAX + 0.5)

fig.suptitle('NSIDE=32 flatTT fsky≈0.5 midnoise: BJK vs Almanac', fontsize=11)
out_png = f'{BASE}/diag_bjk_fsky05_midnoise.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_png}")
plt.close()
