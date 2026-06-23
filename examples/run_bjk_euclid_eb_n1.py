"""
Apply BJK98 pixel-space likelihood to Euclid RR2v2.1 LensMC data with EB.

Estimates EE, BB, and EB bandpowers jointly via Newton-Raphson for tomographic
bin -1 (n_P=1), NSIDE=128 maps.

Run from repo root:
    python3 examples/run_bjk_euclid_eb_n1.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path for pixel_likelihood import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOMBIN    = -1
NSIDE     = 128

# Data files in Almanac directory
ALMANAC_DIR = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs')
DATA_FITS = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_data.fits')
NINV_FITS = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_invnoise.fits')

# Output files
OUT_DIR = os.path.join(ALMANAC_DIR, 'bjk_results')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DAT   = os.path.join(OUT_DIR, f'bjk_euclid_tombin{TOMBIN}_nside{NSIDE}_eb_n1.dat')
OUT_PNG   = os.path.join(OUT_DIR, f'bjk_euclid_tombin{TOMBIN}_nside{NSIDE}_eb_n1.png')

LMIN, LMAX  = 2, 256
BAND_EDGES  = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
# ---------------------------------------------------------------------------


def main():
    print("="*70)
    print("BJK98 Pixel Likelihood: Euclid RR2 EE+BB+EB (n_P=1)")
    print("="*70)
    print(f"Data: {DATA_FITS}")
    print(f"Noise: {NINV_FITS}")
    print(f"Output: {OUT_DAT}")
    print()

    # Build likelihood with on-the-fly kernels and parallelization
    t0 = time.time()
    lik = PixelLikelihood(
        data_fits=DATA_FITS,
        ninv_fits=NINV_FITS,
        lmin=LMIN, lmax=LMAX,
        band_edges=BAND_EDGES,
        n_T=0, n_P=1,
        band_model='Cl',
        include_TB=False,
        include_EB=True,      # Include EB!
        kernel_mode='auto',   # Auto-select based on RAM
        n_threads='auto'      # Use all cores
    )
    t_init = time.time() - t0
    print(f"Initialization time: {t_init:.1f}s")
    print()

    layout = lik.layout
    ell_b  = lik.ell_bands
    nbands = len(ell_b)

    print(f"Problem size:")
    print(f"  N_d = {len(lik.d)} data points")
    print(f"  n_params = {layout.n_params} bandpower parameters")
    print(f"    {nbands} EE + {nbands} BB + {nbands} EB = {3*nbands} total")
    print(f"  Kernel mode: {lik._resolved_mode}")
    print(f"  Threads: {lik.n_threads}")
    print()

    # Initial guess: small positive values
    cl_init = np.full(layout.n_params, 1e-8)

    # Run Newton-Raphson
    print(f"Running Newton-Raphson (max_iter=20)...")
    t0 = time.time()
    cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20, tol=1e-5)
    t_newton = time.time() - t0
    print(f"Total Newton-Raphson time: {t_newton/60:.1f} min")
    print()

    # ---- Report ----
    print(f"--- ML bandpowers (Euclid RR2, tombin {TOMBIN}, NSIDE={NSIDE}) ---")
    for idx, spec, i, j, b in layout.entries():
        ell = ell_b[b]
        print(f"  {spec} band {b} (ell~{ell:.0f}):  "
              f"C_ell = {cl_ml[idx]:.4e} ± {sigma[idx]:.4e}  "
              f"(SNR = {cl_ml[idx]/sigma[idx]:.2f})")

    # ---- Save ----
    rows = [(ell_b[b], spec, i, j, cl_ml[idx], sigma[idx], cl_ml[idx]/sigma[idx])
            for idx, spec, i, j, b in layout.entries()]
    with open(OUT_DAT, 'w') as f:
        f.write('# ell_band  spec  i  j  C_l_ML  sigma_Cl_ML  SNR\n')
        for ell, spec, i, j, cl, sig, snr in rows:
            f.write(f'{ell:.2f}  {spec}  {i}  {j}  {cl:.6e}  {sig:.6e}  {snr:.4f}\n')
    print(f"\nSaved: {OUT_DAT}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'EE': 'steelblue', 'BB': 'darkorange', 'EB': 'green'}

    for ax, spec in zip(axes, ('EE', 'BB', 'EB')):
        idx_list = [idx for idx, s, *_ in layout.entries() if s == spec]
        cl_spec  = cl_ml[idx_list]
        sig_spec = sigma[idx_list]

        ax.errorbar(ell_b, cl_spec, yerr=sig_spec, fmt='o',
                    color=colors[spec], capsize=4, ms=5, lw=1.5,
                    label='BJK98 ML ± 1σ')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$C_\ell^{{{spec}}}$  [rad$^2$]')
        ax.set_title(rf'Euclid RR2 tombin {TOMBIN}, NSIDE={NSIDE} — {spec}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('BJK98 pixel-space ML bandpowers (EE+BB+EB) — real Euclid data',
                 y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close()

    # ---- Summary stats ----
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for spec in ['EE', 'BB', 'EB']:
        idx_list = [idx for idx, s, *_ in layout.entries() if s == spec]
        cl_spec = cl_ml[idx_list]
        sig_spec = sigma[idx_list]
        snr_spec = cl_spec / sig_spec

        print(f"\n{spec}:")
        print(f"  Mean C_ell: {cl_spec.mean():.4e} ± {cl_spec.std():.4e}")
        print(f"  Mean SNR: {snr_spec.mean():.2f} ± {snr_spec.std():.2f}")
        print(f"  Max SNR: {snr_spec.max():.2f} (band {np.argmax(snr_spec)})")
        if spec in ['BB', 'EB']:
            # Test for non-zero detection
            chi2 = np.sum((cl_spec / sig_spec)**2)
            print(f"  χ² vs zero: {chi2:.2f} ({nbands} bands)")
            print(f"  PTE: ~{1.0 - np.minimum(chi2/nbands/2, 0.99):.3f} (crude estimate)")


if __name__ == '__main__':
    main()
