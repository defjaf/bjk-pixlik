"""
Apply BJK98 pixel-space likelihood to Euclid TR1 data with EB, RAW shear
(corrected_shear=False -- no additive/multiplicative shear calibration).

Estimates EE, BB, and EB bandpowers jointly via Newton-Raphson for the TR1
southern-patch shear map (dec<0, tom_bin_id>0, raw she_lensmc_e1/e2), NSIDE=128,
Delta_ell=10 -- matches the Almanac TR1_nside128_dl10_raw_v{1,2,3,4} runs
(see Euclid-Almanac/TR1/runs/RUN_LOG.md).

Run from repo root:
    python3 examples/run_bjk_TR1_dl10_raw.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
# Configuration -- must match Euclid-Almanac/TR1/runs/RUN_LOG.md
# ---------------------------------------------------------------------------
NSIDE = 128

TR1_DIR   = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/TR1')
DATA_FITS = os.path.join(TR1_DIR, 'maps', f'TR1_shear_raw_nside{NSIDE}.fits')
NINV_FITS = os.path.join(TR1_DIR, 'maps', f'TR1_weights_raw_nside{NSIDE}.fits')

OUT_DIR = os.path.join(TR1_DIR, 'runs', 'bjk_results')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DAT = os.path.join(OUT_DIR, f'bjk_TR1_nside{NSIDE}_dl10_raw_eb.dat')
OUT_PNG = os.path.join(OUT_DIR, f'bjk_TR1_nside{NSIDE}_dl10_raw_eb.png')

LMIN, LMAX = 2, 256
BAND_EDGES = np.array([2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122,
                        132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232,
                        242, 252, 257])
# ---------------------------------------------------------------------------


def main():
    print("="*70)
    print("BJK98 Pixel Likelihood: Euclid TR1 EE+BB+EB, Delta_ell=10, RAW shear (n_P=1)")
    print("="*70)
    print(f"Data: {DATA_FITS}")
    print(f"Noise: {NINV_FITS}")
    print(f"Output: {OUT_DAT}")
    print()

    t0 = time.time()
    lik = PixelLikelihood(
        data_fits=DATA_FITS,
        ninv_fits=NINV_FITS,
        lmin=LMIN, lmax=LMAX,
        band_edges=BAND_EDGES,
        n_T=0, n_P=1,
        band_model='Dl',
        include_TB=False,
        include_EB=True,
        kernel_mode='auto',
        n_threads='auto'
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

    cl_init = np.full(layout.n_params, 1e-6)

    print(f"Running Newton-Raphson (max_iter=20)...")
    t0 = time.time()
    cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20, tol=1e-5)
    t_newton = time.time() - t0
    print(f"Total Newton-Raphson time: {t_newton/60:.1f} min")
    print()

    print(f"--- ML bandpowers (Euclid TR1, NSIDE={NSIDE}, Delta_ell=10, RAW shear) ---")
    for idx, spec, i, j, b in layout.entries():
        ell = ell_b[b]
        print(f"  {spec} band {b} (ell~{ell:.0f}):  "
              f"C_ell = {cl_ml[idx]:.4e} +/- {sigma[idx]:.4e}  "
              f"(SNR = {cl_ml[idx]/sigma[idx]:.2f})")

    rows = [(ell_b[b], spec, i, j, cl_ml[idx], sigma[idx], cl_ml[idx]/sigma[idx])
            for idx, spec, i, j, b in layout.entries()]
    with open(OUT_DAT, 'w') as f:
        f.write('# ell_band  spec  i  j  C_l_ML  sigma_Cl_ML  SNR\n')
        for ell, spec, i, j, cl, sig, snr in rows:
            f.write(f'{ell:.2f}  {spec}  {i}  {j}  {cl:.6e}  {sig:.6e}  {snr:.4f}\n')
    print(f"\nSaved: {OUT_DAT}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'EE': 'steelblue', 'BB': 'darkorange', 'EB': 'green'}

    for ax, spec in zip(axes, ('EE', 'BB', 'EB')):
        idx_list = [idx for idx, s, *_ in layout.entries() if s == spec]
        cl_spec  = cl_ml[idx_list]
        sig_spec = sigma[idx_list]

        ax.errorbar(ell_b, cl_spec, yerr=sig_spec, fmt='o',
                    color=colors[spec], capsize=4, ms=5, lw=1.5,
                    label='BJK98 ML +/- 1sigma')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$C_\ell^{{{spec}}}$  [rad$^2$]')
        ax.set_title(rf'Euclid TR1 (raw shear), NSIDE={NSIDE} -- {spec}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('BJK98 pixel-space ML bandpowers (EE+BB+EB) -- Euclid TR1 RAW shear',
                 y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close()

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for spec in ['EE', 'BB', 'EB']:
        idx_list = [idx for idx, s, *_ in layout.entries() if s == spec]
        cl_spec = cl_ml[idx_list]
        sig_spec = sigma[idx_list]
        snr_spec = cl_spec / sig_spec

        print(f"\n{spec}:")
        print(f"  Mean C_ell: {cl_spec.mean():.4e} +/- {cl_spec.std():.4e}")
        print(f"  Mean SNR: {snr_spec.mean():.2f} +/- {snr_spec.std():.2f}")
        print(f"  Max SNR: {snr_spec.max():.2f} (band {np.argmax(snr_spec)})")
        if spec in ['BB', 'EB']:
            chi2 = np.sum((cl_spec / sig_spec)**2)
            print(f"  chi^2 vs zero: {chi2:.2f} ({nbands} bands)")
            print(f"  PTE: ~{1.0 - np.minimum(chi2/nbands/2, 0.99):.3f} (crude estimate)")


if __name__ == '__main__':
    main()
