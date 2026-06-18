"""
Apply BJK98 pixel-space likelihood to real Euclid RR2v2.1 LensMC data.

Estimates EE and BB bandpowers jointly via Newton-Raphson for tomographic
bin -1, NSIDE=128 maps prepared by Almanac.

Run from sim_runs/bjk/:
    python3 run_bjk_euclid_ee.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(HERE, 'almanac_runs')
OUT_DIR   = DATA_DIR

TOMBIN    = -1
NSIDE     = 128

DATA_FITS = os.path.join(DATA_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_data.fits')
NINV_FITS = os.path.join(DATA_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_invnoise.fits')
OUT_PNG   = os.path.join(OUT_DIR,  f'bjk_euclid_tombin{TOMBIN}_nside{NSIDE}_ee.png')
OUT_DAT   = OUT_PNG.replace('.png', '.dat')

LMIN, LMAX  = 2, 256
BAND_EDGES  = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
# ---------------------------------------------------------------------------


def main():
    lik = PixelLikelihood(
        data_fits=DATA_FITS,
        ninv_fits=NINV_FITS,
        lmin=LMIN, lmax=LMAX,
        band_edges=BAND_EDGES,
        n_T=0, n_P=1,
        band_model='Cl',
    )

    layout = lik.layout
    ell_b  = lik.ell_bands
    nbands = len(ell_b)

    cl_init = np.full(layout.n_params, 1e-7)

    print(f"\nRunning Newton-Raphson  ({layout.n_params} params: "
          f"{nbands} EE + {nbands} BB bands) ...")
    cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20, tol=1e-5)

    # ---- Report ----
    print(f"\n--- ML bandpowers (Euclid RR2, tombin {TOMBIN}, NSIDE={NSIDE}) ---")
    for idx, spec, i, j, b in layout.entries():
        ell = ell_b[b]
        print(f"  {spec} band {b} (ell~{ell:.0f}):  "
              f"C_ell = {cl_ml[idx]:.4e} ± {sigma[idx]:.4e}")

    # ---- Save ----
    rows = [(ell_b[b], spec, cl_ml[idx], sigma[idx])
            for idx, spec, i, j, b in layout.entries()]
    with open(OUT_DAT, 'w') as f:
        f.write('# ell_band  spec  C_l_ML  sigma_Cl_ML\n')
        for ell, spec, cl, sig in rows:
            f.write(f'{ell:.2f}  {spec}  {cl:.6e}  {sig:.6e}\n')
    print(f"Saved: {OUT_DAT}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'EE': 'steelblue', 'BB': 'darkorange'}

    for ax, spec in zip(axes, ('EE', 'BB')):
        idx_list = [idx for idx, s, *_ in layout.entries() if s == spec]
        cl_spec  = cl_ml[idx_list]
        sig_spec = sigma[idx_list]

        ax.errorbar(ell_b, cl_spec, yerr=sig_spec, fmt='o',
                    color=colors[spec], capsize=4, ms=5, lw=1.5,
                    label=f'BJK98 ML ± 1σ')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$C_\ell^{{{spec}}}$  [rad$^2$]')
        ax.set_title(rf'Euclid RR2 tombin {TOMBIN}, NSIDE={NSIDE} — {spec}')
        ax.legend()

    fig.suptitle('BJK98 pixel-space ML bandpowers — real Euclid data', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close()


if __name__ == '__main__':
    main()
