"""
Apply BJK98 pixel-space likelihood to the flatEE Almanac simulation.
Estimates EE and BB bandpowers jointly via Newton-Raphson.

Run from the repo root:
    python3 examples/run_bjk_ee.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(HERE, 'data', 'sim_tombin-1_nside128_flatEE')
OUT_DIR = os.path.join(HERE, 'examples')

DATA_FITS  = os.path.join(SIM_DIR, 'sim_out_channel_0_data.fits')
NINV_FITS  = os.path.join(SIM_DIR, 'sim_out_channel_0_nInv.fits')
TRUTH_FILE = os.path.join(SIM_DIR, 'flatEE_input_cls.dat')
OUT_PNG    = os.path.join(OUT_DIR,  'bjk_ee_bandpowers.png')

LMIN, LMAX = 2, 256
BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
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
    n_EE = sum(1 for _, s, *_ in layout.entries() if s == 'EE')
    n_BB = sum(1 for _, s, *_ in layout.entries() if s == 'BB')

    # Initial guess: flat C_l
    cl_init = np.full(layout.n_params, 1e-6)

    print(f"\nRunning Newton-Raphson  ({layout.n_params} params: "
          f"{n_EE} EE + {n_BB} BB bands) ...")
    cl_ml, sigma, F = lik.newton_raphson(cl_init, max_iter=20, tol=1e-5)

    # ---- Report ----
    ell_b = lik.ell_bands
    print("\n--- ML bandpowers ---")
    idx = 0
    for spec in ('EE', 'BB'):
        print(f"  {spec}:")
        for b, ell in enumerate(ell_b):
            print(f"    band {b} (ell~{ell:.0f}):  C_ell = {cl_ml[idx]:.4e} ± {sigma[idx]:.4e}")
            idx += 1

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, spec, offset in zip(axes, ('EE', 'BB'), (0, len(ell_b))):
        sl = slice(offset, offset + len(ell_b))
        ax.errorbar(ell_b, cl_ml[sl], yerr=sigma[sl], fmt='o', color='steelblue',
                    capsize=4, ms=5, lw=1.5, label='BJK98 ML ± 1σ')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$C_\ell^{{{spec}}}$')
        ax.set_title(f'BJK98 — flatEE simulation, {spec}')
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved: {OUT_PNG}")
    plt.close()


if __name__ == '__main__':
    main()
