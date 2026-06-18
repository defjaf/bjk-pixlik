"""
Apply BJK98 pixel-space likelihood to the flatTT Almanac simulation.
Finds ML bandpowers via Newton-Raphson and compares to input truth.

Run from the repo root:
    python3 examples/run_bjk_tt.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(HERE, 'data', 'sim_tombin-1_nside32_flatTT_fsky05_midnoise')
OUT_DIR = os.path.join(HERE, 'examples')

DATA_FITS  = os.path.join(SIM_DIR, 'sim_out_channel_0_data.fits')
NINV_FITS  = os.path.join(SIM_DIR, 'sim_out_channel_0_nInv.fits')
TRUTH_FILE = os.path.join(SIM_DIR, 'flatTT_input_cls.dat')
OUT_PNG    = os.path.join(OUT_DIR,  'bjk_tt_bandpowers.png')

LMIN, LMAX = 2, 256

# Same Delta-ell=32 binning used by NaMaster in the EE experiment
BAND_EDGES = np.arange(2, 66, 2)
# ---------------------------------------------------------------------------


def main():
    # ---- Build likelihood (D_l=const bands: parameter is D_b directly) ----
    lik = PixelLikelihood(
        data_fits=DATA_FITS,
        ninv_fits=NINV_FITS,
        lmin=LMIN, lmax=LMAX,
        band_edges=BAND_EDGES = np.arange(2, 66, 2)
        nfields=1, spin=0,
        band_model='Dl',
    )

    # ---- Initial guess: flat D_l = A/(2pi) ----
    ell_b  = lik.ell_bands
    D_FLAT = 1e-4 / (2.0 * np.pi)
    dl_init = np.full(len(ell_b), D_FLAT)
    print(f"\nInitial D_b (truth): {dl_init}")

    # ---- Newton-Raphson (parameter is D_b) ----
    dl_ml, sigma, F = lik.newton_raphson(dl_init, max_iter=20, tol=1e-5)

    # ---- Report ----
    print("\n--- ML bandpowers ---")
    for b, (ell, db, sb) in enumerate(zip(ell_b, dl_ml, sigma)):
        print(f"  band {b} (ell~{ell:.0f}):  D_ell = {db:.4e} ± {sb:.4e}  "
              f"(truth {D_FLAT:.4e})")

    # ---- Plot ----
    truth = np.loadtxt(TRUTH_FILE, delimiter=',')
    ell_in, cl_in = truth[:, 0], truth[:, 1]

    def D(l, cl):
        return l*(l+1)/(2*np.pi) * cl

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(ell_b, dl_ml, yerr=sigma, fmt='o', color='steelblue',
                capsize=4, ms=5, lw=1.5, label='BJK98 ML ± 1σ (Fisher)')
    ax.plot(ell_in, D(ell_in, cl_in), 'k-', lw=1.5, label='Input truth')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{TT} = \ell(\ell+1)C_\ell^{TT} / 2\pi$')
    ax.set_title('BJK98 pixel-space ML — flatTT simulation (D_l=const bands)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PNG.replace('.png', '.svg'))
    print(f"\nSaved: {OUT_PNG}")
    plt.close()

    # ---- Save ML bandpowers ----
    out_dat = OUT_PNG.replace('.png', '.dat')
    np.savetxt(out_dat,
               np.column_stack([ell_b, dl_ml, sigma]),
               header='ell_band  D_l_ML  sigma_Dl_ML',
               fmt='%.6e')
    print(f"Saved: {out_dat}")


if __name__ == '__main__':
    main()
