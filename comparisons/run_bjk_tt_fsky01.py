"""
BJK98 pixel-space ML bandpowers for the f_sky=0.1 flatTT sim (NSIDE=64, lmax=128).
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bjk'))
from pixel_likelihood import PixelLikelihood

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside64_flatTT_fsky01'

DATA_FITS  = f'{SIM_DIR}/sim_out_channel_0_data.fits'
NINV_FITS  = f'{SIM_DIR}/sim_out_channel_0_nInv.fits'
TRUTH_FILE = f'{SIM_DIR}/flatTT_input_cls.dat'
OUT_DAT    = f'{BASE}/bjk_tt_fsky01_bandpowers.dat'
OUT_PNG    = f'{BASE}/bjk_tt_fsky01_bandpowers.png'

LMIN, LMAX   = 2, 128
BAND_EDGES   = np.array([2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 129])


def main():
    lik = PixelLikelihood(
        data_fits=DATA_FITS, ninv_fits=NINV_FITS,
        lmin=LMIN, lmax=LMAX,
        band_edges=BAND_EDGES,
        nfields=1, spin=0,
        band_model='Dl',
    )

    ell_b  = lik.ell_bands
    D_FLAT = 1e-4 / (2.0 * np.pi)
    dl_init = np.full(len(ell_b), D_FLAT)
    print(f"N_obs = {lik.n_obs}  N_bands = {len(ell_b)}")

    dl_ml, sigma, F = lik.newton_raphson(dl_init, max_iter=20, tol=1e-5)

    print("\n--- ML bandpowers ---")
    for b, (ell, db, sb) in enumerate(zip(ell_b, dl_ml, sigma)):
        print(f"  band {b} (ell~{ell:.0f}):  D_l = {db:.4e} ± {sb:.4e}  (truth {D_FLAT:.4e})")

    np.savetxt(OUT_DAT, np.column_stack([ell_b, dl_ml, sigma]),
               header='ell_band  Dl_ML  sigma_Dl', fmt='%.6e')
    print(f"\nSaved: {OUT_DAT}")

    truth = np.loadtxt(TRUTH_FILE, delimiter=',')
    ell_in, cl_in = truth[:, 0], truth[:, 1]
    D_in = ell_in * (ell_in + 1) / (2 * np.pi) * cl_in

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ell_b, dl_ml, yerr=sigma, fmt='o', color='steelblue',
                capsize=4, ms=5, lw=1.5, label='BJK98 ML ± 1σ')
    ax.plot(ell_in[ell_in <= LMAX], D_in[ell_in <= LMAX], 'k-', lw=1.5, label='Input truth')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{TT}$')
    ax.set_title(rf'BJK98 ML — flatTT f_sky≈0.1 (NSIDE=64, $\Delta\ell$=16)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}")
    plt.close()


if __name__ == '__main__':
    main()
