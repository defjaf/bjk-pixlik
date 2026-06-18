"""
BJK98 pixel-space ML bandpowers for the full-sky flatTT sim (NSIDE=32, lmax=64).

At full sky the BJK ML estimate should equal hp.anafast exactly (to within
Newton-Raphson convergence), providing a clean code-level validation.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bjk'))
from pixel_likelihood import PixelLikelihood

BASE    = '/Users/jaffe/Desktop/Projects/Almanac/Euclid-Almanac/sim_runs'
SIM_DIR = f'{BASE}/sim_tombin-1_nside32_flatTT_fsky1'

DATA_FITS  = f'{SIM_DIR}/sim_out_channel_0_data.fits'
NINV_FITS  = f'{SIM_DIR}/sim_out_channel_0_nInv.fits'
TRUTH_FILE = f'{SIM_DIR}/flatTT_input_cls.dat'
ANAFAST_FILE = f'{BASE}/anafast_tt_fsky1_bandpowers.dat'
OUT_DAT    = f'{BASE}/bjk_tt_fsky1_bandpowers.dat'
OUT_PNG    = f'{BASE}/bjk_tt_fsky1_bandpowers.png'

LMIN, LMAX   = 2, 64
BAND_EDGES   = np.array([2, 10, 18, 26, 34, 42, 50, 58, 65])


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

    # Compare to anafast reference
    if os.path.exists(ANAFAST_FILE):
        af = np.loadtxt(ANAFAST_FILE)
        print("\n--- BJK vs anafast residuals (should be ~0) ---")
        for b in range(len(ell_b)):
            diff = dl_ml[b] - af[b, 1]
            print(f"  band {b} ell~{ell_b[b]:.0f}:  BJK-anafast = {diff:.3e}  "
                  f"({abs(diff)/sigma[b]:.2f} sigma)")

    truth = np.loadtxt(TRUTH_FILE, delimiter=',')
    ell_in, cl_in = truth[:, 0], truth[:, 1]
    D_in = ell_in * (ell_in + 1) / (2 * np.pi) * cl_in

    fig, ax = plt.subplots(figsize=(8, 5))
    if os.path.exists(ANAFAST_FILE):
        af = np.loadtxt(ANAFAST_FILE)
        ax.errorbar(af[:, 0], af[:, 1], yerr=af[:, 2], fmt='gs', ms=5, capsize=4,
                    lw=1.2, label='anafast ± 1σ', zorder=5)
    ax.errorbar(ell_b, dl_ml, yerr=sigma, fmt='o', color='steelblue',
                capsize=4, ms=5, lw=1.5, label='BJK98 ML ± 1σ', zorder=6)
    ax.plot(ell_in[ell_in <= LMAX], D_in[ell_in <= LMAX], 'k-', lw=1.5, label='Input truth')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{TT}$')
    ax.set_title(r'BJK98 vs anafast — full sky flatTT (NSIDE=32, $\Delta\ell$=8)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}")
    plt.close()


if __name__ == '__main__':
    main()
