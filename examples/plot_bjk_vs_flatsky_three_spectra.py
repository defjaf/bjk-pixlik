"""
Three-spectra comparison: BJK vs flat-sky NaMaster (EE, BB, EB).

BJK results from eb_n1.dat are already in D_ℓ (band_model='Dl').
Flat-sky NaMaster computed on the same RR2 data, matched resolution.
"""

import numpy as np
import healpy as hp
from healpy.projector import GnomonicProj
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import os

TOMBIN = -1
NSIDE = 128  # match BJK's resolution (apples-to-apples, matched-res flat-sky)
ALMANAC_DIR = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs')
OUT_DIR = os.path.join(ALMANAC_DIR, 'bjk_results')

# BJK used these band edges (centers 17.5, 49.5, ... 241)
BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])

RESO = np.degrees(hp.nside2resol(NSIDE)) * 60  # ~27.5'
XSIZE = 100
APO_SCALE = 60.0


def load_bjk():
    """Load BJK eb_n1.dat — columns are D_ℓ (band_model='Dl')."""
    f = os.path.join(OUT_DIR, 'bjk_euclid_tombin-1_nside128_eb_n1.dat')
    data = np.genfromtxt(f, comments='#', dtype=None, encoding=None)
    ell, spec, dl, sig = data['f0'], data['f1'], data['f4'], data['f5']

    out = {}
    for s in ('EE', 'BB', 'EB'):
        m = spec == s
        out[s] = {'ell': ell[m], 'D': dl[m], 'sigma': sig[m]}
    return out


def project(map_hp, nside, ra0, dec0, xsize, reso):
    proj = GnomonicProj(rot=[ra0, dec0, 0], xsize=xsize, reso=reso)
    img = proj.projmap(map_hp, vec2pix_func=lambda x, y, z: hp.vec2pix(nside, x, y, z))
    valid = np.isfinite(img) & (img != hp.UNSEEN)
    return np.where(valid, img, 0.0), valid.astype(float)


N_NOISE_SIM = 30  # noise realizations for bias estimation


def run_flatsky():
    """Flat-sky NaMaster on RR2 data with noise debiasing; return D_ℓ for EE, BB, EB."""
    DATA = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_data.fits')
    NINV = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_invnoise.fits')

    Q_hp, U_hp = hp.read_map(DATA, field=[0, 1], verbose=False)
    ninv_Q, ninv_U = hp.read_map(NINV, field=[0, 1], verbose=False)
    mask_hp = (ninv_Q > 0).astype(float)

    obs = np.where(mask_hp)[0]
    th, ph = hp.pix2ang(NSIDE, obs)
    dec, ra = np.pi/2 - th, ph
    rx, ry = np.cos(ra).mean(), np.sin(ra).mean()
    ra0 = np.arctan2(ry, rx)
    if ra0 < 0: ra0 += 2*np.pi
    dec0 = dec.mean()
    ra0d, dec0d = np.degrees(ra0), np.degrees(dec0)

    # Per-pixel noise std in observed pixels
    sigma_Q = np.zeros(len(mask_hp)); sigma_U = np.zeros(len(mask_hp))
    sigma_Q[obs] = 1.0 / np.sqrt(ninv_Q[obs])
    sigma_U[obs] = 1.0 / np.sqrt(ninv_U[obs])

    # --- Project data ---
    Q_in = np.where(mask_hp > 0, Q_hp, hp.UNSEEN)
    U_in = np.where(mask_hp > 0, U_hp, hp.UNSEEN)
    Q, mask = project(Q_in, NSIDE, ra0d, dec0d, XSIZE, RESO)
    U, _ = project(U_in, NSIDE, ra0d, dec0d, XSIZE, RESO)

    reso_rad = np.radians(RESO / 60.0)
    Lx = Ly = XSIZE * reso_rad
    dist = distance_transform_edt(mask)
    mask_apo = np.clip(dist / (APO_SCALE / RESO), 0, 1)

    b = nmt.NmtBinFlat(BAND_EDGES[:-1].astype(float), (BAND_EDGES[1:] - 1).astype(float))
    ell_eff = b.get_effective_ells()

    def coupled(Qm, Um):
        f = nmt.NmtFieldFlat(Lx, Ly, mask_apo, [Qm, Um], purify_b=True)
        return nmt.compute_coupled_cell_flat(f, f, b)

    wsp = nmt.NmtWorkspaceFlat()
    f2 = nmt.NmtFieldFlat(Lx, Ly, mask_apo, [Q, U], purify_b=True)
    wsp.compute_coupling_matrix(f2, f2, b)

    cl_data_coupled = coupled(Q, U)

    # --- Noise bias from simulations ---
    print(f"  Estimating noise bias from {N_NOISE_SIM} realizations...")
    rng = np.random.default_rng(1234)
    nl_coupled = np.zeros_like(cl_data_coupled)
    for k in range(N_NOISE_SIM):
        nQ_hp = np.full(len(mask_hp), hp.UNSEEN)
        nU_hp = np.full(len(mask_hp), hp.UNSEEN)
        nQ_hp[obs] = rng.standard_normal(len(obs)) * sigma_Q[obs]
        nU_hp[obs] = rng.standard_normal(len(obs)) * sigma_U[obs]
        nQ, _ = project(nQ_hp, NSIDE, ra0d, dec0d, XSIZE, RESO)
        nU, _ = project(nU_hp, NSIDE, ra0d, dec0d, XSIZE, RESO)
        nl_coupled += coupled(nQ, nU)
    nl_coupled /= N_NOISE_SIM

    # Debias then decouple
    cl_raw = wsp.decouple_cell(cl_data_coupled)
    cl_sig = wsp.decouple_cell(cl_data_coupled - nl_coupled)
    nl_dec = wsp.decouple_cell(nl_coupled)

    fac = ell_eff * (ell_eff + 1) / (2 * np.pi)
    # cl order for spin2×spin2: [EE, EB, BE, BB]
    return {
        'ell': ell_eff,
        'EE': cl_sig[0] * fac, 'BB': cl_sig[3] * fac, 'EB': cl_sig[1] * fac,
        'EE_raw': cl_raw[0] * fac, 'BB_raw': cl_raw[3] * fac,
        'EE_noise': nl_dec[0] * fac, 'BB_noise': nl_dec[3] * fac,
    }


def main():
    print("Loading BJK (eb_n1.dat, D_ℓ)...")
    bjk = load_bjk()
    print("Running flat-sky NaMaster...")
    nmt_res = run_flatsky()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    specs = ['EE', 'BB', 'EB']
    for ax, spec in zip(axes, specs):
        be = bjk[spec]
        ax.errorbar(be['ell'], be['D'], yerr=be['sigma'], fmt='o-', ms=8, lw=2,
                    capsize=5, label='BJK (HEALPix ML)', color='C0', zorder=3)
        ax.plot(nmt_res['ell'], nmt_res[spec], 's-', ms=8, lw=2,
                label='Flat-sky NaMaster (debiased)', color='C1', zorder=2)
        # Show noise floor and raw (pre-debias) for EE/BB
        if spec in ('EE', 'BB'):
            ax.plot(nmt_res['ell'], nmt_res[f'{spec}_noise'], ':', lw=1.5,
                    color='gray', label='noise bias $N_\\ell$', zorder=1)
            ax.plot(nmt_res['ell'], nmt_res[f'{spec}_raw'], '^--', ms=5, lw=1,
                    color='C1', alpha=0.4, label='flat-sky (raw)', zorder=1)
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$D_\ell^{{{spec}}}$')
        ax.set_title(f'{spec}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if spec == 'EE':
            ax.set_yscale('log')
            ax.set_ylim(1e-6, 5e-5)

    fig.suptitle('Euclid RR2 tombin-1: BJK vs Flat-sky NaMaster (EE, BB, EB)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    outfile = os.path.join(OUT_DIR, 'bjk_vs_flatsky_three_spectra.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")

    # Table
    print("\n" + "="*70)
    print("D_ℓ comparison (×10⁶)")
    print("="*70)
    for spec in specs:
        print(f"\n{spec}:")
        print(f"{'ℓ':<8} {'BJK':<12} {'Flat-sky':<12} {'ratio'}")
        be = bjk[spec]
        for i in range(len(be['ell'])):
            nv = nmt_res[spec][i]
            r = nv/be['D'][i] if be['D'][i] != 0 else 0
            print(f"{be['ell'][i]:<8.1f} {be['D'][i]*1e6:<12.3f} {nv*1e6:<12.3f} {r:.2f}")


if __name__ == '__main__':
    main()
