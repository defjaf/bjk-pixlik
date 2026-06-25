"""
Steelman flat-sky NaMaster vs BJK: give NaMaster every legitimate advantage.

Corrections applied to the flat-sky estimator:
  1. No B-purification (it is unstable/injects power on this ragged patch).
  2. Noise-bias subtraction from noise-only sims (drawn from 1/Ninv, same pipeline).
  3. Transfer-function correction from signal-only sims:
       T_EE = <D_rec_EE> / D_in_EE   (pure-EE input)
       T_BB = <D_rec_BB> / D_in_BB   (pure-BB input)
     D_corrected = (D_data - N_ell) / T
  4. E->B leakage subtraction: spurious BB per unit EE, measured from pure-EE sims,
     scaled by the (corrected) measured EE.

Whatever gap remains after all this is a genuine limit of the flat-sky method.
"""

import numpy as np
import healpy as hp
from healpy.projector import GnomonicProj
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import os

TOMBIN = -1
NSIDE = 128
ALMANAC_DIR = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs')
OUT_DIR = os.path.join(ALMANAC_DIR, 'bjk_results')

BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
RESO = np.degrees(hp.nside2resol(NSIDE)) * 60
XSIZE = 100
APO_SCALE = 60.0
LMAX = 256
PURIFY_B = False
NSIM = 40

D_EE_IN = 5e-6
D_BB_IN = 5e-6  # for transfer function (use sizeable input so recovered is clean)


def project(map_hp, ra0, dec0):
    proj = GnomonicProj(rot=[ra0, dec0, 0], xsize=XSIZE, reso=RESO)
    img = proj.projmap(map_hp, vec2pix_func=lambda x, y, z: hp.vec2pix(NSIDE, x, y, z))
    valid = np.isfinite(img) & (img != hp.UNSEEN)
    return np.where(valid, img, 0.0)


def load_bjk():
    f = os.path.join(OUT_DIR, 'bjk_euclid_tombin-1_nside128_eb_n1.dat')
    data = np.genfromtxt(f, comments='#', dtype=None, encoding=None)
    ell, spec, dl, sig = data['f0'], data['f1'], data['f4'], data['f5']
    out = {}
    for s in ('EE', 'BB', 'EB'):
        m = spec == s
        out[s] = {'ell': ell[m], 'D': dl[m], 'sigma': sig[m]}
    return out


def main():
    NINV = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_invnoise.fits')
    DATA = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_data.fits')
    ninv_Q, ninv_U = hp.read_map(NINV, field=[0, 1], verbose=False)
    Q_dat, U_dat = hp.read_map(DATA, field=[0, 1], verbose=False)
    mask_hp = (ninv_Q > 0).astype(float)
    obs = np.where(mask_hp)[0]

    th, ph = hp.pix2ang(NSIDE, obs)
    dec, ra = np.pi/2 - th, ph
    rx, ry = np.cos(ra).mean(), np.sin(ra).mean()
    ra0 = np.arctan2(ry, rx); ra0 += 2*np.pi if ra0 < 0 else 0
    ra0d, dec0d = np.degrees(ra0), np.degrees(dec.mean())

    # Grid + apodized mask
    mimg = project(np.where(mask_hp > 0, 1.0, hp.UNSEEN), ra0d, dec0d)
    mask = (mimg > 0).astype(float)
    reso_rad = np.radians(RESO / 60.0)
    Lx = Ly = XSIZE * reso_rad
    mask_apo = np.clip(distance_transform_edt(mask) / (APO_SCALE / RESO), 0, 1)

    b = nmt.NmtBinFlat(BAND_EDGES[:-1].astype(float), (BAND_EDGES[1:] - 1).astype(float))
    ell_eff = b.get_effective_ells()
    fac = ell_eff * (ell_eff + 1) / (2 * np.pi)
    wsp = nmt.NmtWorkspaceFlat()
    f0 = nmt.NmtFieldFlat(Lx, Ly, mask_apo, [np.zeros((XSIZE, XSIZE)), np.zeros((XSIZE, XSIZE))],
                          purify_b=PURIFY_B)
    wsp.compute_coupling_matrix(f0, f0, b)

    def coupled(Q, U):
        f = nmt.NmtFieldFlat(Lx, Ly, mask_apo, [Q, U], purify_b=PURIFY_B)
        return nmt.compute_coupled_cell_flat(f, f, b)

    def Dl(coup):
        cl = wsp.decouple_cell(coup)
        return cl[0]*fac, cl[3]*fac  # EE, BB

    def cls_from_Dl(dee, dbb):
        ell = np.arange(LMAX + 1)
        ff = ell*(ell+1)/(2*np.pi); ff[0] = 1.0
        return (np.where(ell >= 2, dee/ff, 0.0), np.where(ell >= 2, dbb/ff, 0.0))

    def sim_signal(cl_ee, cl_bb, seed):
        np.random.seed(seed)
        aE, aB = hp.synalm(cl_ee, lmax=LMAX), hp.synalm(cl_bb, lmax=LMAX)
        Qf, Uf = hp.alm2map_spin([aE, aB], NSIDE, 2, lmax=LMAX)
        return (project(np.where(mask_hp > 0, Qf, hp.UNSEEN), ra0d, dec0d),
                project(np.where(mask_hp > 0, Uf, hp.UNSEEN), ra0d, dec0d))

    # --- Transfer functions + leakage from signal-only sims ---
    print(f"Signal-only sims ({NSIM}) for transfer functions...")
    cl_ee_in, _ = cls_from_Dl(D_EE_IN, 0.0)
    _, cl_bb_in = cls_from_Dl(0.0, D_BB_IN)
    rec_ee_from_ee, rec_bb_from_ee, rec_bb_from_bb = [], [], []
    for s in range(NSIM):
        Q, U = sim_signal(cl_ee_in, np.zeros(LMAX+1), 200+s)
        dEE, dBB = Dl(coupled(Q, U)); rec_ee_from_ee.append(dEE); rec_bb_from_ee.append(dBB)
        Q, U = sim_signal(np.zeros(LMAX+1), cl_bb_in, 400+s)
        _, dBB2 = Dl(coupled(Q, U)); rec_bb_from_bb.append(dBB2)
    rec_ee_from_ee = np.mean(rec_ee_from_ee, 0)
    rec_bb_from_ee = np.mean(rec_bb_from_ee, 0)
    rec_bb_from_bb = np.mean(rec_bb_from_bb, 0)

    T_EE = rec_ee_from_ee / D_EE_IN          # EE transfer
    T_BB = rec_bb_from_bb / D_BB_IN          # BB transfer
    leak_per_EE = rec_bb_from_ee / D_EE_IN   # spurious BB per unit input EE

    # --- Noise bias from noise-only sims ---
    print(f"Noise-only sims ({NSIM}) for noise bias...")
    rng = np.random.default_rng(7)
    nEE, nBB, nEB = [], [], []
    for s in range(NSIM):
        nQ = np.full(len(mask_hp), hp.UNSEEN); nU = np.full(len(mask_hp), hp.UNSEEN)
        nQ[obs] = rng.standard_normal(len(obs)) / np.sqrt(ninv_Q[obs])
        nU[obs] = rng.standard_normal(len(obs)) / np.sqrt(ninv_U[obs])
        cl = wsp.decouple_cell(coupled(project(nQ, ra0d, dec0d), project(nU, ra0d, dec0d)))
        nEE.append(cl[0]*fac); nBB.append(cl[3]*fac); nEB.append(cl[1]*fac)
    N_EE, N_BB = np.mean(nEE, 0), np.mean(nBB, 0)

    # EB noise bias and transfer (EB is parity-odd: no EE->EB leakage by symmetry)
    N_EB = np.mean(nEB, 0)
    T_EB = np.sqrt(np.abs(T_EE * T_BB))

    def Dl3(coup):
        cl = wsp.decouple_cell(coup)
        return cl[0]*fac, cl[3]*fac, cl[1]*fac  # EE, BB, EB

    # --- Data ---
    print("Analyzing data...")
    Qd = project(np.where(mask_hp > 0, Q_dat, hp.UNSEEN), ra0d, dec0d)
    Ud = project(np.where(mask_hp > 0, U_dat, hp.UNSEEN), ra0d, dec0d)
    dEE_raw, dBB_raw, dEB_raw = Dl3(coupled(Qd, Ud))

    # Corrections: noise-subtract, transfer-correct EE; for BB also remove E->B leakage
    EE_corr = (dEE_raw - N_EE) / T_EE
    BB_leak = leak_per_EE * EE_corr           # leakage scales with the true (corrected) EE
    BB_corr = (dBB_raw - N_BB - BB_leak) / T_BB
    EB_corr = (dEB_raw - N_EB) / T_EB

    bjk = load_bjk()

    # --- Error bars: signal(EE fiducial)+noise ensemble through full corrected pipeline ---
    print(f"Fiducial signal+noise ensemble ({NSIM}) for NaMaster error bars...")
    # fiducial per-ℓ EE from BJK bandpowers (interp), BB=0
    fid_ell = bjk['EE']['ell']
    ell_all = np.arange(LMAX + 1)
    ff = ell_all*(ell_all+1)/(2*np.pi); ff[0] = 1.0
    Dl_fid = np.interp(ell_all, fid_ell, bjk['EE']['D'],
                       left=bjk['EE']['D'][0], right=bjk['EE']['D'][-1])
    cl_ee_fid = np.where(ell_all >= 2, Dl_fid / ff, 0.0)
    rng2 = np.random.default_rng(99)
    ens_EE, ens_BB, ens_EB = [], [], []
    for s in range(NSIM):
        np.random.seed(900 + s)
        aE = hp.synalm(cl_ee_fid, lmax=LMAX); aB = hp.synalm(np.zeros(LMAX+1), lmax=LMAX)
        Qf, Uf = hp.alm2map_spin([aE, aB], NSIDE, 2, lmax=LMAX)
        nQ = np.zeros(len(mask_hp)); nU = np.zeros(len(mask_hp))
        nQ[obs] = rng2.standard_normal(len(obs)) / np.sqrt(ninv_Q[obs])
        nU[obs] = rng2.standard_normal(len(obs)) / np.sqrt(ninv_U[obs])
        Qsim = project(np.where(mask_hp > 0, Qf + nQ, hp.UNSEEN), ra0d, dec0d)
        Usim = project(np.where(mask_hp > 0, Uf + nU, hp.UNSEEN), ra0d, dec0d)
        e, bbv, ebv = Dl3(coupled(Qsim, Usim))
        e = (e - N_EE) / T_EE
        bbv = (bbv - N_BB - leak_per_EE*e) / T_BB
        ebv = (ebv - N_EB) / T_EB
        ens_EE.append(e); ens_BB.append(bbv); ens_EB.append(ebv)
    err_EE = np.std(ens_EE, 0); err_BB = np.std(ens_BB, 0); err_EB = np.std(ens_EB, 0)

    # --- Report ---
    print("\nTransfer functions & leakage (median ℓ>=80):")
    print(f"  T_EE = {np.median(T_EE[2:]):.3f},  T_BB = {np.median(T_BB[2:]):.3f}")
    print(f"  E->B leak per unit EE = {np.median(leak_per_EE[2:]):.4f}")

    print(f"\n{'ℓ':<7} {'BJK_EE':<9} {'NMT_EE':<9} {'±err':<7} {'ratio':<7}  "
          f"{'BJK_BB':<9} {'NMT_BB':<9} {'±err'}")
    print("-"*72)
    for i in range(len(ell_eff)):
        rEE = EE_corr[i]/bjk['EE']['D'][i]
        print(f"{ell_eff[i]:<7.1f} {bjk['EE']['D'][i]*1e6:<9.2f} {EE_corr[i]*1e6:<9.2f} "
              f"{err_EE[i]*1e6:<7.2f} {rEE:<7.2f}  {bjk['BB']['D'][i]*1e6:<9.3f} "
              f"{BB_corr[i]*1e6:<9.3f} {err_BB[i]*1e6:.3f}")
    print("\nBB error-bar comparison (median ℓ>=80, D_ℓ×10⁶):")
    print(f"  NaMaster σ(BB) = {np.median(err_BB[2:])*1e6:.3f}")
    print(f"  BJK      σ(BB) = {np.median(bjk['BB']['sigma'][2:])*1e6:.3f}")
    print(f"  → NaMaster/BJK variance ratio ~ "
          f"{(np.median(err_BB[2:])/np.median(bjk['BB']['sigma'][2:]))**1:.2f}x larger error")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [('EE', EE_corr, err_EE), ('BB', BB_corr, err_BB), ('EB', EB_corr, err_EB)]
    for ax, (spec, nmtvals, nmterr) in zip(axes, panels):
        be = bjk[spec]
        ax.errorbar(be['ell'], be['D'], yerr=be['sigma'], fmt='o-', ms=8, lw=2,
                    capsize=5, label='BJK (HEALPix ML)', color='C0', zorder=3)
        ax.errorbar(ell_eff, nmtvals, yerr=nmterr, fmt='s-', ms=8, lw=2, capsize=5,
                    color='C1', zorder=2, label='Flat-sky NaMaster (steelman)')
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel(r'$\ell$'); ax.set_ylabel(rf'$D_\ell^{{{spec}}}$')
        ax.set_title(spec); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if spec == 'EE':
            ax.set_yscale('log'); ax.set_ylim(1e-6, 5e-5)

    fig.suptitle('Euclid RR2 tombin-1: BJK vs steelman Flat-sky NaMaster '
                 '(no purify, transfer-corrected, noise+leakage debiased)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    outfile = os.path.join(OUT_DIR, 'bjk_vs_flatsky_steelman.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
