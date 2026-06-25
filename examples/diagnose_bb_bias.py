"""
Diagnose the flat-sky BB bias: is it E->B leakage or noise?

Tests, all on the real RR2 mask geometry (NSIDE=128, matched resolution):
  A) Pure-EE signal, BB_input=0, NO noise  -> measures E->B leakage floor
  B) Same but purify_b=False               -> shows what purification removes
  C) Pure-BB signal, EE_input=0            -> B->E leakage + BB transfer/normalization
  D) Noise-only                            -> noise floor (the debias target)
Compare leakage floor to the residual BB seen in the data (~2-4e-6).
"""

import numpy as np
import healpy as hp
from healpy.projector import GnomonicProj
import pymaster as nmt
from scipy.ndimage import distance_transform_edt
import os

TOMBIN = -1
NSIDE = 128
ALMANAC_DIR = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs')

BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
RESO = np.degrees(hp.nside2resol(NSIDE)) * 60
XSIZE = 100
APO_SCALE = 60.0
LMAX = 256

D_EE = 5e-6  # flat D_ℓ input


def project(map_hp, nside, ra0, dec0, xsize, reso):
    proj = GnomonicProj(rot=[ra0, dec0, 0], xsize=xsize, reso=reso)
    img = proj.projmap(map_hp, vec2pix_func=lambda x, y, z: hp.vec2pix(nside, x, y, z))
    valid = np.isfinite(img) & (img != hp.UNSEEN)
    return np.where(valid, img, 0.0), valid.astype(float)


def setup():
    NINV = os.path.join(ALMANAC_DIR, f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_invnoise.fits')
    ninv_Q, ninv_U = hp.read_map(NINV, field=[0, 1], verbose=False)
    mask_hp = (ninv_Q > 0).astype(float)
    obs = np.where(mask_hp)[0]
    th, ph = hp.pix2ang(NSIDE, obs)
    dec, ra = np.pi/2 - th, ph
    rx, ry = np.cos(ra).mean(), np.sin(ra).mean()
    ra0 = np.arctan2(ry, rx)
    if ra0 < 0: ra0 += 2*np.pi
    return mask_hp, obs, np.degrees(ra0), np.degrees(dec.mean()), ninv_Q, ninv_U


def make_grid(mask_hp, ra0d, dec0d):
    _, mask = project(np.where(mask_hp > 0, 1.0, hp.UNSEEN), NSIDE, ra0d, dec0d, XSIZE, RESO)
    reso_rad = np.radians(RESO / 60.0)
    Lx = Ly = XSIZE * reso_rad
    dist = distance_transform_edt(mask)
    mask_apo = np.clip(dist / (APO_SCALE / RESO), 0, 1)
    return mask_apo, Lx, Ly


def cls_from_Dl(D_EE_val, D_BB_val):
    ell = np.arange(LMAX + 1)
    fac = ell * (ell + 1) / (2 * np.pi); fac[0] = 1.0
    cl_ee = np.where(ell >= 2, D_EE_val / fac, 0.0)
    cl_bb = np.where(ell >= 2, D_BB_val / fac, 0.0)
    return cl_ee, cl_bb


def analyze(Q, U, mask_apo, Lx, Ly, purify_b=True):
    f2 = nmt.NmtFieldFlat(Lx, Ly, mask_apo, [Q, U], purify_b=purify_b)
    b = nmt.NmtBinFlat(BAND_EDGES[:-1].astype(float), (BAND_EDGES[1:] - 1).astype(float))
    ell_eff = b.get_effective_ells()
    wsp = nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f2, f2, b)
    cl = wsp.decouple_cell(nmt.compute_coupled_cell_flat(f2, f2, b))
    fac = ell_eff * (ell_eff + 1) / (2 * np.pi)
    return ell_eff, cl[0]*fac, cl[3]*fac  # ell, D_EE, D_BB


def sim_maps(cl_ee, cl_bb, mask_hp, ra0d, dec0d, seed):
    np.random.seed(seed)
    almE = hp.synalm(cl_ee, lmax=LMAX)
    almB = hp.synalm(cl_bb, lmax=LMAX)
    Qf, Uf = hp.alm2map_spin([almE, almB], NSIDE, 2, lmax=LMAX)
    Q_in = np.where(mask_hp > 0, Qf, hp.UNSEEN)
    U_in = np.where(mask_hp > 0, Uf, hp.UNSEEN)
    Q, _ = project(Q_in, NSIDE, ra0d, dec0d, XSIZE, RESO)
    U, _ = project(U_in, NSIDE, ra0d, dec0d, XSIZE, RESO)
    return Q, U


def main():
    mask_hp, obs, ra0d, dec0d, ninv_Q, ninv_U = setup()
    mask_apo, Lx, Ly = make_grid(mask_hp, ra0d, dec0d)
    NSIM = 20

    print("="*72)
    print("BB bias diagnosis (NSIDE=128, RR2 mask). D_EE input = 5e-6, D_BB input = 0")
    print("="*72)

    # --- A) pure-EE, no noise, purify_b=True ---
    cl_ee, cl_bb0 = cls_from_Dl(D_EE, 0.0)
    ee_p, bb_p = [], []
    for s in range(NSIM):
        Q, U = sim_maps(cl_ee, cl_bb0, mask_hp, ra0d, dec0d, seed=100+s)
        ell, dEE, dBB = analyze(Q, U, mask_apo, Lx, Ly, purify_b=True)
        ee_p.append(dEE); bb_p.append(dBB)
    ee_p, bb_p = np.mean(ee_p, 0), np.mean(bb_p, 0)

    # --- B) pure-EE, no noise, purify_b=False ---
    ee_np, bb_np = [], []
    for s in range(NSIM):
        Q, U = sim_maps(cl_ee, cl_bb0, mask_hp, ra0d, dec0d, seed=100+s)
        ell, dEE, dBB = analyze(Q, U, mask_apo, Lx, Ly, purify_b=False)
        ee_np.append(dEE); bb_np.append(dBB)
    ee_np, bb_np = np.mean(ee_np, 0), np.mean(bb_np, 0)

    # --- D) noise-only, purify_b=True ---
    bb_noise = []
    rng = np.random.default_rng(7)
    for s in range(NSIM):
        nQ = np.full(len(mask_hp), hp.UNSEEN); nU = np.full(len(mask_hp), hp.UNSEEN)
        nQ[obs] = rng.standard_normal(len(obs)) / np.sqrt(ninv_Q[obs])
        nU[obs] = rng.standard_normal(len(obs)) / np.sqrt(ninv_U[obs])
        Q, _ = project(nQ, NSIDE, ra0d, dec0d, XSIZE, RESO)
        U, _ = project(nU, NSIDE, ra0d, dec0d, XSIZE, RESO)
        _, _, dBB = analyze(Q, U, mask_apo, Lx, Ly, purify_b=True)
        bb_noise.append(dBB)
    bb_noise = np.mean(bb_noise, 0)

    print(f"\n{'ℓ':<7} {'EE_in':<9} {'BB leak':<11} {'BB leak':<11} {'BB noise':<11}")
    print(f"{'':<7} {'(recov)':<9} {'(purify)':<11} {'(no pur)':<11} {'floor':<11}")
    print("-"*54)
    for i in range(len(ell)):
        print(f"{ell[i]:<7.1f} {ee_p[i]*1e6:<9.2f} {bb_p[i]*1e6:<11.3f} "
              f"{bb_np[i]*1e6:<11.3f} {bb_noise[i]*1e6:<11.3f}")

    print("\nInterpretation (compare to data residual BB ~2-4e-6):")
    print(f"  E->B leakage (purified), ℓ>=80: {np.median(bb_p[2:])*1e6:.3f}e-6")
    print(f"  E->B leakage (unpurified),ℓ>=80: {np.median(bb_np[2:])*1e6:.3f}e-6")
    print(f"  Noise floor (single real),ℓ>=80: {np.median(bb_noise[2:])*1e6:.3f}e-6")
    print(f"  EE leakage as % of EE: {np.median(bb_p[2:]/ee_p[2:])*100:.1f}%")


if __name__ == '__main__':
    main()
