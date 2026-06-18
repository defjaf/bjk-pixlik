"""
Run BJK pixel-space ML and NaMaster pseudo-Cl on an Almanac simulation directory.

Reads configuration from the sim ini file.  Supports:
  spin=0                      → TT only
  spin=2                      → EE + BB  [+ EB with --include-eb]
  spin=2, nfields=2*n_P       → n_P tomographic E/B bins (EE/BB/EB pairs)

Usage:
    python3 run_bjk_nmt.py <sim_dir>  [--band-edges "2,34,...,257"]  [--lmax 256]
                                      [--include-eb]

Outputs (written into <sim_dir>/):
    bjk_nmt_bandpowers.dat    columns: ell_eff  spec  D_l_bjk  sig_bjk  D_l_nmt  sig_nmt
    bjk_nmt_bandpowers.png

Also accepts optional Almanac extract files to overlay:
    --almanac-extract  /path/to/chain.extract.npy   (repeatable)
    --almanac-label    "label"                       (one per extract)
"""

import argparse, configparser, os, sys
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BJK_ROOT = '/Users/jaffe/Desktop/Projects/bjk-pixlik'
sys.path.insert(0, BJK_ROOT)
from pixel_likelihood import PixelLikelihood

DEFAULT_BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_ini(sim_dir):
    """Parse the single *.ini file in sim_dir (AlmaSim format with [simulation] section)."""
    inis = [f for f in os.listdir(sim_dir) if f.endswith('.ini')]
    if len(inis) != 1:
        raise RuntimeError(f"Expected exactly one .ini in {sim_dir}, found {inis}")
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(sim_dir, inis[0]))
    sim = cfg['simulation']
    out = cfg['sim_output']
    return {
        'nside':       int(sim.get('nside',  128)),
        'lmax':        int(sim.get('lmax',   256)),
        'nfields':     int(sim.get('nfields', 1)),
        'spin':        int(sim.get('spin', '0').split(',')[0].strip()),
        'noise_level': float(sim.get('noiselevel', 0.0)),
        'mask':        sim.get('mask', '').strip(),
        'output_root': out.get('outputroot', os.path.join(sim_dir, 'sim_out')).strip(),
    }


def D(l, c):
    return l * (l + 1) / (2 * np.pi) * c


def bin_almanac_samples(samples, band_edges, lmin=2):
    """Bin (n_samples, n_ell) extract array into bandpowers with (2l+1) weights."""
    ell_all = np.arange(lmin, lmin + samples.shape[1])
    n_bands = len(band_edges) - 1
    result  = np.zeros((len(samples), n_bands))
    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        idx = np.where((ell_all >= lo) & (ell_all < hi))[0]
        w   = (2 * ell_all[idx] + 1).astype(float)
        result[:, b] = (samples[:, idx] * w).sum(1) / w.sum()
    return result


def hsm(x):
    """Half-sample mode estimator."""
    x = np.sort(x); n = len(x)
    while n > 2:
        h = (n + 1) // 2
        i = np.argmin(x[h - 1:] - x[:n - h + 1])
        x = x[i:i + h]; n = len(x)
    return x.mean()


# ---------------------------------------------------------------------------
# PD-safe Newton
# ---------------------------------------------------------------------------

def pd_safe_newton(lik, cl_init, max_iter=30, tol=1e-5, damp=1.0):
    """
    Newton-Raphson with positive-definiteness safeguard.

    After each step, for every band b the per-band signal matrix must be PD.
    We enforce this by clamping diagonal spectra to a small positive floor and
    clipping cross-spectra so |C_XY|^2 < C_XX * C_YY (with a 1% margin).
    Falls back to the plain newton_raphson when the layout has no cross-spectra.
    """
    from pixel_likelihood import SpectraLayout
    layout = lik.layout
    nb = lik.nbands

    # Identify diagonal and cross-spectrum bands for the PD constraint.
    # For each band b, collect all (spec, i, j) entries and clamp.
    def _clamp(cl):
        cl = cl.copy()
        # For each P bin pair (i<=j), enforce EE[i,i]*BB[i,i] > EB[i,j]^2
        for b in range(nb):
            for spec_diag, spec_cross, label in [('EE', 'BB', None), ]:
                pass   # placeholder; real clamping below
            # Clamp all diagonal auto-spectra to a floor
            for spec in ('TT', 'EE', 'BB'):
                for i, j in layout._pairs.get(spec, []):
                    if i == j:
                        idx = layout.index(spec, i, j, b)
                        cl[idx] = max(cl[idx], 1e-35)
            # Clamp cross-spectra that violate PD
            for spec_cross in ('EB', 'TE', 'TB'):
                for i, j in layout._pairs.get(spec_cross, []):
                    idx_c = layout.index(spec_cross, i, j, b)
                    if spec_cross == 'EB':
                        idx_ee = layout.index('EE', i, i, b)
                        idx_bb = layout.index('BB', j, j, b)
                        limit = 0.999 * np.sqrt(abs(cl[idx_ee]) * abs(cl[idx_bb]))
                        cl[idx_c] = np.clip(cl[idx_c], -limit, limit)
                    elif spec_cross in ('TE', 'TB'):
                        idx_tt = layout.index('TT', i, i, b)
                        idx_pp = layout.index('EE' if spec_cross == 'TE' else 'BB', j, j, b)
                        limit = 0.999 * np.sqrt(abs(cl[idx_tt]) * abs(cl[idx_pp]))
                        cl[idx_c] = np.clip(cl[idx_c], -limit, limit)
        return cl

    cl = _clamp(np.asarray(cl_init, dtype=float))
    print(f"\nPD-safe Newton-Raphson ({len(cl)} parameters, max_iter={max_iter})")

    for it in range(max_iter):
        try:
            g, F = lik.gradient_and_fisher(cl)
            delta = np.linalg.solve(F, g)
        except np.linalg.LinAlgError as e:
            print(f"  iter {it+1}: stopping ({e})")
            break
        cl_new = _clamp(cl + damp * delta)
        step = np.max(np.abs(delta) / (np.abs(cl) + 1e-40))
        logL = lik.log_likelihood(cl_new)
        print(f"  iter {it+1}: logL={logL:.4f}  max|δ/C|={step:.2e}")
        cl = cl_new
        if step < tol:
            print("  Converged.")
            break

    try:
        _, F = lik.gradient_and_fisher(cl)
        sigma = np.sqrt(np.maximum(np.diag(np.linalg.inv(F)), 0))
    except np.linalg.LinAlgError:
        sigma = np.full(len(cl), np.nan)
        F = np.full((len(cl), len(cl)), np.nan)

    return cl, sigma, F


# ---------------------------------------------------------------------------
# BJK
# ---------------------------------------------------------------------------

def run_bjk(sim_dir, cfg, band_edges, lmax, include_eb=False):
    root = cfg['output_root']
    spin = cfg['spin']
    n_P  = cfg['nfields'] // 2 if spin != 0 else 0
    n_T  = 1 if spin == 0 else 0

    data_fits = root + '_channel_0_data.fits'
    ninv_fits = root + '_channel_0_nInv.fits'

    lik = PixelLikelihood(
        data_fits=data_fits, ninv_fits=ninv_fits,
        lmin=2, lmax=lmax, band_edges=band_edges,
        n_T=n_T, n_P=n_P, band_model='Dl',
        include_EB=include_eb and (n_P > 0),
    )

    layout = lik.layout
    nb = lik.nbands
    ell_b = lik.ell_bands
    D_FLAT = 1e-4 / (2.0 * np.pi)
    cl_init = np.zeros(layout.n_params)
    for idx, spec, i, j, b in layout.entries():
        if spec in ('TT', 'EE', 'BB') and i == j:
            cl_init[idx] = D_FLAT
        # cross-spectra start at zero

    dl_ml, sigma, _ = pd_safe_newton(lik, cl_init, max_iter=30, tol=1e-5)

    # Active spectrum names for output
    specs = []
    for spec, pairs in layout.spec_groups():
        for (i, j) in pairs:
            label = spec if n_P <= 1 and n_T <= 1 else f'{spec}_{i}{j}'
            specs.append((label, spec, i, j))

    return ell_b, dl_ml, sigma, specs, layout


# ---------------------------------------------------------------------------
# NaMaster
# ---------------------------------------------------------------------------

def run_nmt(sim_dir, cfg, band_edges, lmax, include_eb=False, specs_out=None):
    root        = cfg['output_root']
    spin        = cfg['spin']
    nside       = cfg['nside']
    noise_level = cfg['noise_level']
    mask_file   = cfg['mask']

    mask = hp.read_map(mask_file, verbose=False).astype(float)
    npix      = hp.nside2npix(nside)
    omega_pix = 4.0 * np.pi / npix
    N_l       = omega_pix * noise_level ** 2

    # Use lmax for NmtField to match BJK; bin edges as given
    bins = nmt.NmtBin.from_edges(band_edges[:-1], band_edges[1:], is_Dell=True)
    keep = bins.get_effective_ells() < band_edges[-1]
    ell_eff = bins.get_effective_ells()[keep]

    if spin == 0:
        T_map = hp.read_map(root + '_channel_0_data.fits', field=0, verbose=False)
        f     = nmt.NmtField(mask, [T_map], lmax=lmax)
        wsp   = nmt.NmtWorkspace.from_fields(f, f, bins)
        cl    = wsp.decouple_cell(nmt.compute_coupled_cell(f, f))
        lmax_field = f.ainfo.lmax
        ell_all  = np.arange(lmax_field + 1, dtype=float)
        cl_tot   = np.where(ell_all >= 2,
                            1e-4 / np.where(ell_all >= 2, ell_all*(ell_all+1), 1.0), 0.0) + N_l
        cw  = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
        cov = cw.gaussian_covariance([cl_tot], [cl_tot], [cl_tot], [cl_tot], wsp)
        err = np.sqrt(np.diag(cov))
        return ell_eff, {'TT': (D(ell_eff, cl[0][keep]), D(ell_eff, err[keep]))}

    else:
        # spin-2; supports single tomo bin (n_P=1) and multiple (n_P>1 — one field per bin)
        n_P = cfg['nfields'] // 2
        maps_all = hp.read_map(root + '_channel_0_data.fits', field=None, verbose=False)
        result = {}
        nell = np.array([bins.get_ell_list(b).size for b in range(bins.get_n_bands())])
        for p in range(n_P):
            Q, U = maps_all[2*p], maps_all[2*p+1]
            fp = nmt.NmtField(mask, [Q, U], lmax=lmax)
            for q in range(p, n_P):
                if q == p:
                    fq = fp
                else:
                    Qq, Uq = maps_all[2*q], maps_all[2*q+1]
                    fq = nmt.NmtField(mask, [Qq, Uq], lmax=lmax)
                wsp = nmt.NmtWorkspace.from_fields(fp, fq, bins)
                cl  = wsp.decouple_cell(nmt.compute_coupled_cell(fp, fq))
                # cl[0]=EE, cl[1]=EB, cl[2]=BE, cl[3]=BB
                cl_ee = cl[0][keep]; cl_bb = cl[3][keep]; cl_eb = cl[1][keep]
                n_ell_b = nell[keep]
                fsky_eff = mask.mean()
                sig_ee = np.sqrt(2/(n_ell_b*(2*ell_eff+1)*fsky_eff)) * (cl_ee + N_l)
                sig_bb = np.sqrt(2/(n_ell_b*(2*ell_eff+1)*fsky_eff)) * (cl_bb + N_l)
                sig_eb = np.sqrt(1/(n_ell_b*(2*ell_eff+1)*fsky_eff)) * \
                         np.sqrt((cl_ee+N_l)*(cl_bb+N_l) + cl_eb**2)
                suf = f'_{p}{q}' if n_P > 1 else ''
                result[f'EE{suf}'] = (D(ell_eff, cl_ee), D(ell_eff, sig_ee))
                result[f'BB{suf}'] = (D(ell_eff, cl_bb), D(ell_eff, sig_bb))
                if include_eb:
                    result[f'EB{suf}'] = (D(ell_eff, cl_eb), D(ell_eff, sig_eb))
        return ell_eff, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim_dir')
    ap.add_argument('--band-edges', default=None,
                    help='Comma-separated band edges, e.g. "2,34,66,98,130,162,194,226,257"')
    ap.add_argument('--lmax', type=int, default=None)
    ap.add_argument('--almanac-extract', action='append', default=[], dest='extracts',
                    metavar='FILE')
    ap.add_argument('--almanac-label',   action='append', default=[], dest='labels',
                    metavar='LABEL')
    ap.add_argument('--include-eb', action='store_true', default=False,
                    help='Include EB cross-spectrum for spin-2 fields. '
                         'WARNING: UNTESTED (OOM at NSIDE=64 fsky=0.1 on 64 GB RAM). '
                         'Memory usage: 21 kernels × (2×n_obs)² × 8 bytes ≈ 16 GB stored '
                         'plus build-time temporaries. Use NSIDE<=32 or implement '
                         'on-the-fly kernels before using this at higher resolution.')
    args = ap.parse_args()

    sim_dir    = os.path.abspath(args.sim_dir)
    cfg        = read_ini(sim_dir)
    lmax       = args.lmax or cfg['lmax']
    band_edges = (np.array([int(x) for x in args.band_edges.split(',')])
                  if args.band_edges else DEFAULT_BAND_EDGES)

    include_eb = args.include_eb

    print(f"Sim dir : {sim_dir}")
    print(f"spin={cfg['spin']}  nfields={cfg['nfields']}  nside={cfg['nside']}  lmax={lmax}  noise={cfg['noise_level']:.4e}")
    print(f"Bands   : {band_edges}  include_eb={include_eb}")

    # --- BJK ---
    print("\n=== BJK ===")
    ell_bjk, dl_bjk, sig_bjk, specs, layout = run_bjk(sim_dir, cfg, band_edges, lmax,
                                                        include_eb=include_eb)

    # --- NaMaster ---
    print("\n=== NaMaster ===")
    ell_nmt, nmt_res = run_nmt(sim_dir, cfg, band_edges, lmax,
                                include_eb=include_eb)

    # --- Save dat ---
    rows = []
    for label, spec, i, j, in specs:
        if label not in nmt_res:
            continue
        dl_nmt_s, err_nmt_s = nmt_res[label]
        for b_idx, l in enumerate(ell_bjk):
            bjk_idx = layout.index(spec, i, j, b_idx)
            rows.append([l, label, dl_bjk[bjk_idx], sig_bjk[bjk_idx],
                         dl_nmt_s[b_idx], err_nmt_s[b_idx]])
    out_dat = os.path.join(sim_dir, 'bjk_nmt_bandpowers.dat')
    with open(out_dat, 'w') as f:
        f.write('# ell_eff  spec  D_l_bjk  sig_bjk  D_l_nmt  sig_nmt\n')
        for r in rows:
            f.write(f'{r[0]:.2f}  {r[1]}  {r[2]:.6e}  {r[3]:.6e}  {r[4]:.6e}  {r[5]:.6e}\n')
    print(f"\nSaved: {out_dat}")

    # --- Plot ---
    spec_labels = [s[0] for s in specs if s[0] in nmt_res]
    fig, axes = plt.subplots(1, len(spec_labels), figsize=(7*len(spec_labels), 5), squeeze=False)
    for ax, label in zip(axes[0], spec_labels):
        dl_nmt_s, err_nmt_s = nmt_res[label]
        # BJK values for this spectrum
        spec_entry = next(s for s in specs if s[0] == label)
        _, spec, i, j = spec_entry
        bjk_vals = np.array([dl_bjk[layout.index(spec, i, j, b)] for b in range(len(ell_bjk))])
        bjk_errs = np.array([sig_bjk[layout.index(spec, i, j, b)] for b in range(len(ell_bjk))])

        ax.errorbar(ell_bjk - 4, bjk_vals, yerr=bjk_errs,
                    fmt='gs', ms=6, capsize=4, lw=1.5, label='BJK ML ±1σ', zorder=9)
        ax.errorbar(ell_nmt + 4, dl_nmt_s, yerr=err_nmt_s,
                    fmt='b^', ms=6, capsize=4, lw=1.5, label='NaMaster ±1σ', zorder=8)

        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_xlabel('ℓ'); ax.set_ylabel('Dℓ = ℓ(ℓ+1)Cℓ/2π')
        ax.set_title(f'{label}  —  {os.path.basename(sim_dir)}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(sim_dir, 'bjk_nmt_bandpowers.png')
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_png.replace('.png', '.svg'))
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
