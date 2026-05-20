"""
BJK (Dl-model) vs NaMaster — sky fraction × band-width comparison.

Three cases:
  A. Full sky      — NSIDE=8,  LMAX=15,  f_sky=1,    synthetic,  Δℓ ∈ {7,4,2,1}
  B. Partial sky   — NSIDE=128,f_sky≈0.01,almasim data, Δℓ ∈ {32,16,8}
  C. Intermediate  — NSIDE=32, LMAX=64,  f_sky≈0.1,  synthetic,  Δℓ ∈ {32,16,8,4,2}

ℓ_char ≈ π/(2√f_sky): NaMaster degrades when Δℓ ≲ ℓ_char.
BJK degrades when the Fisher matrix becomes near-singular (too few observed pixels
per band parameter).

NaMaster uses is_Dell=True so its output D_b = (2l+1)-weighted mean l(l+1)C_l/(2π)
is directly comparable to the BJK D_l=const model.

Run from sim_runs/bjk/:
    python3 test_sky_comparison.py
Output:
    test_sky_comparison.png
"""
import sys, os, tempfile
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixel_likelihood import PixelLikelihood

# ---------------------------------------------------------------------------
# ── Case A: full sky, NSIDE=8 ──────────────────────────────────────────────
NSIDE_A      = 8
LMIN_A       = 2
LMAX_A       = 2 * NSIDE_A - 1       # 15
LMAX_FIELD_A = 3 * NSIDE_A - 1       # 23
NPIX_A       = hp.nside2npix(NSIDE_A)
OMEGA_PIX_A  = 4.0 * np.pi / NPIX_A
A_SIG        = 1e-4                  # flat D_l amplitude (D_l = A/(2π) = const)
SIGMA_PIX_A  = 1e-3
N_L_A        = SIGMA_PIX_A**2 * OMEGA_PIX_A
D_TRUE_A     = A_SIG / (2.0 * np.pi)
DELL_LIST_A  = [7, 4, 2, 1]

# Simulate full-sky data (fixed seed)
np.random.seed(42)
rng_a        = np.random.default_rng(42)
ll_a         = np.arange(LMAX_A + 1, dtype=float)
cl_sig_a     = np.zeros(LMAX_A + 1)
cl_sig_a[2:] = A_SIG / (ll_a[2:] * (ll_a[2:] + 1))  # D_l = A/(2π) = const
data_map_a   = (hp.alm2map(hp.synalm(cl_sig_a, lmax=LMAX_A), NSIDE_A, lmax=LMAX_A)
                + rng_a.normal(0.0, SIGMA_PIX_A, NPIX_A))
mask_a       = np.ones(NPIX_A)       # full sky

# Theory C_l for NaMaster covariance (Case A)
ell_th_a          = np.arange(LMAX_FIELD_A + 1, dtype=float)
cl_sig_th_a       = np.where(ell_th_a >= 2,
                              A_SIG / np.where(ell_th_a >= 2,
                                               ell_th_a * (ell_th_a + 1), 1.0),
                              0.0)
cl_sig_th_a[LMAX_A + 1:] = 0.0
cl_tot_th_a       = cl_sig_th_a + N_L_A

# ── Case C: intermediate sky fraction, NSIDE=32, synthetic ─────────────────
NSIDE_C      = 32
LMIN_C       = 2
LMAX_C       = 2 * NSIDE_C            # 64
LMAX_FIELD_C = 3 * NSIDE_C - 1        # 95
NPIX_C       = hp.nside2npix(NSIDE_C) # 12288
OMEGA_PIX_C  = 4.0 * np.pi / NPIX_C
SIGMA_PIX_C  = 1e-3
N_L_C        = SIGMA_PIX_C**2 * OMEGA_PIX_C
D_TRUE_C     = A_SIG / (2.0 * np.pi)
F_SKY_C_TARGET = 0.10
DELL_LIST_C  = [32, 16, 8, 4, 2]

# Spherical cap around north pole with area ≈ 10% of sphere
# f_sky = (1 - cos r)/2  →  r = arccos(1 - 2*f_sky)
r_cap_c = np.arccos(1.0 - 2.0 * F_SKY_C_TARGET)
cap_pix_c = hp.query_disc(NSIDE_C, np.array([0.0, 0.0, 1.0]), r_cap_c,
                           inclusive=True)
mask_c_hard = np.zeros(NPIX_C)
mask_c_hard[cap_pix_c] = 1.0
fsky_c      = float(mask_c_hard.mean())
n_obs_c     = int(mask_c_hard.sum())
ell_ch_c    = np.pi / (2.0 * np.sqrt(max(fsky_c, 1e-9)))

# Apodized mask for NaMaster
mask_c_nmt  = nmt.mask_apodization(mask_c_hard, 5.0, apotype='C2')

# Simulate partial-sky data (fixed seed, same flat D_l spectrum)
np.random.seed(137)
rng_c        = np.random.default_rng(137)
ll_c         = np.arange(LMAX_C + 1, dtype=float)
cl_sig_c     = np.zeros(LMAX_C + 1)
cl_sig_c[2:] = A_SIG / (ll_c[2:] * (ll_c[2:] + 1))
data_map_c   = (hp.alm2map(hp.synalm(cl_sig_c, lmax=LMAX_C), NSIDE_C, lmax=LMAX_C)
                + rng_c.normal(0.0, SIGMA_PIX_C, NPIX_C))
ninv_c       = np.where(mask_c_hard > 0, 1.0 / SIGMA_PIX_C**2, 0.0)

# Theory C_l for NaMaster covariance (Case C)
ell_th_c     = np.arange(LMAX_FIELD_C + 1, dtype=float)
cl_sig_th_c  = np.where(ell_th_c >= 2,
                         A_SIG / np.where(ell_th_c >= 2,
                                          ell_th_c * (ell_th_c + 1), 1.0),
                         0.0)
cl_sig_th_c[LMAX_C + 1:] = 0.0
cl_tot_th_c  = cl_sig_th_c + N_L_C

print(f"Case C (synthetic): NSIDE={NSIDE_C}  n_obs={n_obs_c}  "
      f"f_sky={fsky_c:.4f}  ℓ_char≈{ell_ch_c:.1f}")

# ── Case B: partial sky, NSIDE=128 ─────────────────────────────────────────
BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
SIM_DIR_B    = os.path.join(BASE_DIR, 'sim_tombin-1_nside128_flatTT')
DATA_FILE_B  = os.path.join(SIM_DIR_B, 'sim_out_channel_0_data.fits')
NINV_FILE_B  = os.path.join(SIM_DIR_B, 'sim_out_channel_0_nInv.fits')
MASK_FILE_B  = os.path.join(BASE_DIR,
               'sim_tombin-1_nside128_flatEE/euclid_rr2_tombin-1_nside128_mask.fits')

NSIDE_B      = 128
LMIN_B       = 2
LMAX_B       = 256
LMAX_FIELD_B = 3 * NSIDE_B - 1       # 383
NPIX_B       = hp.nside2npix(NSIDE_B)
OMEGA_PIX_B  = 4.0 * np.pi / NPIX_B
NOISE_PIX_B  = 1.612931e-3            # per-pixel noise std from ini
N_L_B        = NOISE_PIX_B**2 * OMEGA_PIX_B
D_TRUE_B     = A_SIG / (2.0 * np.pi)
DELL_LIST_B  = [32, 16, 8]

have_b = os.path.exists(DATA_FILE_B) and os.path.exists(NINV_FILE_B)
if have_b:
    print("Loading NSIDE=128 flatTT data (almasim)...")
    data_map_b = hp.read_map(DATA_FILE_B, field=0, verbose=False)
    ninv_map_b = hp.read_map(NINV_FILE_B, field=0, verbose=False)
    mask_b     = (ninv_map_b > 0).astype(float)
    fsky_b     = mask_b.mean()
    n_obs_b    = int(mask_b.sum())
    if os.path.exists(MASK_FILE_B):
        mask_b_nmt = hp.read_map(MASK_FILE_B, verbose=False).astype(float)
    else:
        mask_b_nmt = mask_b
    ell_ch_b = np.pi / (2.0 * np.sqrt(max(fsky_b, 1e-9)))
    print(f"  n_obs={n_obs_b}  f_sky={fsky_b:.4f}"
          f"  N_l={N_L_B:.4e}  D_true={D_TRUE_B:.4e}  ℓ_char≈{ell_ch_b:.1f}")
    # Theory C_l for NaMaster covariance (Case B)
    ell_th_b          = np.arange(LMAX_FIELD_B + 1, dtype=float)
    cl_sig_th_b       = np.where(ell_th_b >= 2,
                                  A_SIG / np.where(ell_th_b >= 2,
                                                   ell_th_b * (ell_th_b + 1), 1.0),
                                  0.0)
    cl_sig_th_b[LMAX_B + 1:] = 0.0
    cl_tot_th_b = cl_sig_th_b + N_L_B
else:
    print("WARNING: NSIDE=128 data not found — skipping Case B.")
    fsky_b, n_obs_b, ell_ch_b = 0.01, 0, 15.3

# ---------------------------------------------------------------------------
# Helpers

def band_edges(lmin, lmax, dlell):
    ed = list(range(lmin, lmax + 1, dlell))
    if ed[-1] < lmax + 1:
        ed.append(lmax + 1)
    return np.array(ed, dtype=int)

def midpoints(edges):
    return 0.5 * (edges[:-1] + edges[1:] - 1)

# ---------------------------------------------------------------------------
# BJK runner

def run_bjk(data_map, ninv_map, lmin, lmax, edges, d_true):
    """BJK D_l-model NR. Returns (ell_b, dl, sigma, ok)."""
    ell_b = midpoints(edges)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d_f = os.path.join(tmp, 'd.fits')
            n_f = os.path.join(tmp, 'n.fits')
            hp.write_map(d_f, data_map, overwrite=True)
            hp.write_map(n_f, ninv_map, overwrite=True)
            lik = PixelLikelihood(d_f, n_f, lmin=lmin, lmax=lmax,
                                  band_edges=edges, spin=0, band_model='Dl')
            dl_init = np.full(len(ell_b), d_true)
            dl, sig, _ = lik.newton_raphson(dl_init, max_iter=50, tol=1e-7)
        # Detect NR divergence: step size exploded and never recovered
        if np.any(np.isnan(dl)) or np.any(np.isnan(sig)) or np.any(sig > 500 * d_true):
            raise ValueError("NR diverged (Fisher matrix near-singular)")
        return ell_b, dl, sig, True
    except Exception as exc:
        print(f"    BJK FAILED: {exc}")
        nan = np.full(len(ell_b), np.nan)
        return ell_b, nan, nan, False

# ---------------------------------------------------------------------------
# NaMaster runner  (is_Dell=True → output is D_b, directly comparable to BJK)

def run_namaster(data_map, mask, lmin, lmax, lmax_field, edges, n_l, cl_tot_th):
    """NaMaster pseudo-Cl with D_l binning. Returns (ell_eff, dl_nmt, ok)."""
    _ini = np.append(edges[:-1], edges[-1])
    _end = np.append(edges[1:],  lmax_field + 1)
    # is_Dell=True: NMT returns (2l+1)-weighted mean of l(l+1)C_l/(2π) = D_l
    b    = nmt.NmtBin.from_edges(_ini, _end, is_Dell=True)
    ell_all = b.get_effective_ells()
    try:
        f0  = nmt.NmtField(mask.astype(float), [data_map],
                           beam=np.ones(lmax_field + 1))
        wsp = nmt.NmtWorkspace.from_fields(f0, f0, b)
        # cl_d is already D_b (NMT applied the l*(l+1)/(2π) weighting internally)
        cl_d = wsp.decouple_cell(nmt.compute_coupled_cell(f0, f0))[0]
        keep = ell_all < edges[-1]
        ell_e  = ell_all[keep]
        # Subtract noise D_l contribution: N_l * eff_ell*(eff_ell+1)/(2π)
        dl_nmt = cl_d[keep] - n_l * ell_e * (ell_e + 1) / (2.0 * np.pi)
        return ell_e, dl_nmt, True
    except Exception as exc:
        print(f"    NaMaster FAILED: {exc}")
        keep = ell_all < edges[-1]
        ell_e = ell_all[keep]
        return ell_e, np.full(ell_e.shape, np.nan), False

# ---------------------------------------------------------------------------
# Run Case A (full sky, NSIDE=8)
print(f"\n{'='*60}")
print(f"Case A: full sky  NSIDE={NSIDE_A}  LMAX={LMAX_A}  f_sky=1.0")
print('='*60)
results_a = {}
for dlell in DELL_LIST_A:
    edges_ = band_edges(LMIN_A, LMAX_A, dlell)
    nbands = len(edges_) - 1
    print(f"  Δℓ={dlell:2d} ({nbands} bands)", end='  ', flush=True)

    ninv_a = np.where(mask_a > 0, 1.0 / SIGMA_PIX_A**2, 0.0)
    ell_b, dl_b, sig_b, ok_b = run_bjk(data_map_a, ninv_a,
                                        LMIN_A, LMAX_A, edges_, D_TRUE_A)
    ell_n, dl_n, ok_n = run_namaster(data_map_a, mask_a,
                                      LMIN_A, LMAX_A, LMAX_FIELD_A,
                                      edges_, N_L_A, cl_tot_th_a)
    with np.errstate(invalid='ignore', divide='ignore'):
        db = np.nanmax(np.abs(dl_b - D_TRUE_A)) / D_TRUE_A
        dn = np.nanmax(np.abs(dl_n - D_TRUE_A)) / D_TRUE_A
    print(f"BJK |dev|/D={db:.2f}  NMT |dev|/D={dn:.2f}")
    results_a[dlell] = dict(ell_b=ell_b, dl_b=dl_b, sig_b=sig_b, ok_b=ok_b,
                             ell_n=ell_n, dl_n=dl_n, ok_n=ok_n, d_true=D_TRUE_A)

# Run Case C (intermediate sky, NSIDE=32, synthetic)
print(f"\n{'='*60}")
print(f"Case C: synthetic partial sky  NSIDE={NSIDE_C}  LMAX={LMAX_C}  "
      f"f_sky≈{fsky_c:.4f}  ℓ_char≈{ell_ch_c:.1f}")
print('='*60)
results_c = {}
for dlell in DELL_LIST_C:
    edges_ = band_edges(LMIN_C, LMAX_C, dlell)
    nbands = len(edges_) - 1
    n_kern = nbands * n_obs_c**2 * 8 / 1e9
    print(f"  Δℓ={dlell:3d} ({nbands} bands, ~{n_kern:.2f} GB kernels)",
          end='  ', flush=True)

    ell_b, dl_b, sig_b, ok_b = run_bjk(data_map_c, ninv_c,
                                         LMIN_C, LMAX_C, edges_, D_TRUE_C)
    ell_n, dl_n, ok_n = run_namaster(data_map_c, mask_c_nmt,
                                      LMIN_C, LMAX_C, LMAX_FIELD_C,
                                      edges_, N_L_C, cl_tot_th_c)
    with np.errstate(invalid='ignore', divide='ignore'):
        db = np.nanmax(np.abs(dl_b - D_TRUE_C)) / D_TRUE_C
        dn = np.nanmax(np.abs(dl_n - D_TRUE_C)) / D_TRUE_C
    print(f"BJK |dev|/D={db:.2f}  NMT |dev|/D={dn:.2f}")
    results_c[dlell] = dict(ell_b=ell_b, dl_b=dl_b, sig_b=sig_b, ok_b=ok_b,
                             ell_n=ell_n, dl_n=dl_n, ok_n=ok_n, d_true=D_TRUE_C)

# Run Case B (partial sky, NSIDE=128, almasim data)
results_b = {}
if have_b:
    print(f"\n{'='*60}")
    print(f"Case B: almasim partial sky  NSIDE={NSIDE_B}  LMAX={LMAX_B}  "
          f"f_sky≈{fsky_b:.4f}  ℓ_char≈{ell_ch_b:.1f}")
    print('='*60)
    for dlell in DELL_LIST_B:
        edges_ = band_edges(LMIN_B, LMAX_B, dlell)
        nbands = len(edges_) - 1
        n_kern = nbands * (int(mask_b.sum()))**2 * 8 / 1e9
        print(f"  Δℓ={dlell:3d} ({nbands} bands, ~{n_kern:.2f} GB kernels)",
              end='  ', flush=True)

        ninv_b = ninv_map_b.copy()
        ell_b, dl_b, sig_b, ok_b = run_bjk(data_map_b, ninv_b,
                                             LMIN_B, LMAX_B, edges_, D_TRUE_B)
        ell_n, dl_n, ok_n = run_namaster(data_map_b, mask_b_nmt,
                                          LMIN_B, LMAX_B, LMAX_FIELD_B,
                                          edges_, N_L_B, cl_tot_th_b)
        with np.errstate(invalid='ignore', divide='ignore'):
            db = np.nanmax(np.abs(dl_b - D_TRUE_B)) / D_TRUE_B
            dn = np.nanmax(np.abs(dl_n - D_TRUE_B)) / D_TRUE_B
        print(f"BJK |dev|/D={db:.2f}  NMT |dev|/D={dn:.2f}")
        results_b[dlell] = dict(ell_b=ell_b, dl_b=dl_b, sig_b=sig_b, ok_b=ok_b,
                                 ell_n=ell_n, dl_n=dl_n, ok_n=ok_n, d_true=D_TRUE_B)

# ---------------------------------------------------------------------------
# Summary tables
def print_table(label, results, dell_list, d_true):
    print(f"\n{label}")
    print(f"{'Δℓ':>5}  {'n_bands':>7}  "
          f"{'BJK |dev|/D':>12}  {'BJK |dev|/σ':>12}  "
          f"{'NMT |dev|/D':>12}  {'NMT neg?':>8}")
    print("-"*62)
    for dlell in dell_list:
        if dlell not in results:
            continue
        r = results[dlell]
        dl_b, sig_b, dl_n = r['dl_b'], r['sig_b'], r['dl_n']
        with np.errstate(invalid='ignore', divide='ignore'):
            dev_b  = np.nanmax(np.abs(dl_b - d_true)) / d_true
            devsig = np.nanmax(np.abs(dl_b - d_true) / (np.abs(sig_b) + 1e-40))
            dev_n  = np.nanmax(np.abs(dl_n - d_true)) / d_true
        neg_n  = bool(np.any(dl_n < 0))
        nbands = len(r['ell_b'])
        ok_b   = r['ok_b']
        print(f"{dlell:>5}  {nbands:>7}  "
              f"{dev_b:>12.2f}  {devsig:>12.2f}  "
              f"{dev_n:>12.2f}  {'YES' if neg_n else 'no':>8}"
              f"{'  [BJK FAIL]' if not ok_b else ''}")

print()
print_table(f"Case A: full sky (NSIDE={NSIDE_A}, LMAX={LMAX_A})",
            results_a, DELL_LIST_A, D_TRUE_A)
if results_c:
    print_table(f"Case C: synthetic partial sky "
                f"(NSIDE={NSIDE_C}, LMAX={LMAX_C}, f_sky≈{fsky_c:.4f})",
                results_c, DELL_LIST_C, D_TRUE_C)
if results_b:
    print_table(f"Case B: almasim partial sky "
                f"(NSIDE={NSIDE_B}, LMAX={LMAX_B}, f_sky≈{fsky_b:.4f})",
                results_b, DELL_LIST_B, D_TRUE_B)

# ---------------------------------------------------------------------------
# Figure: 3 rows (A, C, B), each with its own column count

all_cases = [
    ('a', results_a, DELL_LIST_A, D_TRUE_A, LMIN_A, LMAX_A,
     f'Full sky\nNSIDE={NSIDE_A}', 1.0, np.inf),
    ('c', results_c, DELL_LIST_C, D_TRUE_C, LMIN_C, LMAX_C,
     f'Synth. partial\n($f_{{\\rm sky}}$≈{fsky_c:.2f}, NSIDE={NSIDE_C})',
     fsky_c, ell_ch_c),
]
if results_b:
    all_cases.append(
        ('b', results_b, DELL_LIST_B, D_TRUE_B, LMIN_B, LMAX_B,
         f'almasim partial\n($f_{{\\rm sky}}$≈{fsky_b:.3f}, NSIDE={NSIDE_B})',
         fsky_b, ell_ch_b)
    )

ncols_max = max(len(r[2]) for r in all_cases)
nrows     = len(all_cases)
fig_w = ncols_max * 3.1
fig_h = nrows * 3.0 + 0.9

fig  = plt.figure(figsize=(fig_w, fig_h))
gs   = fig.add_gridspec(nrows, ncols_max, hspace=0.55, wspace=0.30)

for row, (tag, results, dell_list, d_true, lmin, lmax, row_label,
          fsky, ell_char) in enumerate(all_cases):

    active_cols = [c for c, d in enumerate(dell_list) if d in results]

    for col, dlell in enumerate(dell_list):
        if dlell not in results:
            continue
        ax = fig.add_subplot(gs[row, col])
        r  = results[dlell]

        ax.axhline(d_true, color='k', lw=1.2, ls='-', zorder=0)
        ax.axhline(0,      color='k', lw=0.4, ls=':', zorder=0)
        if np.isfinite(ell_char) and lmin < ell_char < lmax:
            ax.axvline(ell_char, color='grey', lw=0.8, ls='--', zorder=0)

        ell_n, dl_n = r['ell_n'], r['dl_n']
        ax.plot(ell_n, dl_n, 's', color='darkorange', ms=4, zorder=2)

        ell_b, dl_b, sig_b = r['ell_b'], r['dl_b'], r['sig_b']
        failed = not r['ok_b']
        bjk_color = 'steelblue' if not failed else 'lightblue'
        ax.errorbar(ell_b, dl_b, yerr=sig_b, fmt='o', color=bjk_color,
                    ms=3, lw=0.8, capsize=2, zorder=3)
        if failed:
            ax.text(0.5, 0.97, 'BJK: no conv.', transform=ax.transAxes,
                    ha='center', va='top', fontsize=6.5, color='red')

        # Y limits: centre on d_true, scale to BJK σ; clip NaMaster excursions
        sig_scale = float(np.nanmax(np.abs(sig_b))) if not np.all(np.isnan(sig_b)) else d_true
        ylo = min(-1.5 * sig_scale, -d_true)
        yhi = max(d_true + 2.5 * sig_scale, 3.0 * d_true)
        if not np.all(np.isnan(dl_n)):
            ylo = max(min(ylo, float(np.nanmin(dl_n)) * 1.3), -8 * d_true)
            yhi = min(max(yhi, float(np.nanmax(dl_n)) * 1.3),  8 * d_true)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(lmin - 0.5, lmax + 0.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f'Δℓ = {dlell}', fontsize=9)
        ax.set_xlabel(r'$\ell$', fontsize=8)
        if col == 0:
            ax.set_ylabel(row_label + '\n' + r'$D_\ell$', fontsize=7)

handles = [
    Line2D([0], [0], marker='o', color='steelblue', lw=1, ms=4,
           label='BJK (Fisher σ)'),
    Line2D([0], [0], marker='s', color='darkorange', lw=0, ms=4,
           label='NaMaster ($D_\\ell$ bins)'),
    Line2D([0], [0], color='k', lw=1.2, label=r'$D_\ell^{\rm true}$'),
    Line2D([0], [0], color='grey', lw=0.8, ls='--',
           label=r'$\ell_{\rm char}=\pi/(2\sqrt{f_{\rm sky}})$'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    'BJK ($D_\\ell$-model) vs NaMaster: band-width × sky-fraction\n'
    r'Signal: flat $D_\ell = A/2\pi$;  NMT uses $D_\ell$ bins (is\_Dell=True)',
    fontsize=10)
fig.subplots_adjust(bottom=0.10, top=0.88)

out_png = os.path.join(os.path.dirname(__file__), 'test_sky_comparison.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_png}")
plt.close()
