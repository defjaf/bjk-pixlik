"""
Compare BJK98 ML estimates with Almanac HMC posterior for Euclid RR2 data.

Compares EE, BB, and EB bandpowers from:
- BJK98: Maximum likelihood point estimate + Fisher uncertainties
- Almanac: HMC posterior mean + posterior standard deviation

Run from repo root:
    python3 examples/compare_bjk_almanac_eb.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOMBIN = -1
NSIDE  = 128

ALMANAC_DIR = os.path.expanduser('~/Desktop/Projects/Almanac/Euclid-Almanac/almanac_runs')
BJK_DIR = os.path.join(ALMANAC_DIR, 'bjk_results')

# Input files
BJK_DAT = os.path.join(BJK_DIR, f'bjk_euclid_tombin{TOMBIN}_nside{NSIDE}_eb_n1.dat')
ALM_RUN = f'euclid_rr2_tombin{TOMBIN}_nside{NSIDE}_bandpower_v1'
ALM_ELLIP = os.path.join(ALMANAC_DIR, f'{ALM_RUN}.ellip.extract.npy')

# Output
OUT_PNG = os.path.join(BJK_DIR, f'compare_bjk_almanac_tombin{TOMBIN}_nside{NSIDE}_eb.png')
OUT_TXT = os.path.join(BJK_DIR, f'compare_bjk_almanac_tombin{TOMBIN}_nside{NSIDE}_eb.txt')

BAND_EDGES = np.array([2, 34, 66, 98, 130, 162, 194, 226, 257])
ell_b = 0.5 * (BAND_EDGES[:-1] + BAND_EDGES[1:] - 1)
# ---------------------------------------------------------------------------


def load_bjk_results(filename):
    """Load BJK output: ell_band spec i j C_l sigma SNR"""
    data = {'EE': [], 'BB': [], 'EB': []}
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            ell, spec = float(parts[0]), parts[1]
            cl, sig = float(parts[4]), float(parts[5])
            data[spec].append((ell, cl, sig))

    result = {}
    for spec in data:
        if data[spec]:
            arr = np.array(data[spec])
            result[spec] = {'ell': arr[:, 0], 'cl': arr[:, 1], 'sigma': arr[:, 2]}
    return result


def load_almanac_bandpowers(ellip_file):
    """
    Extract bandpowers from Almanac ellip.extract.npy file.

    Almanac stores bandpowers in Cholesky (ellip) parameterization:
    - Multi-field nf=2: 3 params per band (λ_EE, λ_EB, λ_BB)
    - Conversion:
        θ_EE = exp(2*λ_00)
        θ_EB = λ_10 * exp(λ_00)
        θ_BB = λ_10² + exp(2*λ_11)
    """
    print(f"Loading Almanac bandpowers from: {ellip_file}")
    ellip = np.load(ellip_file)  # (nsamples, nbands * 3)
    print(f"  Shape: {ellip.shape}")

    nsamples = ellip.shape[0]
    nbands = len(BAND_EDGES) - 1
    nparams_per_band = 3  # EE, EB, BB

    if ellip.shape[1] != nbands * nparams_per_band:
        raise ValueError(f"Expected {nbands * nparams_per_band} columns, got {ellip.shape[1]}")

    # Convert from λ to θ (bandpower) space
    theta_samples = np.zeros((nsamples, nbands, 3))  # (samples, bands, [EE, EB, BB])

    for b in range(nbands):
        offset = b * nparams_per_band
        lam_ee = ellip[:, offset + 0]     # λ(0,0)
        lam_eb = ellip[:, offset + 1]     # λ(1,0)
        lam_bb = ellip[:, offset + 2]     # λ(1,1)

        theta_ee = np.exp(2.0 * lam_ee)
        theta_eb = lam_eb * np.exp(lam_ee)
        theta_bb = lam_eb**2 + np.exp(2.0 * lam_bb)

        theta_samples[:, b, 0] = theta_ee
        theta_samples[:, b, 1] = theta_eb
        theta_samples[:, b, 2] = theta_bb

    # Compute posterior statistics
    theta_mean = theta_samples.mean(axis=0)  # (nbands, 3)
    theta_std = theta_samples.std(axis=0)

    result = {
        'EE': {'ell': ell_b, 'cl': theta_mean[:, 0], 'sigma': theta_std[:, 0]},
        'EB': {'ell': ell_b, 'cl': theta_mean[:, 1], 'sigma': theta_std[:, 1]},
        'BB': {'ell': ell_b, 'cl': theta_mean[:, 2], 'sigma': theta_std[:, 2]},
    }

    return result


def compute_chi2(bjk, alm):
    """
    Compute χ² between BJK and Almanac estimates.

    χ² = Σ (C_bjk - C_alm)² / (σ_bjk² + σ_alm²)
    """
    chi2_stats = {}
    for spec in ['EE', 'BB', 'EB']:
        if spec not in bjk or spec not in alm:
            continue

        cl_bjk = bjk[spec]['cl']
        sig_bjk = bjk[spec]['sigma']
        cl_alm = alm[spec]['cl']
        sig_alm = alm[spec]['sigma']

        # Combined uncertainty
        sig_combined = np.sqrt(sig_bjk**2 + sig_alm**2)

        # Chi-square
        diff = cl_bjk - cl_alm
        chi2 = np.sum((diff / sig_combined)**2)
        ndof = len(diff)

        chi2_stats[spec] = {
            'chi2': chi2,
            'ndof': ndof,
            'chi2_per_dof': chi2 / ndof,
            'diff': diff,
            'sig_combined': sig_combined,
            'pull': diff / sig_combined
        }

    return chi2_stats


def main():
    print("="*70)
    print("BJK vs Almanac Comparison: Euclid RR2 EE+BB+EB")
    print("="*70)

    # Load results
    print(f"\nLoading BJK results from: {BJK_DAT}")
    if not os.path.exists(BJK_DAT):
        print(f"ERROR: BJK results not found!")
        print(f"Run: python3 examples/run_bjk_euclid_eb_n1.py first")
        return
    bjk = load_bjk_results(BJK_DAT)

    if not os.path.exists(ALM_ELLIP):
        print(f"ERROR: Almanac results not found at: {ALM_ELLIP}")
        return
    alm = load_almanac_bandpowers(ALM_ELLIP)

    # Compute chi-square
    print("\n" + "="*70)
    print("CHI-SQUARE COMPARISON")
    print("="*70)
    chi2_stats = compute_chi2(bjk, alm)

    with open(OUT_TXT, 'w') as f:
        f.write("BJK vs Almanac Comparison\n")
        f.write("="*70 + "\n\n")

        for spec in ['EE', 'BB', 'EB']:
            if spec not in chi2_stats:
                continue
            stats = chi2_stats[spec]

            msg = f"{spec}:\n"
            msg += f"  χ² = {stats['chi2']:.2f} ({stats['ndof']} bands)\n"
            msg += f"  χ²/dof = {stats['chi2_per_dof']:.3f}\n"
            msg += f"  RMS pull = {np.sqrt(np.mean(stats['pull']**2)):.3f}\n"
            msg += f"  Max |pull| = {np.abs(stats['pull']).max():.3f}\n"

            print(msg)
            f.write(msg + "\n")

        # Band-by-band comparison
        f.write("\n" + "="*70 + "\n")
        f.write("BAND-BY-BAND COMPARISON\n")
        f.write("="*70 + "\n\n")

        for spec in ['EE', 'BB', 'EB']:
            if spec not in bjk:
                continue
            f.write(f"\n{spec}:\n")
            f.write("  ell_b    BJK_mean    BJK_sigma    ALM_mean    ALM_sigma    diff/sigma\n")

            for i in range(len(ell_b)):
                f.write(f"  {ell_b[i]:5.0f}  {bjk[spec]['cl'][i]:10.3e}  "
                       f"{bjk[spec]['sigma'][i]:10.3e}  {alm[spec]['cl'][i]:10.3e}  "
                       f"{alm[spec]['sigma'][i]:10.3e}  {chi2_stats[spec]['pull'][i]:7.3f}\n")

    print(f"\nSaved text comparison: {OUT_TXT}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'EE': 'steelblue', 'BB': 'darkorange', 'EB': 'green'}

    for col, spec in enumerate(['EE', 'BB', 'EB']):
        if spec not in bjk:
            continue

        ax_cl = axes[0, col]
        ax_diff = axes[1, col]

        # Top: C_ell comparison
        ax_cl.errorbar(ell_b - 2, bjk[spec]['cl'], yerr=bjk[spec]['sigma'],
                      fmt='o', color=colors[spec], capsize=4, ms=6, lw=1.5,
                      label='BJK ML ± Fisher σ', alpha=0.8)
        ax_cl.errorbar(ell_b + 2, alm[spec]['cl'], yerr=alm[spec]['sigma'],
                      fmt='s', color='black', capsize=4, ms=5, lw=1.5,
                      label='Almanac HMC ± post. σ', alpha=0.6)
        ax_cl.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
        ax_cl.set_xlabel(r'$\ell$')
        ax_cl.set_ylabel(rf'$C_\ell^{{{spec}}}$  [rad$^2$]')
        ax_cl.set_title(rf'{spec}: BJK vs Almanac')
        ax_cl.legend(fontsize=9)
        ax_cl.grid(True, alpha=0.3)

        # Bottom: Residuals (pull plot)
        stats = chi2_stats[spec]
        ax_diff.errorbar(ell_b, stats['pull'], yerr=np.ones_like(ell_b),
                        fmt='o', color=colors[spec], capsize=4, ms=6, lw=1.5)
        ax_diff.axhline(0, color='k', lw=1, ls='-')
        ax_diff.axhline(+2, color='r', lw=0.5, ls='--', alpha=0.5)
        ax_diff.axhline(-2, color='r', lw=0.5, ls='--', alpha=0.5)
        ax_diff.fill_between([0, 300], -1, 1, color='gray', alpha=0.2)
        ax_diff.set_xlabel(r'$\ell$')
        ax_diff.set_ylabel(r'$(C_{BJK} - C_{ALM})/\sigma_{combined}$')
        ax_diff.set_title(rf'Pull: χ²/dof = {stats["chi2_per_dof"]:.2f}')
        ax_diff.grid(True, alpha=0.3)
        ax_diff.set_ylim(-4, 4)

    fig.suptitle(f'BJK vs Almanac: Euclid RR2 tombin {TOMBIN}, NSIDE={NSIDE} (EE+BB+EB)',
                 y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {OUT_PNG}")
    plt.close()

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == '__main__':
    main()
