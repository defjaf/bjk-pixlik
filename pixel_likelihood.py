"""
BJK98 pixel-space marginalised likelihood for angular power spectrum estimation.

Reference: Bond, Jaffe & Knox 1998, PhysRevD 57, 2117  (10.1103/physrevd.57.2117)

Model:  d ~ N(0, C(cl) + N)

Log-likelihood: L = -1/2 [log|M| + d^T M^{-1} d],  M = C + N

Newton-Raphson (BJK98 eq. 14-15):
  g_b  = 1/2 [d^T M^{-1} K_b M^{-1} d  -  Tr(M^{-1} K_b)]
  F_bb' = 1/2 Tr[M^{-1} K_b M^{-1} K_b']
  delta = F^{-1} g

All linear algebra uses Cholesky decomposition (M = L L^T) rather than
explicit matrix inversion, for numerical stability and efficiency.

Field types and parameter ordering
-----------------------------------
Supports any combination of n_T spin-0 (T) bins and n_P spin-2 (P) bins.
Data vector:
  d = [ T_0 | T_1 | ... | Q_0 U_0 | Q_1 U_1 | ... ]
      ←  n_T * n_obs  →  ←  n_P * 2 * n_obs  →
Total N_d = (n_T + 2*n_P) * n_obs.

Bandpower parameter vector (flat, length n_params):
  TT pairs  (i,j), i<=j   : n_T*(n_T+1)/2 pairs
  TE pairs  (i,j) all i,j : n_T*n_P pairs
  TB pairs  (if include_TB): n_T*n_P pairs
  EE pairs  (i,j), i<=j   : n_P*(n_P+1)/2 pairs
  BB pairs  (i,j), i<=j   : n_P*(n_P+1)/2 pairs
  EB pairs  (if include_EB): n_P*(n_P+1)/2 pairs
Each group repeated for nbands, so parameter layout is:
  [TT(0,0,0), TT(0,0,1), ..., TT(0,0,nb-1),  TT(0,1,0), ...,  TE(0,0,0), ...]

Memory: O(N_d^2).  For n_T=n_P=6, n_obs~2000: N_d~36000; N_d^2~1e9 doubles ~8 GB.
For n_T=n_P<=2 or n_obs<=~500, comfortably under 16 GB.

** MEMORY WARNING for spin-2 multi-field (n_P >= 1, include_EB=False) **
With N_d = 2*n_obs and n_params kernels, all kernel matrices are allocated upfront:
  n_params × N_d² × 8 bytes.
Example: n_P=1, include_EB=False, 7 bands → 21 kernels × (2×4900)² × 8 ≈ 16 GB stored,
plus substantial temporary arrays during kernel construction — peak RAM usage can
exceed 50 GB at NSIDE=64/fsky=0.1.  On a 64 GB machine this will OOM.

Mitigation options:
  1. Reduce NSIDE (NSIDE=32 → n_obs ≈ 1000 → ~1 GB total).
  2. On-the-fly kernels: build each kernel inside gradient_and_fisher and discard
     immediately.  Reduces peak to ~2 GB at the cost of ~30 rebuilds per Newton run.
     This fix is NOT yet implemented.

** MULTI-FIELD STATUS: run_bjk_nmt.py wiring for --include-eb is written but UNTESTED **
(crashed with OOM before completing a single Newton iteration, June 2026).
Single-field TT and EE/BB (without include_EB) are validated and production-ready.

Complexity per Newton-Raphson iteration: O(n_params * N_d^3) dominated by the
triangular solves to form A_b = L^{-1} K_b (L^{-1})^T for each kernel K_b.
"""

import numpy as np
import healpy as hp
from scipy.linalg import solve_triangular


# ===========================================================================
# Spin-0 (scalar) kernels
# ===========================================================================

def _build_scalar_kernels(obs_pix, nside, lmin, lmax, band_edges, beam2,
                          ell_weights=None):
    """
    Build band kernels for the scalar (spin-0) pixel likelihood.

    With ell_weights=None (default): C_l = C_b = const within band b.
      K_b[p,p'] = sum_{l in b} (2l+1)/(4pi) B_l^2 P_l(cos theta)

    With ell_weights[l] = 2pi/(l*(l+1)): D_l = D_b = const within band b.

    Uses Legendre recurrence: O(n_obs^2 * lmax) fully vectorised.
    """
    if ell_weights is None:
        ell_weights = np.ones(lmax + 1)

    vx, vy, vz = hp.pix2vec(nside, obs_pix)
    vec = np.column_stack([vx, vy, vz])
    n = len(obs_pix)

    # Exploit pixelisation regularity: many pixel pairs share the same cos θ.
    # Run the Legendre recurrence only on unique values, then expand.
    cos_theta_full = np.clip(vec @ vec.T, -1.0, 1.0).ravel()
    unique_ct, inverse = np.unique(cos_theta_full, return_inverse=True)
    inverse = inverse.astype(np.int32)   # halve index memory
    del cos_theta_full

    nbands = len(band_edges) - 1
    kernels = [np.zeros(len(unique_ct)) for _ in range(nbands)]

    def _add(l, Pl):
        if l < lmin or l > lmax:
            return
        b = int(np.searchsorted(band_edges, l, side='right')) - 1
        if 0 <= b < nbands:
            kernels[b] += (2*l + 1) / (4*np.pi) * beam2[l] * ell_weights[l] * Pl

    x = unique_ct
    Pl_prev = np.ones_like(x)
    Pl_curr = x.copy()
    _add(0, Pl_prev)
    _add(1, Pl_curr)
    for l in range(2, lmax + 1):
        Pl_next = ((2*l - 1) * x * Pl_curr - (l - 1) * Pl_prev) / l
        _add(l, Pl_next)
        Pl_prev, Pl_curr = Pl_curr, Pl_next

    return [k[inverse].reshape(n, n) for k in kernels]


# ===========================================================================
# Spin-2 geometry: pixel-pair position angles
# ===========================================================================

def _compute_spin2_geometry(obs_pix, nside):
    """
    For every pair (p, p') of observed pixels compute position angles.

    Returns a 6-tuple (cos2dpsi, sin2dpsi, cos2spsi, sin2spsi, c2j, s2j):
      dpsi = psi_p - psi_p'   (difference angle at pixel p)
      spsi = psi_p + psi_p'   (sum angle)
      c2j = cos(2*psi_{p'})   (angle at the P-field pixel, needed for TE/TB)
      s2j = sin(2*psi_{p'})

    All shape (n_obs, n_obs).
    """
    theta, phi = hp.pix2ang(nside, obs_pix)
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    vec = np.column_stack([vx, vy, vz])          # (n, 3)

    Nx = np.cos(theta) * np.cos(phi)
    Ny = np.cos(theta) * np.sin(phi)
    Nz = -np.sin(theta)
    North = np.column_stack([Nx, Ny, Nz])         # (n, 3)

    Ex = -np.sin(phi)
    Ey = np.cos(phi)
    East = np.column_stack([Ex, Ey, np.zeros(len(phi))])  # (n, 3)

    cos_theta = np.clip(vec @ vec.T, -1.0, 1.0)   # (n, n)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    safe_denom = np.where(sin_theta > 1e-10, sin_theta, 1.0)

    VN = vec @ North.T                             # (n,n): VN[i,j] = vec[j].North[i]
    vn = (vec * North).sum(1)                      # (n,):  = 0 (tangent plane)
    cos_psi_i = (VN - cos_theta * vn[:, None]) / safe_denom

    VE = vec @ East.T                              # (n,n): VE[i,j] = vec[j].East[i]
    ve = (vec * East).sum(1)                       # (n,): = 0
    sin_psi_i = (VE - cos_theta * ve[:, None]) / safe_denom

    cos_psi_j = (VN.T - cos_theta * vn[None, :]) / safe_denom
    sin_psi_j = (VE.T - cos_theta * ve[None, :]) / safe_denom

    c2i = 2*cos_psi_i**2 - 1
    s2i = 2*cos_psi_i * sin_psi_i
    c2j = 2*cos_psi_j**2 - 1
    s2j = 2*cos_psi_j * sin_psi_j

    cos2dpsi = c2i * c2j + s2i * s2j
    sin2dpsi = s2i * c2j - c2i * s2j
    cos2spsi = c2i * c2j - s2i * s2j
    sin2spsi = s2i * c2j + c2i * s2j

    return cos2dpsi, sin2dpsi, cos2spsi, sin2spsi, c2j, s2j


# ===========================================================================
# Spin-2 Wigner d-matrix recurrences
# ===========================================================================

def _wigner_d22_recurrence(x, lmax):
    """
    Compute d^l_{2,2}(x) and d^l_{2,-2}(x) for l=2..lmax.
    Returns dp[l], dm[l] lists indexed from l=0 (entries below l=2 are zero arrays).
    """
    x = np.asarray(x, dtype=float).ravel()
    dp = [None] * (lmax + 1)
    dm = [None] * (lmax + 1)

    dp[1] = np.zeros_like(x)
    dm[1] = np.zeros_like(x)
    dp[2] = ((1.0 + x) / 2.0) ** 2
    dm[2] = ((1.0 - x) / 2.0) ** 2

    for l in range(2, lmax):
        lp1 = l + 1
        prefac  = lp1 / (lp1**2 - 4.0)
        coeff1_p = (2*l + 1) * (x - 4.0 / (l * lp1))
        coeff1_m = (2*l + 1) * (x + 4.0 / (l * lp1))
        coeff2   = (l**2 - 4.0) / l
        dp[lp1] = prefac * (coeff1_p * dp[l] - coeff2 * dp[l-1])
        dm[lp1] = prefac * (coeff1_m * dm[l] - coeff2 * dm[l-1])

    return dp, dm


def _wigner_d20_recurrence(x, lmax):
    """
    Compute d^l_{2,0}(x) for l=2..lmax.
    """
    x  = np.asarray(x, dtype=float).ravel()
    dx = [None] * (lmax + 1)
    dx[1] = np.zeros_like(x)
    dx[2] = np.sqrt(3.0 / 8.0) * (1.0 - x**2)

    for l in range(2, lmax):
        lp1 = l + 1
        prefac  = 1.0 / np.sqrt(float(lp1**2 - 4))
        dx[lp1] = prefac * ((2*l + 1) * x * dx[l] - np.sqrt(float(l**2 - 4)) * dx[l - 1])

    return dx


def _build_spin2_kernels(obs_pix, nside, lmin, lmax, band_edges, beam2,
                          ell_weights=None):
    """
    Build the Wigner d-matrix band kernels and pixel-pair geometry.

    With ell_weights=None (default): C_l = C_b = const within band b.
    With ell_weights[l] = 2pi/(l*(l+1)): D_l = D_b = const within band b.

    Returns:
      Kp_bands, Km_bands, Kx_bands : lists of (n_obs, n_obs) per band
        Kp[b] = sum_{l in b} (2l+1)/(8pi) B_l^2 ell_weights[l] d^l_{2,+2}(cos theta)
        Km[b] = same for d^l_{2,-2}
        Kx[b] = same for d^l_{2,0}   [used for EB; TE uses 2*Kx]
      geom : 6-tuple from _compute_spin2_geometry
    """
    if ell_weights is None:
        ell_weights = np.ones(lmax + 1)
    n = len(obs_pix)
    vx, vy, vz = hp.pix2vec(nside, obs_pix)
    vec = np.column_stack([vx, vy, vz])
    cos_theta = np.clip(vec @ vec.T, -1.0, 1.0)

    print("  Computing Wigner d-matrix recurrence...")
    x_flat = cos_theta.ravel()
    dp, dm = _wigner_d22_recurrence(x_flat, lmax)
    dx     = _wigner_d20_recurrence(x_flat, lmax)

    nbands = len(band_edges) - 1
    Kp_bands = [np.zeros(n * n) for _ in range(nbands)]
    Km_bands = [np.zeros(n * n) for _ in range(nbands)]
    Kx_bands = [np.zeros(n * n) for _ in range(nbands)]

    for l in range(lmin, lmax + 1):
        b = int(np.searchsorted(band_edges, l, side='right')) - 1
        if not (0 <= b < nbands):
            continue
        fac = (2*l + 1) / (8*np.pi) * beam2[l] * ell_weights[l]
        Kp_bands[b] += fac * dp[l]
        Km_bands[b] += fac * dm[l]
        Kx_bands[b] += fac * dx[l]

    Kp_bands = [k.reshape(n, n) for k in Kp_bands]
    Km_bands = [k.reshape(n, n) for k in Km_bands]
    Kx_bands = [k.reshape(n, n) for k in Kx_bands]

    print("  Computing pixel-pair geometry (position angles)...")
    geom = _compute_spin2_geometry(obs_pix, nside)

    return Kp_bands, Km_bands, Kx_bands, geom


# ===========================================================================
# TE / TB cross-field kernel blocks
# ===========================================================================

def _build_te_block(Kx_b, c2j, s2j):
    """
    TE derivative matrix block for T_i vs P_j: shape (n_obs, 2*n_obs).

    C^{Ti,Qj}[p,p'] = -K_TE * cos(2*psi at P pixel p' toward T pixel p)
    C^{Ti,Uj}[p,p'] = -K_TE * sin(2*psi at P pixel p' toward T pixel p)

    c2j[i,j] is the angle at pixel i (T pixel) toward pixel j (P pixel), so
    the angle at the P pixel toward T is c2j.T.
    The minus sign comes from HEALPix's convention a_E = -Re(alm transform).
    Reference: Zaldarriaga & Seljak 1997, Appendix A.
    """
    K = 2.0 * Kx_b
    return np.concatenate([-K * c2j.T, -K * s2j.T], axis=1)


def _build_tb_block(Kx_b, c2j, s2j):
    """
    TB derivative matrix block for T_i vs P_j: shape (n_obs, 2*n_obs).

    C^{Ti,Qj}[p,p'] = +K_TE * sin(2*psi at P pixel p' toward T pixel p)
    C^{Ti,Uj}[p,p'] = -K_TE * cos(2*psi at P pixel p' toward T pixel p)

    MC-verified in derivations/verify_eb_tb.py (rms_rel ~6% = MC noise floor).
    Signs confirmed by parity: TB = TE with E→B (cos→sin, sin→-cos) plus the
    HEALPix a_E = -Re(...) convention from Zaldarriaga & Seljak 1997.
    """
    K = 2.0 * Kx_b
    return np.concatenate([K * s2j.T, -K * c2j.T], axis=1)


# ===========================================================================
# Spin-2 covariance block assembly (P_i vs P_j)
# ===========================================================================

def _spin2_cov_block(Kp_b, Km_b, Kx_b, geom, cEE, cBB=0.0, cEB=0.0):
    """
    Assemble the 2*n_obs x 2*n_obs signal covariance block for one band,
    one pair of P bins, at bandpower values cEE, cBB, cEB.
    """
    cos2dp, sin2dp, cos2sp, sin2sp, _, _ = geom
    QQ = cEE * (Kp_b * cos2dp + Km_b * cos2sp)
    QU = cEE * (Kp_b * sin2dp + Km_b * sin2sp)
    UU = cEE * (Kp_b * cos2dp - Km_b * cos2sp)

    QQ += cBB * (Kp_b * cos2dp - Km_b * cos2sp)
    QU += cBB * (Kp_b * sin2dp - Km_b * sin2sp)
    UU += cBB * (Kp_b * cos2dp + Km_b * cos2sp)

    if cEB != 0.0:
        QQ -= cEB * Km_b * sin2sp
        QU += cEB * Km_b * cos2sp
        UU += cEB * Km_b * sin2sp

    return np.block([[QQ, QU], [QU.T, UU]])


# ===========================================================================
# Derivative kernel matrices for each spectrum type
# ===========================================================================

def _ee_kernel(Kp_b, Km_b, geom):
    """EE derivative kernel: d(C_PP)/d(C^EE_b), shape (2n, 2n)."""
    cos2dp, sin2dp, cos2sp, sin2sp, _, _ = geom
    QQ =  Kp_b * cos2dp + Km_b * cos2sp
    QU =  Kp_b * sin2dp + Km_b * sin2sp
    UU =  Kp_b * cos2dp - Km_b * cos2sp
    return np.block([[QQ, QU], [QU.T, UU]])


def _bb_kernel(Kp_b, Km_b, geom):
    """BB derivative kernel: d(C_PP)/d(C^BB_b), shape (2n, 2n)."""
    cos2dp, sin2dp, cos2sp, sin2sp, _, _ = geom
    QQ =  Kp_b * cos2dp - Km_b * cos2sp
    QU =  Kp_b * sin2dp - Km_b * sin2sp
    UU =  Kp_b * cos2dp + Km_b * cos2sp
    return np.block([[QQ, QU], [QU.T, UU]])


def _eb_kernel(Km_b, geom):
    """EB derivative kernel: d(C_PP)/d(C^EB_b), shape (2n, 2n).

    Derived from the HEALPix spin-2 convention (Q+iU) = -sum (a_E+ia_B) _{+2}Y:
    B-mode maps satisfy Q_B = -U_E (the -i factor introduces a sign flip vs E).
    Only the d^l_{2,-2} (Km) Wigner kernel with sum angles contributes; d^l_{2,0}
    (Kx) does not appear.  MC-verified in derivations/verify_eb_tb.py.
    """
    _, _, cos2sp, sin2sp, _, _ = geom
    QQ = -Km_b * sin2sp
    QU =  Km_b * cos2sp
    UU =  Km_b * sin2sp
    return np.block([[QQ, QU], [QU.T, UU]])


# ===========================================================================
# SpectraLayout: parameter index bookkeeping
# ===========================================================================

class SpectraLayout:
    """
    Maps (spec_type, bin_i, bin_j, band_b) <-> flat parameter index.

    Parameter groups (in order within each band):
      'TT': n_T*(n_T+1)/2 pairs (i,j) with i<=j
      'TE': n_T*n_P pairs (i,j) all i in [0,n_T), j in [0,n_P)
      'TB': n_T*n_P pairs  (only if include_TB)
      'EE': n_P*(n_P+1)/2 pairs (i,j) with i<=j
      'BB': n_P*(n_P+1)/2 pairs
      'EB': n_P*(n_P+1)/2 pairs  (only if include_EB)

    Ordering within each group: pairs first, then bands.
    Parameter index = group_offset + pair_index * nbands + b
    """

    def __init__(self, n_T, n_P, nbands, include_TB=False, include_EB=False):
        self.n_T = n_T
        self.n_P = n_P
        self.nbands = nbands
        self.include_TB = include_TB
        self.include_EB = include_EB

        # Build pair lists per spec type
        self._pairs = {}
        self._pairs['TT'] = [(i, j) for i in range(n_T) for j in range(i, n_T)]
        self._pairs['TE'] = [(i, j) for i in range(n_T) for j in range(n_P)]
        self._pairs['TB'] = [(i, j) for i in range(n_T) for j in range(n_P)] if include_TB else []
        self._pairs['EE'] = [(i, j) for i in range(n_P) for j in range(i, n_P)]
        self._pairs['BB'] = [(i, j) for i in range(n_P) for j in range(i, n_P)]
        self._pairs['EB'] = [(i, j) for i in range(n_P) for j in range(i, n_P)] if include_EB else []

        # Group offsets in the flat vector
        order = ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']
        self._group_offset = {}
        offset = 0
        for s in order:
            self._group_offset[s] = offset
            offset += len(self._pairs[s]) * nbands
        self._n_params = offset

        # Build decode cache: idx -> (spec, i, j, b)
        self._decode_cache = {}
        for idx, spec, i, j, b in self.entries():
            self._decode_cache[idx] = (spec, i, j, b)

    @property
    def n_params(self):
        return self._n_params

    def decode(self, idx):
        """Return (spec, i, j, b) for flat index idx."""
        return self._decode_cache[idx]

    def index(self, spec, i, j, b):
        """Return flat index for (spec, bin_i, bin_j, band_b).
        For symmetric specs (TT, EE, BB, EB) i<=j is enforced automatically."""
        if spec in ('TT', 'EE', 'BB', 'EB'):
            i, j = min(i, j), max(i, j)
        pairs = self._pairs[spec]
        pair_idx = pairs.index((i, j))
        return self._group_offset[spec] + pair_idx * self.nbands + b

    def entries(self):
        """Yield (idx, spec, i, j, b) for every parameter in order."""
        order = ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']
        for spec in order:
            for pi, (i, j) in enumerate(self._pairs[spec]):
                for b in range(self.nbands):
                    idx = self._group_offset[spec] + pi * self.nbands + b
                    yield idx, spec, i, j, b

    def spec_groups(self):
        """Yield (spec, pairs_list) for active spectrum types."""
        for s in ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']:
            if self._pairs[s]:
                yield s, self._pairs[s]


# ===========================================================================
# Main class
# ===========================================================================

class PixelLikelihood:
    """
    BJK98 pixel-space likelihood supporting arbitrary combinations of
    n_T spin-0 (T) bins and n_P spin-2 (P/Q+U) bins.

    Parameters
    ----------
    data_fits  : path to FITS data map  (or None if using from_arrays)
    ninv_fits  : path to FITS inverse-noise map  (or None if using from_arrays)
    lmin, lmax : multipole range
    band_edges : 1-D int array, length nbands+1
    n_T        : number of temperature (spin-0) bins
    n_P        : number of polarisation (spin-2) bins
    beam       : B_l array (length >= lmax+1); default 1
    band_model : 'Cl' (default), 'Dl'/'D_l', or array of ell weights
    include_TB : include TB cross-spectrum parameters (default False)
    include_EB : include EB cross-spectrum parameters (default False)
    N_matrix   : (N_d, N_d) full noise matrix; if None, uses diagonal from ninv

    Backward-compatibility shims
    ----------------------------
    spin=0        → n_T=1, n_P=0
    spin=2, ntomo → n_T=0, n_P=ntomo
    nfields       : ignored (inferred from n_T, n_P)
    """

    def __init__(self, data_fits, ninv_fits, lmin, lmax, band_edges,
                 n_T=1, n_P=0, beam=None, band_model='Cl',
                 include_TB=False, include_EB=False,
                 N_matrix=None,
                 kernel_mode='auto',
                 n_threads='auto',
                 # backward-compat
                 nfields=None, spin=None, ntomo=None, ell_weights=None):

        # --- Backward-compat shims ---
        if spin is not None:
            if spin == 0:
                n_T, n_P = 1, 0
            elif spin == 2:
                n_T = 0
                n_P = ntomo if ntomo is not None else 1
        elif ntomo is not None and n_P == 0:
            n_T, n_P = 0, ntomo

        self.lmin   = lmin
        self.lmax   = lmax
        self.band_edges = np.asarray(band_edges, dtype=int)
        self.nbands = len(band_edges) - 1
        self.n_T    = n_T
        self.n_P    = n_P
        self.include_TB = include_TB
        self.include_EB = include_EB

        if kernel_mode not in ('precompute', 'onthefly', 'auto'):
            raise ValueError(f"kernel_mode must be 'precompute', 'onthefly', or 'auto', got {kernel_mode!r}")
        self.kernel_mode = kernel_mode

        # Thread configuration
        if n_threads == 'auto':
            import os
            self.n_threads = os.cpu_count() or 1
        elif isinstance(n_threads, int) and n_threads > 0:
            self.n_threads = n_threads
        else:
            raise ValueError(f"n_threads must be 'auto' or positive int, got {n_threads!r}")

        if self.n_threads > 1:
            import multiprocessing
            # Use physical cores, not hyperthreads
            physical_cores = multiprocessing.cpu_count() // 2
            if self.n_threads > physical_cores:
                print(f"Note: Using {self.n_threads} threads but only {physical_cores} physical cores")

        self.beam2 = np.ones(lmax + 1) if beam is None else np.asarray(beam)[:lmax+1]**2

        # --- Resolve band_model → ell_weights ---
        if ell_weights is not None:
            _ew = np.asarray(ell_weights, dtype=float)
        elif isinstance(band_model, str):
            bm = band_model.replace('_', '')
            if bm == 'Cl':
                _ew = None
            elif bm == 'Dl':
                ll  = np.arange(lmax + 1, dtype=float)
                _ew = np.where(ll >= 2,
                               2.0 * np.pi / np.where(ll >= 2, ll * (ll + 1), 1.0),
                               1.0)
            else:
                raise ValueError(f"Unknown band_model {band_model!r}; use 'Cl', 'Dl', or an array.")
        else:
            _ew = np.asarray(band_model, dtype=float)[:lmax + 1]
        self._ew = _ew

        if data_fits is not None:
            self._load_fits(data_fits, ninv_fits, N_matrix)
        # else: caller (from_arrays) sets self.d, self.N_diag, self.obs_pix, self.nside, self.n_obs

    def _load_fits(self, data_fits, ninv_fits, N_matrix):
        import warnings
        raw_data = hp.read_map(data_fits, field=None, verbose=False)
        raw_ninv = hp.read_map(ninv_fits, field=None, verbose=False)
        if raw_data.ndim == 1:
            raw_data = raw_data[np.newaxis, :]
        if raw_ninv.ndim == 1:
            raw_ninv = raw_ninv[np.newaxis, :]

        self.nside = hp.get_nside(raw_data[0])
        npix = hp.nside2npix(self.nside)

        self.obs_pix = np.where(raw_ninv[0] > 0)[0]
        self.n_obs   = len(self.obs_pix)
        print(f"Observed pixels: {self.n_obs}  (f_sky = {self.n_obs/npix:.4f})")

        # Detect ninv format: diagonal (n_f fields) or precision matrix (n_f*(n_f+1)//2 fields).
        # Almanac writes precision matrices, so for EE (n_T=0, n_P=1) ninv has 3 fields:
        # [N^{-1}_{QQ}, N^{-1}_{QU}, N^{-1}_{UU}], not 2.  Using the wrong field (QU=0
        # instead of UU) would give infinite U noise.
        n_f    = self.n_T + 2 * self.n_P
        n_diag = n_f
        n_prec = n_f * (n_f + 1) // 2
        n_ninv = raw_ninv.shape[0]

        if n_ninv == n_prec and n_prec > n_diag:
            is_prec = True
            # Upper-triangle row-major: diagonal element k is at field k*n_f - k*(k-1)//2
            def _diag_field(k):
                return k * n_f - k * (k - 1) // 2
            # Warn if any off-diagonal field is non-zero over observed pixels
            diag_set = {_diag_field(k) for k in range(n_f)}
            for fi in range(n_prec):
                if fi not in diag_set and np.any(raw_ninv[fi][self.obs_pix] != 0):
                    warnings.warn(
                        f"ninv FITS field {fi} contains non-zero off-diagonal precision "
                        f"elements. Correlated pixel noise is not yet supported; these "
                        f"off-diagonal terms will be ignored.",
                        UserWarning, stacklevel=2)
                    break
        elif n_ninv == n_diag:
            is_prec = False
            def _diag_field(k):
                return k
        else:
            raise ValueError(
                f"ninv FITS has {n_ninv} fields; expected {n_diag} (diagonal noise) "
                f"or {n_prec} (precision matrix) for n_T={self.n_T}, n_P={self.n_P}")

        # T maps: fields 0 .. n_T-1
        # P maps: fields n_T, n_T+1, ...  in pairs (Q_0,U_0,Q_1,U_1,...)
        dlist, nlist = [], []
        for i in range(self.n_T):
            dlist.append(raw_data[i][self.obs_pix])
            nlist.append(1.0 / raw_ninv[_diag_field(i)][self.obs_pix])
        for j in range(self.n_P):
            qi = self.n_T + 2 * j
            ui = self.n_T + 2 * j + 1
            dlist.append(raw_data[qi][self.obs_pix])
            dlist.append(raw_data[ui][self.obs_pix])
            nlist.append(1.0 / raw_ninv[_diag_field(qi)][self.obs_pix])
            nlist.append(1.0 / raw_ninv[_diag_field(ui)][self.obs_pix])

        self.d = np.concatenate(dlist)
        self.N_diag = np.concatenate(nlist)
        self._N_matrix = N_matrix
        self._finish_init()

    def _estimate_kernel_memory_gb(self, n_params):
        """Estimate memory needed for all kernel matrices."""
        Nd = (self.n_T + 2 * self.n_P) * self.n_obs
        kernel_gb = n_params * Nd**2 * 8 / 1e9
        return kernel_gb

    def _detect_available_memory_gb(self):
        """Detect available RAM with safety margin."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / 1e9
            safe_available = available_gb * 0.8
            return safe_available
        except ImportError:
            print("Warning: psutil not installed, cannot detect memory. "
                  "Defaulting to onthefly mode for safety.")
            return 0.0

    def _choose_kernel_mode(self, n_params):
        """Auto-select kernel mode based on resources."""
        kernel_mem = self._estimate_kernel_memory_gb(n_params)
        available_mem = self._detect_available_memory_gb()

        total_needed = kernel_mem * 2.0

        if total_needed < available_mem:
            mode = 'precompute'
            print(f"Auto mode: precompute ({kernel_mem:.1f} GB kernels, "
                  f"{available_mem:.1f} GB available)")
        else:
            mode = 'onthefly'
            print(f"Auto mode: onthefly ({total_needed:.1f} GB needed, "
                  f"{available_mem:.1f} GB available)")

        return mode

    def _finish_init(self):
        """Build kernels and layout after d, N_diag, obs_pix, nside, n_obs are set."""
        n = self.n_obs
        print(f"Building kernels (n_T={self.n_T}, n_P={self.n_P}, "
              f"lmin={self.lmin}, lmax={self.lmax}, nbands={self.nbands}, "
              f"threads={self.n_threads})...")

        self._TT_kernels = None
        self._Kp = self._Km = self._Kx = self._geom = None

        if self.n_T > 0:
            self._TT_kernels = _build_scalar_kernels(
                self.obs_pix, self.nside, self.lmin, self.lmax,
                self.band_edges, self.beam2, ell_weights=self._ew)

        if self.n_P > 0:
            self._Kp, self._Km, self._Kx, self._geom = _build_spin2_kernels(
                self.obs_pix, self.nside, self.lmin, self.lmax,
                self.band_edges, self.beam2, ell_weights=self._ew)

        self.layout = SpectraLayout(
            self.n_T, self.n_P, self.nbands,
            include_TB=self.include_TB, include_EB=self.include_EB)

        # Resolve kernel mode
        if self.kernel_mode == 'auto':
            self._resolved_mode = self._choose_kernel_mode(self.layout.n_params)
        else:
            self._resolved_mode = self.kernel_mode
            kernel_mem = self._estimate_kernel_memory_gb(self.layout.n_params)
            print(f"Kernel mode: {self._resolved_mode} ({kernel_mem:.1f} GB for kernels)")

        # Pre-build all derivative kernel matrices or store None for on-the-fly
        if self._resolved_mode == 'precompute':
            self._kernel_matrices = self._build_all_kernels()
        else:
            self._kernel_matrices = None

        # Noise matrix
        if self._N_matrix is not None:
            raise NotImplementedError(
                "N_matrix is not yet supported. Pass per-pixel diagonal noise via "
                "N_Q_list/N_U_list (from_arrays) or a FITS ninv map (_load_fits).")
        self.N_mat = np.diag(self.N_diag)

        print("Done.")

    @classmethod
    def from_arrays(cls, d_T_list, d_Q_list, d_U_list,
                    obs_pix, nside,
                    N_T_list, N_Q_list, N_U_list,
                    lmin, lmax, band_edges,
                    beam=None, band_model='Cl',
                    include_TB=False, include_EB=False,
                    N_matrix=None, kernel_mode='auto', n_threads='auto'):
        """
        Construct without FITS files, for testing.

        d_T_list  : list of n_T arrays, each shape (n_obs,)
        d_Q_list  : list of n_P arrays, each shape (n_obs,)
        d_U_list  : list of n_P arrays, each shape (n_obs,)
        N_T_list  : list of n_T noise variance arrays (sigma^2 per pixel)
        N_Q_list, N_U_list : same for Q, U
        obs_pix   : (n_obs,) HEALPix pixel indices
        nside     : HEALPix NSIDE
        """
        n_T = len(d_T_list)
        n_P = len(d_Q_list)
        obj = cls.__new__(cls)
        obj.lmin   = lmin
        obj.lmax   = lmax
        obj.band_edges = np.asarray(band_edges, dtype=int)
        obj.nbands = len(band_edges) - 1
        obj.n_T    = n_T
        obj.n_P    = n_P
        obj.include_TB = include_TB
        obj.include_EB = include_EB
        obj.kernel_mode = kernel_mode
        obj.beam2  = np.ones(lmax + 1) if beam is None else np.asarray(beam)[:lmax+1]**2

        # Thread configuration
        if n_threads == 'auto':
            import os
            obj.n_threads = os.cpu_count() or 1
        elif isinstance(n_threads, int) and n_threads > 0:
            obj.n_threads = n_threads
        else:
            raise ValueError(f"n_threads must be 'auto' or positive int, got {n_threads!r}")

        bm = band_model
        if isinstance(bm, str):
            bm = bm.replace('_', '')
            if bm == 'Cl':
                obj._ew = None
            elif bm == 'Dl':
                ll = np.arange(lmax + 1, dtype=float)
                obj._ew = np.where(ll >= 2,
                                   2.0*np.pi / np.where(ll >= 2, ll*(ll+1), 1.0), 1.0)
            else:
                raise ValueError(f"Unknown band_model {bm!r}")
        else:
            obj._ew = np.asarray(bm, dtype=float)[:lmax+1]

        obj.obs_pix = np.asarray(obs_pix)
        obj.nside   = nside
        obj.n_obs   = len(obs_pix)

        dparts, nparts = [], []
        for arr in d_T_list:
            dparts.append(np.asarray(arr))
        for q, u in zip(d_Q_list, d_U_list):
            dparts.append(np.asarray(q))
            dparts.append(np.asarray(u))
        obj.d = np.concatenate(dparts) if dparts else np.zeros(0)

        nparts = []
        for arr in N_T_list:
            nparts.append(np.asarray(arr))
        for nq, nu in zip(N_Q_list, N_U_list):
            nparts.append(np.asarray(nq))
            nparts.append(np.asarray(nu))
        obj.N_diag = np.concatenate(nparts) if nparts else np.zeros(0)
        obj._N_matrix = N_matrix

        obj._finish_init()
        return obj

    # ------------------------------------------------------------------
    # Block offset helpers
    # ------------------------------------------------------------------

    def _T_slice(self, i):
        n = self.n_obs
        return slice(i * n, (i + 1) * n)

    def _P_slice(self, j):
        n = self.n_obs
        start = self.n_T * n + j * 2 * n
        return slice(start, start + 2 * n)

    def _P_Q_slice(self, j):
        n = self.n_obs
        start = self.n_T * n + j * 2 * n
        return slice(start, start + n)

    def _P_U_slice(self, j):
        n = self.n_obs
        start = self.n_T * n + j * 2 * n + n
        return slice(start, start + n)

    # ------------------------------------------------------------------
    # Build derivative kernel matrices
    # ------------------------------------------------------------------

    def _build_kernel(self, idx):
        """
        Build a single derivative kernel matrix for parameter index idx.
        Returns an (N_d, N_d) matrix.
        """
        Nd = len(self.d)
        n = self.n_obs
        geom = self._geom

        spec, i, j, b = self.layout.decode(idx)
        K = np.zeros((Nd, Nd))

        if spec == 'TT':
            si = self._T_slice(i)
            sj = self._T_slice(j)
            K_tt = self._TT_kernels[b]
            K[si, sj] = K_tt
            if i != j:
                K[sj, si] = K_tt.T

        elif spec in ('TE', 'TB'):
            si = self._T_slice(i)
            sj = self._P_slice(j)
            Kx_b = self._Kx[b]
            _, _, _, _, c2j, s2j = geom
            if spec == 'TE':
                blk = _build_te_block(Kx_b, c2j, s2j)
            else:
                blk = _build_tb_block(Kx_b, c2j, s2j)
            K[si, sj] = blk
            K[sj, si] = blk.T

        elif spec == 'EE':
            sj = self._P_slice(i)
            sk = self._P_slice(j)
            Kp_b, Km_b = self._Kp[b], self._Km[b]
            blk = _ee_kernel(Kp_b, Km_b, geom)
            K[sj, sk] = blk
            if i != j:
                K[sk, sj] = blk.T

        elif spec == 'BB':
            sj = self._P_slice(i)
            sk = self._P_slice(j)
            Kp_b, Km_b = self._Kp[b], self._Km[b]
            blk = _bb_kernel(Kp_b, Km_b, geom)
            K[sj, sk] = blk
            if i != j:
                K[sk, sj] = blk.T

        elif spec == 'EB':
            sj = self._P_slice(i)
            sk = self._P_slice(j)
            blk = _eb_kernel(self._Km[b], geom)
            K[sj, sk] = blk
            if i != j:
                K[sk, sj] = blk.T

        return K

    def _get_kernel(self, idx):
        """
        Get kernel matrix for parameter index idx.
        In precompute mode: fetch from cache.
        In onthefly mode: build on demand.
        """
        if self._resolved_mode == 'precompute':
            return self._kernel_matrices[idx]
        else:
            return self._build_kernel(idx)

    def _build_all_kernels(self):
        """
        Returns a list of n_params full (N_d, N_d) derivative matrices,
        one per parameter, indexed identically to layout.entries().
        """
        kernels = [self._build_kernel(idx) for idx in range(self.layout.n_params)]
        return kernels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_signal_cov(self, cl_bands):
        """
        Build signal covariance from flat bandpower vector.

        cl_bands : shape (layout.n_params,)
        Returns  : (N_d, N_d) symmetric positive-semidefinite matrix.
        """
        cl_bands = np.asarray(cl_bands, dtype=float)
        Nd = len(self.d)
        C  = np.zeros((Nd, Nd))
        for idx, _, _, _, _ in self.layout.entries():
            if cl_bands[idx] != 0.0:
                C += cl_bands[idx] * self._get_kernel(idx)
        return C

    def log_likelihood(self, cl_bands):
        C = self.build_signal_cov(cl_bands)
        M = C + self.N_mat
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf
        logdet = 2.0 * np.log(np.diag(L)).sum()
        y = solve_triangular(L, self.d, lower=True)
        return -0.5 * (logdet + y @ y)

    def gradient_and_fisher(self, cl_bands):
        """
        Returns gradient g (n_params,) and Fisher matrix F (n_params, n_params).

        Uses Cholesky decomposition M = L L^T throughout.
        For each kernel K_b: A_b = L^{-1} K_b (L^{-1})^T via triangular solves.
          g_b  = 0.5 * [v^T K_b v - Tr(A_b)]     where v = M^{-1} d
          F_bb' = 0.5 * Tr[A_b A_b'] = 0.5 * (A_b * A_b'.T).sum()
        """
        C = self.build_signal_cov(cl_bands)
        M = C + self.N_mat

        L = np.linalg.cholesky(M)               # may raise LinAlgError

        # v = M^{-1} d via two triangular solves
        y = solve_triangular(L,   self.d, lower=True)   # L y = d
        v = solve_triangular(L.T, y,      lower=False)  # L^T v = y

        np_total = self.layout.n_params
        A = [None] * np_total

        # Build A_b = L^{-1} K_b (L^{-1})^T for each kernel
        if self.n_threads == 1:
            # Serial computation
            for idx in range(np_total):
                K_b = self._get_kernel(idx)
                W   = solve_triangular(L,   K_b,   lower=True)        # L W = K_b
                A[idx] = solve_triangular(L, W.T, lower=True).T       # L A^T = W^T
        else:
            # Parallel computation
            from concurrent.futures import ThreadPoolExecutor

            def compute_A_b(idx):
                K_b = self._get_kernel(idx)
                W   = solve_triangular(L,   K_b,   lower=True)
                return solve_triangular(L, W.T, lower=True).T

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                A = list(executor.map(compute_A_b, range(np_total)))

        g = np.array([
            0.5 * (v @ (self._get_kernel(b) @ v) - np.trace(A[b]))
            for b in range(np_total)
        ])

        F = np.zeros((np_total, np_total))
        for b in range(np_total):
            for bp in range(b, np_total):
                val = 0.5 * (A[b] * A[bp].T).sum()
                F[b, bp] = F[bp, b] = val

        return g, F

    def newton_raphson(self, cl_init, max_iter=15, tol=1e-4, damp=1.0):
        """
        Newton-Raphson for ML bandpowers with backtracking line search.

        cl_init : flat array length layout.n_params
        Returns (cl_ml, sigma, F) where sigma = sqrt(diag(F^{-1})).
        """
        cl = np.array(cl_init, dtype=float)
        logL_best = self.log_likelihood(cl)
        print(f"\nNewton-Raphson ({len(cl)} parameters, max_iter={max_iter})")

        for it in range(max_iter):
            try:
                g, F = self.gradient_and_fisher(cl)
                delta = np.linalg.solve(F, g)
            except np.linalg.LinAlgError as e:
                print(f"  iter {it+1}: stopping ({e})")
                break

            # Backtracking line search: try damp, damp/2, damp/4, ...
            alpha = damp
            for _ in range(5):
                cl_new = cl + alpha * delta
                logL_new = self.log_likelihood(cl_new)
                if np.isfinite(logL_new) and logL_new > logL_best:
                    break
                alpha *= 0.5
            else:
                # No improvement found
                print(f"  iter {it+1}: no finite improvement, stopping")
                break

            step = np.max(np.abs(alpha * delta) / (np.abs(cl) + 1e-40))
            print(f"  iter {it+1}: logL={logL_new:.4f}  max|δ/C|={step:.2e}")
            cl = cl_new
            logL_best = logL_new

            if step < tol:
                print("  Converged.")
                break

        try:
            _, F = self.gradient_and_fisher(cl)
            sigma = np.sqrt(np.maximum(np.diag(np.linalg.inv(F)), 0))
        except np.linalg.LinAlgError:
            sigma = np.full(len(cl), np.nan)
            F = np.full((len(cl), len(cl)), np.nan)

        return cl, sigma, F

    @property
    def ell_bands(self):
        return 0.5 * (self.band_edges[:-1] + self.band_edges[1:] - 1)
