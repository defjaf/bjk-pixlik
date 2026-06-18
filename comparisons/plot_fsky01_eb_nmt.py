#!/usr/bin/env python3
"""fsky=0.1 spin-2 EB bandpower: Almanac vs NaMaster, with HSM mode."""
import numpy as np, pymaster as nmt, healpy as hp, arviz as az
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

edges_arr = [2,20,38,56,74,92,110,129]
nb = len(edges_arr)-1; lmax = 128; NSIDE = 64
truth = {'EE': 1e-4/(2*np.pi), 'EB': 0.3*np.sqrt(0.5)*1e-4/(2*np.pi), 'BB': 0.5e-4/(2*np.pi)}
centers = np.array([0.5*(edges_arr[b]+edges_arr[b+1]-1) for b in range(nb)])
hw = np.array([0.5*(edges_arr[b+1]-edges_arr[b]) for b in range(nb)])

mask = hp.read_map('sim_tombin-1_nside64_flatTT_fsky01/cap_nside64_fsky01_mask.fits')
fsky = mask.mean()
maps = hp.read_map('sim_eb_nside64_fsky01/sim_eb03_out_channel_0_data.fits', field=None)
Q, U = maps[0], maps[1]
Nl = (1.612931e-03)**2 * 4*np.pi/len(Q)

bins = nmt.NmtBin.from_edges(edges_arr[:-1], edges_arr[1:], is_Dell=True)
f2 = nmt.NmtField(mask, [Q, U], n_iter=0, lmax=lmax)
wsp = nmt.NmtWorkspace.from_fields(f2, f2, bins)
cl_dec = wsp.decouple_cell(nmt.compute_coupled_cell(f2, f2))
ell_eff = bins.get_effective_ells()
nmt_d = {'EE': cl_dec[0], 'EB': cl_dec[1], 'BB': cl_dec[3]}
nell = np.array([bins.get_ell_list(b).size for b in range(nb)])
sig = {'EE': np.sqrt(2/(nell*(2*ell_eff+1)*fsky))*(nmt_d['EE']+Nl),
       'BB': np.sqrt(2/(nell*(2*ell_eff+1)*fsky))*(nmt_d['BB']+Nl),
       'EB': np.sqrt(1/(nell*(2*ell_eff+1)*fsky))*np.sqrt((nmt_d['EE']+Nl)*(nmt_d['BB']+Nl)+nmt_d['EB']**2)}

lam = np.load('almanac_runs/NSIDE64_fsky01_bandpower_eb03_v1.ellip.extract.npy')
lam = lam[len(lam)//5:]
th = {'EE': np.exp(2*lam[:,0::3]),
      'EB': lam[:,1::3]*np.exp(lam[:,0::3]),
      'BB': lam[:,1::3]**2+np.exp(2*lam[:,2::3])}

def hsm(x):
    x = np.sort(x)
    while len(x) > 3:
        n = (len(x)+1)//2; w = x[n-1:]-x[:len(x)-n+1]; i = np.argmin(w); x = x[i:i+n]
    return x.mean()

idata = az.from_dict({'posterior': {f'{s}{b}': th[s][:,b][None,:] for s in ('EE','EB','BB') for b in range(nb)}})
er = az.ess(idata, method='bulk')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
off = 0.25*hw

for ax, sp in zip(axes, ('EE','EB','BB')):
    mean = th[sp].mean(axis=0)*1e5
    std  = th[sp].std(axis=0)*1e5
    mode = np.array([hsm(th[sp][::5,b])*1e5 for b in range(nb)])
    ess  = np.array([float(er[f'{sp}{b}'].values) for b in range(nb)])

    ax.errorbar(centers-off, mean, yerr=std, xerr=hw, fmt='o', color='C0',
                capsize=2, label='Almanac (mean±std)')
    ax.scatter(centers-off, mode, marker='D', s=22, color='navy', zorder=5,
               label='Almanac HSM mode')
    ax.errorbar(centers+off, nmt_d[sp]*1e5, yerr=sig[sp]*1e5, xerr=hw,
                fmt='s', color='C3', capsize=2, markerfacecolor='none',
                label='NaMaster (MLE±σ)')
    ax.hlines([truth[sp]*1e5]*nb, edges_arr[:-1], edges_arr[1:],
              color='green', ls='--', lw=1.2, label='truth')
    for b in range(nb):
        ax.annotate(f'{ess[b]:.0f}', (centers[b], (mean[b]+std[b])),
                    textcoords='offset points', xytext=(0,3),
                    ha='center', fontsize=6.5)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell\ [\times 10^{-5}]$')
    ax.set_title(f'$\\theta^{{{sp}}}$  (ESS annotated)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(r'fsky=0.1 spin-2 $E/B$ bandpowers: Almanac vs NaMaster ($r_{EB}=0.3$)', y=1.01)
fig.tight_layout()
out = 'almanac_runs/fsky01_eb03_nmt_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
