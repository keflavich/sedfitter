"""
Fit one source's SED with the spubsmi emulator directly, without a stored
model grid.

    python fit_with_emulator.py          # runs the mock-source demo below

The fit has three stages:

1. Scan: chi^2 for n_scan draws from the R24 priors, with the foreground A_V
   solved analytically for each draw.
2. Polish: bounded L-BFGS-B minimisation (12 parameters + A_V) from the few
   best draws.
3. Sample: adaptive Metropolis chains (all chains evaluated in one batch per
   step) started around the minimum, sampling exp(-chi^2 / 2) times the
   uniform R24 priors.  The chains give posterior percentiles for every
   parameter and for log L and log M_core.

The posterior is much narrower than the prior for a well-sampled SED, so
weighting the prior draws of stage 1 by exp(-chi^2 / 2) leaves an effective
sample of a few draws; stage 3 replaces that.

Fluxes are compared in log10 space.  Each band's uncertainty is the data
error (converted to dex) and the emulator error (``model_err_dex``) added in
quadrature; see docs/emulator.rst for the emulator's accuracy.
"""
import numpy as np
from scipy.optimize import minimize

from yso_emulator import YSOEmulator


def band_readout(emu, wavelengths, apertures_au):
    """Stack one readout row per band, so each band can have its own aperture."""
    rows = [emu.readout([lam], ap) for lam, ap in zip(wavelengths, apertures_au)]
    return (np.vstack([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
            np.concatenate([r[2] for r in rows]))


class EmulatorFit:
    """chi^2 of the emulator against one source.

    wavelengths : micron;  flux, error : mJy;  apertures_au : aperture radius
    of each band in AU (= arcsec x distance in pc);  upper : boolean mask of
    bands whose ``flux`` is an upper limit (the model is penalised only above
    it).
    """

    def __init__(self, emu, wavelengths, flux, error, apertures_au, distance_kpc,
                 upper=None, av_range=(0., 100.), model_err_dex=0.05):
        self.emu = emu
        self.readout = band_readout(emu, wavelengths, apertures_au)
        self.ext = self.readout[2]                       # A_lambda / A_V
        flux, error = np.asarray(flux, float), np.asarray(error, float)
        self.upper = np.zeros(len(flux), bool) if upper is None else np.asarray(upper, bool)
        self.logf = np.log10(flux)
        sig_data = np.where(self.upper, 0., error / (flux * np.log(10.)))
        self.sig = np.sqrt(sig_data ** 2 + model_err_dex ** 2)
        self.dshift = -2. * np.log10(distance_kpc)       # models are at 1 kpc
        self.lo = np.append(emu.lo, av_range[0])          # bounds of (theta, A_V)
        self.hi = np.append(emu.hi, av_range[1])

    def model(self, theta, av):
        """log10 model flux (mJy) at the source distance, behind A_V."""
        m = self.emu.log_flux_at(theta, self.readout) + self.dshift
        return m - 0.4 * np.atleast_1d(av)[:, None] * self.ext

    def chi2(self, theta, av):
        m = self.model(theta, av)
        r = (self.logf - m) / self.sig
        r = np.where(self.upper & (m < self.logf), 0., r)
        return (r ** 2).sum(1)

    def best_av(self, theta):
        """A_V minimising chi^2 over the detections (clipped to the bounds)."""
        m0 = self.model(theta, 0.)
        det = ~self.upper
        a = -0.4 * self.ext[det] / self.sig[det]
        r = (self.logf[det] - m0[:, det]) / self.sig[det]
        return np.clip((r * a).sum(1) / (a * a).sum(), self.lo[-1], self.hi[-1])

    def scan(self, n_scan=100000, seed=0, chunk=20000):
        theta = self.emu.prior_draws(n_scan, seed=seed)
        av = np.concatenate([self.best_av(theta[i:i + chunk]) for i in range(0, n_scan, chunk)])
        chi2 = np.concatenate([self.chi2(theta[i:i + chunk], av[i:i + chunk])
                               for i in range(0, n_scan, chunk)])
        return np.column_stack([theta, av]), chi2

    def polish(self, p0):
        scale = self.hi - self.lo                         # optimise in [0, 1] units

        def f(u):
            p = self.lo + u * scale
            return float(self.chi2(p[:-1], p[-1])[0])

        # the network runs in float32, so the finite-difference step must be
        # well above the default 1e-8 (1e-5 of each prior range works)
        res = minimize(f, (p0 - self.lo) / scale, method='L-BFGS-B',
                       bounds=[(0., 1.)] * len(p0), options={'eps': 1e-5})
        return self.lo + res.x * scale, res.fun

    def sample(self, p0, n_chains=64, n_steps=3000, n_burn=1000, seed=0):
        """Adaptive Metropolis (Haario et al. 2001) over (theta, A_V)."""
        rng = np.random.default_rng(seed)
        nd = len(p0)
        scale = self.hi - self.lo
        cov = np.diag((1e-3 * scale) ** 2)
        p = np.clip(p0 + rng.multivariate_normal(np.zeros(nd), cov, n_chains), self.lo, self.hi)
        c = self.chi2(p[:, :-1], p[:, -1])
        chain = np.empty((n_steps, n_chains, nd))
        chi2 = np.empty((n_steps, n_chains))
        n_acc = 0
        for k in range(n_steps):
            if k >= 200 and k % 100 == 0 and k <= n_burn:
                hist = chain[k // 2:k].reshape(-1, nd)
                cov = 2.38 ** 2 / nd * np.cov(hist.T) + np.diag((1e-6 * scale) ** 2)
            q = p + rng.multivariate_normal(np.zeros(nd), cov, n_chains)
            inside = np.all((q >= self.lo) & (q <= self.hi), axis=1)
            cq = np.full(n_chains, np.inf)
            cq[inside] = self.chi2(q[inside, :-1], q[inside, -1])
            acc = np.log(rng.random(n_chains)) < -0.5 * (cq - c)
            p[acc], c[acc] = q[acc], cq[acc]
            if k >= n_burn:
                n_acc += acc.sum()
            chain[k], chi2[k] = p, c
        acc_rate = n_acc / ((n_steps - n_burn) * n_chains)
        return chain[n_burn:].reshape(-1, nd), chi2[n_burn:].ravel(), acc_rate


def fit_source(emu, *args, n_scan=100000, n_polish=10, seed=0, n_steps=3000, **kwargs):
    fitter = EmulatorFit(emu, *args, **kwargs)
    draws, chi2 = fitter.scan(n_scan, seed=seed)
    best_p, best_chi2 = None, np.inf
    for i in np.argsort(chi2)[:n_polish]:
        p, c = fitter.polish(draws[i])
        if c < best_chi2:
            best_p, best_chi2 = p, c
    chain, chain_chi2, acc = fitter.sample(best_p, n_steps=n_steps, seed=seed)
    if chain_chi2.min() < best_chi2:
        best_p, best_chi2 = chain[chain_chi2.argmin()], chain_chi2.min()

    def derived(p):
        p = np.atleast_2d(p)
        cols = {name: p[:, j] for j, name in enumerate(emu.names)}
        cols['log L'] = emu.luminosity(p[:, :-1])
        cols['log Mcore'] = emu.log_mcore(p[:, :-1])
        cols['A_V'] = p[:, -1]
        return cols

    best = {k: float(v[0]) for k, v in derived(best_p).items()}
    posterior = {k: np.percentile(v, [16, 50, 84]) for k, v in derived(chain).items()}
    return {'best': best, 'chi2_best': best_chi2, 'posterior_16_50_84': posterior,
            'chain': chain, 'chain_chi2': chain_chi2, 'acceptance': acc, 'fitter': fitter}


if __name__ == '__main__':
    import time

    emu = YSOEmulator.download('emulator_files')
    emu.check()

    # Mock source: an emulated model at 2 kpc behind A_V = 20, observed in a
    # 3 arcsec (6000 AU) aperture, with 10% errors and 10% scatter.
    rng = np.random.default_rng(42)
    truth = emu.prior_draws(1, seed=7)[0]
    wav = np.array([1.25, 2.2, 3.6, 4.5, 8., 24., 70., 160., 250., 500., 870., 1300.])
    ap = np.full(len(wav), 3. * 2000.)
    d_kpc, av_true = 2., 20.
    mock = EmulatorFit(emu, wav, np.ones(len(wav)), np.ones(len(wav)), ap, d_kpc)
    f_true = 10. ** mock.model(truth, av_true)[0]
    flux = f_true * (1. + 0.1 * rng.standard_normal(len(wav)))
    error = 0.1 * f_true

    t0 = time.time()
    out = fit_source(emu, wav, flux, error, ap, d_kpc)
    fitter = out['fitter']
    print(f'fit took {time.time() - t0:.1f} s; best chi^2 {out["chi2_best"]:.2f} for {len(wav)} bands '
          f'(chi^2 at the truth {fitter.chi2(truth, av_true)[0]:.2f}); '
          f'MCMC acceptance {out["acceptance"]:.2f}')
    true_vals = list(truth) + [float(emu.luminosity(truth)[0]), float(emu.log_mcore(truth)[0]), av_true]
    print(f'{"parameter":18s} {"true":>9s} {"best":>9s}   posterior 16/50/84')
    for (name, pct), t in zip(out['posterior_16_50_84'].items(), true_vals):
        print(f'{name:18s} {t:9.3f} {out["best"][name]:9.3f}   ' + ' '.join(f'{v:8.3f}' for v in pct))
