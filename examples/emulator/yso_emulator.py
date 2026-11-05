"""
Pure-numpy evaluation of the spubsmi SED emulator.

The emulator is a neural network trained on the Richardson et al. (2024)
update of the Robitaille (2017) YSO model grid, geometry ``spubsmi``
(star + power-law envelope + Ulrich envelope + bipolar cavity + passive disk
+ ambient medium).  It maps the 11 free physical parameters and the viewing
inclination to the model SED (log10 flux density in mJy at 1 kpc) in the 20
apertures (100 to 10^6 AU) and at the 165 wavelengths >= 0.1 micron of the
published grid.  See ``docs/emulator.rst`` for its accuracy and limits.

The weights (one ensemble member, float16) and the metadata are public:

    https://data.rc.ufl.edu/pub/adamginsburg/sedfitter/meta.json
    https://data.rc.ufl.edu/pub/adamginsburg/sedfitter/emu0_f16.b64.txt

Usage::

    from yso_emulator import YSOEmulator
    emu = YSOEmulator.download('emulator_files')    # or YSOEmulator('emulator_files')
    theta = emu.prior_draws(1000)                     # (1000, 12) fit-space rows
    logf = emu.log_flux(theta)                        # (1000, 20, 165) log10 mJy at 1 kpc

Fit-space rows ``theta`` have 12 columns, in ``emu.names`` order:
log10 of star.radius (Rsun), star.temperature (K), disk.mass (Msun),
disk.rmax (AU), then disk.beta, disk.p, log10 disk.h100 (AU), log10
envelope.rho_0 (g/cm^3), cavity.power, cavity.theta_0 (deg), log10
cavity.rho_0 (g/cm^3) and the inclination (deg).  ``emu.lo`` / ``emu.hi``
are the bounds of the R24 priors (uniform in these coordinates); the
emulator is not valid outside them.  The Ulrich envelope's centrifugal
radius equals the disk outer radius, as in the grid.
"""
import base64
import json
import os
import urllib.request

import numpy as np

BASE_URL = 'https://data.rc.ufl.edu/pub/adamginsburg/sedfitter/'
FILES = ('meta.json', 'emu0_f16.b64.txt')
LOG_TSUN = np.log10(5772.)


def _layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _silu(x):
    return x * 0.5 * (1. + np.tanh(0.5 * x))       # x * sigmoid(x), overflow-free


class YSOEmulator:
    """The spubsmi SED emulator (one network, float16 weights evaluated in
    float32).  ``directory`` holds meta.json and emu0_f16.b64.txt."""

    def __init__(self, directory):
        with open(os.path.join(directory, 'meta.json')) as f:
            self.meta = M = json.load(f)
        with open(os.path.join(directory, 'emu0_f16.b64.txt'), 'rb') as f:
            blob = np.frombuffer(base64.b64decode(f.read()), dtype='<f2').astype(np.float32)
        if blob.size != M['n_values']:
            raise ValueError(f"weight file has {blob.size} values, meta.json expects {M['n_values']}")
        self.W = {l['name']: blob[l['offset']:l['offset'] + int(np.prod(l['shape']))].reshape(l['shape'])
                  for l in M['layers']}
        self.nblocks = M['nblocks']
        self.wav = np.array(M['wav'])                 # micron, descending
        self.apertures = np.array(M['aps_au'])        # AU, ascending
        self.names = M['fit_names']
        self.is_log = np.array(M['fit_log'])
        self.lo, self.hi = np.array(M['lo']), np.array(M['hi'])
        self.src = np.array(M['src'])
        self.x_lo, self.x_hi = np.array(M['x_lo']), np.array(M['x_hi'])
        self.out_mu = np.array(M['out_mu'], np.float32)
        self.out_sd = np.array(M['out_sd'], np.float32)
        self.ext = np.array(M['ext'])                 # A_lambda / A_V of the grid's dust, on self.wav
        self.mc = [np.array(l['data'], np.float32).reshape(l['shape']) for l in M['mcore']['layers']]

    @classmethod
    def download(cls, directory, base_url=BASE_URL):
        """Fetch meta.json (3.4 MB) and the weights (13 MB) into directory
        (skipped for files already there) and load them."""
        os.makedirs(directory, exist_ok=True)
        for name in FILES:
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(base_url + name, path)
        return cls(directory)

    # ------------------------------------------------------------------ inputs
    def prior_draws(self, n, seed=None):
        """n rows drawn from the R24 priors (uniform in fit space)."""
        rng = np.random.default_rng(seed)
        return self.lo + (self.hi - self.lo) * rng.random((n, len(self.lo)))

    def theta_from_physical(self, **pars):
        """One fit-space row from physical values, e.g.
        theta_from_physical(**{'star.radius': 5., 'star.temperature': 4000., ...,
        'inclination': 45.}); every name in self.names is required."""
        return np.array([np.log10(pars[n]) if lg else pars[n] for n, lg in zip(self.names, self.is_log)])

    def luminosity(self, theta):
        """log10 L / Lsun of fit-space rows (L = R^2 (T / 5772 K)^4)."""
        theta = np.atleast_2d(theta)
        return 2 * theta[:, 0] + 4 * (theta[:, 1] - LOG_TSUN)

    def _hidden(self, theta):
        theta = np.atleast_2d(np.asarray(theta, float))
        x = np.column_stack([theta[:, self.src], self.luminosity(theta), np.cos(np.radians(theta[:, -1]))])
        x = (2 * (x - self.x_lo) / (self.x_hi - self.x_lo) - 1).astype(np.float32)
        W = self.W
        h = x @ W['inp.weight'].T + W['inp.bias']
        for i in range(self.nblocks):
            p = f'blocks.{i}.'
            a = _layer_norm(h, W[p + '0.weight'], W[p + '0.bias'])
            a = _silu(a @ W[p + '1.weight'].T + W[p + '1.bias'])
            h = h + a @ W[p + '3.weight'].T + W[p + '3.bias']
        return _layer_norm(h, W['out.0.weight'], W['out.0.bias'])

    # ------------------------------------------------------------------ outputs
    def log_flux(self, theta, chunk=4096):
        """log10 flux density (mJy at 1 kpc, no foreground extinction) of
        fit-space rows theta: shape (n, 20 apertures, 165 wavelengths), on
        self.apertures and self.wav (descending)."""
        theta = np.atleast_2d(theta)
        out = np.empty((len(theta), len(self.apertures), len(self.wav)), np.float32)
        Wo, bo = self.W['out.1.weight'], self.W['out.1.bias']
        for i in range(0, len(theta), chunk):
            y = self._hidden(theta[i:i + chunk]) @ Wo.T + bo
            out[i:i + chunk] = (y * self.out_sd + self.out_mu).reshape(-1, len(self.apertures), len(self.wav))
        return out

    def readout(self, wavelengths, aperture_au):
        """Effective output layer (weights, biases, A_lambda/A_V) for given
        wavelengths (micron) in one aperture (AU), linear in log wavelength
        and log aperture between the tabulated values (clamped at the ends).
        Evaluating only these outputs is ~3300 / len(wavelengths) times
        cheaper in the last layer than log_flux."""
        lw, la = np.log(self.wav[::-1]), np.log(self.apertures)        # ascending
        nw = len(self.wav)
        ia = int(np.clip(np.searchsorted(la, np.log(aperture_au)) - 1, 0, len(la) - 2))
        ta = float(np.clip((np.log(aperture_au) - la[ia]) / (la[ia + 1] - la[ia]), 0, 1))
        Wo, bo = self.W['out.1.weight'], self.W['out.1.bias']
        w = np.zeros((len(wavelengths), Wo.shape[1]), np.float32)
        b = np.zeros(len(wavelengths))
        ext = np.zeros(len(wavelengths))
        for q, lam in enumerate(np.atleast_1d(wavelengths)):
            j = int(np.clip(np.searchsorted(lw, np.log(lam)) - 1, 0, nw - 2))
            tw = float(np.clip((np.log(lam) - lw[j]) / (lw[j + 1] - lw[j]), 0, 1))
            jd = [nw - 1 - j, nw - 2 - j]                 # descending indices of the bracketing wavelengths
            ext[q] = self.ext[jd[0]] * (1 - tw) + self.ext[jd[1]] * tw
            for a, ca in ((ia, 1 - ta), (ia + 1, ta)):
                for jj, cw in zip(jd, (1 - tw, tw)):
                    c = ca * cw
                    if c == 0:
                        continue
                    row = a * nw + jj
                    w[q] += c * self.out_sd[row] * Wo[row]
                    b[q] += c * (self.out_sd[row] * bo[row] + self.out_mu[row])
        return w, b, ext

    def log_flux_at(self, theta, readout):
        """log10 mJy at 1 kpc of rows theta at a readout(): shape (n, n_wav)."""
        w, b, _ = readout
        return self._hidden(theta) @ w.T + b

    def log_mcore(self, theta):
        """log10 core mass (Msun) in the 12,743 AU aperture, as defined by
        Richardson et al. (2025, Paper II), from a small companion network."""
        theta = np.atleast_2d(theta)
        mc = self.meta['mcore']
        v = ((theta[:, self.src] - np.array(mc['mu'])) / np.array(mc['sd'])).astype(np.float32)
        for k in range(0, len(self.mc), 2):
            v = v @ self.mc[k].T + self.mc[k + 1]
            if k + 2 < len(self.mc):
                v = _silu(v)
        return v[:, 0] * mc['ys'] + mc['ym']

    def check(self, tol=0.01):
        """Compare with the reference outputs stored in meta.json; returns the
        largest absolute difference (dex)."""
        tp = self.meta['test_points']
        d = np.abs(self.log_flux(np.array(tp['theta'])).reshape(len(tp['theta']), -1) - np.array(tp['logf']))
        if d.max() > tol:
            raise ValueError(f'emulator check failed: max difference {d.max():.3g} dex')
        return float(d.max())
