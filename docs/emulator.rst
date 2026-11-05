.. _emulator:

==========================================
Fitting with an emulator instead of a grid
==========================================

A model grid samples parameter space at fixed points, and its density is set
by the cost of the radiative transfer runs behind it. An *emulator* is a
neural network trained on such a grid that returns the model SED for any
parameter values inside the grid's priors in about 0.1 ms. It can replace the
stored grid in two ways:

* **Route 1 — emulated model package.** Draw as many models as needed,
  evaluate them with the emulator, and write them in the version-2 model
  package format (:doc:`creating_model_packages`). :func:`~sedfitter.fit`,
  :func:`~sedfitter.plot` and the other tools then work unchanged.
* **Route 2 — direct fit.** Minimise :math:`\chi^2` and sample the posterior
  with the emulator inside the likelihood. No grid is stored; each source
  takes about a minute on 8 cores.

The scripts used below are in ``examples/emulator/`` of the repository:

==============================  ===================================================
``yso_emulator.py``             pure-numpy emulator (needs only numpy)
``make_model_package.py``       Route 1: writes ``flux.fits``, ``parameters.fits``
                                and ``models.conf``
``fit_with_emulator.py``        Route 2: scan, L-BFGS-B polish and MCMC for one source
==============================  ===================================================

The available emulator
======================

The public emulator covers the ``spubsmi`` geometry of the `Robitaille (2017)
<https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..11R>`_ YSO models as
recomputed by Richardson et al. (2024; "R24"): a star, a passive flared disk,
a power-law (Ulrich) envelope with a bipolar cavity, and an ambient medium.
It returns :math:`\log_{10}` of the flux density in mJy at 1 kpc, without
foreground extinction, at the 165 grid wavelengths from 0.1 to 4838 µm and in
the 20 grid apertures from 100 to :math:`10^6` AU. The other 15 geometries of
the grid would each need their own emulator; none are public yet.

The network takes 12 inputs ("fit-space" coordinates), with uniform priors
over these ranges:

==================  =================================  ============
parameter           coordinate                         range
==================  =================================  ============
star.radius         log10 R* / Rsun                    -1 to 2
star.temperature    log10 T* / K                       3.30 to 4.48
disk.mass           log10 M_disk / Msun                -8 to -1
disk.rmax           log10 R_disk / AU                  1.70 to 3.70
disk.beta           flaring power                      1.0 to 1.3
disk.p              surface density power              -2 to 0
disk.h100           log10 h(100 AU) / AU               0 to 1.30
envelope.rho_0      log10 rho_0 / g cm^-3              -24 to -16
cavity.power        cavity shape power                 1 to 2
cavity.theta_0      cavity opening angle / deg         0 to 60
cavity.rho_0        log10 rho_cav / g cm^-3            -23 to -20
inclination         viewing angle / deg                0 to 90
==================  =================================  ============

The envelope's centrifugal radius equals the disk outer radius, as in the
grid, and the luminosity follows from :math:`L = R_*^2 (T_*/5772\,{\rm K})^4`.
The emulator is valid only inside these ranges. A second, small network
returns the core mass :math:`\log_{10} M_{\rm core}` inside 12,743 AU, as
defined by Richardson et al. (Paper II).

The weights (one network, float16, 13 MB) and the metadata (3.4 MB) are at
https://data.rc.ufl.edu/pub/adamginsburg/sedfitter/ , which also hosts a
browser version of the Route 2 fitter for quick checks.

Getting the emulator
====================

::

    from yso_emulator import YSOEmulator

    emu = YSOEmulator.download('emulator_files')   # fetches meta.json and emu0_f16.b64.txt once
    emu.check()                                     # compares with reference outputs; max difference in dex

    theta = emu.prior_draws(1000, seed=0)           # (1000, 12) fit-space rows
    logf = emu.log_flux(theta)                      # (1000, 20, 165): log10 mJy at 1 kpc
    emu.wav, emu.apertures                          # micron (descending), AU (ascending)
    emu.luminosity(theta), emu.log_mcore(theta)     # log10 L/Lsun, log10 M_core/Msun

.. note:: If the download fails with ``CERTIFICATE_VERIFY_FAILED`` (some
          conda Python builds lack the system certificates), point Python at
          the system bundle, e.g. ``export
          SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt``, or fetch the two
          files with ``curl`` or ``wget`` into ``emulator_files/``.

Rows can be built from physical values with
``emu.theta_from_physical(**{'star.radius': 5., 'star.temperature': 4000.,
..., 'inclination': 45.})``.

Route 1: an emulated model package
==================================

``make_model_package.py`` draws ``N`` rows from the priors, evaluates them,
and writes a version-2 model package::

    python make_model_package.py model_emulated 200000

Each row is one (model, inclination) pair. Flux values more than 7.9 dex below
an SED's peak are set to zero, following the grid's convention for bins
without photons. ``parameters.fits`` holds the physical parameters,
``inclination``, ``Source Luminosity`` and ``log Mcore``. To concentrate rows
where the data need them, replace ``emu.prior_draws`` with any sampling of
fit space inside the ranges above.

The fit then proceeds as with the published grid. Wavelengths given as
:class:`~astropy.units.Quantity` objects in the filter list use the nearest
emulated wavelength, so monochromatic fits need no convolution step; for
broadband photometry, convolve the package with
:func:`~sedfitter.convolve.convolve_model_dir` as described in
:doc:`convolution`. The extinction law that matches the grid's dust is stored
with the emulator (as :math:`A_\lambda/A_V`; :class:`~sedfitter.extinction.Extinction`
uses only its shape)::

    import numpy as np
    from astropy import units as u
    from sedfitter import fit
    from sedfitter.extinction import Extinction

    extinction = Extinction()
    extinction.wav = emu.wav[::-1] * u.micron       # Extinction needs ascending wavelengths
    extinction.chi = emu.ext[::-1] * u.cm**2 / u.g

    wav = [1.25, 2.2, 3.6, 4.5, 8., 24., 70., 160., 250., 500., 870., 1300.] * u.micron
    fit('data_file', list(wav), [3.] * len(wav) * u.arcsec, 'model_emulated',
        'output.fitinfo', extinction_law=extinction,
        distance_range=[1.9, 2.1] * u.kpc, av_range=[0., 100.])

:func:`~sedfitter.fit` computes :math:`\chi^2` from the data errors only. Add
the emulator error (see `Accuracy`_) to the flux errors in the data file, in
quadrature, before fitting: :math:`\sigma^2 \rightarrow \sigma^2 + (F \ln 10\,
\sigma_{\rm emu})^2` with :math:`\sigma_{\rm emu}` in dex.

Each row takes 26 kB in ``flux.fits`` (float32 values and uncertainties), so
:math:`10^5` rows need 2.6 GB of disk and are read into memory when fitting.
A 20,000-row package was written in about 5 s on 8 cores; the evaluation runs
at about 10,000 SEDs per second.

Route 2: fitting directly with the emulator
===========================================

``fit_with_emulator.py`` fits one source without a stored grid. Each band has
a wavelength, a flux and error in mJy, and an aperture radius in AU
(arcseconds times the distance in pc); upper limits are marked with a boolean
mask::

    from yso_emulator import YSOEmulator
    from fit_with_emulator import fit_source

    emu = YSOEmulator.download('emulator_files')
    out = fit_source(emu, wavelengths, flux, error, apertures_au, distance_kpc,
                     upper=None, av_range=(0., 100.), model_err_dex=0.05)

    out['best']                    # best-fit parameters, log L, log Mcore and A_V
    out['posterior_16_50_84']      # percentiles of the same quantities from the chains
    out['chain'], out['chain_chi2']

The fit runs in three stages:

1. a scan of 100,000 prior draws, with :math:`A_V` solved analytically for
   each draw;
2. a bounded L-BFGS-B minimisation over the 12 parameters and :math:`A_V`,
   started from the 10 best draws;
3. 64 adaptive Metropolis chains of 3,000 steps (1,000 burn-in), started at
   the minimum, sampling :math:`\exp(-\chi^2/2)` times the uniform priors.

The fluxes are compared in :math:`\log_{10}`, with the data error and
``model_err_dex`` added in quadrature. The default, 0.05 dex, is close to the
emulator's typical error at 1–24 µm (see `Accuracy`_); smaller values suit
fits that use only far-infrared and (sub)millimetre bands.

Running ``python fit_with_emulator.py`` fits a mock source: an emulated model
at 2 kpc behind :math:`A_V=20`, in 12 bands from 1.25 µm to 1.3 mm with 10%
errors. The run takes about 55 s on 8 cores. The best fit reaches
:math:`\chi^2=0.7` (4.5 at the true parameters), and the 16–84% ranges of
:math:`\log L` (3.99–4.21; true 4.13), :math:`\log M_{\rm core}`
(0.85–1.04; true 0.91) and :math:`A_V` (11–22; true 20) contain the true
values. The chains accept 2–3% of the proposals because the posterior is
curved and partly degenerate in 13 dimensions; for publication-quality
posteriors, pass ``fitter.chi2`` to a dedicated sampler (e.g. ``emcee`` or
``dynesty``) as :math:`\log \mathcal{L} = -\chi^2/2`.

Accuracy
========

The errors below are emulator minus Hyperion :math:`\log_{10}` fluxes for
the public network, measured on about 34,000 held-out R24 models (models the
network never saw in training), in the 12,743 AU aperture, for bins within
6 dex of each SED's peak. Other apertures give similar values.

==============  ===========  ===========  ===========
wavelength      68% half-    95% half-    rms (dex)
                width (dex)  width (dex)
==============  ===========  ===========  ===========
1.0 µm          0.030        0.12         0.073
2.2 µm          0.030        0.10         0.084
3.6 µm          0.036        0.12         0.108
4.6 µm          0.040        0.15         0.121
7.8 µm          0.046        0.18         0.163
24 µm           0.029        0.10         0.094
70 µm           0.020        0.063        0.050
250 µm          0.016        0.039        0.021
880 µm          0.017        0.044        0.021
1.3 mm          0.017        0.046        0.022
3.1 mm          0.018        0.051        0.024
==============  ===========  ===========  ===========

The median error is below 0.002 dex at every wavelength, and the Monte Carlo
noise of the Hyperion models themselves is 0.003–0.008 dex. In the near- and
mid-infrared the rms exceeds the 68% half-width because a few per cent of the
models have errors of several tenths of a dex. Reasonable choices of
``model_err_dex`` (Route 2) or :math:`\sigma_{\rm emu}` (Route 1) are
0.05 dex for fits that include 1–24 µm bands and 0.02 dex for far-infrared
and (sub)millimetre fits. An ensemble of five such networks, averaged, reduces
the 68% half-widths by a factor of 1.3–1.6; only one network is public.

Choosing between the routes
===========================

* Route 1 keeps the familiar :func:`~sedfitter.fit` workflow, output files and
  plots, and fits many sources against one package in seconds. Its resolution
  is set by the number of rows, and its :math:`\chi^2` is the grid
  :math:`\chi^2` of :func:`~sedfitter.fit` (best models and
  :math:`\Delta\chi^2` cuts, as described in :doc:`fitting`).
* Route 2 has no resolution limit, gives posterior samples, and handles a
  different aperture for every band. It costs about a minute per source.

Caveats
=======

* The emulator reproduces the R24 models; every limitation of the models
  themselves (see :doc:`common_model_packages`) applies unchanged.
* Outside the prior ranges the network extrapolates without warning.
  ``fit_with_emulator.py`` keeps all parameters inside the ranges; for Route 1
  this holds by construction.
