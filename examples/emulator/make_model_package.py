"""
Write an emulated spubsmi model package that sedfitter.fit reads like the
published grid: flux.fits (SEDCube), parameters.fits and models.conf
(version 2).

    python make_model_package.py model_emulated 200000

The models are drawn from the R24 priors, so a package with N models is an
R24-like grid of any density.  Each row here is one (model, inclination)
pair and takes 26 kB in flux.fits (20 apertures x 165 wavelengths x
values + uncertainties in float32), so 10^5 rows need 2.6 GB.  Rows can
instead be concentrated where the data need them (e.g. a range of
luminosities) by changing ``theta`` below.
"""
import os
import sys

import numpy as np
from astropy import units as u
from astropy.table import Table

from sedfitter.sed import SEDCube
from yso_emulator import YSOEmulator

# Stored as the SED uncertainty for reference.  sedfitter's chi^2 uses only
# the data errors, so add the emulator error to those in quadrature instead
# (see docs/emulator.rst).
MODEL_ERR_DEX = 0.03


def make_package(model_dir, n_models, emulator_dir='emulator_files', seed=0, chunk=20000):
    emu = YSOEmulator.download(emulator_dir)
    emu.check()
    theta = emu.prior_draws(n_models, seed=seed)

    os.makedirs(model_dir, exist_ok=True)
    val = np.empty((n_models, len(emu.apertures), len(emu.wav)), np.float32)
    for i in range(0, n_models, chunk):
        lf = emu.log_flux(theta[i:i + chunk])
        # the grid sets photon-free bins (here: within 0.1 dex of the
        # emulator's training floor, 8 dex below each SED's peak) to zero
        floor = lf < lf.max(axis=2, keepdims=True) - 7.9
        v = 10. ** lf
        v[floor] = 0.
        val[i:i + chunk] = v

    names = np.array([f'emu{i:08d}' for i in range(n_models)])
    cube = SEDCube()               # set n_models (names) before 'valid'
    cube.names = names
    cube.distance = (1. * u.kpc).to(u.cm)
    cube.wav = emu.wav * u.micron
    cube.apertures = emu.apertures * u.au
    cube.val = val * u.mJy
    cube.unc = val * np.float32(np.log(10.) * MODEL_ERR_DEX) * u.mJy     # float32 halves the file
    cube.valid = np.ones(n_models, int)
    cube.write(os.path.join(model_dir, 'flux.fits'), overwrite=True)

    t = Table()
    t['MODEL_NAME'] = names.astype('S20')
    for j, (name, lg) in enumerate(zip(emu.names, emu.is_log)):
        t[name] = 10. ** theta[:, j] if lg else theta[:, j]
    t['envelope.rc'] = t['disk.rmax']
    t['Source Luminosity'] = 10. ** emu.luminosity(theta)
    t['log Mcore'] = emu.log_mcore(theta)
    t.write(os.path.join(model_dir, 'parameters.fits'), overwrite=True)

    with open(os.path.join(model_dir, 'models.conf'), 'w') as f:
        f.write('name = spubsmi (emulated)\n'
                'length_subdir = 2\n'
                'aperture_dependent = yes\n'
                'logd_step = 0.02\n'
                'version = 2\n')
    return model_dir


if __name__ == '__main__':
    model_dir = sys.argv[1] if len(sys.argv) > 1 else 'model_emulated'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    make_package(model_dir, n)

    # Version-2 packages need no convolution for monochromatic fits: pass
    # wavelengths (astropy Quantities) as filter_names in sedfitter.fit and the
    # nearest tabulated wavelength is used.  For broadband photometry, build
    # sedfitter.filter.Filter objects and run
    # sedfitter.convolve.convolve_model_dir(model_dir, filters).
