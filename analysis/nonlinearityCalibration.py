"""
Non-linearity Calibration
=========================

This simple script can be used to study the error in the non-linearity correction that can be tolerated given the
requirements.

The following requirements related to the non-linearity has been taken from GDPRD.

R-GDP-CAL-058: The contribution of the residuals of the non-linearity correction on the error on the determination
of each ellipticity component of the local PSF shall not exceed 3x10**-5 (one sigma).

R-GDP-CAL-068: The contribution of the residuals of the non-linearity correction on the error on the relative
error sigma(R**2)/R**2 on the determination of the local PSF R**2 shall not exceed 1x10**-4 (one sigma).

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools, glob, os, sys
from scipy.ndimage.interpolation import zoom
from scipy import interpolate
from analysis import shape
from support import logger as lg
from support import files as fileIO


def testNonlinearity(log, file='data/psf12x.fits', oversample=12.0, slopes=500):
    """

    """
    #read in PSF and renormalize it
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 1e5

    #derive reference values
    settings = dict(sampling=1.0/oversample)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()
    print reference

    out = {}
    de1 = []
    de2 = []
    de = []
    R2 = []
    dR2 = []
    e1 = []
    e2 = []
    e = []
    for slope in xrange(slopes):
        print'%i / %i' % (slope, slopes)

        #non-linearity residual after correction
        residual = data.copy()

        #measure e and R2 from the postage stamp image
        sh = shape.shapeMeasurement(residual.copy(), log, **settings)
        results = sh.measureRefinedEllipticity()

        #save values
        e1.append(results['e1'])
        e2.append(results['e2'])
        e.append(results['ellipticity'])
        R2.append(results['R2'])
        de1.append(results['e1'] - reference['e1'])
        de2.append(results['e2'] - reference['e2'])
        de.append(results['ellipticity'] - reference['ellipticity'])
        dR2.append(results['R2'] - reference['R2'])

    out[1] = [e1, e2, e, R2, de1, de2, de, dR2]


if __name__ == '__main__':
    #start the script
    log = lg.setUpLogger('nonlinearityCalibration.log')
    log.info('Testing non-linearity calibration...')

    testNonlinearity(log)

    log.info('Run finished...\n\n\n')
