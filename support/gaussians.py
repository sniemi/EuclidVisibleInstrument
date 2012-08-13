"""
Functions related to 2D gaussian functions and comparing ellipticities
derived either analytically or using quadrupole moments.

:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.2
"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import math, datetime, pprint
from analysis import shape
from support import logger as lg
from support import files


def Gaussian2D(x, y, sizex, sizey, sigmax, sigmay):
    """
    Create a circular symmetric Gaussian centered on x, y.

    :param x: x coordinate of the centre
    :type x: float
    :param y: y coordinate of the centre
    :type y: float
    :param sigmax: standard deviation of the Gaussian in x-direction
    :type sigmax: float
    :param sigmay: standard deviation of the Gaussian in y-direction
    :type sigmay: float

    :return: circular Gaussian 2D profile and x and y mesh grid
    :rtype: dict
    """
    #x and y coordinate vectors
    Gyvect = np.arange(1, sizey + 1)
    Gxvect = np.arange(1, sizex + 1)

    #meshgrid
    Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)

    #normalizers
    sigx = 1. / (2. * sigmax**2)
    sigy = 1. / (2. * sigmay**2)

    #gaussian
    exponent = (sigx * (Gxmesh - x)**2 + sigy * (Gymesh - y)**2)
    Gaussian = np.exp(-exponent) / (2. * math.pi * sigmax*sigmay)

    output = dict(GaussianXmesh=Gxmesh, GaussianYmesh=Gymesh, Gaussian=Gaussian)

    return output


def plot3D(data):
    """
    Plot a 3d image of the input data. Assumes that the input dictionary
    contains X, Y, and Z.

    :param data: input data including X and Y mesh and Z-values
    :type data: dict
    """
    fig = plt.figure(figsize=(12,12))
    rect = fig.add_subplot(111, visible=False).get_position()
    ax = Axes3D(fig, rect)
    surf = ax.plot_surface(data['GaussianXmesh'],
                           data['GaussianYmesh'],
                           data['Gaussian'],
                           rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.savefig('gaussian.pdf')


def plotEllipticityDependency(data, ellipticity, log):
    """
    Generate a simple plot: size of the Gaussian weighting function vs. derived ellipticity.
    """
    x = []
    y = []
    for sigma in range(1, 50):
        settings = dict(sigma=sigma)
        sh = shape.shapeMeasurement(data, log, **settings)
        results = sh.measureRefinedEllipticity()
        x.append(sigma)
        y.append(results['ellipticity'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'bo-')
    ax.plot([min(x), max(x)], [ellipticity, ellipticity], 'k--')
    ax.set_xlabel(r'Gaussian Weighting $\sigma$ [arcseconds]')
    ax.set_ylabel('Measured Ellipticity')
    ax.set_ylim(0, 1.01)
    plt.savefig('EvsSigma.pdf')
    plt.close()


def ellipticityFromSigmas(sigmax, sigmay):
    """
    Calculate ellipticity from standard deviations of a 2D Gaussian.

    :param sigmax: standard deviation in x direction
    :type sigmax: float or ndarray
    :param sigmay: standard deviation in y direction
    :type sigmay: float or ndarray

    :return: ellipticity
    :rtype: float or ndarray
    """
    e = (np.float(sigmax)**2 - sigmay**2) / (sigmax**2 + sigmay**2)
    return np.abs(e)


def size():
    """
    :requires: sympy
    """
    from sympy import Symbol
    from sympy import integrate, exp, pi

    x = Symbol('x')
    y = Symbol('y')
    mu = Symbol('mu')
    sigma = Symbol('sigma')

    tmpx = (x - mu)
    tmpy = (y - mu)
    integrand = (1/(2*pi*sigma**2)) * exp(-((tmpx**2 + tmpy**2) / (2*sigma**2) ))

    res = integrate(integrand, x)
    pprint.pprint(res)


def measureGaussianR2(log):
    #gaussian
    sigma = 2. / (2. * math.sqrt(2.*math.log(2)))
    Gaussian = shape.shapeMeasurement(np.zeros((100, 100)), log).circular2DGaussian(50, 50, sigma)['Gaussian']

    settings = dict(sigma=sigma, weighted=False)
    sh = shape.shapeMeasurement(Gaussian, log, **settings)
    results = sh.measureRefinedEllipticity()
    print
    print results['R2']
    print
    #sh.writeFITS(Gaussian, 'GaussianSmall.fits')


def testFiles():
    #testing part, looks for blob?.fits and psf.fits to derive centroids and ellipticity
    import pyfits as pf
    import glob as g
    from support import logger as lg
    import sys

    files = g.glob('blob?.fits')

    log = lg.setUpLogger('shape.log')
    log.info('Testing shape measuring class...')

    for file in files:
        log.info('Processing file %s' % file)
        data = pf.getdata(file)
        sh = shape.shapeMeasurement(data, log)
        results = sh.measureRefinedEllipticity()
        sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))

        print file
        pprint.pprint(results)
        print

    file = 'psf1x.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    sh = shape.shapeMeasurement(data, log)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    print file
    pprint.pprint(results)
    print

    file = 'stamp.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    settings = dict(sigma=10.0)
    sh = shape.shapeMeasurement(data, log, **settings)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    print file
    pprint.pprint(results)
    print

    file = 'gaussian.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    settings = dict(sampling=0.2)
    sh = shape.shapeMeasurement(data, log, **settings)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    print file
    pprint.pprint(results)
    print

    log.info('All done\n\n')


if __name__ == '__main__':
    log = lg.setUpLogger('gaussians.log')
    log.info('Testing gaussians...')

    xsize, ysize = 300, 300
    xcen, ycen = 150, 150
    sigmax = 27.25
    sigmay = 14.15

    #calculate ellipticity from Sigmas
    e = ellipticityFromSigmas(sigmax, sigmay)

    #generate a 2D gaussian with given properties...
    gaussian2d = Gaussian2D(xcen, ycen, xsize, ysize, sigmax, sigmay)

    #plot
    plot3D(gaussian2d)

    #write FITS file
    files.writeFITS(gaussian2d['Gaussian'], 'gaussian.fits')

    #calculate shape and printout results
    settings = dict(sigma=15., weighted=False)
    sh = shape.shapeMeasurement(gaussian2d['Gaussian'], log, **settings)
    results = sh.measureRefinedEllipticity()
    print
    pprint.pprint(results)
    print e, (e - results['ellipticity']) / e * 100.

    #generate a plot sigma vs ellipticity for a given Gaussian
    plotEllipticityDependency(gaussian2d['Gaussian'], e, log)

    #measureGaussianR2
    measureGaussianR2(log)

    #derive FWHM - R2 relation... not really working
    #size()

    #test many files
    testFiles()

    log.info('All done\n\n')