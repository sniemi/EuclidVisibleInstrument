"""
Measuring a shape of an object
==============================

Simple class to measure quadrupole moments and ellipticity of an object.

:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import math, pprint
import numpy as np


class shapeMeasurement():
    """
    Provives methods to measure the shape of an object.

    :param data: name of the FITS file to be analysed.
    :type data: ndarray
    :param log: logger
    :type log: instance
    :param kwargs: additional keyword arguments
    :type kwargs: dict

    Settings dictionary contains all parameter values needed.
    """

    def __init__(self, data, log, **kwargs):
        """
        :param data: name of the FITS file to be analysed.
        :type data: ndarray
        :param log: logger
        :type log: instance
        :param kwargs: additional keyword arguments
        :type kwargs: dict

        Settings dictionary contains all parameter values needed.
        """
        self.data = data
        self.log = log

        sizeY, sizeX = self.data.shape
        self.settings = dict(sizeX=sizeX,
                             sizeY=sizeY,
                             iterations=4,
                             sampling=1.0,
                             platescale=120.0,
                             pixelSize=12.0,
                             sigma=0.75)
        self.settings.update(kwargs)
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))


    def quadrupoles(self, image):
        """
        Derive quadrupole moments and ellipticity from the input image.

        :param image: input image data
        :type image: ndarray

        :return: quadrupoles, centroid, and ellipticity (also the projected components e1, e2)
        :rtype: dict
        """
        self.log.info('Deriving quadrupole moments')

        #normalization factor
        imsum = float(np.sum(image))

        #generate a mesh coordinate grid
        sizeY, sizeX = image.shape
        Xvector = np.arange(0, sizeX)
        Yvector = np.arange(0, sizeY)
        Xmesh, Ymesh = np.meshgrid(Xvector, Yvector)

        # No centroid given, take from data and weighting with input image
        Xcentre = np.sum(Xmesh.copy() * image) / imsum
        Ycentre = np.sum(Ymesh.copy() * image) / imsum

        #coordinate array
        Xarray = Xcentre * np.ones([sizeY, sizeX])
        Yarray = Ycentre * np.ones([sizeY, sizeX])

        #centroided positions
        Xpos = Xmesh - Xarray
        Ypos = Ymesh - Yarray

        #squared and cross term
        Xpos2 = Xpos * Xpos
        Ypos2 = Ypos * Ypos
        XYpos2 = Ypos * Xpos

        #integrand
        Qyyint = Ypos2 * image
        Qxxint = Xpos2 * image
        Qlxint = XYpos2 * image

        #summ over and normalize to get the quadrupole moments
        Qyy = np.sum(Qyyint) / imsum
        Qxx = np.sum(Qxxint) / imsum
        Qxy = np.sum(Qlxint) / imsum

        self.log.info('(Qxx, Qyy, Qxy) = (%f, %f, %f)' % (Qxx, Qyy, Qxy))

        #derive projections and ellipticity
        denom = Qxx + Qyy
        e1 = (Qxx - Qyy) / denom
        e2 = 2. * Qxy / denom
        ellipticity = math.sqrt(e1*e1 + e2*e2)

        if ellipticity > 1.0:
            self.log.error('Ellipticity greater than 1 derived, will set it to unity!')
            ellipticity = 1.0

        self.log.info('Centroiding (x, y) = (%f, %f) and ellipticity = %.4f' % (Ycentre, Xcentre, ellipticity))

        out = dict(ellipticity=ellipticity, e1=e1, e2=e2, Qxx=Qxx, Qyy=Qyy, Qxy=Qxy, centreY=Ycentre, centreX=Xcentre)
        return out


    def circular2DGaussian(self, x, y, sigma):
        """
        Create a circular symmetric Gaussian centered on x, y.

        :param x: x coordinate of the centre
        :type x: float
        :param y: y coordinate of the centre
        :type y: float
        :param sigma: standard deviation of the Gaussian, note that sigma_x = sigma_y = sigma
        :type sigma: float

        :return: circular Gaussian 2D profile and x and y mesh grid
        :rtype: dict
        """
        self.log.info('Creating a circular symmetric 2D Gaussian with sigma=%.3f centered on (x, y) = (%f, %f)' % (sigma, x, y))

        #x and y coordinate vectors
        Gyvect = np.arange(1, self.settings['sizeY'] + 1)
        Gxvect = np.arange(1, self.settings['sizeX'] + 1)

        #meshgrid
        Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)

        #normalizers
        sigmax = 1. / (2. * sigma**2)
        sigmay = sigmax #same sigma in both directions, thus same normalizer

        #gaussian
        exponent = (sigmax * (Gxmesh - x)**2 + sigmay * (Gymesh - y)**2)
        Gaussian = np.exp(-exponent) / (2. * math.pi * sigma**2)

        output = dict(GaussianXmesh=Gxmesh, GaussianYmesh=Gymesh, Gaussian=Gaussian)

        return output


    def measureRefinedEllipticity(self):
        """
        Derive a refined iterated ellipticity measurement for a given object.
        Ellipticity is defined in terms of the Gaussian weighted quadrupole moments.

        The number of iterations is defined in self.settings['iterations'].

        :return centroids, ellipticity (including projected e1 and e2), and R2
        :rtype: dict
        """
        self.settings['sampleSigma'] = self.settings['sigma'] / self.settings['pixelSize'] * \
                                       self.settings['platescale'] / self.settings['sampling']

        self.log.info('Sample sigma used for weighting = %f' % self.settings['sampleSigma'])

        self.log.info('The intial estimate for the mean values are taken from the unweighted quadrupole moments.')
        quad = self.quadrupoles(self.data)

        for x in range(self.settings['iterations']):
            self.log.info('Iteration %i with circular symmetric Gaussian weights' % x)
            gaussian = self.circular2DGaussian(quad['centreX']+1, quad['centreY']+1, self.settings['sampleSigma'])
            GaussianWeighted = self.data * gaussian['Gaussian']
            quad = self.quadrupoles(GaussianWeighted)

        # R2 in um2
        R2 = quad['Qxx'] * self.settings['sampling']**2 + quad['Qyy'] * self.settings['sampling']**2

        out = dict(centreX=quad['centreX']+1, centreY=quad['centreY']+1,
                   e1=quad['e1'], e2=quad['e2'], ellipticity=quad['ellipticity'], R2=R2)

        return out


if __name__ == '__main__':
    #testing part, looks for blob?.fits and psf.fits to derive centroids and ellipticity
    import pyfits as pf
    import glob as g
    from support import logger as lg

    files = g.glob('blob*.fits')

    log = lg.setUpLogger('shape.log')
    log.info('Testing shape measuring class...')

    for file in files:
        log.info('Processing file %s' % file)
        data = pf.getdata(file)
        sh = shapeMeasurement(data, log)
        results = sh.measureRefinedEllipticity()

        print file
        pprint.pprint(results)
        print

    file = 'psf.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    sh = shapeMeasurement(data, log)
    results = sh.measureRefinedEllipticity()

    print file
    pprint.pprint(results)
    print

    file = 'stamp.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    sh = shapeMeasurement(data, log)
    results = sh.measureRefinedEllipticity()

    print file
    pprint.pprint(results)
    print

    file = 'gaussian.fits'
    log.info('Processing file %s' % file)
    data = pf.getdata(file)
    sh = shapeMeasurement(data, log)
    results = sh.measureRefinedEllipticity()

    print file
    pprint.pprint(results)
    print

    log.info('All done\n\n')