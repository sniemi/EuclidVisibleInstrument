"""
Measuring a shape of an object
==============================

Simple class to measure quadrupole moments and ellipticity of an object.

.. Note:: Double check that the e1 component is not flipped in sense that Qxx and Qyy would be reversed
          because NumPy arrays are column major.

:requires: NumPy
:requres: PyFITS

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.3
"""
import math, os, datetime, unittest
import numpy as np
import pyfits as pf


class shapeMeasurement():
    """
    Provides methods to measure the shape of an object.

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
        self.data = data.copy()
        self.log = log

        sizeY, sizeX = self.data.shape
        self.settings = dict(sizeX=sizeX,
                             sizeY=sizeY,
                             iterations=4,
                             sampling=1.0,
                             platescale=120.0,
                             pixelSize=12.0,
                             sigma=0.75,
                             weighted=True)
        self.settings.update(kwargs)
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))


    def quadrupoles(self, img):
        """
        Derive quadrupole moments and ellipticity from the input image.

        :param img: input image data
        :type ig: ndarray

        :return: quadrupoles, centroid, and ellipticity (also the projected components e1, e2)
        :rtype: dict
        """
        self.log.info('Deriving quadrupole moments')
        image = img.copy()

        #normalization factor
        imsum = float(np.sum(image))

        #generate a mesh coordinate grid
        sizeY, sizeX = image.shape
        Xvector = np.arange(0, sizeX)
        Yvector = np.arange(0, sizeY)
        Xmesh, Ymesh = np.meshgrid(Xvector, Yvector)

        # No centroid given, take from data and weighting with input image
        Xcentre = np.sum(Xmesh.copy() * image.copy()) / imsum
        Ycentre = np.sum(Ymesh.copy() * image.copy()) / imsum

        #coordinate array
        Xarray = Xcentre * np.ones([sizeY, sizeX])
        Yarray = Ycentre * np.ones([sizeY, sizeX])

        #centroided positions
        Xpos = Xmesh - Xarray
        Ypos = Ymesh - Yarray

        #squared and cross term
        Xpos2 = Xpos * Xpos
        Ypos2 = Ypos * Ypos
        XYpos = Ypos * Xpos

        #integrand
        Qyyint = Ypos2 * image.copy()
        Qxxint = Xpos2 * image.copy()
        Qxyint = XYpos * image.copy()

        #sum over and normalize to get the quadrupole moments
        Qyy = np.sum(Qyyint) / imsum
        Qxx = np.sum(Qxxint) / imsum
        Qxy = np.sum(Qxyint) / imsum

        self.log.info('(Qxx, Qyy, Qxy) = (%f, %f, %f)' % (Qxx, Qyy, Qxy))

        #derive projections and ellipticity
        denom = Qxx + Qyy
        e1 = (Qxx - Qyy) / denom
        e2 = 2. * Qxy / denom
        ellipticity = math.sqrt(e1*e1 + e2*e2)

        if ellipticity > 1.0:
            self.log.error('Ellipticity greater than 1 derived, will set it to unity!')
            ellipticity = 1.0

        self.log.info('Centroiding (x, y) = (%f, %f) and ellipticity = %.4f (%.4f, %.4f)' %
                      (Ycentre+1, Xcentre+1, ellipticity, e1, e2))

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
        Gaussian = np.exp(-exponent) / (2. * math.pi * sigma*sigma)

        output = dict(GaussianXmesh=Gxmesh, GaussianYmesh=Gymesh, Gaussian=Gaussian)

        return output


    def Gaussian2D(self, x, y, sigmax, sigmay):
        """
        Create a two-dimensional Gaussian centered on x, y.

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
        self.log.info('Creating a 2D Gaussian with sigmax=%.3f and sigmay=%.3f centered on (x, y) = (%f, %f)' %
                      (sigmax, sigmay, x, y))

        #x and y coordinate vectors
        Gyvect = np.arange(1, self.settings['sizeY'] + 1)
        Gxvect = np.arange(1, self.settings['sizeX'] + 1)

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


    def measureRefinedEllipticity(self):
        """
        Derive a refined iterated ellipticity measurement for a given object.

        By default ellipticity is defined in terms of the Gaussian weighted quadrupole moments.
        If self.settings['weigted'] is False then no weighting scheme is used.
        The number of iterations is defined in self.settings['iterations'].

        :return centroids [indexing stars from 1], ellipticity (including projected e1 and e2), and R2
        :rtype: dict
        """
        self.settings['sampleSigma'] = self.settings['sigma'] / self.settings['pixelSize'] * \
                                       self.settings['platescale'] / self.settings['sampling']

        self.log.info('Sample sigma used for weighting = %f' % self.settings['sampleSigma'])

        self.log.info('The intial estimate for the mean values are taken from the unweighted quadrupole moments.')
        quad = self.quadrupoles(self.data.copy())

        for x in range(self.settings['iterations']):
            if self.settings['weighted']:
                self.log.info('Iteration %i with circular symmetric Gaussian weights' % x)
                gaussian = self.circular2DGaussian(quad['centreX']+1, quad['centreY']+1, self.settings['sampleSigma'])
                GaussianWeighted = self.data.copy() * gaussian['Gaussian'].copy()
            else:
                self.log.info('Iteration %i with no weighting' % x)
                GaussianWeighted = self.data.copy()

            quad = self.quadrupoles(GaussianWeighted.copy())

        # The squared radius R2 in um2
        R2 = quad['Qxx'] * self.settings['sampling']**2 + quad['Qyy'] * self.settings['sampling']**2

        out = dict(centreX=quad['centreX']+1, centreY=quad['centreY']+1,
                   e1=quad['e1'], e2=quad['e2'],
                   ellipticity=quad['ellipticity'],
                   R2=R2,
                   GaussianWeighted=GaussianWeighted)
        return out


    def writeFITS(self, data, output):
        """
        Write out a FITS file using PyFITS.

        :param data: data to write to a FITS file
        :type data: ndarray
        :param output: name of the output file
        :type output: string

        :return: None
        """
        if os.path.isfile(output):
            os.remove(output)

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=data)

        #update and verify the header
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(output)
        self.log.info('Wrote %s' % output)


class TestShape(unittest.TestCase):
    """
    Unit tests for the shape class.
    """
    def setUp(self):
        from support import logger as lg
        self.log = lg.setUpLogger('shapeTesting.log')

        self.psffile12x = '../data/psf12x.fits'
        self.psffile = '../data/psf1x.fits'
        self.tolerance = 1.e-6
        self.sigma = 40.0
        self.sigmax = 27.25
        self.sigmay = 14.15
        self.sigmax2 = 37.12343
        self.sigmay2 = 8.34543
        self.xcent = 150.
        self.ycent = 150.

        #create 2D Gaussians that will be used for testing
        self.GaussianCirc = shapeMeasurement(np.zeros((300, 300)), self.log).circular2DGaussian(self.xcent,
                                                                                                self.ycent,
                                                                                                self.sigma)['Gaussian']
        self.Gaussian = shapeMeasurement(np.zeros((300, 300)), self.log).Gaussian2D(self.xcent,
                                                                                    self.ycent,
                                                                                    self.sigmax,
                                                                                    self.sigmay)['Gaussian']
        self.Gaussian2 = shapeMeasurement(np.zeros((300, 300)), self.log).Gaussian2D(self.xcent,
                                                                                     self.ycent,
                                                                                     self.sigmax2,
                                                                                     self.sigmay2)['Gaussian']

    def test_ellipticity_noweighting_circular_Gaussian(self):
        expected = 0.0
        settings = dict(weighted=False)
        actual = shapeMeasurement(self.GaussianCirc, self.log, **settings).measureRefinedEllipticity()['ellipticity']
        self.assertAlmostEqual(expected, actual, msg='exp=%f, got=%f' % (expected, actual), delta=self.tolerance)


    def test_noweighting_Gaussian(self):
        expected = math.fabs((self.sigmax**2 - self.sigmay**2) / (self.sigmax**2 + self.sigmay**2))
        settings = dict(weighted=False)
        actual = shapeMeasurement(self.Gaussian, self.log, **settings).measureRefinedEllipticity()
        ae = actual['ellipticity']
        ae1 = actual['e1']
        ae2 = actual['e2']
        R2 = actual['R2']
        R2exp = 942.78413888701903
        self.assertAlmostEqual(expected, ae, msg='exp=%f, got=%f' % (expected, ae), delta=self.tolerance)
        self.assertAlmostEqual(expected, ae1, msg='exp=%f, got=%f' % (expected, ae1), delta=self.tolerance)
        self.assertAlmostEqual(0.0, ae2, msg='exp=%f, got=%f' % (expected, ae2), delta=self.tolerance)
        self.assertAlmostEqual(R2exp, R2, msg='exp=%f, got=%f' % (R2exp, R2), delta=self.tolerance)


    def test_noweighting_Gaussian2(self):
        expected = math.fabs((self.sigmax2**2 - self.sigmay2**2) / (self.sigmax2**2 + self.sigmay2**2))
        settings = dict(weighted=False, iterations=10)
        actual = shapeMeasurement(self.Gaussian2, self.log, **settings).measureRefinedEllipticity()
        ae = actual['ellipticity']
        ae1 = actual['e1']
        ae2 = actual['e2']
        R2 = actual['R2']
        R2exp = 1446.528033
        self.assertAlmostEqual(expected, ae, msg='exp=%f, got=%f' % (expected, ae), delta=self.tolerance*100)
        self.assertAlmostEqual(expected, ae1, msg='exp=%f, got=%f' % (expected, ae1), delta=self.tolerance*100)
        self.assertAlmostEqual(0.0, ae2, msg='exp=%f, got=%f' % (expected, ae2), delta=self.tolerance)
        self.assertAlmostEqual(R2exp, R2, msg='exp=%f, got=%f' % (R2exp, R2), delta=self.tolerance)


    def test_ellipticity_Gaussian(self):
        expected = 0.557664 #0.5752531064876935
        settings = dict(sigma=10.)
        actual = shapeMeasurement(self.Gaussian, self.log, **settings).measureRefinedEllipticity()['ellipticity']
        self.assertAlmostEqual(expected, actual, msg='exp=%f, got=%f' % (expected, actual), delta=self.tolerance)


    def test_centroiding_weighting_Gaussian(self):
        expected = self.xcent, self.ycent
        actual = shapeMeasurement(self.Gaussian, self.log).measureRefinedEllipticity()
        self.assertAlmostEqual(expected[0], actual['centreX'],
                               msg='exp=%f, got=%f' % (expected[0], actual['centreX']), delta=self.tolerance)
        self.assertAlmostEqual(expected[1], actual['centreY'],
                               msg='exp=%f, got=%f' % (expected[1], actual['centreY']), delta=self.tolerance)


    def test_R2_noweighting_circular_Gaussian(self):
        expected = 3191.531326
        settings = dict(weighted=False)
        actual = shapeMeasurement(self.GaussianCirc, self.log, **settings).measureRefinedEllipticity()['R2']
        self.assertAlmostEqual(expected, actual, msg='exp=%f, got=%f' % (expected, actual), delta=self.tolerance)


    def test_PSF12x(self):
        expected = 0.045536
        R2exp = 5.010087
        data = pf.getdata(self.psffile12x)
        settings = dict(sampling=1/12.0)
        actual = shapeMeasurement(data, self.log, **settings).measureRefinedEllipticity()
        self.assertAlmostEqual(expected, actual['ellipticity'],
                               msg='exp=%f, got=%f' % (expected, actual['ellipticity']), delta=self.tolerance)
        self.assertAlmostEqual(R2exp, actual['R2'],
                               msg='exp=%f, got=%f' % (R2exp, actual['R2']), delta=self.tolerance)

    def test_PSF(self):
        expected = 0.045437
        R2exp = 4.959904
        data = pf.getdata(self.psffile)
        actual = shapeMeasurement(data, self.log).measureRefinedEllipticity()
        self.assertAlmostEqual(expected, actual['ellipticity'],
                               msg='exp=%f, got=%f' % (expected, actual['ellipticity']), delta=self.tolerance)
        self.assertAlmostEqual(R2exp, actual['R2'],
                               msg='exp=%f, got=%f' % (R2exp, actual['R2']), delta=self.tolerance)



if __name__ == '__main__':
    #testing section
    suite = unittest.TestLoader().loadTestsFromTestCase(TestShape)
    unittest.TextTestRunner(verbosity=3).run(suite)
