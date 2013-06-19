"""
The Euclid Visible Instrument Image Simulator
=============================================

This file contains an image simulator for the Euclid VISible instrument.

The approximate sequence of events in the simulator is as follows:

      #. Read in a configuration file, which defines for example,
         detector characteristics (bias, dark and readout noise, gain,
         plate scale and pixel scale, oversampling factor, exposure time etc.).
      #. Read in another file containing charge trap definitions (for CTI modelling).
      #. Read in a file defining the cosmic rays (trail lengths and cumulative distributions).
      #. Read in CCD offset information, displace the image, and modify
         the output file name to contain the CCD and quadrant information
         (note that VIS has a focal plane of 6 x 6 detectors).
      #. Read in a source list and determine the number of different object types.
      #. Read in a file which assigns data to a given object index.
      #. Load the PSF model (a 2D map with a given over sampling or field dependent maps).
      #. Generate a finemap (oversampled image) for each object type. If an object
         is a 2D image then calculate the shape tensor to be used for size scaling.
         Each type of an object is then placed onto its own finely sampled finemap.
      #. Loop over the number of exposures to co-add and for each object in the object catalog:

            * determine the number of electrons an object should have by scaling the object's magnitude
              with the given zeropoint and exposure time.
            * determine whether the object lands on to the detector or not and if it is
              a star or an extended source (i.e. a galaxy).
            * if object is extended determine the size (using a size-magnitude relation) and scale counts,
              convolve with the PSF, and finally overlay onto the detector according to its position.
            * if object is a star, scale counts according to the derived
              scaling (first step), and finally overlay onto the detector according to its position.

      #. Apply calibration unit flux to mimic flat field exposures [optional].
      #. Apply a multiplicative flat-field map to emulate pixel-to-pixel non-uniformity [optional].
      #. Add a charge injection line (horizontal and/or vertical) [optional].
      #. Add cosmic ray tracks onto the CCD with random positions but known distribution [optional].
      #. Apply detector charge bleeding in column direction [optional].
      #. Add constant dark current and background light from Zodiacal light [optional].
      #. Include spatially uniform scattered light to the pixel grid [optional].
      #. Add photon (Poisson) noise [optional]
      #. Add cosmetic defects from an input file [optional].
      #. Add pre- and overscan regions in the serial direction [optional].
      #. Apply the CDM03 radiation damage model [optional].
      #. Apply CCD273 non-linearity model to the pixel data [optional].
      #. Add readout noise selected from a Gaussian distribution [optional].
      #. Convert from electrons to ADUs using a given gain factor.
      #. Add a given bias level and discretise the counts (the output is going to be in 16bit unsigned integers).
      #. Finally the simulated image is converted to a FITS file, a WCS is assigned
         and the output is saved to the current working directory.

.. Warning:: The code is still work in progress and new features are being added.
             The code has been tested, but nevertheless bugs may be lurking in corners, so
             please report any weird or inconsistent simulations to the author.


Dependencies
------------

This script depends on the following packages:

:requires: PyFITS (tested with 3.0.6)
:requires: NumPy (tested with 1.6.1)
:requires: numexpr (tested with 2.0.1)
:requires: SciPy (tested with 0.10.1)
:requires: vissim-python package

.. Note:: This class is not Python 3 compatible. For example, xrange does not exist
          in Python 3 (but is used here for speed and memory consumption improvements).
          In addition, at least the string formatting should be changed if moved to
          Python 3.x.


Testing
-------

Before trying to run the code, please make sure that you have compiled the
cdm03.f90 Fortran code using f2py (f2py -c -m cdm03 cdm03.f90). For testing,
please run the unittest as follows::

    python simulator.py -t

This will run the unittest and compare the result to a previously calculated image.
Please inspect the standard output for results.

Running the test will produce an image representing VIS lower left (0th) quadrant of the CCD (x, y) = (0, 0). Because
noise and cosmic rays are randomised one cannot directly compare the science
outputs but we must rely on the outputs that are free from random effects.

In the data subdirectory there is a file called "nonoisenocrQ0_00_00testscience.fits",
which is the comparison image without any noise or cosmic rays. To test the functionality,
please divide your nonoise and no cosmic ray track output image with the on in the data
folder. This should lead to a uniformly unity image or at least very close given some
numerical rounding uncertainties, especially in the FFT convolution (which is float32 not
float64).


Benchmarking
------------

A minimal benchmarking has been performed using the TESTSCIENCE1X section of the test.config input file::

    Galaxy: 26753/26753 magnitude=26.710577 intscale=177.489159281 FWHM=0.117285374813 arc sec
    6791 objects were place on the detector

    real	2m19.378s
    user	2m13.237s
    sys	        0m4.394s


These numbers have been obtained with my laptop (2.2 GHz Intel Core i7) with
64-bit Python 2.7.2 installation. Further speed testing can be performed using the cProfile module
as follows::

    python -m cProfile -o vissim.profile simulator.py -c data/test.config -s TESTSCIENCE3X

and then analysing the results with e.g. RunSnakeRun.

.. Note: The result above was obtained with nominally sampled PSF, however, that is only good for
         testing purposes. If instead one uses say three times over sampled PSF (TESTSCIENCE3x) then the
         execution time rises significantly (to about 20 minutes). This is mostly due to the fact that convolution
         becomes rather expensive when done in the finely sampled PSF domain.


Change Log
----------

:version: 1.2

Version and change logs::

    0.1: pre-development backbone.
    0.4: first version with most pieces together.
    0.5: this version has all the basic features present, but not fully tested.
    0.6: implemented pre/overscan, fixed a bug when an object was getting close to the upper right corner of an
         image it was not overlaid correctly. Included multiplicative flat fielding effect (pixel non-uniformity).
    0.7: implemented bleeding.
    0.8: cleaned up the code and improved documentation. Fixed a bug related to checking if object falls on the CCD.
         Improved the information that is being written to the FITS header.
    0.9: fixed a problem with the CTI model swapping Q1 with Q2. Fixed a bug that caused the pre- and overscan to
         be identical for each quadrant even though Q1 and 3 needs the regions to be mirrored.
    1.0: First release. The code can now take an over sampled PSF and use that for convolutions. Implemented a WCS
         to the header.
    1.05: included an option to add flux from the calibration unit to allow flat field exposures to be generated.
          Now scaled the number of cosmic rays with the exposure time so that 10s flats have an appropriate number
          of cosmic ray tracks.
    1.06: changed how stars are laid down on the CCD. Now the PSF is interpolated to a new coordinate grid in the
          oversampled frame after which it is downsampled to the CCD grid. This should increase the centroiding
          accuracy.
    1.07: included an option to apply non-linearity model. Cleaned the documentation.
    1.08: optimised some of the operations with numexpr (only a minor improvement).
    1.1: Fixed a bug related to adding the system readout noise. In previous versions the readout noise was
         being underestimated due to the fact that it was included as a variance not standard deviation.
    1.2: Included a spatially uniform scattered light. Changed how the image pixel values are rounded before
         deriving the Poisson noise. Included focal plane CCD gaps. Included a unittest.
    1.21: included an option to exclude cosmic background; separated dark current from background.


Future Work
-----------

.. todo::

    #. objects.dat is now hard coded into the code, this should be read from the config file
    #. implement spatially variable PSF
    #. test that the WCS is correctly implemented and allows CCD offsets
    #. charge injection line positions are now hardcoded to the code, read from the config file
    #. include rotation in metrology
    #. implement optional dithered offsets
    #. try to further improve the convolution speed (look into fftw package)
    #. CCD273 has 4 pixel row gap between the top and bottom half, this is not taken into account in coordinate shifts


Contact Information
-------------------

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import os, sys, datetime, math, pprint, unittest
import ConfigParser
from optparse import OptionParser
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import interpolation
from scipy import ndimage
from scipy import signal
import pyfits as pf
import numpy as np
import numexpr as ne
from CTI import CTI
from support import logger as lg
from support import VISinstrumentModel

__author__ = 'Sami-Matias Niemi'
__version__ = 1.21


class VISsimulator():
    """
    Euclid Visible Instrument Image Simulator

    The image that is being build is in::

        self.image

    :param opts: OptionParser instance
    :type opts: OptionParser instance
    """

    def __init__(self, opts):
        """
        Class Constructor.

        :param opts: OptionParser instance
        :type opts: OptionParser instance
        """
        self.configfile = opts.configfile

        if opts.section is None:
            self.section = 'DEFAULT'
        else:
            self.section = opts.section

        if opts.debug is None:
            self.debug = False
        else:
            self.debug = opts.debug

        try:
            self.random = opts.testing
        except:
            self.random = False

        #load instrument model
        self.information = VISinstrumentModel.VISinformation()

        #update settings with defaults
        self.information.update(dict(quadrant=int(opts.quadrant),
                                     ccdx=int(opts.xCCD),
                                     ccdy=int(opts.yCCD),
                                     psfoversampling=1.0,
                                     ccdxgap=1.643,
                                     ccdygap=8.116,
                                     xsize=2048,
                                     ysize=2066,
                                     prescanx=50,
                                     ovrscanx=20,
                                     fullwellcapacity=200000,
                                     dark=0.001,
                                     readout=4.5,
                                     bias=1000.0,
                                     cosmic_bkgd=0.182758225257,
                                     scattered_light=2.96e-2,
                                     e_adu=3.1,
                                     magzero=15182880871.225231,
                                     exposures=1,
                                     exptime=565.0,
                                     ra=123.0,
                                     dec=45.0,
                                     flatflux='data/VIScalibrationUnitflux.fits',
                                     cosmicraylengths='data/cdf_cr_length.dat',
                                     cosmicraydistance='data/cdf_cr_total.dat',
                                     flatfieldfile='data/VISFlatField2percent.fits',
                                     trapfile='data/cdm_euclid.dat'))

        #setup logger
        self.log = lg.setUpLogger('VISsim.log')


    def readConfigs(self):
        """
        Reads the config file information using configParser.
        """
        self.config = ConfigParser.RawConfigParser()
        self.config.readfp(open(self.configfile))


    def processConfigs(self):
        """
        Processes configuration information and save the information to a dictionary self.information.

        The configuration file may look as follows::

            [TEST]
            quadrant = 0
            CCDx = 0
            CCDy = 0
            CCDxgap = 1.643
            CCDygap = 8.116
            xsize = 2048
            ysize = 2066
            prescanx = 50
            ovrscanx = 20
            fullwellcapacity = 200000
            dark = 0.001
            readout = 4.5
            bias = 1000.0
            cosmic_bkgd = 0.182758225257
            e_ADU = 3.1
            injection = 150000.0
            magzero = 15182880871.225231
            exposures = 1
            exptime = 565.0
            RA = 145.95
            DEC = -38.16
            sourcelist = data/source_test.dat
            PSFfile = data/interpolated_psf.fits
            trapfile = data/cdm_euclid.dat
            cosmeticsFile = data/cosmetics.dat
            flatfieldfile = data/VISFlatField2percent.fits
            output = test.fits
            addSources = yes
            noise = yes
            cosmetics = no
            chargeInjectionx = no
            chargeInjectiony = no
            radiationDamage = yes
            cosmicRays = yes
            overscans = yes
            bleeding = yes
            flatfieldM = yes
            random = yes
            background = yes

        For explanation of each field, see /data/test.config. Note that if an input field does not exist,
        then the values are taken from the default instrument model as described in
        support.VISinstrumentModel.VISinformation(). Any of the defaults can be overwritten by providing
        a config file with a correct field name.
        """
        #parse options and update the information dictionary
        options = self.config.options(self.section)
        settings = {}
        for option in options:
            try:
                settings[option] = self.config.getint(self.section, option)
            except ValueError:
                try:
                    settings[option] = self.config.getfloat(self.section, option)
                except ValueError:
                    settings[option] = self.config.get(self.section, option)

        self.information.update(settings)

        #force gain to be float
        self.information['e_adu'] = float(self.information['e_adu'])

        #name of the output file, include quadrants and CCDs
        self.information['output'] = 'Q%i_0%i_0%i%s' % (self.information['quadrant'],
                                                        self.information['ccdx'],
                                                        self.information['ccdy'],
                                                        self.config.get(self.section, 'output'))

        #booleans to control the flow
        self.chargeInjectionx = self.config.getboolean(self.section, 'chargeInjectionx')
        self.chargeInjectiony = self.config.getboolean(self.section, 'chargeInjectiony')
        self.cosmicRays = self.config.getboolean(self.section, 'cosmicRays')
        self.noise = self.config.getboolean(self.section, 'noise')
        self.cosmetics = self.config.getboolean(self.section, 'cosmetics')
        self.radiationDamage = self.config.getboolean(self.section, 'radiationDamage')
        self.addsources = self.config.getboolean(self.section, 'addSources')
        self.bleeding = self.config.getboolean(self.section, 'bleeding')
        self.overscans = self.config.getboolean(self.section, 'overscans')

        #these don't need to be in the config file
        try:
            self.lampFlux = self.config.getboolean(self.section, 'lampFlux')
        except:
            self.lampFlux = False
        try:
            self.nonlinearity = self.config.getboolean(self.section, 'nonlinearity')
        except:
            self.nonlinearity = False
        try:
            self.flatfieldM = self.config.getboolean(self.section, 'flatfieldM')
        except:
            self.flatfieldM = False
        try:
            self.scatteredlight = self.config.getboolean(self.section, 'scatteredLight')
        except:
            self.scatteredlight = True
        try:
            self.readoutNoise =  self.config.getboolean(self.section, 'readoutNoise')
        except:
            self.readoutNoise = True
        try:
            self.postages = self.config.getboolean(self.section, 'postageStamps')
        except:
            self.postages = True
        try:
            self.random = self.config.getboolean(self.section, 'random')
        except:
            self.random = False
        try:
            self.background = self.config.getboolean(self.section, 'background')
        except:
            self.background = True
        try:
            self.intscale = self.config.getboolean(self.section, 'intscale')
        except:
            self.intscale = True


        self.information['variablePSF'] = False

        self.booleans = dict(nonlinearity=self.nonlinearity,
                             flatfieldM=self.flatfieldM,
                             lampFlux=self.lampFlux,
                             chargeInjectionx=self.chargeInjectionx,
                             chargeInjectiony=self.chargeInjectiony,
                             cosmicRays=self.cosmicRays,
                             noise=self.noise,
                             cosmetics=self.cosmetics,
                             radiationDamage=self.radiationDamage,
                             addsources=self.addsources,
                             bleeding=self.bleeding,
                             overscans=self.overscans,
                             random=self.random,
                             background=self.background,
                             intscale=self.intscale)

        if self.debug:
            pprint.pprint(self.information)

        self.log.info('Using the following input values:')
        for key, value in self.information.iteritems():
            self.log.info('%s = %s' % (key, value))
        self.log.info('Using the following booleans:')
        for key, value in self.booleans.iteritems():
            self.log.info('%s = %s' % (key, value))


    def _createEmpty(self):
        """
        Creates and empty array of a given x and y size full of zeros.
        """
        self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)


    def cosmicRayIntercepts(self, lum, x0, y0, l, phi):
        """
        Derive cosmic ray streak intercept points.

        :param lum: luminosities of the cosmic ray tracks
        :param x0: central positions of the cosmic ray tracks in x-direction
        :param y0: central positions of the cosmic ray tracks in y-direction
        :param l: lengths of the cosmic ray tracks
        :param phi: orientation angles of the cosmic ray tracks

        :return: map
        :rtype: nd-array
        """
        #create empty array
        crImage = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)

        #this is very slow way to do this
        for cosmics in xrange(0, len(l)):
            #delta x and y
            dx = l[cosmics] * np.cos(phi[cosmics])
            dy = l[cosmics] * np.sin(phi[cosmics])

            #pixels in x-direction
            ilo = np.floor(x0[cosmics] - l[cosmics])

            if ilo < 1.:
                ilo = 1

            ihi = 1 + np.floor(x0[cosmics] + l[cosmics])

            if ihi > self.information['xsize']:
                ihi = self.information['xsize']

            #pixels in y-directions
            jlo = np.floor(y0[cosmics] - l[cosmics])

            if jlo < 1.:
                jlo = 1

            jhi = 1 + np.floor(y0[cosmics] + l[cosmics])
            if jhi > self.information['ysize']:
                jhi = self.information['ysize']

            u = []
            x = []
            y = []

            n = 0  # count the intercepts

            #Compute X intercepts on the pixel grid
            if dx > 0.:
                for j in xrange(int(ilo), int(ihi)):
                    ok = (j - x0[cosmics]) / dx
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(j)
                        y.append(y0[cosmics] + ok * dy)

            #Compute Y intercepts on the pixel grid
            if dy > 0.:
                for j in xrange(int(jlo), int(jhi)):
                    ok = (j - y0[cosmics]) / dy
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(x0[cosmics] + ok * dx)
                        y.append(j)

            #check if no intercepts were found
            if n < 1:
                i = np.floor(x0[cosmics])
                j = np.floor(y0[cosmics])
                crImage[j, i] += lum[cosmics]

            #Find the arguments that sort the intersections along the track.
            u = np.asarray(u)
            x = np.asarray(x)
            y = np.asarray(y)

            args = np.argsort(u)

            u = u[args]
            x = x[args]
            y = y[args]

            #Decide which cell each interval traverses, and the path length.
            for i in xrange(1, n - 1):
                w = u[i + 1] - u[i]
                cx = 1 + np.floor((x[i + 1] + x[i]) / 2.0)
                cy = 1 + np.floor((y[i + 1] + y[i]) / 2.0)

                if cx >= 0 and cx < self.information['xsize'] and cy >= 0 and cy < self.information['ysize']:
                    crImage[cy, cx] += (w * lum[cosmics])

        return crImage


    def readCosmicRayInformation(self):
        """
        Reads in the cosmic ray track information from two input files.

        Stores the information to a dictionary called cr.
        """
        self.log.info('Reading in cosmic ray information from %s and %s' % (self.information['cosmicraylengths'],
                                                                            self.information['cosmicraydistance']))

        crLengths = np.loadtxt(self.information['cosmicraylengths'])
        crDists = np.loadtxt(self.information['cosmicraydistance'])

        self.cr = dict(cr_u=crLengths[:, 0], cr_cdf=crLengths[:, 1], cr_cdfn=np.shape(crLengths)[0],
                       cr_v=crDists[:, 0], cr_cde=crDists[:, 1], cr_cden=np.shape(crDists)[0])


    def _writeFITSfile(self, image, filename):
        """
        :param image: image array to save
        :type image: ndarray
        :param filename: name of the output file, e.g. file.fits
        :type filename: str

        :return: None
        """
        if os.path.isfile(filename):
            os.remove(filename)

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=image)

        #update and verify the header
        hdu.header.add_history('Created by VISsim at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(filename)


    def objectOnDetector(self, object):
        """
        Tests if the object falls on the detector.

        :param object: object to be placed to the self.image.

        :return: whether the object falls on the detector or not
        :rtype: boolean
        """
        ny, nx = self.finemap[object[3]].shape
        mx = self.information['xsize']
        my = self.information['ysize']
        xt = object[0]
        yt = object[1]

        if object[3] > 0:
            #galaxy
            fac = (0.2**((object[2] - 22.)/7.)) / self.shapey[object[3]] / 2.
        else:
            #star
            fac = 1.0

        #Assess the boundary box of the input image.
        xlo = (1 - nx) * 0.5 * fac + xt
        xhi = (nx - 1) * 0.5 * fac + xt
        ylo = (1 - ny) * 0.5 * fac + yt
        yhi = (ny - 1) * 0.5 * fac + yt

        i1 = np.floor(xlo + 0.5)
        i2 = np.floor(xhi + 0.5) + 1
        j1 = np.floor(ylo + 0.5)
        j2 = np.floor(yhi + 0.5) + 1

        if i2 < 1 or i1 > mx:
            return False

        if j2 < 1 or j1 > my:
            return False

        return True


    def overlayToCCD(self, data, obj):
        """
        Overlay data from a source object onto the self.image.

        :param data: ndarray of data to be overlaid on to self.image
        :type data: ndarray
        :param obj: object information such as x,y position
        :type obj: list
        """
        #object centre x and y coordinates (only in full pixels, fractional has been taken into account already)
        xt = np.floor(obj[0])
        yt = np.floor(obj[1])

        #input array size
        nx = data.shape[1]
        ny = data.shape[0]

        # Assess the boundary box of the input image
        xlo = (1 - nx) * 0.5 + xt
        xhi = (nx - 1) * 0.5 + xt + 1
        ylo = (1 - ny) * 0.5 + yt
        yhi = (ny - 1) * 0.5 + yt + 1

        i1 = int(np.floor(xlo + 0.5))
        if i1 < 1:
            i1 = 0

        i2 = int(np.floor(xhi + 0.5))
        if i2 > self.information['xsize']:
            i2 = self.information['xsize']

        j1 = int(np.floor(ylo + 0.5))
        if j1 < 1:
            j1 = 0

        j2 = int(np.floor(yhi + 0.5))
        if j2 > self.information['ysize']:
            j2 = self.information['ysize']

        if i1 > i2 or j1 > j2:
            self.log.info('Object does not fall on the detector...')
            return

        ni = i2 - i1
        nj = j2 - j1

        self.log.info('Adding an object to (x,y)=({0:.4f}, {1:.4f})'.format(xt, yt))
        self.log.info('Bounding box = [%i, %i : %i, %i]' % (i1, i2, j1, j2))

        #add to the image
        if ni == nx and nj == ny:
            #full frame will fit
            self.image[j1:j2, i1:i2] += data
        elif ni < nx and nj == ny:
            #x dimensions shorter
            if int(np.floor(xlo + 0.5)) < 1:
                #small values, left side
                self.image[j1:j2, i1:i2] += data[:, nx-ni:]
            else:
                #large values, right side
                self.image[j1:j2, i1:i2] += data[:, :ni]
        elif nj < ny and ni == nx:
            #y dimensions shorter
            if int(np.floor(ylo + 0.5)) < 1:
                #small values, bottom
                self.image[j1:j2, i1:i2] += data[ny-nj:, :]
            else:
                #large values, top
                self.image[j1:j2, i1:i2] += data[:nj, :]
        else:
            #both lengths smaller, can be in any of the four corners
            if int(np.floor(xlo + 0.5)) < 1 > int(np.floor(ylo + 0.5)):
                #left lower
                self.image[j1:j2, i1:i2] += data[ny-nj:, nx-ni:]
            elif int(np.floor(xlo + 0.5)) < 1 and int(np.floor(yhi + 0.5)) > self.information['ysize']:
                #left upper
                self.image[j1:j2, i1:i2] += data[:nj, nx-ni:]
            elif int(np.floor(xhi + 0.5)) > self.information['xsize'] and int(np.floor(ylo + 0.5)) < 1:
                #right lower
                self.image[j1:j2, i1:i2] += data[ny-nj:, :ni]
            else:
                #right upper
                self.image[j1:j2, i1:i2] += data[:nj, :ni]


    def writeFITSfile(self, data, filename, unsigned16bit=False):
        """
        Writes out a simple FITS file.

        :param data: data to be written
        :type data: ndarray
        :param filename: name of the output file
        :type filename: str
        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool

        :return: None
        """
        if os.path.isfile(filename):
            os.remove(filename)

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=data)

        #convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #add input keywords to the header
        for key, value in self.information.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.update(key.upper(), value)
            except:
                try:
                    hdu.header.update(key.upper(), str(value))
                except:
                    pass

        #write booleans
        for key, value in self.booleans.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.update(key.upper(), str(value), 'Boolean Flags')

        #update and verify the header
        hdu.header.add_history('This is an itermediate data product no the final output!')
        hdu.header.add_history('Created by VISsim (version=%.2f) at %s' % (__version__,
                                                                           datetime.datetime.isoformat(datetime.datetime.now())))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(filename)


    def configure(self):
        """
        Configures the simulator with input information and creates and empty array to which the final image will
        be build on.
        """
        self.readConfigs()
        self.processConfigs()
        self._createEmpty()
        self.log.info('Read in the configuration files and created an empty array')


    def readObjectlist(self):
        """
        Reads object list using numpy.loadtxt, determines the number of object types,
        and finds the file that corresponds to a given object type.

        The input catalog is assumed to contain the following columns:

            #. x coordinate
            #. y coordinate
            #. apparent magnitude of the object
            #. type of the object [0=star, number=type defined in the objects.dat]
            #. rotation [0 for stars, [0, 360] for galaxies]

        This method also displaces the object coordinates based on the quadrant and the
        CCD to be simulated.

        .. Note:: If even a single object type does not have a corresponding input then this method
                  forces the program to exit.
        """
        self.objects = np.loadtxt(self.information['sourcelist'])

        #if only a single object in the input, must force it to 2D
        try:
            tmp_ = self.objects.shape[1]
        except:
            self.objects = self.objects[np.newaxis, :]

        str = '{0:d} sources read from {1:s}'.format(np.shape(self.objects)[0], self.information['sourcelist'])
        self.log.info(str)

        #read in object types
        data = open('data/objects.dat').readlines()

        #only 2D array will have second dimension, so this will trigger the exception if only one input source
        tmp_ = self.objects.shape[1]
        #find all object types
        self.sp = np.asarray(np.unique(self.objects[:, 3]), dtype=np.int)

        #generate mapping between object type and data
        objectMapping = {}
        for stype in self.sp:
            if stype == 0:
                #delta function
                objectMapping[stype] = 'PSF'
            else:
                for line in data:
                    tmp = line.split()
                    if int(tmp[0]) == stype:
                        #found match
                        if tmp[2].endswith('.fits'):
                            d = pf.getdata(tmp[2])
                        else:
                            d = np.loadtxt(tmp[2], skiprows=2)
                        objectMapping[stype] = dict(file=tmp[2], data=d)
                        break

        self.objectMapping = objectMapping

        #test that we have input data for each object type, if not exit with error
        if not np.array_equal(self.sp, np.asarray(list(objectMapping.keys()), dtype=np.int)):
            print self.sp
            print self.objectMapping
            print data
            print np.asarray(list(objectMapping.keys()))
            self.log.error('No all object types available, will exit!')
            sys.exit('No all object types available')

        #change the image coordinates based on the CCD being simulated
        if self.information['ccdx'] > 0:
            #x coordinate shift
            self.objects[:, 0] -= (self.information['ccdx'] * (4196. + (self.information['ccdxgap'] * 1000 / 12.)))
        if self.information['ccdy'] > 0:
            #y coordinate shift
            self.objects[:, 1] -= (self.information['ccdy'] * (4132. + (self.information['ccdxgap'] * 1000 / 12.)))

        #and quadrant
        if self.information['quadrant'] > 0:
            if self.information['quadrant'] > 1:
                #change y coordinate value
                self.log.info('Changing y coordinates to take into account quadrant')
                self.objects[:, 1] -= self.information['ysize']

            if self.information['quadrant'] % 2 != 0:
                self.log.info('Changing x coordinates to take into account quadrant')
                self.objects[:, 0] -= self.information['xsize']

        self.log.info('Object types:')
        self.log.info(self.sp)
        self.log.info('Total number of object types is %i' % len(self.sp))


    def readPSFs(self):
        """
        Reads in a PSF from a FITS file.

        .. Note:: at the moment this method supports only a single PSF file.
        """
        if self.information['variablePSF']:
            #grid of PSFs
            self.log.debug('Spatially variable PSF:')
            self.log.error('NOT IMPLEMENTED!')
            print 'Spatially variable PSF not implemented -- exiting'
            sys.exit(-9)
        else:
            #single PSF
            self.log.debug('Spatially static PSF:')
            self.log.info('Opening PSF file %s' % self.information['psffile'])
            self.PSF = pf.getdata(self.information['psffile']).astype(np.float64)
            self.PSF /= np.sum(self.PSF)
            self.PSFx = self.PSF.shape[1]
            self.PSFy = self.PSF.shape[0]
            self.log.info('PSF sampling (x,y) = (%i, %i) ' % (self.PSFx, self.PSFy))


    def generateFinemaps(self):
        """
        Generates finely sampled images of the input data.
        """
        self.finemap = {}
        self.shapex = {}
        self.shapey = {}

        for k, stype in enumerate(self.sp):
            if stype == 0:
                data = self.PSF.copy().astype(np.float64)
                data /= np.sum(data)
                self.finemap[stype] = data
                self.shapex[stype] = 0
                self.shapey[stype] = 0
            else:
                if self.information['psfoversampling'] > 1.0:
                    data = scipy.ndimage.zoom(self.objectMapping[stype]['data'],
                                              self.information['psfoversampling'],
                                              order=0)
                else:
                    data = self.objectMapping[stype]['data']

                #suppress background
                data[data < 7e-5] = 0.0

                #calculate shape tensor -- used later for size scaling
                #make a copy
                image = data.copy()

                #normalization factor
                imsum = float(np.sum(image))

                #generate a mesh coordinate grid
                sizeY, sizeX = image.shape
                Xvector = np.arange(0, sizeX)
                Yvector = np.arange(0, sizeY)
                Xmesh, Ymesh = np.meshgrid(Xvector, Yvector)

                #take centroid from data and weighting with input image
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

                shx = (Qxx + Qyy + np.sqrt((Qxx - Qyy) ** 2 + 4. * Qxy * Qxy)) / 2.
                shy = (Qxx + Qyy - np.sqrt((Qxx - Qyy) ** 2 + 4. * Qxy * Qxy)) / 2.

                #recentroid -- interpolation, not good for weak lensing etc.
                ceny, cenx = data.shape
                ceny /= 2
                cenx /= 2
                shiftx = -Xcentre + cenx
                shifty = -Ycentre + ceny

                #one should do sinc-interpolation instead...
                data = interpolation.shift(data, [shifty, shiftx], order=3, cval=0.0, mode='constant')

                data[data < 0.] = 0.0
                data /= np.sum(data)
                self.finemap[stype] = data

                self.shapex[stype] = (np.sqrt(shx / np.sum(data)))
                self.shapey[stype] = (np.sqrt(shy / np.sum(data)))

                self.log.info('shapex = %5f, shapey = %5f' % (self.shapex[stype], self.shapey[stype]))

            if self.debug:
                scipy.misc.imsave('finemap%i.jpg' % (k + 1), (data / np.max(data) * 255))


    def addObjects(self):
        """
        Add objects from the object list to the CCD image (self.image).

        Scale the object's brightness in electrons and size using the input catalog magnitude.
        The size-magnitude scaling relation is taken to be the equation B1 from Miller et al. 2012 (1210.8201v1;
        Appendix "prior distributions"). The spread is estimated from Figure 1 to be around 0".1 (1 sigma).
        A random draw from a Gaussian distribution with spread of 0".1 arc sec is performed so that galaxies
        of the same brightness would not be exactly the same size.

        .. Note:: scipy.signal.fftconvolve seems to be significantly faster than scipy.signal.convolve2d.

        .. Warning:: If random Gaussian dispersion is added to the scale-magnitude relation, then one cannot
                     simulate several dithers. The random dispersion can be turned off by setting random=no in
                     the configuration file so that dithers can be simulated and co-added correctly.
        """
        #total number of objects in the input catalogue and counter for visible objects
        n_objects = self.objects.shape[0]
        visible = 0

        self.log.info('Number of CCD transits = %i' % self.information['exposures'])
        self.log.info('Total number of objects in the input catalog = %i' % n_objects)

        #calculate the scaling factors from the magnitudes
        intscales = 10.0**(-0.4 * self.objects[:, 2]) * self.information['magzero'] * self.information['exptime']

        if ~self.random:
            self.log.info('Using a fixed size-magnitude relation (equation B1 from Miller et al. 2012 (1210.8201v1).')
            #testin mode will bypass the small random scaling in the size-mag relation
            #loop over exposures
            for i in xrange(self.information['exposures']):
                #loop over the number of objects
                for j, obj in enumerate(self.objects):

                    stype = obj[3]

                    if self.objectOnDetector(obj):
                        visible += 1
                        if stype == 0:
                            #point source, apply PSF
                            txt = "Star: " + str(j + 1) + "/" + str(n_objects) + " intscale=" + str(intscales[j])
                            print txt
                            self.log.info(txt)

                            data = self.finemap[stype].copy()

                            #map the data to new grid aligned with the centre of the object and scale
                            yind, xind = np.indices(data.shape)
                            yi = yind.astype(np.float) + (obj[0] % 1)
                            xi = xind.astype(np.float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                            if self.information['psfoversampling'] != 1.0:
                                data = scipy.ndimage.zoom(data, 1. / self.information['psfoversampling'], order=1)

                            #suppress negative numbers, renormalise and scale with the intscale
                            data[data < 0.0] = 0.0
                            sum = np.sum(data)
                            sca = intscales[j] / sum
                            data = ne.evaluate("data * sca")

                            self.log.info('Maximum value of the data added is %.2f electrons' % np.max(data))

                            #overlay the scaled PSF on the image
                            self.overlayToCCD(data, obj)
                        else:
                            #extended source, rename finemap
                            data = self.finemap[stype].copy()
                            #map the data to new grid aligned with the centre of the object
                            yind, xind = np.indices(data.shape)
                            yi = yind.astype(np.float) + (obj[0] % 1)
                            xi = xind.astype(np.float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')

                            #size-magnitude scaling
                            sbig = np.e ** (-1.145 - 0.269 * (obj[2] - 23.))  #from Miller et al. 2012 (1210.8201v1)
                            #take into account the size of the finemap galaxy -- the shape tensor is in pixels
                            #so convert to arc seconds prior to scaling
                            smin = float(min(self.shapex[stype], self.shapey[stype])) / 10.  #1 pix = 0".1
                            sbig /= smin

                            txt = "Galaxy: " +str(j+1) + "/" + str(n_objects) + " magnitude=" + str(obj[2]) + \
                                  " intscale=" + str(intscales[j]) + " FWHM=" + str(sbig*smin) + " arc sec"
                            print txt
                            self.log.info(txt)

                            #rotate the image using interpolation and suppress negative values
                            if math.fabs(obj[4]) > 1e-5:
                                data = ndimage.interpolation.rotate(data, obj[4], reshape=False)

                            #scale the size of the galaxy before convolution
                            if sbig != 1.0:
                                data = scipy.ndimage.zoom(data, self.information['psfoversampling'] * sbig, order=0)
                                data[data < 0.0] = 0.0

                            if self.debug:
                                self.writeFITSfile(data, 'beforeconv%i.fits' % (j + 1))

                            if self.information['variablePSF']:
                                sys.exit('Spatially variable PSF not implemented yet!')
                            else:
                                #conv = ndimage.filters.convolve(data, self.PSF) #would need manual padding?
                                #conv = signal.convolve2d(data, self.PSF, mode='full') #slow!
                                conv = signal.fftconvolve(data, self.PSF, mode='full')

                            #scale the galaxy image size with the inverse of the PSF over sampling factor
                            if self.information['psfoversampling'] != 1.0:
                                conv = scipy.ndimage.zoom(conv, 1. / self.information['psfoversampling'], order=1)

                            #suppress negative numbers
                            conv[conv < 0.0] = 0.0

                            #renormalise and scale to the right magnitude
                            sum = np.sum(conv)
                            sca = intscales[j] / sum
                            conv = ne.evaluate("conv * sca")

                            #tiny galaxies sometimes end up with completely zero array
                            #checking this costs time, so perhaps this could be removed
                            if np.isnan(np.sum(conv)):
                                continue

                            if self.debug:
                                scipy.misc.imsave('image%i.jpg' % (j + 1), conv / np.max(conv) * 255)
                                self.writeFITSfile(conv, 'afterconv%i.fits' % (j + 1))

                            self.log.info('Maximum value of the data added is %.3f electrons' % np.max(conv))

                            #overlay the convolved image on the image
                            self.overlayToCCD(conv, obj)

                    else:
                        #not on the screen
                        self.log.info('Object %i was outside the detector area' % (j + 1))

        else:
            #loop over exposures
            self.log.info('Using equation B1 from Miller et al. 2012 (1210.8201v1) '
                          'for scale-magnitude relation with Gaussian random dispersion.')
            for i in xrange(self.information['exposures']):
                #loop over the number of objects
                for j, obj in enumerate(self.objects):

                    stype = obj[3]

                    if self.objectOnDetector(obj):
                        visible += 1
                        if stype == 0:
                            #point source, apply PSF
                            txt = "Star: " + str(j+1) + "/" + str(n_objects) + " intscale=" + str(intscales[j])
                            print txt
                            self.log.info(txt)

                            data = self.finemap[stype].copy()

                            #map the data to new grid aligned with the centre of the object and scale
                            #def shift_func(output_coords):
                            #    return output_coords[0] - (obj[0] % 1), output_coords[1] - (obj[1] % 1)
                            #data = ndimage.geometric_transform(data, shift_func, order=0)
                            yind, xind = np.indices(data.shape)
                            yi = yind.astype(np.float) + (obj[0] % 1)
                            xi = xind.astype(np.float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                            if self.information['psfoversampling'] != 1.0:
                                data = scipy.ndimage.zoom(data, 1./self.information['psfoversampling'], order=1)

                            #suppress negative numbers, renormalise and scale with the intscale
                            data[data < 0.0] = 0.0
                            sum = np.sum(data)
                            sca = intscales[j] / sum
                            data = ne.evaluate("data * sca")

                            self.log.info('Maximum value of the data added is %.2f electrons' % np.max(data))

                            #overlay the scaled PSF on the image
                            self.overlayToCCD(data, obj)
                        else:
                            #extended source, rename finemap
                            data = self.finemap[stype].copy()

                            #map the data to new grid aligned with the centre of the object
                            yind, xind = np.indices(data.shape)
                            yi = yind.astype(np.float) + (obj[0] % 1)
                            xi = xind.astype(np.float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')

                            #size scaling relation and random draw
                            sbig = np.e**(-1.145-0.269*(obj[2] - 23.))  #from Miller et al. 2012 (1210.8201v1)
                            rshift = np.random.normal(sbig, 0.2)  #gaussian random draw to mimic the spread
                            if rshift > 0.15:  #no negative numbers or tiny tiny galaxies
                                sbig = rshift

                            #take into account the size of the finemap galaxy -- the shape tensor is in pixels
                            #so convert to arc seconds prior to scaling
                            smin = float(min(self.shapex[stype], self.shapey[stype])) / 10.  #1 pix = 0".1
                            sbig /= smin

                            txt = "Galaxy: " +str(j+1) + "/" + str(n_objects) + " magnitude=" + str(obj[2]) + \
                                  " intscale=" + str(intscales[j]) + " FWHM=" + str(sbig*smin) + " arc sec"
                            print txt
                            self.log.info(txt)

                            #rotate the image using interpolation
                            if math.fabs(obj[4]) > 1e-5:
                                data = ndimage.interpolation.rotate(data, obj[4], reshape=False)

                            #scale the size of the galaxy before convolution
                            if sbig != 1.0:
                                data = scipy.ndimage.zoom(data, self.information['psfoversampling']*sbig, order=0,
                                                          cval=0.0)
                                data[data < 0.0] = 0.0

                            if self.debug:
                                self.writeFITSfile(data, 'beforeconv%i.fits' % (j+1))

                            if self.information['variablePSF']:
                                sys.exit('Spatially variable PSF not implemented yet!')
                            else:
                                #conv = ndimage.filters.convolve(data, self.PSF) #would need manual padding?
                                #conv = signal.convolve2d(data, self.PSF, mode='full') #slow!
                                conv = signal.fftconvolve(data, self.PSF, mode='full')

                            #scale the galaxy image size with the inverse of the PSF over sampling factor
                            if self.information['psfoversampling'] != 1.0:
                                conv = scipy.ndimage.zoom(conv, 1./self.information['psfoversampling'], order=1)

                            #suppress negative numbers
                            conv[conv < 0.0] = 0.0

                            #renormalise and scale to the right magnitude
                            sum = np.sum(conv)
                            sca = intscales[j] / sum
                            conv = ne.evaluate("conv * sca")

                            #tiny galaxies sometimes end up with completely zero array
                            #checking this costs time, so perhaps this could be removed
                            if np.isnan(np.sum(conv)):
                                continue

                            if self.debug:
                                scipy.misc.imsave('image%i.jpg' % (j+1), conv/np.max(conv)*255)
                                self.writeFITSfile(conv, 'afterconv%i.fits' % (j+1))

                            self.log.info('Maximum value of the data added is %.3f electrons' % np.max(conv))

                            #overlay the convolved image on the image
                            self.overlayToCCD(conv, obj)

                    else:
                        #not on the screen
                        self.log.info('Object %i was outside the detector area' % (j+1))

        self.log.info('%i objects were place on the detector' % visible)
        print '%i objects were place on the detector' % visible


    def addLampFlux(self):
        """
        Include flux from the calibration source.
        """
        self.image += pf.getdata(self.information['flatflux'])
        self.log.info('Flux from the calibration unit included (%s)' % self.information['flatflux'])


    def applyFlatfield(self):
        """
        Applies multiplicative flat field to emulate pixel-to-pixel non-uniformity.

        Because the pixel-to-pixel non-uniformity effect (i.e. multiplicative) flat fielding takes place
        before CTI and other effects, the flat field file must be the same size as the pixels that see
        the sky. Thus, in case of a single quadrant (x, y) = (2048, 2066).
        """
        flat = pf.getdata(self.information['flatfieldfile'])
        self.image *= flat
        self.log.info('Applied multiplicative flat (pixel-to-pixel non-uniformity) from %s...' %
                      self.information['flatfieldfile'])


    def addChargeInjection(self):
        """
        Add either horizontal or vertical charge injection line to the image.
        """
        if self.chargeInjectionx:
            #self.image[self.information['ysize']/2self.information['ysize']/2-10:self.information['ysize']/2, :] = self.information['injection']
            self.image[1500:1511, :] = self.information['injection']
            self.log.info('Adding vertical charge injection line')
        if self.chargeInjectiony:
            #self.image[:, self.information['xsize']/2-10:self.information['xsize']/2] = self.information['injection']
            self.image[:, 1500:1511] = self.information['injection']
            #self.image[:, 1950:1961] = self.information['injection']
            self.log.info('Adding horizontal charge injection line')


    def addCosmicRays(self):
        """
        Add cosmic rays to the arrays based on a power-law intensity distribution for tracks.

        Cosmic ray properties (such as location and angle) are chosen from random Uniform distribution.
        """
        self.readCosmicRayInformation()

        #estimate the number of cosmics
        cr_n = self.information['xsize'] * self.information['ysize'] * 0.014 / 43.263316 * 2.
        #scale with exposure time, the above numbers are for the nominal 565s exposure
        cr_n *= (self.information['exptime'] / 565.0)

        #assume a power-law intensity distribution for tracks
        fit = dict(cr_lo=1.0e3, cr_hi=1.0e5, cr_q=2.0e0)
        fit['q1'] = 1.0e0 - fit['cr_q']
        fit['en1'] = fit['cr_lo'] ** fit['q1']
        fit['en2'] = fit['cr_hi'] ** fit['q1']

        #choose the length of the tracks
        #pseudo-random number taken from a uniform distribution between 0 and 1
        luck = np.random.rand(int(np.floor(cr_n)))

        if self.cr['cr_cdfn'] > 1:
            ius = InterpolatedUnivariateSpline(self.cr['cr_cdf'], self.cr['cr_u'])
            self.cr['cr_l'] = ius(luck)
        else:
            self.cr['cr_l'] = np.sqrt(1.0 - luck ** 2) / luck

        if self.cr['cr_cden'] > 1:
            ius = InterpolatedUnivariateSpline(self.cr['cr_cde'], self.cr['cr_v'])
            self.cr['cr_e'] = ius(luck)
        else:
            self.cr['cr_e'] = (fit['en1'] + (fit['en2'] - fit['en1']) *
                                        np.random.rand(int(np.floor(cr_n)))) ** (1.0 / fit['q1'])

        #Choose the properties such as positions and an angle from a random Uniform dist
        cr_x = self.information['xsize'] * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.information['ysize'] * np.random.rand(int(np.floor(cr_n)))
        cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

        #find the intercepts
        CCD_cr = self.cosmicRayIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)

        #save image without cosmics rays
        if self.nonlinearity:
            tmp = VISinstrumentModel.CCDnonLinearityModel(self.image.copy())
            self.writeFITSfile(tmp, 'nonoisenocr' + self.information['output'])
        else:
            self.writeFITSfile(self.image, 'nonoisenocr' + self.information['output'])

        #image without cosmic rays
        self.imagenoCR = self.image.copy()

        #paste the information
        self.image += CCD_cr

        #save cosmic ray image map
        self.cosmicMap = CCD_cr

        #count the covering factor
        area_cr = np.count_nonzero(self.cosmicMap)
        self.log.info('The cosmic ray covering factor is %i pixels ' % area_cr)

        #output information to a FITS file
        self.writeFITSfile(self.cosmicMap, 'cosmicraymap' + self.information['output'])


    def applyDarkCurrent(self):
        """
        Apply dark current. Scales the dark with the exposure time.

        Additionally saves the image without noise to a FITS file.
        """
        #save no noise image
        self.writeFITSfile(self.image, 'nonoise' + self.information['output'])

        #add dark
        dark = self.information['exptime'] * self.information['dark']
        self.image += dark
        self.log.info('Added dark current = %f' % dark)

        if self.cosmicRays:
            self.imagenoCR += dark


    def applyCosmicBackground(self):
        """
        Apply dark the cosmic background. Scales the background with the exposure time.

        Additionally saves the image without noise to a FITS file.
        """
        #save no noise image
        self.writeFITSfile(self.image, 'nobackground' + self.information['output'])

        #add background
        bcgr = self.information['exptime'] * self.information['cosmic_bkgd']
        self.image += bcgr
        self.log.info('Added cosmic background = %f' % bcgr)

        if self.cosmicRays:
            self.imagenoCR += bcgr


    def applyScatteredLight(self):
        """
        Adds spatially uniform scattered light to the image.
        """
        sl = self.information['exptime'] * self.information['scattered_light']
        self.image += sl
        self.log.info('Added scattered light = %f' % sl)


    def applyPoissonNoise(self):
        """
        Add Poisson noise to the image.
        """
        rounded = np.rint(self.image)
        residual = self.image.copy() - rounded #ugly workaround for multiple rounding operations...
        rounded[rounded < 0.0] = 0.0
        self.image = np.random.poisson(rounded).astype(np.float64)
        self.log.info('Added Poisson noise')
        self.image += residual

        if self.cosmicRays:
            #self.imagenoCR[ self.imagenoCR < 0.0] = 0.0
            self.imagenoCR = np.random.poisson(np.rint(self.imagenoCR)).astype(np.float64)


    def applyCosmetics(self):
        """
        Apply cosmetic defects described in the input file.

        .. Warning:: This method does not work if the input file has exactly one line.
        """
        cosmetics = np.loadtxt(self.information['cosmeticsFile'], delimiter=',')

        for line in cosmetics:
            x = int(np.floor(line[1]))
            y = int(np.floor(line[2]))
            value = line[3]
            self.image[y, x] = value

            self.log.info('Adding cosmetic defects from %s:' % input)
            self.log.info('x=%i, y=%i, value=%f' % (x, y, value))


    def applyRadiationDamage(self):
        """
        Applies CDM03 radiation model to the image being constructed.

        .. seealso:: Class :`CDM03`
        """
        #save image without CTI
        self.noCTI = self.image.copy()
        self.writeFITSfile(self.noCTI, 'noctinonoise' + self.information['output'])

        self.log.debug('Starting to apply radiation damage model...')
        #at this point we can give fake data...
        cti = CTI.CDM03(dict(trapfile=(self.information['trapfile'])), [-1,], log=self.log)
        #here we need the right input data
        self.image = cti.applyRadiationDamage(self.image, iquadrant=self.information['quadrant'])
        self.log.info('Radiation damage added.')

        if self.cosmicRays:
            self.log.info('Adding radiation damage to the no cosmic rays image...')
            self.imagenoCR = cti.applyRadiationDamage(self.imagenoCR,
                                                      iquadrant=self.information['quadrant'])


    def applyNonlinearity(self):
        """
        Applies a CCD273 non-linearity model to the image being constructed.
        """
        #save fully linear image
        self.writeFITSfile(self.image, 'nononlinearity' + self.information['output'])

        self.log.debug('Starting to apply non-linearity model...')
        self.image = VISinstrumentModel.CCDnonLinearityModel(self.image.copy())

        self.log.info('Non-linearity effects included.')

        if self.cosmicRays:
            self.imagenoCR = VISinstrumentModel.CCDnonLinearityModel(self.imagenoCR.copy())


    def applyReadoutNoise(self):
        """
        Applies readout noise to the image being constructed.

        The noise is drawn from a Normal (Gaussian) distribution with average=0.0 and std=readout noise.
        """
        noise = np.random.normal(loc=0.0, scale=self.information['readout'], size=self.image.shape)
        self.log.info('Sum of readnoise = %f' % np.sum(noise))

        #save the readout noise image
        self.writeFITSfile(noise, 'readoutnoise' + self.information['output'])

        #add to the image
        self.image += noise

        if self.radiationDamage:
            self.noCTI += noise
            self.noCTI /= self.information['e_adu']

        if self.cosmicRays:
            self.imagenoCR += noise


    def electrons2ADU(self):
        """
        Convert from electrons to ADUs using the value read from the configuration file.
        """
        if self.debug:
            #save the image without converting to integers
            self.writeFITSfile(self.image, 'floatsNoGain' + self.information['output'])

        self.image /= self.information['e_adu']
        self.log.info('Converting from electrons to ADUs using a factor of %f' % self.information['e_adu'])

        if self.cosmicRays:
            self.imagenoCR /= self.information['e_adu']


    def applyBias(self):
        """
        Adds a bias level to the image being constructed.

        The value of bias is read from the configure file and stored
        in the information dictionary (key bias).
        """
        self.image += self.information['bias']
        self.log.info('Bias of %i counts were added to the image' % self.information['bias'])

        if self.cosmicRays:
            self.imagenoCR += self.information['bias']


    def addPreOverScans(self):
        """
        Add pre- and overscan regions to the self.image. These areas are added only in the serial direction.
        Because the 1st and 3rd quadrant are read out in to a different serial direction than the nominal
        orientation, in these images the regions are mirrored.

        The size of prescan and overscan regions are defined by the prescanx and overscanx keywords, respectively.
        """
        self.log.info('Adding pre- and overscan regions')

        canvas = np.zeros((self.information['ysize'],
                          (self.information['xsize'] + self.information['prescanx'] + self.information['ovrscanx'])))

        #because the pre- and overscans are in x-direction this needs to be taken into account for the
        # 1st and 3rd quadrant
        if self.information['quadrant'] in (0, 2):
            canvas[:, self.information['prescanx']: self.information['prescanx']+self.information['xsize']] = self.image
            self.image = canvas
        elif self.information['quadrant'] in (1, 3):
            canvas[:, self.information['ovrscanx']: self.information['ovrscanx']+self.information['xsize']] = self.image
            self.image = canvas
        else:
            self.log.error('Cannot include pre- and overscan because of an unknown quadrant!')

        if self.cosmicRays:
            canvas = np.zeros((self.information['ysize'],
                              (self.information['xsize'] + self.information['prescanx'] + self.information['ovrscanx'])))

            if self.information['quadrant'] in (0, 2):
                canvas[:, self.information['prescanx']: self.information['prescanx']+self.information['xsize']] = self.imagenoCR
            else:
                canvas[:, self.information['ovrscanx']: self.information['ovrscanx']+self.information['xsize']] = self.imagenoCR

            self.imagenoCR = canvas


    def applyBleeding(self):
        """
        Apply bleeding along the CCD columns if the number of electrons in a pixel exceeds the full-well capacity.

        Bleeding is modelled in the parallel direction only, because the CCD273s are assumed not to bleed in
        serial direction.

        :return: None
        """
        self.log.info('Applying column bleeding...')
        #loop over each column, as bleeding is modelled column-wise
        for i, column in enumerate(self.image.T):
            sum = 0.
            for j, value in enumerate(column):
                #first round - from bottom to top (need to half the bleeding)
                overload = value - self.information['fullwellcapacity']
                if overload > 0.:
                    overload /= 2.
                    self.image[j, i] -= overload
                    sum += overload
                elif sum > 0.:
                    if -overload > sum:
                        overload = -sum
                    self.image[j, i] -= overload
                    sum += overload

        for i, column in enumerate(self.image.T):
            sum = 0.
            for j, value in enumerate(column[::-1]):
                #second round - from top to bottom (bleeding was half'd already, so now full)
                overload = value - self.information['fullwellcapacity']
                if overload > 0.:
                    self.image[-j-1, i] -= overload
                    sum += overload
                elif sum > 0.:
                    if -overload > sum:
                        overload = -sum
                    self.image[-j-1, i] -= overload
                    sum += overload


    def discretise(self, max=2**16-1):
        """
        Converts a floating point image array (self.image) to an integer array with max values
        defined by the argument max.

        :param max: maximum value the the integer array may contain [default 65k]
        :type max: float

        :return: None
        """
        #also write out an image without cosmics if those were added
        if self.cosmicRays:
            self.imagenoCR = np.rint(self.imagenoCR).astype(np.int)
            self.imagenoCR[self.imagenoCR > max] = max
            self.writeFITSfile(self.imagenoCR, 'nocr' + self.information['output'], unsigned16bit=True)

        if self.debug:
            #save the image without converting to integers
            self.writeFITSfile(self.image, 'floats' + self.information['output'])

        #avoid negative numbers in case bias level was not added
        #self.image[self.image < 0.0] = 0.
        #cut of the values larger than max
        self.image[self.image > max] = max

        self.image = np.rint(self.image).astype(np.int)
        self.log.info('Maximum and total values of the image are %i and %i, respectively' % (np.max(self.image),
                                                                                             np.sum(self.image)))
        if self.radiationDamage:
            self.noCTI = np.rint(self.noCTI).astype(np.int)
            self.noCTI[self.noCTI > max] = max
            self.writeFITSfile(self.noCTI, 'nocti' + self.information['output'], unsigned16bit=True)


    def writeOutputs(self):
        """
        Writes out a FITS file using PyFITS and converts the image array to 16bit unsigned integer as
        appropriate for VIS.

        Updates header with the input values and flags used during simulation.
        """
        if os.path.isfile(self.information['output']):
            os.remove(self.information['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=self.image)

        #convert to unsigned 16bit
        if self.intscale:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #add WCS to the header
        hdu.header.update('WCSAXES', 2)
        hdu.header.update('CRPIX1', self.image.shape[1]/2.)
        hdu.header.update('CRPIX2', self.image.shape[0]/2.)
        hdu.header.update('CRVAL1', self.information['ra'])
        hdu.header.update('CRVAL2', self.information['dec'])
        hdu.header.update('CTYPE1', 'RA---TAN')
        hdu.header.update('CTYPE2', 'DEC--TAN')
        #north is up, east is left
        hdu.header.update('CD1_1', -0.1 / 3600.) #pix size in arc sec / deg
        hdu.header.update('CD1_2', 0.0)
        hdu.header.update('CD2_1', 0.0)
        hdu.header.update('CD2_2', 0.1 / 3600.)

        hdu.header.update('DATE-OBS', datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.header.update('INSTRUME', 'VISsim')

        #add input keywords to the header
        for key, value in self.information.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.update(key.upper(), value)
            except:
                try:
                    hdu.header.update(key.upper(), str(value))
                except:
                    pass

        hdu.header.update('NRPUFILE', self.information['flatfieldfile'])

        #write booleans
        for key, value in self.booleans.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.update(key.upper(), str(value), 'Boolean Flags')

        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (s.niemi at ucl.ac.uk).')
        hdu.header.add_history('Created by VISsim (version=%.2f) at %s' % (__version__,
                                                                           datetime.datetime.isoformat(datetime.datetime.now())))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.information['output'])


    def simulate(self):
        """
        Create a single simulated image of a quadrant defined by the configuration file.
        Will do all steps defined in the config file sequentially.

        :return: None
        """
        self.configure()
        self.readObjectlist()
        self.readPSFs()
        self.generateFinemaps()

        if self.addsources:
            self.addObjects()

        if self.lampFlux:
            self.addLampFlux()

        if self.flatfieldM:
            self.applyFlatfield()

        if self.chargeInjectionx or self.chargeInjectiony:
            self.addChargeInjection()

        if self.cosmicRays:
            self.addCosmicRays()

        if self.bleeding:
            self.applyBleeding()

        if self.noise:
            self.applyDarkCurrent()

        if self.background:
            self.applyCosmicBackground()

        if self.scatteredlight:
            self.applyScatteredLight()

        if self.noise:
            self.applyPoissonNoise()

        if self.cosmetics:
            self.applyCosmetics()

        if self.overscans:
            self.addPreOverScans()

        if self.radiationDamage:
            self.applyRadiationDamage()

        if self.nonlinearity:
            self.applyNonlinearity()

        if self.readoutNoise:
            self.applyReadoutNoise()

        self.electrons2ADU()

        if self.information['bias'] <= 0.0:
            self.log.info('Bias level less or equal to zero, will not add bias!')
        else:
            self.applyBias()

        if self.intscale:
            self.discretise()

        self.writeOutputs()


class Test(unittest.TestCase):
    """
    Unit tests for the shape class.
    """
    def setUp(self):
        class dummy:
            pass
        opts = dummy() #ugly hack...
        opts.quadrant = '0'
        opts.xCCD = '0'
        opts.yCCD = '0'
        opts.configfile = 'data/test.config'
        opts.section = 'TESTSCIENCE1X'
        opts.debug = False
        opts.testing = True
        self.simulate = VISsimulator(opts)


    def test(self):
        """
        Runs a test case and compares it the previously calculated results.

        :return: None
        """
        #run simulator
        self.simulate.simulate()
        #load generated file
        new = pf.open('nonoisenocrQ0_00_00testscience.fits')[1].data
        #load test file
        expected = pf.open('data/nonoisenocrQ0_00_00testscience.fits')[1].data
        #assert
        print 'Asserting...'
        np.testing.assert_array_almost_equal(new, expected, decimal=7, err_msg='', verbose=True)


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-c', '--configfile', dest='configfile',
                      help="Name of the configuration file", metavar="string")
    parser.add_option('-s', '--section', dest='section',
                      help="Name of the section of the config file [SCIENCE]", metavar="string")
    parser.add_option('-q', '--quadrant', dest='quadrant', help='CCD quadrant to simulate [0, 1, 2, 3]',
                      metavar='int')
    parser.add_option('-x', '--xCCD', dest='xCCD', help='CCD number in X-direction within the FPA matrix',
                      metavar='int')
    parser.add_option('-y', '--yCCD', dest='yCCD', help='CCD number in Y-direction within the FPA matrix',
                      metavar='int')
    parser.add_option('-d', '--debug', dest='debug', action='store_true',
                      help='Debugging mode on')
    parser.add_option('-t', '--test', dest='test', action='store_true',
                      help='Run unittest')
    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    #run unittest and exit
    if opts.test is not None:
        suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        unittest.TextTestRunner(verbosity=3).run(suite)
        sys.exit(1)

    #no input file, exti
    if opts.configfile is None:
        processArgs(True)
        sys.exit(1)

    #set defaults if not given
    if opts.quadrant is None:
        opts.quadrant = '0'
    if opts.xCCD is None:
        opts.xCCD = '0'
    if opts.yCCD is None:
        opts.yCCD = '0'

    #run the simulator
    simulate = VISsimulator(opts)
    simulate.simulate()