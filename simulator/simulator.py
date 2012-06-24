"""
The Euclid Visible Instrument Image Simulator
=============================================

This file contains an image simulator for the Euclid VISible instrument.

.. Warning:: This code has not been fully developed or tested. This is a prototype
             that can be used to test different aspects of image simulations.

.. todo::

    1. test that the radiation damage is correctly implemented
    2. start using oversampled PSF (need to modify the overlay part)
    3. implement bleeding
    4. implement spatially variable PSF
    5. test that the cosmic rays are correctly implemented


Dependencies
------------

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: vissim-python package


Testing
-------

Benchmark using the SCIENCE section of test.config::

        Galaxy: 26753/26753 intscale=199.421150298 size=0.0329649781423
        6772 objects were place on the detector

        real	8m28.781s
        user	8m24.792s
        sys	    0m1.361s

These numbers have been obtained with my laptop (2.2 GHz Intel Core i7).


Current Version
---------------

:version: 0.5

Version and change logs::

    0.1: pre-development backbone.
    0.4: first version with most pieces together.
    0.5: this version has all the basic features present, but not fully tested.



Contact Information
-------------------

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import os, sys, datetime, math
import ConfigParser
from optparse import OptionParser
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import ndimage
from scipy import signal
import pyfits as pf
import numpy as np
from CTI import CTI
from support import logger as lg


class VISsimulator():
    """
    Euclid Visible Instrument Simulator.

    The image that is being build is in::

        self.image

    :param configfile: name of the configuration file
    :type configfile: string
    :param debug: debugging mode on/off
    :type debug: boolean
    :param section: name of the section of the configuration file to process
    :type section: str
    """

    def __init__(self, configfile, debug, section='SCIENCE'):
        """
        Class Constructor.

        :param configfile: name of the configuration file
        :type configfile: string
        :param debug: debugging mode on/off
        :type debug: boolean
        :param section: name of the section of the configuration file to process
        :type section: string
        """
        self.configfile = configfile
        self.section = section
        self.debug = debug

        #setup logger
        self.log = lg.setUpLogger('VISsim.log')

        #dictionary to hold basic information
        self.information = {}


    def _readConfigs(self):
        """
        Reads the config file information using configParser.
        """
        self.config = ConfigParser.RawConfigParser()
        self.config.readfp(open(self.configfile))


    def _processConfigs(self):
        """
        Processes configuration information and save the information to a dictionary self.information.

        The configuration file may look as follows::

            [SCIENCE]
            quadrant = 0
            CCDx = 0
            CCDy = 0
            xsize = 2048
            ysize = 2066
            dark = 0.001
            readout = 4.5
            bias = 1000.0
            cosmic_bkgd = 0.172
            e_ADU = 3.5
            magzero = 1.7059e10
            exposures = 1
            exptime = 565.0
            RA = 145.95
            DEC = -38.16
            sourcelist = source_ngp.dat
            PSFfile = data/interpolated_psf.fits
            trapfile = data/cdm_euclid.dat
            cosmeticsFile = ./data/cosmetics.dat
            output = science.fits
            noise = yes
            cosmetics = no
            chargeInjection = no
            radiationDamage = yes
            cosmicRays = yes
            flatfieldM = no
            flatfieldA = no

        For explanation of each field, see /data/test.config.

        """
        #sizes
        self.information['xsize'] = self.config.getint(self.section, 'xsize')
        self.information['ysize'] = self.config.getint(self.section, 'ysize')

        #quadrant and CCD
        self.information['quadrant'] = self.config.getint(self.section, 'quadrant')
        self.information['CCDx'] = self.config.getint(self.section, 'CCDx')
        self.information['CCDy'] = self.config.getint(self.section, 'CCDy')

        #noise values
        self.information['dark'] = self.config.getfloat(self.section, 'dark')
        self.information['cosmic_bkgd'] = self.config.getfloat(self.section, 'cosmic_bkgd')
        self.information['readout'] = self.config.getfloat(self.section, 'readout')

        #bias and conversions
        self.information['bias'] = self.config.getfloat(self.section, 'bias')
        self.information['e_ADU'] = self.config.getfloat(self.section, 'e_ADU')

        #exposure time and position on the sky
        self.information['exposures'] = self.config.getint(self.section, 'exposures')
        self.information['exptime'] = self.config.getfloat(self.section, 'exptime')
        self.information['RA'] = self.config.getfloat(self.section, 'RA')
        self.information['DEC'] = self.config.getfloat(self.section, 'DEC')

        #charge injection value
        self.information['injection'] = self.config.getfloat(self.section, 'injection')

        #zeropoint
        self.information['magzero'] = self.config.getfloat(self.section, 'magzero')

        #name of the input source list
        self.information['sourcelist'] = self.config.get(self.section, 'sourcelist')

        #name of the output file
        self.information['output'] = self.config.get(self.section, 'output')

        #PSF related information
        self.information['PSFfile'] = self.config.get(self.section, 'PSFfile')
        self.information['variablePSF'] = False

        #charge traps input file
        self.information['trapfile'] = self.config.get(self.section, 'trapfile')

        #cosmetic defects input file
        self.information['cosmeticsFile'] = self.config.get(self.section, 'cosmeticsFile')

        #booleans to control the flow
        self.flatfieldM = self.config.getboolean(self.section, 'flatfieldM')
        self.flatfieldA = self.config.getboolean(self.section, 'flatfieldA')
        self.chargeInjectionx = self.config.getboolean(self.section, 'chargeInjectionx')
        self.chargeInjectiony = self.config.getboolean(self.section, 'chargeInjectiony')
        self.cosmicRays = self.config.getboolean(self.section, 'cosmicRays')
        self.noise = self.config.getboolean(self.section, 'noise')
        self.cosmetics = self.config.getboolean(self.section, 'cosmetics')
        self.radiationDamage = self.config.getboolean(self.section, 'radiationDamage')
        self.addsources = self.config.getboolean(self.section, 'addSources')

        if self.debug:
            print self.information

        self.log.info('Using the following inputs:')
        for key, value in self.information.iteritems():
            self.log.info('%s = %s' % (key, value))


    def _createEmpty(self):
        """
        Creates and empty array with zeros.
        """
        self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)


    def _crIntercepts(self, lum, x0, y0, l, phi):
        """
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
        for cosmics in range(0, len(l)):
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
                for j in range(int(ilo), int(ihi)):
                    ok = (j - x0[cosmics]) / dx
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(j)
                        y.append(y0[cosmics] + ok * dy)

            #Compute Y intercepts on the pixel grid
            if dy > 0.:
                for j in range(int(jlo), int(jhi)):
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
            for i in range(1, n - 1):
                w = u[i + 1] - u[i]
                cx = 1 + np.floor((x[i + 1] + x[i]) / 2.0)
                cy = 1 + np.floor((y[i + 1] + y[i]) / 2.0)

                if cx >= 0 and cx < self.information['xsize'] and cy >= 0 and cy < self.information['ysize']:
                    crImage[cy, cx] += (w * lum[cosmics])

        return crImage


    def _readCosmicRayInformation(self):
        """
        Reads in the cosmic ray track information from two input files.
        Stores the information to a dictionary called cr.
        """
        length = 'data/cdf_cr_length.dat'
        dist = 'data/cdf_cr_total.dat'

        crLengths = np.loadtxt(length)
        crDists = np.loadtxt(dist)

        self.cr = dict(cr_u=crLengths[:, 0], cr_cdf=crLengths[:, 1],
            cr_cdfn=np.shape(crLengths)[0],
            cr_v=crDists[:, 0], cr_cde=crDists[:, 1],
            cr_cden=np.shape(crDists)[0])


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


    def _objectOnDetector(self, object):
        """
        Tests if the object falls on the detector.

        :param object: object to be placed to the self.image.

        :return: whether the object falls on the detector or not
        :rtype: boolean
        """
        ny = self.finemap[object[3]].shape[1]
        nx = self.finemap[object[3]].shape[0]
        mx = self.information['xsize']
        my = self.information['ysize']
        fac = object[4]
        xt = object[0]
        yt = object[1]

        #Assess the boundary box of the input image.
        xlo = (1 - nx) * 0.5 * fac  + xt
        xhi = (nx - 1) * 0.5 * fac  + xt
        ylo = (1 - ny) * 0.5 * fac  + yt
        yhi = (ny - 1) * 0.5 * fac  + yt

        i1 = np.floor(xlo + 0.5)
        i2 = np.floor(xhi + 0.5) + 1
        j1 = np.floor(ylo + 0.5)
        j2 = np.floor(yhi + 0.5) + 1

        if i2 < 1 or i1 > mx:
            return False

        if j2 < 1 or j1 > my:
            return False

        return True


    def _gammln(self, xx):
        """
        Logarithm of the gamma function.
        """
        gammln_cof = [76.18009173,
                      -86.50532033,
                      24.01409822,
                      -1.231739516e0,
                      0.120858003e-2,
                      -0.536382e-5]
        gammln_stp = 2.50662827465
        x = xx
        y = x
        tmp = x + 5.5
        tmp = (x + .5) * np.log(tmp) - tmp
        ser = 1.000000000190015
        for j in range(6):
            y = y + 1.0
            ser += gammln_cof[j] / y
        return tmp + np.log(gammln_stp * ser / x)


    def _slowPoissonNoise(self, value):
        """

        """
        if value < 12:
            g = np.exp(-value)
            em = -1
            t = 1
            while t > g:
                em += 1
                t *= np.random.unform()
                return em
        else:
            sq = np.sqrt(2.*value)
            al = np.log(value)
            g = value * al - self._gammln(value+1.)

            em = -1
            while em < 0:
                y = np.tan(math.pi * np.random.uniform())
                em = int(sq*y + value)

            t = 0.9 * (1. + y**2) * np.exp(em * al - self._gammln(em+1.) - g)
            while np.random.uniform() > t:
                t = 0.9 * (1. + y**2) * np.exp(em * al - self._gammln(em+1.) - g)
                y = np.tan(math.pi * np.random.uniform())
                em = int(sq * y + value)

            return em


    def _overlayToCCD(self, data, obj):
        """
        Overlay data from a source object onto the self.image.

        :param data: ndarray of data to be overlaid on to self.image
        :type data: ndarray
        :param obj: object information such as position
        :type obj: list
        """
        #object magnification and center x and y coordinates
        xt = obj[0]
        yt = obj[1]

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

        ni = i2 - i1
        nj = j2 - j1

        self.log.info('Adding an object to (x,y)=({0:.1f}, {1:.1f})'.format(xt, yt))

        #add to the image
        if ni == nx and nj == ny:
            #full frame
            self.image[j1:j2, i1:i2] += data
        elif ni < nx and nj == ny:
            #x smaller
            if int(np.floor(xlo + 0.5)) < 1:
                #small values, left side
                self.image[j1:j2, i1:i2] += data[:, nx-ni:]
            else:
                #large values, right side
                self.image[j1:j2, i1:i2] += data[:, :ni]
            pass
        elif nj < ny and ni == nx:
            #y smaller
            if int(np.floor(ylo + 0.5)) < 1:
                #small values, bottom
                self.image[j1:j2, i1:i2] += data[ny-nj:, :]
            else:
                #large values, top
                self.image[j1:j2, i1:i2] += data[:nj, :]
            pass
        else:
            #both smaller, can be in any of the four corners
            if int(np.floor(xlo + 0.5)) < 1 and int(np.floor(ylo + 0.5)) < 1:
                self.image[j1:j2, i1:i2] += data[ny-nj:, nx-ni:]
            elif int(np.floor(xlo + 0.5)) < 1 and int(np.floor(ylo + 0.5)) > self.information['ysize']:
                self.image[j1:j2, i1:i2] += data[:nj, nx-ni:]
            elif int(np.floor(xlo + 0.5)) > self.information['xsize'] and int(np.floor(ylo + 0.5)) < 1:
                self.image[j1:j2, i1:i2] += data[ny-nj:, :ni]
            else:
                self.image[j1:j2, i1:i2] += data[ny-nj:, nx-ni:]


    def _writeFITSfile(self, data, filename, unsigned16bit=False):
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

        #update and verify the header
        hdu.header.add_history('Created by VISsim at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(filename)


    def configure(self):
        """
        Configure the simulator with input information and
        create and empty array to which the final image will
        be build on.
        """
        self._readConfigs()
        self._processConfigs()
        self._createEmpty()

        self.log.info('Read in the configuration files and created and empty array')


    def readObjectlist(self):
        """
        Read object list using numpy.loadtxt, determine the number of spectral types,
        and find the file that corresponds to a given spectral type.
        """
        self.objects = np.loadtxt(self.information['sourcelist'])

        str = '{0:d} sources read from {1:s}'.format(np.shape(self.objects)[0], self.information['sourcelist'])
        self.log.info(str)

        #find all spectral types
        self.sp = np.asarray(np.unique(self.objects[:, 3]), dtype=np.int)

        #generate mapping between spectral type and data
        spectraMapping = {}
        data = open('data/objects.dat').readlines()
        for stype in self.sp:
            if stype == 0:
                #delta function
                spectraMapping[stype] = 'PSF'
            else:
                for line in data:
                    tmp = line.split()
                    if int(tmp[0]) == stype:
                        #found match
                        if tmp[2].endswith('.fits'):
                            d = pf.getdata(tmp[2])
                        else:
                            d = np.loadtxt(tmp[2], skiprows=2)
                        spectraMapping[stype] = dict(file=tmp[2], data=d)
                        break

        self.spectraMapping = spectraMapping

        #TODO: write a check if there are missing spectra and then stop
        #msk = np.asarray(self.sp, dtype=np.int) != np.asarray(list(spectraMapping.keys()), dtype=np.int)
        #print msk
        #if len(msk > 0):
        #    print 'Missing spectra...'

        self.log.info('Spectral types:')
        self.log.info(self.sp)
        self.log.info('Total number of spectral types is %i' % len(self.sp))


    def readPSFs(self):
        """
        Reads in the PSF from a FITS file.

        .. note:: at the moment this method supports only a single PSF file.
        """
        if self.information['variablePSF']:
            #grid of PSFs
            self.log.debug('Spatially variable PSF:')
            self.log.error('NOT IMPLEMENTED!')
        else:
            #single PSF
            self.log.debug('Spatially static PSF:')
            self.log.info('Opening PSF file %s' % self.information['PSFfile'])
            self.PSF = pf.getdata(self.information['PSFfile'])
            self.PSFx = self.PSF.shape[1]
            self.PSFy = self.PSF.shape[0]
            self.log.info('PSF sampling (x,y) = (%i, %i) ' % (self.PSFx, self.PSFy))


    def generateFinemaps(self):
        """
        Generate finely sampled images of the input data.

        .. Warning:: This should be rewritten. Now a direct conversion from FORTRAN, and thus
                     not probably very effective. Also, this now assumed that the PSF is the
                     most finely sampled so it sampled all the postage stamps to the same
                     grid as the PSF.
        """
        self.finemap = {}
        self.shapex = {}
        self.shapey = {}

        for k, stype in enumerate(self.sp):

            #now assumes that PSF is the most finely sampled
            #and places all the postage stamps to PSF grid
            fm = np.zeros((self.PSFy, self.PSFx))

            if stype == 0:
                #PSF
                i = self.PSFx
                j = self.PSFy
                i1 = (self.PSFx - i) / 2
                i2 = i1 + self.PSFx
                j1 = (self.PSFy - j) / 2
                j2 = j1 + self.PSFy
                fm[j1:j2, i1:i2] = self.PSF

                fm /= np.sum(fm)

                self.finemap[stype] = fm
                self.shapex[stype] = 0
                self.shapey[stype] = 0
            else:
                data = self.spectraMapping[stype]['data']
                i = data.shape[1]
                j = data.shape[0]

                i1 = 1 + (self.PSFx - i) / 2
                i2 = i1 + i
                j1 = 1 + (self.PSFy - j) / 2
                j2 = j1 + j

                fm[j1:j2, i1:i2] = data

                #normalize to sum to unity
                fm /= np.sum(fm)
                self.finemap[stype] = fm

                #Compute a shape tensor, translated from Fortran so not very
                #effective.
                Qxx = 0.
                Qxy = 0.
                Qyy = 0.
                for i in range(0, self.PSFx):
                    for j in range(0, self.PSFy):
                        Qxx += fm[j,i]*(i-0.5*(self.PSFx-1))*(i-0.5*(self.PSFx-1))
                        Qxy += fm[j,i]*(i-0.5*(self.PSFx-1))*(j-0.5*(self.PSFy-1))
                        Qyy += fm[j,i]*(j-0.5*(self.PSFy-1))*(j-0.5*(self.PSFy-1))

                shx = (Qxx + Qyy + np.sqrt((Qxx - Qyy)**2 + 4.*Qxy*Qxy ))/2.
                shy = (Qxx + Qyy - np.sqrt((Qxx - Qyy)**2 + 4.*Qxy*Qxy ))/2.
                self.shapex[stype] = (np.sqrt(shx / np.sum(fm)))
                self.shapey[stype] = (np.sqrt(shy / np.sum(fm)))

        #sum of the finemaps, this should be exactly the number of finemaps
        #because these have been normalized to sum to unity.
        self.log.info('finemap sum = %f' %np.sum(np.asarray(self.finemap.values())))


    def addObjects(self):
        """
        Add objects from the object list to the CCD image (self.image).

        Scale the object's brightness based on its magnitude. The size of the object
        is scaled using the brightness.

        For fainter objects the convolved results are on the same sized grid as the
        input. However, for large values (intscale > 1e5) a box with sharp edges
        would be visible if convolution were not done on the full scale. The problem
        is that the values outside the input grid are not very reliable, so these
        objects should be treated with caution.
        """
        #total number of objects in the input catalogue and counter for visible objects
        n_objects = self.objects.shape[0]
        visible = 0

        self.log.info('Number of CCD transits = %i' % self.information['exposures'])
        self.log.info('Number of objects to be inserted is %i' % n_objects)

        #calculate the scaling factors from the magnitudes
        intscales = 10.0**(-0.4 * self.objects[:, 2]) * \
                    self.information['magzero'] * \
                    self.information['exptime']

        #loop over exposures
        for i in range(self.information['exposures']):
            #loop over the number of objects
            for j, obj in enumerate(self.objects):
                stype = obj[3]

                if self._objectOnDetector(obj):
                    visible += 1
                    if stype == 0:
                        #point source, apply PSF
                        txt = "Star: " + str(j+1) + "/" + str(n_objects) + " intscale=" + str(intscales[j])
                        print txt
                        self.log.info(txt)

                        #blending of PSF no implemented

                        #renormalise and scale with the intscale
                        data = self.finemap[stype] / np.sum(self.finemap[stype]) * intscales[j]

                        self.log.info('Maximum value of the data added is %.2f electrons' % np.max(data))

                        #overlays on the image
                        self._overlayToCCD(data, obj)
                    else:
                        #extended source
                        sbig = (0.2**((obj[2] - 22.)/7.)) / self.shapey[stype] / 2.

                        txt = "Galaxy: " +str(j+1) + "/" + str(n_objects) + \
                              " intscale=" + str(intscales[j]) + " size=" + str(sbig)
                        print txt
                        self.log.info(txt)


                        #rotate and zoom the image using interpolation and remove negative values
                        data = ndimage.interpolation.rotate(self.finemap[stype], obj[4], reshape=False)
                        data = ndimage.interpolation.zoom(data, sbig)
                        data[data < 0.0] = 0.0

                        #renormalise and scale to the right magnitude
                        sum = np.sum(data)
                        data *= intscales[j] / sum

                        self.log.info('Maximum value of the data added is %.2f electrons' % np.max(data))

                        if self.information['variablePSF']:
                            #spatially variable PSF, we need to convolve with the appropriate PSF
                            #data = ndimage.filters.convolve(data, self.PSF) #would need padding?
                            #data = signal.fftconvolve(data, self.PSF) #does not work with 64-bit?
                            data = signal.convolve2d(data, self.PSF, mode='same')
                        else:
                            if intscales[j] > 1e5:
                                #For very large values, a boxing effect is
                                #visible unless full convolution is being used.
                                data = signal.convolve2d(data, self.PSF, mode='full')
                            else:
                                data = signal.convolve2d(data, self.PSF, mode='same')

                        #overlay on the image
                        self._overlayToCCD(data, obj)

                else:
                    #not on the screen
                    print 'OFFscreen: ', j+1
                    self.log.info('Object %i was outside the detector area' % (j+1))


        self.log.info('%i objects were place on the detector' % visible)
        print '%i objects were place on the detector' % visible


    def applyFlatfield(self):
        """
        Applies multiplicative and/or additive flat field.
        Assumes that these flat fields are found from the
        data folder and are called flat_field_mul.fits and
        flat_field_add.fits, respectively.
        """
        if self.flatfieldM:
            flatM = pf.getdata('data/flat_field_mul.fits')
            self.image *= flatM
            self.log.info('Applied multiplicative flat... ')

        if self.flatfieldA:
            flatA = pf.getdata('data/flat_field_add.fits')
            self.image += flatA
            self.log.info('Applied additive flat... ')


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
            self.log.info('Adding horizontal charge injection line')


    def addCosmicRays(self):
        """
        Add cosmic rays to the arrays based on a power-law intensity distribution for tracks.

        Cosmic ray properties (such as location and angle) are chosen from random Uniform distribution.
        """
        self._readCosmicRayInformation()

        #estimate the number of cosmics
        cr_n = self.information['xsize'] * self.information['ysize'] * 0.014 / 43.263316

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
        CCD_cr = self._crIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)

        #save image without cosmics rays
        self._writeFITSfile(self.image, 'nonoisenocr'+self.information['output'])

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
        self._writeFITSfile(self.cosmicMap, 'cosmicraymap.fits')


    def applyNoise(self):
        """
        Apply dark current, the cosmic background, and Poisson noise.
        Scales dark and background with the exposure time.

        Additionally saves the image without noise to a FITS file.
        """
        #save no noise image
        self._writeFITSfile(self.image, 'nonoise'+self.information['output'])

        #add dark and background
        noise = self.information['exptime'] * (self.information['dark'] + self.information['cosmic_bkgd'])
        self.image += noise
        if self.cosmicRays:
            self.imagenoCR += noise

        self.log.info('Added dark noise and cosmic background = %f' % noise)

        #for a in range(self.information['xsize']):
        #    for b in range(self.information['ysize']):
        #        self.image[b, a] = self._slowPoissonNoise(self.image[b, a])
        self.image = np.random.poisson(self.image)
        self.image[self.image < 0.0] = 0.0
        self.log.info('Added Poisson noise')
        if self.cosmicRays:
            self.imagenoCR = np.random.poisson(self.imagenoCR)
            self.imagenoCR[ self.imagenoCR < 0.0] = 0.0


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
        """
        #save image without CTI
        self._writeFITSfile(self.image, 'nocti' + self.information['output'])

        self.log.debug('Starting to apply radiation damage model...')
        #at this point we can give fake data...
        cti = CTI.CDM03(dict(trapfile=(self.information['trapfile'])), [-1,], log=self.log)
        #here we need the right input data
        self.image = cti.applyRadiationDamage(self.image,
                                              iquadrant=self.information['quadrant'])
        self.log.info('Radiation damage added.')
        if self.cosmicRays:
            self.imagenoCR = cti.applyRadiationDamage(self.imagenoCR,
                                                      iquadrant=self.information['quadrant'])


    def applyReadoutNoise(self):
        """
        Applies readout noise to the image being constructed.

        The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout noise).
        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(self.information['readout']),
                                 size=(self.information['ysize'], self.information['xsize']))
        self.log.info('Sum of readnoise = %f' % np.sum(noise))

        #save the readout noise image
        self._writeFITSfile(noise, 'readoutnoise.fits')

        #add to the image
        self.image += noise
        if self.cosmicRays:
            self.imagenoCR += noise


    def electrons2ADU(self):
        """
        Convert from electrons to ADUs using the value read from the configuration file.
        """
        self.image /= self.information['e_ADU']
        self.log.info('Converting from electrons to ADUs using a factor of %f' % self.information['e_ADU'])
        if self.cosmicRays:
            self.imagenoCR /= self.information['e_ADU']


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


    def discretise(self, max=2**16-1):
        """
        Convert floating point arrays to integer arrays.

        :param max: maximum value the the integer array may contain [default 65k]
        :type max: float
        """
        #also write out an image without cosmics if those were added
        if self.cosmicRays:
            self.imagenoCR = self.imagenoCR.astype(np.int)
            self.imagenoCR[self.imagenoCR > max] = 0.0
            self._writeFITSfile(self.imagenoCR, 'nocr' + self.information['output'], unsigned16bit=True)

        self.image = self.image.astype(np.int)
        self.image[self.image > max] = max
        self.log.info('Maximum and total values of the image are %i and %i, respectively' % (np.max(self.image),
                                                                                             np.sum(self.image)))


    def writeOutputs(self):
        """
        Write out FITS files using PyFITS.
        """
        if os.path.isfile(self.information['output']):
            os.remove(self.information['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=self.image)

        #convert to unsigned 16bit
        hdu.scale('int16', '', bzero=32768)
        hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #update and verify the header
        hdu.header.update('RA', self.information['RA'], 'RA of the center of the chip')
        hdu.header.update('DEC', self.information['DEC'], 'DEC of the center of the chip')

        #add input keywords to the header
        for key, value in self.information.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.update(key, value)
            except:
                try:
                    hdu.header.update(key, str(value))
                except:
                    pass

        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.information['output'])


    def simulate(self):
        """
        Create a single simulated image defined by the configuration file.
        """
        self.configure()
        self.readObjectlist()
        self.readPSFs()
        self.generateFinemaps()

        if self.addsources:
            self.addObjects()

        if self.flatfieldA or self.flatfieldM:
            self.applyFlatfield()

        if self.chargeInjectionx or self.chargeInjectiony:
            self.addChargeInjection()

        if self.cosmicRays:
            self.addCosmicRays()

        if self.noise:
            self.applyNoise()

        if self.cosmetics:
            self.applyCosmetics()

        if self.radiationDamage:
            self.applyRadiationDamage()

        if self.noise:
            self.applyReadoutNoise()

        self.electrons2ADU()
        self.applyBias()
        self.discretise()
        self.writeOutputs()


    def runAll(self):
        """
        Driver function, which runs all the steps independent of the boolean flags.

        .. Note: Use this for debugging only!
        """
        self.configure()
        self.readObjectlist()
        self.readPSFs()
        self.generateFinemaps()
        self.addObjects()
        self.applyFlatfield()
        self.addChargeInjection()
        self.addCosmicRays()
        self.applyNoise()
        self.applyCosmetics()
        self.applyRadiationDamage()
        self.applyReadoutNoise()
        self.electrons2ADU()
        self.applyBias()
        self.discretise()
        self.writeOutputs()


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-c', '--configfile', dest='configfile',
                      help="Name of the configuration file", metavar="string")
    parser.add_option('-s', '--section', dest='section',
                      help="Name of the section of the config file [SCIENCE]", metavar="string")
    parser.add_option('-d', '--debug', dest='debug', action='store_true',
                      help='Debugging mode on')
    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.configfile is None:
        processArgs(True)
        sys.exit(1)

    if opts.section is None:
        simulate = VISsimulator(opts.configfile, opts.debug)
    else:
        simulate = VISsimulator(opts.configfile, opts.debug, opts.section)

    simulate.simulate()
