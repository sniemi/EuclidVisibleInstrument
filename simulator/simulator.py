"""
Main code of the Euclid Visible Instrument Simulator

:requires: PyFITS
:requires: NumPy
:requires: SciPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.2
"""
import os, sys, datetime, math
import ConfigParser
from optparse import OptionParser
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import griddata
from scipy import ndimage
import logger as lg
import pyfits as pf
import numpy as np


class VISsim():
    """
    Euclid Visible Instrument Simulator.

    The image that is being build is in::

        self.image

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
        Processes configuration information and produces several dictionaries.
        """
        #sizes
        self.information['xsize'] = self.config.getint(self.section, 'xsize')
        self.information['ysize'] = self.config.getint(self.section, 'ysize')

        #noises
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

        #zeropoint
        self.information['magzero'] = self.config.getfloat(self.section, 'magzero')

        #inputs
        self.information['sourcelist'] = self.config.get(self.section, 'sourcelist')

        #output
        self.information['output'] = self.config.get(self.section, 'output')

        #booleans to control the flow
        self.flatfieldM = self.config.getboolean(self.section, 'flatfieldM')
        self.flatfieldA = self.config.getboolean(self.section, 'flatfieldA')
        self.chargeInjection = self.config.getboolean(self.section, 'chargeInjection')
        self.cosmicRays = self.config.getboolean(self.section, 'cosmicRays')
        self.noise = self.config.getboolean(self.section, 'noise')
        self.cosmetics = self.config.getboolean(self.section, 'cosmetics')
        self.radiationDamage = self.config.getboolean(self.section, 'radiationDamage')

        if self.debug:
            print self.information


    def _createEmpty(self):
        """
        Generates and empty array with zeros.
        """
        self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)


    def _crIntercepts(self, lum, x0, y0, l, phi):
        """
        :param lum: luminosities of cosmic ray tracks
        :param x0: central positions of the cosmic ray tracks in x-direction
        :param y0: central positions of the cosmic ray tracks in y-direction
        :param l: lengths of the cosmic ray tracks
        :param phi: orientation angles of cosmic ray tracks

        :return: map
        :rtype: nd-array
        """
        #create empty array
        crImage = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)

#        #delta x and y
#        dxs = l * np.cos(phi)
#        dys = l * np.sin(phi)
#
#        #pixels in x-direction
#        ilos = np.floor(x0 - l)
#        ilos[ilos < 1.0] = 1.0
#        ihis = 1 + np.floor(x0 + l)
#        ihis[ihis > self.information['xsize']] = self.information['xsize']
#
#        #pixels in y-directions
#        jlos = np.floor(y0 - l)
#        jlos[jlos < 1.0] = 1.0
#        jhis = 1 + np.floor(y0 + l)
#        jhis[jhis > self.information['ysize']] = self.information['ysize']

        #this is very slow way to do this
        for cosmics in range(0, len(l)):
            #delta x and y
            dx = l[cosmics] * np.cos(phi[cosmics])
            dy = l[cosmics] * np.sin(phi[cosmics])

            #pixels in x-direction
            ilo = np.floor(x0[cosmics] - l[cosmics])

            if ilo < 1.:
                ilo = 1.0
            ihi = 1 + np.floor(x0[cosmics] + l[cosmics])
            if ihi > self.information['xsize']:
                ihi = self.information['xsize']

            #pixels in y-directions
            jlo = np.floor(y0[cosmics] - l[cosmics])
            if jlo < 1.0:
                jlo = 1.0
            jhi = 1 + np.floor(y0[cosmics] + l[cosmics])
            if jhi > self.information['ysize']:
                jhi = self.information['ysize']

            u = []
            x = []
            y = []

            n = 0  # count the intercepts

            #Compute X intercepts on the pixel grid
            if dx > 0.0:
                for j in range(int(ilo), int(ihi)):
                    ok = (j - x0[cosmics]) / dx
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(j)
                        y.append(y0[cosmics] + ok * dy)

            #Compute Y intercepts on the pixel grid
            if dy > 0.0:
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
        #TODO: double check that cosmic ray information is read correctly
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


    def _objectOnDetector(self, object, nx, ny):
        """
        Test if the object falls on the detector
        """
        mx = self.information['xsize']
        my = self.information['ysize']
        fac = object[6]
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

        if (i2 < 1) or (i1 >= mx):
            return False
        if (j2 < 1) or (j1 >= my):
            return False

        return True


    def _overlayToCCD(self, data, obj):
        """
        Overlay data of a source object to the image.

        :param data: imaging data of the object
        :type data: ndarray
        :param obj: object information
        :type obj: list

        :Note: rewrite this, now a direct copy from FORTRAN

        :return: None
        """
        fac = obj[6]
        xt = obj[0]
        yt = obj[1]

        #Copy object to 1D array, and calculate its transformed coordinates.
        npoints = self.information['xsize'] * self.information['ysize']
        xi = np.zeros(npoints)
        yi = np.zeros(npoints)
        zi = np.zeros(npoints)

        for i in range(self.information['xsize']):
            for j in range(self.information['ysize']):
                xi[i + self.information['xsize']*(j-1)] = ( 2*i - self.information['xsize'] - 1 ) * 0.5 * fac  + xt
                yi[i + self.information['xsize']*(j-1)] = ( 2*j - self.information['ysize'] - 1 ) * 0.5 * fac  + yt
                zi[i + self.information['xsize']*(j-1)] = self.image[j, i]

        # Assess the boundary box of the input image.
        xlo = (1 - self.information['xsize']) * 0.5 * fac  + xt
        xhi = (self.information['xsize']-1) * 0.5 * fac  + xt
        ylo = (1 - self.information['ysize']) * 0.5 * fac  + yt
        yhi = (self.information['ysize']-1) * 0.5 * fac  + yt

        i1 = int(max([np.floor(xlo + 0.5), 1]))
        i2 = int(min([np.floor(xhi + 0.5) + 1, self.information['xsize']]))
        j1 = int(max([np.floor(ylo + 0.5), 1]))
        j2 = int(min([np.floor(yhi + 0.5) + 1, self.information['ysize']]))

        ni = i2 - i1 + 1
        nj = j2 - j1 + 1
        nij = ni * nj

        #Initialise the output image coordinates.
        xo = np.zeros(nij)
        yo = np.zeros(nij)
        for i in range(i1, i2):
            for j in range(j1, j2):
                xo[1 + (i - i1) + ni * (j - j1)] = i - 0.5
                yo[1 + (i - i1) + ni * (j - j1)] = j - 0.5

        #Select the destination points that overlap the input box.
        N = 0
        for k in range(1, nij):
            if xo[k] >= xlo and xo[k] <= xhi and yo[k] >= ylo and yo[k] <= yhi:
                N += 1
        if N <= 0:
            #Quit if there are no overlapping points.
            return None

        QX = np.zeros(N)
        QY = np.zeros(N)
        xq = np.zeros(N)
        yq = np.zeros(N)
        zq = np.zeros(N)
        ic = np.zeros(N)
        jc = np.zeros(N)

        i = 0
        for k in range(nij):
            if xo[k] >= xlo and xo[k] <= xhi and yo[k] >= ylo and yo[k] <= yhi:
                xq[i] = xo[k]
                yq[i] = yo[k]
                ic[i] = 1 + np.mod((k - 1), ni)
                jc[i] = j1 + ((k - ic[i]) / ni)
                i += 1

        ic = ic - 1 + i1

#        LIQ = 2 * np + 1.
#        LRQ = 6 * np + 5.

        if N >= 1:
#            CALL E01SGF(np,xi,yi,zi,NW,NQ,IQ,LIQ,RQ,LRQ,IFAIL)
#            #Evaluate the interpolant using E01SHF.
#            #write(*,*) 'OVERLAY, N =',n
#            CALL E01SHF(np,xi,yi,zi,IQ,LIQ,RQ,LRQ,N,xq,yq,zq,QX,QY,IFAIL)
            zq = ndimage.map_coordinates(data,
                                         np.mgrid[0:self.information['ysize'], 0:self.information['xsize']],
                                         order=3,
                                         mode='nearest')

            #Suppress any negative values.
            zq[zq < 0.0] = 0.0

            #Add transformed image into the 2D output array.
            for k in range(1, N):
                self.image[jc[k], ic[k]] += zq[jc[k], ic[k]]

        else:
            i = int(math.floor(xt))
            j = int(math.floor(yt))

            fi = xt - i
            fj = yt - j

            if i >= 1 and i <= self.information['xsize']:
               if j >= 1  and j <= self.information['ysize']:
                   self.image[j, i] += np.sum(data)*fi*fj
               if j >= 0 and j <= self.information['ysize'] - 1:
                   self.image[j+1, i] += np.sum(data)*fi*(1. - fj)

            if i >= 0 and i <= self.information['xsize'] - 1:
               if j >= 1 and j <= self.information['ysize']:
                   self.image[j, i+1] += np.sum(data)*(1. - fi)*fj
               if j >= 0 and j <= self.information['ysize'] - 1:
                   self.image[j+1, i+1]+= np.sum(data)*(1. - fi)*(1. - fj)



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


    def readPSFs(self, psffile='data/psf.fits'):
        """
        Reads in the PSFs from a file.

        :Note: at the moment this supports only a single PSF file
        """
        self.log.info('Opening PSF file %s' % psffile)
        self.PSF = pf.getdata(psffile)
        self.PSFx = self.PSF.shape[1]
        self.PSFy = self.PSF.shape[0]
        self.log.info('PSF sampling (x,y) = (%i, %i) ' % (self.PSFx, self.PSFy))


    def generateFinemaps(self):
        """
        Generate finely sampled images of the input data
        """
        #allocate (finemap(fm_nx,fm_ny,n_sp_type))
        self.finemap = {}
        self.shapex = {}
        self.shapey = {}

        #TODO: rewrite this one
        for k, stype in enumerate(self.sp):
            fm = np.zeros((self.PSFy, self.PSFx))

            if stype == 0:
                i = self.PSFx
                j = self.PSFy
                i1 = 1 + (self.PSFx - i) / 2
                i2 = i1 + self.PSFx - 1
                j1 = 1 + (self.PSFy - j) / 2
                j2 = j1 + self.PSFy - 1
                fm[j1:j2, i1:i2] = self.PSF[1:j, 1:i]
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

                fm /= np.sum(fm)
                self.finemap[stype] = fm

                #Compute a shape tensor.
                Qxx=0.
                Qxy=0.
                Qyy=0.
                for i in range(0, self.PSFx):
                    for j in range(0, self.PSFy):
                        Qxx += fm[j,i]*(i-0.5*(self.PSFx-1))*(i-0.5*(self.PSFx-1))
                        Qxy += fm[j,i]*(i-0.5*(self.PSFx-1))*(j-0.5*(self.PSFy-1))
                        Qyy += fm[j,i]*(j-0.5*(self.PSFy-1))*(j-0.5*(self.PSFy-1))

                shx = (Qxx + Qyy + math.sqrt((Qxx - Qyy)**2 + 4.*Qxy*Qxy ))/2.
                shy = (Qxx + Qyy - math.sqrt((Qxx - Qyy)**2 + 4.*Qxy*Qxy ))/2.
                self.shapex[stype] = (math.sqrt(shx / np.sum(fm)))
                self.shapey[stype] = (math.sqrt(shy / np.sum(fm)))

        #sum of the finemaps
        self.log.info('finemap sum = %f' %np.sum(np.asarray(self.finemap.values())))


    def addObjects(self):
        """

        """
        self.log.info('Number of CCD transits = %i' % self.information['exposures'])

        n_objects = self.objects.shape[0]

        #loop over exposures
        for i in range(self.information['exposures']):
            #loop over the number of objects
            for j, obj in enumerate(self.objects):

                if j > 8:
                    break

                stype = obj[3]
                #TODO: needs to change this based on object
                inputx = self.PSFx
                inputy = self.PSFy

                if self._objectOnDetector(obj, inputx, inputy):

                    intscale = 10.0 ** (-0.4 * obj[2]) *\
                               self.information['magzero'] * self.information['exptime']

                    if stype == 0:
                        #point source, apply PSF
                        print "Star:   ",j+1,"/",n_objects, "  intscale=", intscale

                        #blending of PSF no implemented

                        #normalise the input data and scale with the intscale
                        data = self.finemap[stype] /np.sum(self.finemap[stype]) * intscale

                        #Find the bounding box of nonzero pixels

                        #overlays on the image
                        self._overlayToCCD(data, obj)


                    else:
                        #extended source
                        sbig = ( 0.2**((22. - obj[2])/7.) )/self.shapey[stype]/2.
                        print "Extended object: ",j+1,"/",n_objects," size=",sbig

                        #renormalize
                        data = self.finemap[stype] /np.sum(self.finemap[stype]) * intscale

                        #convolve with PSF

                        #overlay on the image
                        self._overlayToCCD(data, obj)

                else:
                    #not on the screen
                    print 'OFFscreen: ', j+1


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


    def applyChargeInjection(self):
        """
        Apply either horizontal or vertical charge injection line to the image.
        """
        #TODO: write the code
        pass
        #        if injection == 1:
        #            self.image[self.information['ysize']/2-10:self.information['ysize']/2, :] = inject_num
        #            self.log.info('Adding vertical charge injection line')
        #        if injection == 2:
        #            self.image[:, self.information['xsize']/2-10:self.information['xsize']/2] = inject_num
        #            self.log.info('Adding horizontal charge injection line')


    def applyCosmicRays(self):
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
        #number of points = cr_cdfn
        #independent values = cr_cdf
        #dependent values = cr_u
        #derivatices = cr_d
        #the number of points at which to interpolate = cr_n
        #at which the interpolant is to be evaluated = luck
        #contains the value of the interpolant evaluated at the bolve point = cr_l
        #call e01bef(cr_cdfn,cr_cdf,cr_u,cr_d,ifail)
        #call e01bff(cr_cdfn,cr_cdf,cr_u,cr_d,cr_n,luck,cr_l,ifail)
        #Not the same but with SciPy
            ius = InterpolatedUnivariateSpline(self.cr['cr_cdf'], self.cr['cr_u'])
            self.cr['cr_l'] = ius(luck)

        else:
            self.cr['cr_l'] = np.sqrt(1.0 - luck ** 2) / luck

        if self.cr['cr_cden'] > 1:
        #TODO: add monotonicity-preserving piecewise cubic Hermite interpolation here
        #call e01bef(cr_cden,cr_cde,cr_v,cr_de,ifail)
        #call e01bff(cr_cden,cr_cde,cr_v,cr_de,cr_n,luck,cr_e,ifail)
            ius = InterpolatedUnivariateSpline(self.cr['cr_cde'], self.cr['cr_v'])
            self.cr['cr_e'] = ius(luck)
        else:
            self.cr['cr_e'] = (fit['en1'] + (fit['en2'] - fit['en1']) *
                                        np.random.rand(int(np.floor(cr_n)))) ** (1.0 / fit['q1'])

        #write out the cosmics information

        #Choose the properties such as positions and an angle from a random Uniform dist
        cr_x = self.information['xsize'] * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.information['ysize'] * np.random.rand(int(np.floor(cr_n)))
        cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

        #find the intercepts
        CCD_cr = self._crIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)

        #paste the information
        self.image += CCD_cr

        #count the covering factor
        area_cr = np.count_nonzero(CCD_cr)
        self.log.info('The cosmic ray covering factor is %i pixels ' % area_cr)

        #output information to a FITS file
        self._writeFITSfile(CCD_cr, 'cosmicraymap.fits')


    def applyNoise(self):
        """
        Apply dark current and the cosmic background.

        Scale the dark current with the exposure time, but apply the cosmic background as given.
        """
        noise = self.information['exptime'] * self.information['dark'] + self.information['cosmic_bkgd']
        self.image += noise

        self.log.info('Added dark noise and cosmic background, total noise = %f' % noise)


    def applyCosmetics(self, input='./data/cosmetics.dat'):
        """
        Apply cosmetic defects described in the input file.

        :param input: name of the input file, the file should be csv type
        :type input: str
        """
        cosmetics = np.loadtxt(input, delimiter=',')

        #TODO: this does not work if there is exactly one line in the file
        for line in cosmetics:
            x = int(np.floor(line[1]))
            y = int(np.floor(line[2]))
            value = line[3]
            self.image[y, x] = value

            self.log.info('Adding cosmetic defects from %s:' % input)
            self.log.info('x=%i, y=%i, value=%f' % (x, y, value))


    def applyRadiationDamage(self):
        #TODO: write the radiation damage part
        pass


    def applyReadoutNoise(self):
        """
        Applies readout noise. The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout noise).
        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(self.information['readout']),
                                 size=(self.information['ysize'], self.information['xsize']))
        self.log.info('Sum of readnoise = %f' % np.sum(noise))
        self.image += noise


    def electrons2ADU(self):
        """
        Convert from electrons to ADU using the value read from the configuration file.
        """
        self.image /= self.information['e_ADU']
        self.log.info('Converting from electrons to ADUs using a factor of %f' % self.information['e_ADU'])


    def applyBias(self):
        """
        Add bias level to the image.

        The value of bias is read from the configure file and stored
        in the information dictionary (key bias).
        """
        self.image += self.information['bias']
        self.log.info('Bias of %i counts were added to the image' % self.information['bias'])


    def discretise(self):
        """
        Convert floating point arrays to integer arrays.
        """
        self.image = self.image.astype(np.int)
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

        #update and verify the header
        hdu.header.update('RA', self.information['RA'], 'RA of the center of the chip')
        hdu.header.update('DEC', self.information['DEC'], 'DEC of the center of the chip')
        hdu.header.add_history('Created by VISsim at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.information['output'])


    def simulate(self):
        """
        Create a simulated image
        """
        self.configure()
        self.readObjectlist()
        self.readPSFs()
        self.generateFinemaps()
        self.addObjects()

        if self.flatfieldA or self.flatfieldM:
            self.applyFlatfield()

        if self.chargeInjection:
            self.applyChargeInjection()

        if self.cosmicRays:
            self.applyCosmicRays()

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

        :Note: Use this for debugging only!
        """
        self.configure()
        self.readObjectlist()
        self.readPSFs()
        self.generateFinemaps()
        self.addObjects()
        self.applyFlatfield()
        self.applyChargeInjection()
        self.applyCosmicRays()
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
        simulate = VISsim(opts.configfile, opts.debug)
    else:
        simulate = VISsim(opts.configfile, opts.debug, opts.section)

    simulate.runAll()