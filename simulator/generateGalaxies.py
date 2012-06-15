"""
Generating Objects
==================

This script provides a class that can be used to generate objects such as galaxies.

:requires: PyRAF
:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
from pyraf import iraf
from iraf import artdata
import numpy as np
import pyfits as pf
from support import logger as lg
import os, datetime


class generateFakeData():
    """

    """
    def __init__(self, log, **kwargs):
        """

        """
        self.log = log
        self.settings = dict(dynrange=1e4,
                             gain=3.5,
                             magzero=25.58,
                             exptime=565.0,
                             rdnoise=4.5,
                             background=0.049,
                             xdim=4096,
                             ydim=4132,
                             star='gaussian',
                             beta=2.5,
                             radius=0.18,
                             ar=1.0,
                             pa=0.0,
                             poisson=iraf.yes,
                             egalmix=0.4,
                             output='image.fits')
        self.settings.update(kwargs)
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))
        #self._createEmptyImage()


    def _createEmptyImage(self, unsigned16bit=True):
        """

        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool
        """
        self.image = np.zeros((self.settings['ydim'], self.settings['xdim']))

        if os.path.isfile(self.settings['output']):
            os.remove(self.settings['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=self.image)

        #convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #update and verify the header
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.settings['output'])
        self.log.info('Wrote %s' % self.settings['output'])


    def createStarlist(self, nstars=20, output='stars.dat'):
        """
        Generates an ascii file with uniform random x and y positions.
        The magnitudes of stars are taken from an isotropic and homogeneous power-law distribution.

        The output ascii file contains the following columns: xc yc magnitude

        :param nstars: number of stars to include
        :type nstars: int
        :param output: name of the output ascii file
        :type output: str
        """
        self.log.info('Generating a list of stars; including %i stars to %s' % (nstars, output))
        if os.path.isfile(output):
            os.remove(output)
        iraf.starlist(output, nstars, xmax=self.settings['xdim'], ymax=self.settings['ydim'])#,
                      #minmag=5, maxmag=15)


    def createGalaxylist(self, ngalaxies=150, output='galaxies.dat'):
        """
        Generates an ascii file with uniform random x and y positions.
        The magnitudes of galaxies are taken from an isotropic and homogeneous power-law distribution.

        The output ascii file contains the following columns: xc yc magnitude model radius ar pa <save>

        :param ngalaxies: number of galaxies to include
        :type ngalaxies: int
        :param output: name of the output ascii file
        :type output: str
        """
        self.log.info('Generating a list of galaxies; including %i galaxies to %s' % (ngalaxies, output))
        if os.path.isfile(output):
            os.remove(output)
        iraf.gallist(output, ngalaxies, xmax=self.settings['xdim'], ymax=self.settings['ydim'],
                     egalmix=self.settings['egalmix'], maxmag=23.0, minmag=10)


    def addObjects(self, inputlist='galaxies.dat'):
        """
        Add objects from inputlist to the output image.

        :param inputlist: name of the input list
        :type inputlist: str

        """
        self.log.info('Adding objects from %s to %s' % (inputlist, self.settings['output']))
        iraf.artdata.dynrange = self.settings['dynrange']
        iraf.mkobjects(self.settings['output'],
                       output='',
                        ncols=self.settings['xdim'],
                        nlines=self.settings['ydim'],
                        background=self.settings['background'],
                        objects=inputlist,
                        xoffset=0.0,
                        yoffset=0.0,
                        star=self.settings['star'],
                        radius=self.settings['radius'],
                        beta=self.settings['beta'],
                        ar=self.settings['ar'],
                        pa=self.settings['pa'],
                        distance=1.0,
                        exptime=self.settings['exptime'],
                        magzero=self.settings['magzero'],
                        gain=self.settings['gain'],
                        rdnoise=self.settings['rdnoise'],
                        poisson=self.settings['poisson'],
                        seed=2,
                        comments=iraf.yes)


    def maskCrazyValues(self, filename=None):
        """
        For some reason mkobjects sometimes adds crazy values to an image.
        This method tries to remove those values and set them to more reasonable ones.
        The values > 65k are set to the median of the image.

        :param filename: name of the input file to modify [default = self.settings['output']]
        :type filename: str

        :return: None
        """
        if filename is None:
            filename = self.settings['output']

        fh = pf.open(filename, mode='update')
        hdu = fh[0]
        data = fh[0].data

        msk = data > 65000.
        median = np.median(data)
        data[msk] = median

        hdu.scale('int16', '', bzero=32768)
        hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #update the header
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))

        fh.close()


    def runAll(self, nostars=True):
        """
        Run all methods sequentially.
        """
        if nostars:
            self.createStarlist()
            self.addObjects(inputlist='stars.dat')
        self.createGalaxylist()
        self.addObjects()
        self.maskCrazyValues()


if __name__ == '__main__':
    log = lg.setUpLogger('generateGalaxies.log')
    log.info('Starting to create fake galaxies')

    fakedata = generateFakeData(log)
    fakedata.runAll()

    #no noise or background
    settings = dict(rdnoise=0.0, background=1/565., output='nonoise.fits', poisson=iraf.no)
    fakedata = generateFakeData(log, **settings)
    fakedata.runAll()

    #postage stamp galaxy
    settings = dict(rdnoise=0.0, background=0.0, output='stamp.fits', poisson=iraf.no, xdim=200, ydim=200)
    fakedata = generateFakeData(log, **settings)
    fakedata.addObjects(inputlist='singlegalaxy.dat')
    fakedata.maskCrazyValues('stamp.fits')

    log.info('All done...\n\n\n')