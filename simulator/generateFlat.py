"""
Generating a flat field image
=============================

This script provides a method to generate a flat fielding image that mimics the one expected
for Euclid VIS. The field is a Gaussian random field with a few per cent fluctuations.

:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import os, datetime
import pyfits as pf
import numpy as np
from support import logger as lg


class flatField():
    """
    This class can be used to generate a flat field that mimics the pixel size uniformity
    assumed for VIS.

    :param kwargs: The following arguments can be given::

                     * loc = centre of the distribution
                     * sigma = standard deviation of the distribution
                     * xsize = size of the flat field image in x direction
                     y xsize = size of the flat field image in y direction
    type kwargs: dict
    """
    def __init__(self, log, **kwargs):
        """
        Class constructor.
        """
        self.log = log
        self.settings = dict(loc=1.0,
                             sigma=0.02,
                             xsize=2048,
                             ysize=2066)
        self.settings.update(kwargs)

        #write the input to the log
        self.log.info('The following input parameters were used:')
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))


    def generateFlat(self):
        """
        Creates a flat field image with given properties.

        :return: flat field image
        :rtype: ndarray
        """
        self.log.info('Generating a flat field...')
        self.flat = np.random.normal(loc=self.settings['loc'], scale=self.settings['sigma'],
                                     size=(self.settings['ysize'], self.settings['xsize']))
        return self.flat


    def writeFITS(self, data, output):
        """
        Writes given imaging data to a FITS file.

        :param data: image array
        :type data: ndarray
        :param output: name of the output file
        :type output: str

        :return: None
        """
        self.log.info('Writing data to a FITS file %s' % output)
        if os.path.isfile(output):
            os.remove(output)

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=data)

        #add info
        for key, value in self.settings.iteritems():
            hdu.header.update(key.upper(), value)

        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(output)


if __name__ == '__main__':
    log = lg.setUpLogger('generateFlat.log')

    settings = dict(sigma=0.01)
    flat = flatField(log, **settings)
    data = flat.generateFlat()
    flat.writeFITS(data, 'VISFlatField1percent.fits')

    log.info('Run finished...\n\n\n')