"""
VIS Data Reduction and Processing
=================================

This simple script can be used to reduce (simulated) VIS data.

Does the following steps::

    1 Bias correction
    2 Flat fielding
    3 CTI correction (conversion to electrons and back to ADUs)

To Run::

    python reduceVISdata.py -i VISCCD.fits -b superBiasVIS.fits -f SuperFlatField.fits

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.4

.. todo::

    1. FITS extension should probably be read from the command line
    2. implement background/sky subtraction
"""
import numpy as np
import pyfits as pf
from optparse import OptionParser
import sys, os, datetime, math
from support import logger as lg
from CTI import CTI


class reduceVISdata():
    """
    Simple class to reduce VIS data.
    """
    def __init__(self, values, log):
        """
        Class constructor.
        """
        self.values = values
        self.log = log
        self._readData()

        if self.values['biasframe'] is None:
           self._generateSuperBias()
        else:
            self._readBiasframe()


    def _readData(self):
        """
        Reads in data from all the input files.
        Input files are taken from the input dictionary given when class was initiated.
        """
        fh = pf.open(self.values['input'])
        self.data = fh[self.values['ext']].data
        self.hdu = fh[self.values['ext']].header
        self.log.info('Read data from {0:>s} extension {1:d}'.format(self.values['input'], self.values['ext']))


    def _generateSuperBias(self, nbiases=20):
        """
        This method creates a super bias with given readout noise on fly.
        This method is called only if no input bias file name was given otherwise
        the super bias will be read from a file.
        """
        self.log.debug('Generating a super bias on fly...')
        biases = []
        for x in range(nbiases):
            noise = np.random.normal(loc=0.0, scale=self.values['rnoise'], size=self.data.shape)
            biases.append(noise+self.values['bias'])
        biases = np.asarray(biases)
        self.bias = np.median(biases, axis=0)

        self.log.debug('Converted the on-fly super bias to integers...')
        #convert to integers
        self.bias = self.bias.astype(np.int)

        #save to a file
        ofd = pf.HDUList(pf.PrimaryHDU())
        hdu = pf.ImageHDU(data=self.bias)
        hdu.scale('int16', '', bzero=32768)
        hdu.header.update('Nbias', nbiases, 'Number of bias frames used to make this super bias')
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        if os.path.isfile('superBias.fits'):
            os.remove('superBias.fits')
        ofd.writeto('superBias.fits')
        self.log.debug('Saved on on-fly generated super bias to superBias.fits')


    def _readBiasframe(self):
        """
        Read bias frame from a FITS file.
        """
        self.log.info('Reading a bias frame from %s' % self.values['biasframe'])
        self.bias = pf.getdata(self.values['biasframe'])


    def subtractBias(self):
        """
        Simply subtracts self.bias from the input data.
        """
        self.log.info('Subtracting bias (average value = %f)' % np.mean(self.bias))
        self.data -= self.bias


    def flatfield(self):
        """
        Take into account pixel-to-pixel non-uniformity through multipicative flat fielding.
        """
        if self.values['flatfield'] is None:
            self.log.warning('No flat field given, cannot flat field!')
            self.data = self.data
            return

        self.log.info('Flat fielding data (multiplicative)')
        fh = pf.open(self.values['flatfield'])
        flat = fh[1].data

        self.data /= flat


    def applyCTICorrection(self):
        """
        Applies a CTI correction in electrons using CDM03 CTI model.
        Converts the data to electrons using the gain value given in self.values.
        The number of forward reads is defined by self.values['order'] parameter.

        Bristow & Alexov (2003) algorithm further developed for HST data
        processing by Massey, Rhodes et al.

        There is probably an excess of .copy() calls here, but I had some problems
        when calling the Fortran code so I added them for now.
        """
        if self.values['order'] < 1:
            self.log.warning('Order < 1, no CTI correction (forward modeling) applied!')
            return

        #multiply with the gain
        self.log.info('Multiplying the data with the gain factor = %.3f to convert electrons' % self.values['gain'])
        self.data *= self.values['gain']

        #make a copy
        out = self.data.copy()

        self.log.info('Applying %i forward reads for CTI correction' % self.values['order'])
        for x in range(self.values['order']):
            rd = CTI.CDM03(self.values, out.copy(), self.log)
            damaged = rd.radiateFullCCD()
            out += self.data.copy() - damaged
            self.log.info('Forward read %i performed'% (x + 1))
            del(rd)

        #divide with the gain
        self.log.info('Dividing the data with the gain factor = %.3f to convert ADUs' % self.values['gain'])
        self.data = out / self.values['gain']


    def writeFITSfile(self):
        """
        Write out FITS files using PyFITS.
        """
        if os.path.isfile(self.values['output']):
            os.remove(self.values['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList()

        #new image HDU, this will go to zero th extension now
        hdu = pf.PrimaryHDU(self.data, self.hdu)

        #convert to unsigned 16bit int if requested
        if self.values['unsigned16bit']:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #add keywords
        for key, value in self.values.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            if value is None:
                continue
            hdu.header.update(key, value)

        #update and verify the header
        hdu.header.add_history('The following processing steps have been performed:')
        hdu.header.add_history('1)Bias correction')
        hdu.header.add_history('2)Flat fielding (not done currently)')
        hdu.header.add_history('3)CTI correction  (%i forward reads)' % self.values['order'])
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.values['output'])
        self.log.info('Wrote %s' % self.values['output'])


    def doAll(self):
        self.subtractBias()
        self.flatfield()
        self.applyCTICorrection()
        self.writeFITSfile()


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-i', '--input', dest='input',
                      help="Input file to reduce", metavar='string')
    parser.add_option('-b', '--bias', dest='bias',
                      help='Name of the super bias to use in data reduction', metavar='string')
    parser.add_option('-f', '--flat', dest='flat',
                      help='Name of the super flat field to use in data reduction', metavar='string')
    parser.add_option('-o', '--output', dest='output',
                      help="Name of the output file, default=inputReduced.fits", metavar='string')
    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.input is None:
        processArgs(True)
        sys.exit(8)

    log = lg.setUpLogger('reduction.log')
    log.info('\n\nStarting to reduce data...')

    if opts.output is None:
        output = opts.input.replace('.fits', '') + 'Reduced.fits'
    else:
        output = opts.output

    #input values that are used in processing and save to the FITS headers
    values = dict(rnoise=4.5, dob=0, rdose=3e10, trapfile='data/cdm_euclid.dat', bias=1000.0, beta=0.6, fwc=175000,
                  vth=1.168e7, t=1.024e-2, vg=6.e-11, st=5.e-6, sfwc=730000., svg=1.0e-10, output=output,
                  input=opts.input, unsigned16bit=True, ext=0, biasframe=opts.bias, gain=3.5, exposure=565.0,
                  exptime=565.0, order=3, flatfield=opts.flat)

    reduce = reduceVISdata(values, log)
    reduce.doAll()

    log.info('Reduction done, script will exit...')