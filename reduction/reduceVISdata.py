"""
Simple script to reduce VIS data.

Does the following steps::

    1 Bias correction
    2 Flat fielding (not yet implemented)
    3 CTI correction (conversion to electrons and back to ADUs)

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.2
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
        Input files are taken from the input dictionary given
        when class was initiated.
        """
        self.data = pf.getdata(self.values['input'], self.values['ext'])
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
            noise = np.random.normal(loc=0.0, scale=math.sqrt(self.values['rnoise']), size=self.data.shape)
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
        :Warning: Note written yet
        """
        #TODO: write this part
        self.log.warning('Flat fielding not implement yet...')
        self.data = self.data


    def applyCTICorrection(self):
        """
        Applies a third order (three forward reads) CTI correction in electrons using CDM03 CTI model.

        Converts the data to electrons using the gain value given in self.values.
        Applies CTI correction and converts back to ADUs using the same gain factor.

        Bristow & Alexov (2003) algorithm further developed for HST data
        processing by Massey, Rhodes et al.
        """
        #multiply with the gain
        self.log.info('Multiplying the data with the gain factor = %.3f to convert electrons' % self.values['gain'])
        self.data *= self.values['gain']

        self.log.info('Applying 3rd order CTI correction')
        #first order
        cti1 = CTI.CDM03(self.values, self.data, self.log).radiateFullCCD()
        corrected1 = 2.*self.data.copy() - cti1
        #second order
        cti2 = CTI.CDM03(self.values, corrected1, self.log).radiateFullCCD()
        corrected2 = self.data.copy() + corrected1 - cti2
        #third order
        cti3 = CTI.CDM03(self.values, corrected2, self.log).radiateFullCCD()
        self.data += corrected2 - cti3

        #divide with the gain
        self.log.info('Dividing the data with the gain factor = %.3f to convert ADUs' % self.values['gain'])
        self.data /= self.values['gain']



    def writeFITSfile(self):
        """
        Write out FITS files using PyFITS.
        """
        if os.path.isfile(self.values['output']):
            os.remove(self.values['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=self.data)

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
        hdu.header.add_history('3)CTI correction  (three forward reads)')
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

    parser.add_option('-f', '--file', dest='input',
                      help="Input file to reduce", metavar='string')
    parser.add_option('-b', '--bias', dest='bias',
                  help='Name of the super bias to use in data reduction', metavar='string')
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
    values = dict(rnoise=4.5, dob=0, rdose=3e10, trapfile='cdm_euclid.dat', bias=1000.0, beta=0.6, fwc=175000,
                  vth=1.168e7, t=1.024e-2, vg=6.e-11, st=5.e-6, sfwc=730000., svg=1.0e-10, output=output,
                  input=opts.input, unsigned16bit=True, ext=1, biasframe=opts.bias, gain=3.5, exptime=565.0)

    reduce = reduceVISdata(values, log)
    reduce.doAll()

    log.info('Reduction done, script will exit...')