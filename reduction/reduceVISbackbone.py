"""
VIS Data Reduction Pipeline
===========================

This simple backbone can be used as a VIS data reduction pipeline.

The pipeline performs the following data reduction steps that can be controlled using a config file:

    #. Nonlinearity correction
    #. Bias subtraction
    #. Charge transfer inefficiency correction
    #. Photo-response non-uniformity correction (p-flat)
    #. Background subtraction
    #. Illumination correction (l-flat)
    #. Cosmic ray rejection

Input information is read from a section of an input config file. The call to the script is as follows::

    python reduceVISbackbone.py -c config/pipelineConfig.cfg -s EXAMPLE

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, must be compiled: f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi (MSSL)
:contact: s.niemi@ucl.ac.uk

:version: 0.1a
"""
import ConfigParser, os, datetime, sys, logging, logging.handlers
from optparse import OptionParser
import numpy as np
import pyfits as pf
from CTI import CTI


class VISdataReductionBackbone():
    """
    Simple class that provides VIS data reduction pipeline backbone.
    """
    def __init__(self, configfile, section, instrument, log, debug):
        """
        Class constructor.
        """
        self.instrument = instrument
        self.log = log
        self.debug = debug
        self.section = section
        self.configfile = configfile

        #read the config file
        self.config = ConfigParser.RawConfigParser()
        self.config.readfp(open(self.configfile))

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
                    try:
                        settings[option] = self.config.getboolean(self.section, option)
                    except ValueError:
                        settings[option] = self.config.get(self.section, option)

        self.instrument.update(settings)

        for key, value in self.instrument.iteritems():
            self.log.info('%s = %s' % (key, value))

        if self.debug:
            print '\nInstrument information:'
            print self.instrument


    def readData(self, file, extension=1):
        """
        Reads in data from a FITS file.

        :param file: name of the input FITS file
        :type file: str
        :param extension: FITS extension number
        :type extension: int

        :return: image data, FITS header
        :rtype: tuple
        """
        fh = pf.open(file)
        data = fh[extension].data
        hdu = fh[extension].header

        self.log.info('Read data from {0:>s} extension {1:d}'.format(file, extension))
        self.log.debug('Read data dimensions are {0:d} times {1:d}'.format(*data.shape))

        try:
            overscan = hdu['OVERSCA']
        except:
            overscan = ''

        if 'rue' in overscan and data.shape[0] < 2500:
            #a single quadrant; over and prescans were simulated, need to be removed...
            #note that this removes the overscan data... should be used in bias subtraction...
            self.log.info('Trying to remove pre- and overscan regions from the given in input data...')
            self.log.info('Quadrant is {0:d}'.format(hdu['QUADRANT']))

            if hdu['QUADRANT'] in (0, 2):
                data = data[:, hdu['PRESCANX']:-hdu['OVRSCANX']].copy()
            else:
                data = data[:, hdu['OVRSCANX']:-hdu['PRESCANX']].copy()

        return data, hdu


    def applyNonlinearityCorrection(self):
        """
        Perform nonlinearity correction.

        .. warning:: a dummy procedure now.

        .. todo:: Replace with a real  nonlinearity correction algorithm.
        """
        self.log.info('Performing nonlinearity correction [DUMMY, nothing done]...')


    def applyBiasSubtraction(self):
        """
        Simply subtracts a bias image from the input data.

        .. todo:: Replace with a real bias subtraction algorithm.
        """
        bias, _ = self.readData(self.instrument['biasfile'])
        self.log.info('Subtracting bias (average value = %f, median = %f)' % (np.mean(bias), np.median(bias)))
        self.data -= bias


    def applyPRNUCorrection(self):
        """
        Take into account pixel-to-pixel photo-response non-uniformity (PRNU) through multiplicative flat fielding.

        .. todo:: Replace with a real flat fielding algorithm.
        """
        flat, _ = self.readData(self.instrument['flatfieldfile'])
        self.log.info('Removing pixel-to-pixel variation via flat fielding...')
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
        self.instrument['trapfile'] = self.instrument['ctifile']

        if self.instrument['order'] < 1:
            self.log.warning('Order < 1, no CTI correction (forward modeling) applied!')
            return

        #multiply with the gain
        self.log.info('Multiplying the data with the gain factor = %.3f to convert electrons' % self.instrument['gain'])
        self.data *= self.instrument['gain']

        #make a copy, not necessary...?
        out = self.data.copy()

        if out.shape[0] < 2500:
            #this must be quadrant
            self.log.info('Applying %i forward reads to a quadrant to perform a CTI correction' % self.instrument['order'])
            for x in range(self.instrument['order']):
                rd = CTI.CDM03(self.instrument, [], self.log)
                damaged = rd.applyRadiationDamage(out.copy(), iquadrant=self.hdu['QUADRANT'])
                out += self.data.copy() - damaged
                self.log.info('Forward read %i performed'% (x + 1))
                del(rd)
        else:
            self.log.info('Applying %i forward reads to the full CCD to perform a CTI correction' % self.instrument['order'])
            for x in range(self.instrument['order']):
                rd = CTI.CDM03(self.instrument, out.copy(), self.log)
                damaged = rd.radiateFullCCD2()
                out += self.data.copy() - damaged
                self.log.info('Forward read %i performed'% (x + 1))
                del(rd)

        #divide with the gain
        self.log.info('Dividing the data with the gain factor = %.3f to convert ADUs' % self.instrument['gain'])
        self.data = out / self.instrument['gain']


    def applyBackgroundSubtraction(self):
        """
        Perform background subtraction.

        .. warning:: a dummy procedure now.

        .. todo:: Replace with a real background subtraction algorithm.
        """
        self.log.info('Subtracting background [DUMMY, nothing done]...')


    def applyIlluminationCorrection(self):
        """
        Perform illumination (l-flat) correction.

        .. warning:: a dummy procedure now.

        .. todo:: Replace with a real illumination correction algorithm.
        """
        self.log.info('Applying illumination correction [DUMMY, nothing done]...')


    def applyCosmicrayRejection(self):
        """
        Perform cosmic ray rejection.

        .. warning:: a dummy procedure now.

        .. todo:: Replace with a real cosmic ray rejection algorithm.
        """
        self.log.info('Applying cosmic ray rejection and flagging [DUMMY, nothing done]...')


    def writeFITSfile(self):
        """
        Write out FITS files using PyFITS.
        """
        if os.path.isfile(self.instrument['output']):
            os.remove(self.instrument['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList()

        #new image HDU, this will go to zero th extension now
        hdu = pf.PrimaryHDU(self.data, self.hdu)

        #convert to unsigned 16bit int if requested
        if self.instrument['unsigned16bit']:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #add keywords
        for key, value in self.instrument.iteritems():
            #truncate long keys
            if len(key) > 8:
                key = key[:7]
            #covernt to a string
            if value is None:
                value = 'N/A'
            #take only the first in the list as this is the quadrant
            if 'quads' in key:
                value = value[0]
            hdu.header.update(key, value)

        #update and verify the header
        hdu.header.add_history('The following processing steps have been performed:')
        hdu.header.add_history('- Nonlinearity correction')
        hdu.header.add_history('- Bias subtraction')
        hdu.header.add_history('- Charge transfer inefficiency correction')
        hdu.header.add_history('- Photo-response non-uniformity correction (p-flat)')
        hdu.header.add_history('- Background subtraction')
        hdu.header.add_history('- Illumination correction (l-flat)')
        hdu.header.add_history('- CTI correction (with %i forward reads)' % self.instrument['order'])
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.instrument['output'])
        self.log.info('Wrote %s' % self.instrument['output'])


    def execute(self):
        """
        Executes the backbone based on the config file arguments.
        """
        #read in the image data and header that is going to be reduced...
        self.data, self.hdu = self.readData(self.instrument['input'], self.instrument['extension'])

        if self.instrument['nonlinearitycorrection']:
            self.applyNonlinearityCorrection()

        if self.instrument['biassubtraction']:
            self.applyBiasSubtraction()

        if self.instrument['cticorrection']:
            self.applyCTICorrection()

        if self.instrument['flatfielding']:
            self.applyPRNUCorrection()

        if self.instrument['backgroundsubtraction']:
            self.applyBackgroundSubtraction()

        if self.instrument['illuminationcorrection']:
            self.applyIlluminationCorrection()

        if self.instrument['cosmicrayrejection']:
            self.applyCosmicrayRejection()

        self.writeFITSfile()


def setUpLogger(log_filename, loggername='logger'):
    """
    Sets up a logger.

    :param: log_filename: name of the file to save the log.
    :param: loggername: name of the logger

    :return: logger instance
    """
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def VISinformation():
    """
    Returns a dictionary describing VIS.

    :return: instrument model parameters
    :rtype: dict
    """
    out = dict(pixel_size=0.1, gain=3.1, xsize=2048, ysize=2066, prescanx=50, ovrscanx=20, unsigned16bit=False)
    out.update({'dob' : 0, 'rdose' : 3e10, 'trapfile' : 'dummy',
                'beta' : 0.6, 'fwc' : 175000, 'vth' : 1.168e7, 't' : 1.024e-2, 'vg' : 6.e-11,
                'st' : 5.e-6, 'sfwc' : 730000., 'svg' : 1.0e-10})
    return out


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-c', '--configfile', dest='configfile',
        help="Name of the configuration file", metavar="string")
    parser.add_option('-s', '--section', dest='section',
        help="Name of the section of the config file [EXAMPLE]", metavar="string")
    parser.add_option('-d', '--debug', dest='debug', action='store_true',
        help='Debugging mode on')
    if printHelp:
        parser.print_help()
        print '\nThe config file should contain a section (in this case called EXAMPLE) with the following information:'
        print """[EXAMPLE]
#file to reduce and output
input = Q0_00_00science.fits
extension = 1
output = reduced.fits

#reference files and settings
biasFile = refs/superBiasVIS.fits
flatfieldFile = refs/superFlatVIS.fits
CTIfile = refs/CDM03VIS.dat
order = 5

#processes to apply?
nonlinearitycorrection = yes
biasSubtraction = yes
CTIcorrection = yes
flatfielding = yes
backgroundSubtraction = yes
illuminationcorrection = yes
cosmicrayRejection = yes"""
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.configfile is None:
        processArgs(True)
        sys.exit(-9)

    if opts.debug is None:
        opts.debug = False

    #information describing the instrument, should come from database
    instrument = VISinformation()

    #set up logger, this could be a common...
    log = setUpLogger('logs/reduceVISbackbone.log')
    log.info('\n\nStarting to reduce data...')

    reduce = VISdataReductionBackbone(opts.configfile, opts.section, instrument, log, opts.debug)
    reduce.execute()

    log.info('Run finished, script will exit...')