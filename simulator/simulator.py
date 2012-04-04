"""
Main code of the Euclid Visible Instrument Simulator

:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import os, sys, datetime
import ConfigParser
from optparse import OptionParser
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
        self.information['e_ADU'] = self.config.getfloat(self.section, 'bias')

        #exposure time and position on the sky
        self.information['exptime'] = self.config.getfloat(self.section, 'exptime')
        self.information['RA'] = self.config.getfloat(self.section, 'RA')
        self.information['DEC'] = self.config.getfloat(self.section, 'DEC')

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

        self.cr = dict(cr_u=crLengths[0, :], cr_cdf=crLengths[1, :],
            cr_cdfn=np.shape(crLengths)[0],
            cr_v=crDists[0, :], cr_cde=crDists[1, :],
            cr_cden=np.shape(crDists)[0])


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
        Read object list using numpy.loadtxt
        """
        self.objects = np.loadtxt(self.information['sourcelist'])

        str = '{0:d} sources read from '.format(np.shape(self.objects)[0], self.information['sourcelist'])
        self.log.info(str)

        if self.debug:
            print str


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
        pass


    def applyCosmicRays(self):
        """

        """
        self._readCosmicRayInformation()

        #create empty array
        CCD_cr = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)

        #estimate the number of cosmics
        cr_n = self.information['xsize'] * self.information['ysize'] * 0.014 / 43.263316

        #assume a power-law intensity distribution for tracks
        fit = dict(cr_lo=1.0e3, cr_hi=1.0e5, cr_q=2.0e0)
        fit['q1'] = 1.0e0 - fit['cr_q']
        fit['en1'] = fit['cr_lo'] ** fit['q1']
        fit['en2'] = fit['cr_hi'] ** fit['q1']

        #choose the length of hte tracks
        #pseudo-random number taken from a uniform distribution between 0 and 1
        luck = np.random.rand(int(np.floor(cr_n)))

        if self.cr['cr_cdfn'] > 1:
            pass
            #TODO: add monotonicity-preserving piecewise cubic Hermite interpolantion here
        else:
            self.cr['cr_l'] = np.sqrt(1.0 - luck ** 2) / luck

        if self.cr['cr_cden'] > 1:
            pass
            #TODO: add monotonicity-preserving piecewise cubic Hermite interpolantion here
        else:
            self.cr['cr_e'] = (fit['en1'] + (fit['en2'] - fit['en1']) * np.random.rand(int(np.floor(cr_n)))) ** (
            1.0 / fit['q1'])

        #write out the cosmics information

        #Choose the properties such as positions and an angle from a random Uniform dist
        val = self.cr['cr_e']
        cr_x = self.information['xsize'] * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.information['ysize'] * np.random.rand(int(np.floor(cr_n)))
        cr_pi = np.pi * np.random.rand(int(np.floor(cr_n)))

        #find the intercepts

        #past the information
        self.image += CCD_cr


        #count the covering factor

        #output information to ascii and FITS file


    def applyNoise(self):
        """
        Apply dark current and the cosmic background.
        Both values are scaled with the exposure time
        """
        self.image += self.information['exptime'] * (self.information['dark'] + self.information['cosmic_bkgd'])


    def applyCosmetics(self, input='./data/cosmetics.dat'):
        """
        Apply cosmetic defects described in the input file.

        :param input: name of the input file, the file should be csv type
        :type input: str
        """
        if self.debug:
            print 'Adding cosmetics from %s, the added info:' % input

        cosmetics = np.loadtxt(input, delimiter=',')
        for line in cosmetics:
            x = int(np.floor(line[1]))
            y = int(np.floor(line[2]))
            value = line[3]

            if self.debug:
                print x, y, value

            self.image[y, x] = value


    def applyRadiationDamage(self):
        pass


    def applyReadoutNoise(self):
        """
        Applies readout noise. The noise is drawn from xxx distribution.
        """
        noise = np.random.randn(self.information['ysize'], self.information['xsize'])
        noise += self.information['readout']
        self.image += noise


    def electrons2ADU(self):
        """
        Convert from electrons to ADU using the value read from the configuration file.
        """
        self.image /= self.information['e_ADU']


    def applyBias(self):
        """
        Add bias level to the image.
        The value of bias is read from the configure file and stored
        in the information dictionary (key bias).
        """
        self.image += self.information['bias']

        if self.debug:
            print 'Bias of %i counts were added to the image' % self.information['bias']


    def discretise(self):
        """
        Convert floating point arrays to integer arrays
        """
        self.image = self.image.astype(np.int)

        if self.debug:
            print 'Maximum and total values of the image are %i and %i, respectively' % (np.max(self.image),
                                                                                         np.sum(self.image))

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

        #update header

        hdu.header.update('RA', self.information['RA'], 'RA of the center of the chip')
        hdu.header.update('DEC', self.information['DEC'], 'DEC of the center of the chip')

        hdu.header.add_history('Created by VISsim at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.information['output'])#, output_verify='ignore')


    def simulate(self):
        """
        Create a simulated image
        """
        self.configure()
        self.readObjectlist()

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