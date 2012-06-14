"""
Genrating a mosaic
==================

This file contains a class to create a single VIS CCD image from separate files one for each quadrant.

:requires: NumPy
:requires: PyFITS

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

To Run::

    python tileCCD.py -f 'Q*_00_00_VISscience.fits'

:version: 0.2
"""
import pyfits as pf
import numpy as np
from optparse import OptionParser
import sys, os, datetime, re
import glob as g
from support import logger as lg


class tileCCD():
    """
    Class to create a single VIS CCD image from separate quadrants files.
    """
    def __init__(self, inputs, log):
        """
        Class constructor.
        """
        self.inputs = inputs
        self.log = log


    def readData(self):
        """
        Reads in data from all the input files and the header from the first file.
        Input files are taken from the input dictionary given when class was initiated.
        """
        data = {}
        for i, file in enumerate(self.inputs['files']):
            fh = pf.open(file)
            if i == 0:
                self.hdu = fh[self.inputs['ext']].header
            data[file] = fh[self.inputs['ext']].data
            self.log.info('Read data from {0:>s} extension {1:d}'.format(file, self.inputs['ext']))
            fh.close()

        self.data = data
        return self.data


    def tileCCD(self, xsize=2048, ysize=2066):
        """
        Tiles quadrants to a single CCD.

        Assume that the input file naming convention is Qx_CCDX_CCDY_name.fits.

        :param xsize: length of a quadrant in column direction
        :type xsize: int
        :param ysize: length of a quadrant in row direction
        :type ysize: int
        """
        self.CCDdata = np.zeros((ysize*2, xsize*2))


        for key, data in self.data.iteritems():
            #use regular expression to find the numbers
            p = re.compile('\d+')
            ls = p.findall(key)
            #in ls, numbers are [quadrant, CCDx, CCDy]
            if len(ls) != 3:
                print 'Problem when parsing the file name!'
                print 'Filenames should be in format:'
                print 'Qx_CCDX_CCDY_filename.fits'
                self.log.error('Problem when parsing the file name!')
                return self.CCDdata

            #note here that because we will save a FITS file the numbering
            #starts from the lower left corner!
            if int(ls[0]) == 0:
                self.CCDdata[0:ysize, 0:xsize] = data
            elif int(ls[0]) == 1:
                self.CCDdata[:ysize, xsize:] = data
            elif int(ls[0]) == 2:
                self.CCDdata[ysize:, :xsize] = data
            elif int(ls[0]) == 3:
                self.CCDdata[ysize:, xsize:] = data

        return self.CCDdata


    def writeFITSfile(self, data=None, unsigned16bit=True):
        """
        Write out FITS files using PyFITS.

        :param data: data to write to a FITS file, if None use self.data
        :type data: ndarray
        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool

        :return: None
        """
        if os.path.isfile(self.inputs['output']):
            self.log.info('Deleted existing file %s to generate a new output' % self.inputs['output'])
            os.remove(self.inputs['output'])
        else:
            self.log.info('Writing output to %s' % self.inputs['output'])

        if data is None:
            data = self.CCDdata

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList()

        #new image HDU
        hdu = pf.PrimaryHDU(data, self.hdu)

        #convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #add keywords
        for key, value in self.inputs.iteritems():
            try:
                hdu.header.add_history('{0:>s} = {1:>s}'.format(key, value))
            except:
                hdu.header.add_history('{0:>s} = {1:d}'.format(key, value))

        #update and verify the header
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.inputs['output'])
        self.log.info('Wrote %s' % self.inputs['output'])


    def runAll(self):
        """
        Wrapper to perform all class methods.
        """
        self.readData()
        self.tileCCD()
        self.writeFITSfile()



def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-f', '--files', dest='files',
                      help="Input files to compile e.g. 'Q*_00_00_VISscience.fits'", metavar='string')
    parser.add_option('-o', '--output', dest='output',
                      help="Name of the output file, default=VISCCD.fits", metavar='string')
    parser.add_option('-e', '--extension', type='int', dest='ext',
                     help='FITS extension from which to look for data, default=0', metavar='int')
    parser.add_option('-d', '--debug', dest='debug', action='store_true',
                      help='Debugging mode on')
    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.files is None:
        processArgs(True)
        sys.exit(8)

    #FITS extension
    if opts.ext is None:
        ext = 0
    else:
        ext = opts.ext

    #name of the output file
    if opts.output is None:
        output = 'VISCCD.fits'
    else:
        output = opts.output

    #logger
    log = lg.setUpLogger('tileCCDs.log')

    #look for files
    files = g.glob(opts.files)
    files.sort()
    if len(files) / 4. > 1.0 or len(files) == 0:
        print 'Detected %i input files, but the current version does not support anything but tiling four files...' % len(files)
        sys.exit(9)

    #write to the log what files were used
    log.info('Input files:')
    for file in files:
        log.info(file)

    #intputs
    inputs = dict(files=files, ext=ext, output=output)

    #class call
    tile = tileCCD(inputs, log)
    tile.runAll()

    log.info('CCD tiled, script will exit...')

