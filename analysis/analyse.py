"""
Simple method to find objects and measure their ellipticity.

:reqiures: PyFITS

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import pyfits as pf
import sys
from optparse import OptionParser
from sourceFinder import sourceFinder
from support import logger as lg


class analyseVISdata():
    """
    """
    def __init__(self, filename, log, **kwargs):
        """
        :param filename: name of the FITS file to be analysed.
        :type filename: string
        :param log: logger
        :type log: instance
        :param kwargs: additional keyword arguments
        :type kwargs: dict

        Settings dictionary contains all parameter values needed
        for source finding and analysis.
        """
        self.log = log
        self.settings = dict(filename=filename,
                             extension=0,
                             above_background=2.0,
                             clean_size_min=2,
                             clean_size_max=250,
                             sigma=2.6,
                             disk_struct=3,
                             output='foundsources.txt')
        self.settings.update(kwargs)
        self._loadData()


    def _loadData(self):
        """
        Load data from a given file. Assumes that the data are in a FITS file.
        Filename and FITS extension are read from the settings dictionary.

        Read data are stored in self.data
        """
        fh = pf.open(self.settings['filename'])
        self.data = fh[self.settings['extension']].data


    def findSources(self):
        """
        Finds sources from data that has been read in when the class was initiated.
        Saves results such as x and y coordinates of the objects to self.sources.
        """
        source = sourceFinder(self.data, self.log, **self.settings)
        self.sources = source.runAll()


    def measureEllipticity(self):
        """
        """
        pass

    def doAll(self):
        """
        """
        self.findSources()
        self.measureEllipticity()



def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-f', '--file', dest='input',
                      help="Input file to process", metavar='string')
    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.input is None:
        processArgs(True)
        sys.exit(8)

    log = lg.setUpLogger('analyse.log')
    log.info('\n\nStarting to analyse %s' % opts.input)

    analyse = analyseVISdata(opts.input, log)
    analyse.doAll()