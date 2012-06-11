"""
Simple method to find objects and measure their ellipticity.

One can either choose to use a Python based source finding algorithm or
give a SExtractor catalog as an input. If an input catalog is provided
then the program assumes that X_IMAGE and Y_IMAGE columns are present
in the input file.

:requires: PyFITS

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import pyfits as pf
import sys
from optparse import OptionParser
from sourceFinder import sourceFinder
from support import sextutils
from support import logger as lg
from analysis import shape


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
                             output='foundsources.txt',
                             ellipticityOutput='ellipticities.txt')
        self.settings.update(kwargs)
        self._loadData()


    def _loadData(self):
        """
        Load data from a given file. Assumes that the data are in a FITS file.
        Filename and FITS extension are read from the settings dictionary.

        Read data are stored in self.data
        """
        self.log.info('Reading data from %s extension=%i' % (self.settings['filename'], self.settings['extension']))
        fh = pf.open(self.settings['filename'])
        self.data = fh[self.settings['extension']].data
        ysize, xsize = self.data.shape
        self.settings['sizeX'] = xsize
        self.settings['sizeY'] = ysize


    def findSources(self):
        """
        Finds sources from data that has been read in when the class was initiated.
        Saves results such as x and y coordinates of the objects to self.sources.
        x and y coordinates are also available directly in self.x and self.y.
        """
        self.log.info('Finding sources from data...')
        source = sourceFinder(self.data, self.log, **self.settings)
        self.sources = source.runAll()

        self.x = self.sources['xcms']
        self.y = self.sources['ycms']


    def readSources(self):
        """
        Reads in a list of sources from an external file. This method assumes
        that the input source file is in SExtractor format. Input catalog is
        saves to self.sources. x and y coordinates are also available directly in self.x and self.y.
        """
        self.log.info('Reading source information from %s' % self.settings['sourceFile'])
        self.sources = sextutils.se_catalog(self.settings['sourceFile'])

        self.x = self.sources.x_image
        self.y = self.sources.y_image

        #write out a DS reg file
        rg = open('sources.reg', 'w')
        for x, y  in zip(self.sources.x_image, self.sources.y_image):
            rg.write('circle({0:.3f},{1:.3f},5)\n'.format(x, y))
        rg.close()


    def measureEllipticity(self):
        """
        Measures ellipticity for all objects with coordinates (self.x, self.y).

        Ellipticity is measured using Guassian weighted quadrupole moments.
        See shape.py and especially the ShapeMeasurement class for more details.
        """
        ells = []
        xs = []
        ys = []
        R2s = []
        for x, y in zip(self.x, self.y):
            #cut out a square region around x and y coordinates
            xmin = int(max(x - 30., 0.))
            ymin = int(max(y - 30., 0.))
            xmax = int(min(x + 31., self.settings['sizeX']))
            ymax = int(min(y + 31., self.settings['sizeY']))

            if xmax - xmin < 20 or ymax - ymin < 20:
                self.log.warning('Very little pixels around the object..')

            self.log.info('Measuring ellipticity of an object located at (x, y) = (%f, %f)' % (x, y))

            img = self.data[ymin:ymax, xmin:xmax].copy()
            sh = shape.shapeMeasurement(img, self.log)
            results = sh.measureRefinedEllipticity()

            #get shifts for x and y centroids for the cutout image
            cutsizey, cutsizex = img.shape
            xcent = int(x - cutsizex/2.)
            ycent = int(y - cutsizey/2.)

            self.log.info('Centroiding (x, y) = (%f, %f), e=%f, R2=%f' % (results['centreX']+xcent,
                                                                          results['centreY']+ycent,
                                                                          results['ellipticity'],
                                                                          results['R2']))

            #save the results
            ells.append(results['ellipticity'])
            xs.append(results['centreX']+xcent)
            ys.append(results['centreY']+ycent)
            R2s.append(results['R2'])

        out = dict(Xcentres=xs, Ycentres=ys, ellipticities=ells, R2s=R2s)
        self.results = out

        return self.results


    def writeResults(self):
        """
        Outputs results to an ascii file defined in self.settings.
        """
        fh = open(self.settings['ellipticityOutput'], 'w')
        fh.write('#X, Y, ellipticity R2\n')
        for x, y, e, R2 in zip(self.results['Xcentres'],
                               self.results['Ycentres'],
                               self.results['ellipticities'],
                               self.results['R2s']):
            fh.write('%f %f %f %f\n' % (x, y, e, R2))
        fh.close()


    def doAll(self):
        """
        """
        if self.settings['sourceFile']:
            self.readSources()
        else:
            self.findSources()
        results = self.measureEllipticity()
        self.writeResults()
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))
        return results


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-f', '--file', dest='input',
                      help="Input file to process", metavar='string')
    parser.add_option('-s', '--sourcefile', dest='sourcefile',
                      help='Name of input source file [optional]', metavar='string')

    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.input is None:
        processArgs(True)
        sys.exit(8)

    settings = {}
    if opts.sourcefile is None:
        settings['sourceFile'] = None
    else:
        settings['sourceFile'] = opts.sourcefile

    log = lg.setUpLogger('analyse.log')
    log.info('\n\nStarting to analyse %s' % opts.input)

    analyse = analyseVISdata(opts.input, log, **settings)
    results = analyse.doAll()

    log.info('All done...')