"""
Source finding
==============

Simple source finder that can be used to find objects from astronomical images.

:reqiures: NumPy
:requires: SciPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.6
"""
import matplotlib
matplotlib.use('PDF')
import datetime, sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import interpolation
from analysis import shape


class sourceFinder():
    """
    This class provides methods for source finding.

    :param image: 2D image array
    :type image: numpy.ndarray
    :param log: logger
    :type log: instance

    :param kwargs: additional keyword arguments
    :type kwargs: dictionary
    """

    def __init__(self, image, log, **kwargs):
        """
        Init.

        :param image: 2D image array
        :type image: numpy.ndarray
        :param log: logger
        :type log: instance

        :param kwargs: additional keyword arguments
        :type kwargs: dictionary
        """
        self.image = image.copy()
        self.origimage = image.copy()
        self.log = log
        #set default parameter values and then update using kwargs
        self.settings = dict(above_background=4.0,
                             clean_size_min=2,
                             clean_size_max=20,
                             sigma=1.5,
                             disk_struct=3,
                             aperture=6.5,
                             oversample=10,
                             gain=3.1,
                             exptime=565.,
                             aperture_correction=0.925969,
                             magzero=1.5182880871e10,
                             output='objects.txt')
        self.settings.update(kwargs)

        #write the used parameters to the log
        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))


    def _diskStructure(self, n):
        """
        """
        struct = np.zeros((2 * n + 1, 2 * n + 1))
        x, y = np.indices((2 * n + 1, 2 * n + 1))
        mask = (x - n) ** 2 + (y - n) ** 2 <= n ** 2
        struct[mask] = 1
        return struct.astype(np.bool)


    def find(self):
        """
        Find all pixels above the median pixel after smoothing with a Gaussian filter.

        .. note:: maybe one should use mode instead of median?

        .. warning:: The background estimation is very crude and should not be trusted. One should probably write
                     an iterative scheme were the background is recalculated after identifying the objects.
        """
        #smooth the image
        img = ndimage.gaussian_filter(self.image, sigma=self.settings['sigma'])
        med = np.median(img)
        self.log.info('Median of the gaussian filtered image = %f' % med)

        #find pixels above 1.5*median, these are most likely from sources so assume that the rest is background
        msk = self.image > 1.5*med
        #get background image
        backgrd = self.origimage[~msk].copy()
        #only take values greater than zero
        backgrd = backgrd[backgrd > 0.0]

        if len(backgrd) < 1:
            #could not find background...
            std = 1.0
            #find objects above the background
            self.mask = self.image > std * self.settings['above_background']
            self.settings['clean_size_min'] = 1.0
            self.settings['clean_size_max'] = min(self.image.shape) / 1.5
            self.log.warning('Trouble finding background, will modify cleaning sizes...')
        else:
            std = np.std(backgrd).item() #items required if image was memmap'ed by pyfits
            med = np.median(backgrd).item() #items required if image was memmap'ed by pyfits
            #mean = np.mean(backgrd).item() #items required if image was memmap'ed by pyfits

            #find objects above the background
            filtered = ndimage.median_filter(self.image, self.settings['sigma'])
            #self.mask = filtered - mean > std * self.settings['above_background']
            self.mask = filtered - med > std * self.settings['above_background']

        #TODO: these are very very crude estimates about the background, improved ones should be developed!
        self.background = np.mean(backgrd  * self.settings['gain']).item() /  self.settings['gain']
        self.background_std = np.std(self.origimage[~self.mask])
        self.log.info('Background: mean={0:.4f} and std={1:.4f}'.format(self.background, self.background_std))
        print 'Background: mean={0:.4f} and std={1:.4f}'.format(self.background, self.background_std)

        #get labels
        self.label_im, self.nb_labels = ndimage.label(self.mask)

        self.log.info('Finished the initial run and found {0:d} objects...'.format(self.nb_labels))
        print 'Finished the initial run and found {0:d} objects...'.format(self.nb_labels)

        if self.nb_labels < 1:
            self.log.error('Cannot find any objects, will abort')
            sys.exit(-9)

        return self.mask, self.label_im, self.nb_labels


    def getContours(self):
        """
        Derive contours using the diskStructure function.
        """
        if not hasattr(self, 'mask'):
            self.find()

        self.opened = ndimage.binary_opening(self.mask,
                                             structure=self._diskStructure(self.settings['disk_struct']))
        return self.opened


    def getSizes(self):
        """
        Derives sizes for each object.
        """
        if not hasattr(self, 'label_im'):
            self.find()

        self.sizes = np.asarray(ndimage.sum(self.mask, self.label_im, range(self.nb_labels + 1)))
        return self.sizes


    def getFluxes(self):
        """
        Derive fluxes or counts.
        """
        if not hasattr(self, 'label_im'):
            self.find()

        self.fluxes = np.asarray(ndimage.sum(self.image, self.label_im, range(1, self.nb_labels + 1)))
        return self.fluxes


    def doAperturePhotometry(self, pixel_based=True):
        """
        Perform aperture photometry and calculate the shape of the object based on quadrupole moments.
        This method also calculates refined centroid for each object.

        .. Warning:: Results are rather sensitive to the background subtraction, while the errors depend
                     strongly on the noise estimate from the background. Thus, great care should be exercised
                     when applying this method.

        :param pixel_based: whether to do a global or pixel based background subtraction
        :type pixel_based: boolean

        :return: photometry, error_in_photometry, ellipticity, refined_x_pos, refined_y_pos
        :return: ndarray, ndarray, ndarray, ndarray, ndarray
        """
        if not hasattr(self, 'xcms'):
            self.getCenterOfMass()

        #box around the source, make it larger than the aperture to allow recentroiding
        size = np.ceil(self.settings['aperture']*1.5)

        #area that the aperture covers
        area = np.pi * self.settings['aperture']**2

        #background
        if pixel_based:
            bcg = self.background
        else:
            bcg = self.background * area

        photom = []
        ell = []
        refx = []
        refy = []
        error = []
        for x, y in zip(self.xcms, self.ycms):
            xint = int(x)
            yint = int(y)

            if x-size < 0 or y-size < 0 or x+size > self.origimage.shape[1] or y+size > self.origimage.shape[0]:
                #too close to the edge for aperture photometry
                photom.append(-999)
                ell.append(-999)
                refx.append(-999)
                refy.append(-999)
                error.append(-999)
            else:
                #cut out a small region and subtract the sky
                if pixel_based:
                    small = self.origimage[yint-size:yint+size+1, xint-size:xint+size+1].copy().astype(np.float64) - bcg
                else:
                    small = self.origimage[yint-size:yint+size+1, xint-size:xint+size+1].copy().astype(np.float64)

                if self.settings['oversample'] < 1.5:
                    oversampled = small
                else:
                    sum = np.sum(small)
                    oversampled = interpolation.zoom(small, self.settings['oversample'], order=0)
                    oversampled = oversampled / np.sum(oversampled) * sum

                #indeces of the oversampled image
                yind, xind = np.indices(oversampled.shape)

                #assume that centre is the same as the peak pixel (zero indexed)
                #ycen1, xcen1 = ndimage.measurements.maximum_position(oversampled)

                #calculate centre and shape
                settings = dict(sampling=1./self.settings['oversample'])
                sh = shape.shapeMeasurement(oversampled.copy(), self.log, **settings)
                results = sh.measureRefinedEllipticity()
                xcen = results['centreX'] - 1.
                ycen = results['centreY'] - 1.
                ell.append(results['ellipticity'])

                #refined x and y positions
                refx.append(xint - size + (xcen / self.settings['oversample']))
                refy.append(yint - size + (ycen / self.settings['oversample']))

                #change the peak to be 0, 0 and calculate radius
                xind -= xcen
                yind -= ycen
                rad = np.sqrt(xind**2 + yind**2)

                #calculate flux in the apertures
                mask = rad <= (self.settings['oversample'] * self.settings['aperture'])
                counts = oversampled[np.where(mask)].sum()

                #global background subtraction
                if ~pixel_based:
                    counts -= bcg

                #calculate the error in magnitudes
                err = 1.0857 * np.sqrt(area * self.background_std**2  + (counts / self.settings['gain'])) / counts

                #convert to electrons
                counts *= self.settings['gain']

                photom.append(counts)
                error.append(err)


        self.photometry = np.asarray(photom)
        self.ellipticity = np.asarray(ell)
        self.refx = np.asarray(refx)
        self.refy = np.asarray(refy)
        self.error = np.asarray(error)
        return self.photometry, self.error, self.ellipticity, self.refx, self.refy


    def cleanSample(self):
        """
        Cleans up small connected components and large structures.
        """
        if not hasattr(self, 'sizes'):
            self.getSizes()

        mask_size = (self.sizes < self.settings['clean_size_min']) | (self.sizes > self.settings['clean_size_max'])
        remove_pixel = mask_size[self.label_im]
        self.label_im[remove_pixel] = 0
        labels = np.unique(self.label_im)
        self.label_clean = np.searchsorted(labels, self.label_im)


    def getCenterOfMass(self):
        """
        Finds the center-of-mass for all objects using numpy.ndimage.center_of_mass method.

        .. Note:: these positions are zero indexed!

        :return: xposition, yposition, center-of-masses
        :rtype: list
        """
        if not hasattr(self, 'label_clean'):
            self.cleanSample()

        self.cms = ndimage.center_of_mass(self.image,
                                          labels=self.label_clean,
                                          index=np.unique(self.label_clean))
        self.xcms = [c[1] for c in self.cms][1:]
        self.ycms = [c[0] for c in self.cms][1:]

        self.log.info('After cleaning found {0:d} objects'.format(len(self.xcms)))
        print 'After cleaning found {0:d} objects'.format(len(self.xcms))

        return self.xcms, self.ycms, self.cms


    def plot(self):
        """
        Generates a diagnostic plot.

        :return: None
        """
        if not hasattr(self, 'opened'):
            self.getContours()

        if not hasattr(self, 'xcms'):
            self.getCenterOfMass()

        plt.figure(1, figsize=(30,11))
        s1 = plt.subplot(131)
        s1.imshow(np.log10(np.sqrt(self.image)), interpolation=None, origin='lower')
        s1.plot(self.xcms, self.ycms, 'x', ms=4)
        s1.contour(self.opened, [0.2], c='b', linewidths=1.2, linestyles='dotted')
        s1.axis('off')
        s1.set_title('log10(sqrt(IMAGE))')

        s2 = plt.subplot(132)
        s2.imshow(self.mask, cmap=plt.cm.gray, interpolation=None, origin='lower')
        s2.axis('off')
        s2.set_title('Object Mask')

        s3 = plt.subplot(133)
        s3.imshow(self.label_clean, cmap=plt.cm.spectral, interpolation=None, origin='lower')
        s3.axis('off')
        s3.set_title('Cleaned Object Mask')

        plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
        plt.savefig('SourceFinder.pdf')
        plt.close()


    def generateOutput(self):
        """
        Outputs the found positions to an ascii and a DS9 reg file.

        :return: None
        """
        if not hasattr(self, 'xcms'):
            self.getCenterOfMass()

        fh = open(self.settings['output'], 'w')
        rg = open(self.settings['output'].split('.')[0]+'.reg', 'w')
        fh.write('#1 X coordinate in pixels [starts from 1]\n')
        fh.write('#2 Y coordinate in pixels [starts from 1]\n')
        rg.write('#File written on {0:>s}\n'.format(datetime.datetime.isoformat(datetime.datetime.now())))
        for x, y, size in zip(self.xcms, self.ycms, self.sizes):
            fh.write('%10.3f %10.3f\n' % (x + 1, y + 1))
            rg.write('circle({0:.3f},{1:.3f},{2:.3f})\n'.format(x + 1, y + 1, size))
        fh.close()
        rg.close()


    def writePhotometry(self):
        """
        Outputs the photometric results to an ascii file.

        :return: None
        """
        fh1 = open(self.settings['output'].split('.')[0]+'.phot', 'w')
        fh1.write('# 1 X coordinate in pixels [starts from 1]\n')
        fh1.write('# 2 XREF refined coordinate in pixels [starts from 1]\n')
        fh1.write('# 3 Y coordinate in pixels [starts from 1]\n')
        fh1.write('# 4 YREF refined coordinate in pixels [starts from 1]\n')
        fh1.write('# 5 COUNTS [electrons]\n')
        fh1.write('# 6 ADUS [ADUs]\n')
        fh1.write('# 7 MAGNITUDE [aperture corrected]\n')
        fh1.write('# 8 MAGERR []\n')
        fh1.write('# 9 SNR\n')
        fh1.write('# 10 ELLIPTICITY\n')

        fh = open(self.settings['output'].split('.')[0]+'.photAll', 'w')
        fh.write('# 1 X coordinate in pixels [starts from 1]\n')
        fh.write('# 2 XREF refined coordinate in pixels [starts from 1]\n')
        fh.write('# 3 Y coordinate in pixels [starts from 1]\n')
        fh.write('# 4 YREF refined coordinate in pixels [starts from 1]\n')
        fh.write('# 5 COUNTS [electrons]\n')
        fh.write('# 6 ADUS [ADUs]\n')
        fh.write('# 7 MAGNITUDE [aperture corrected]\n')
        fh.write('# 8 MAGERR []\n')
        fh.write('# 9 SNR\n')
        fh.write('# 10 ELLIPTICITY\n')
        for x, xref, y, yref, phot, e, err in zip(self.xcms, self.refx, self.ycms, self.refy,
                                                  self.photometry, self.ellipticity, self.error):
            mag = -2.5*np.log10(phot / self.settings['aperture_correction'] /
                                self.settings['exptime'] / self.settings['magzero'])
            txt = '%10.3f %10.3f %10.3f %10.3f %15.2f %15.2f %12.7f %12.8f %15.2f %10.6f\n' % \
                  (x + 1, xref + 1, y + 1, yref + 1, phot, phot/self.settings['gain'], mag, err, 1./err, e)
            fh.write(txt)
            if ~(yref < 0.0 or xref < 0.0 or phot < 0.0):
                fh1.write(txt)
        fh.close()
        fh1.close()


    def runAll(self):
        """
        Performs all steps of source finding at one go.

        :return: source finding results such as positions, sizes, fluxes, etc.
        :rtype: dictionary
        """
        self.find()
        self.getContours()
        self.getSizes()
        self.getFluxes()
        self.cleanSample()
        self.getCenterOfMass()
        self.plot()
        self.generateOutput()
        self.doAperturePhotometry()
        self.writePhotometry()

        results = dict(xcms=self.xcms, ycms=self.ycms, cms=self.cms,
                       sizes=self.sizes, fluxes=self.fluxes,
                       photometry=self.photometry)

        return results


if __name__ == '__main__':
    from support import logger as lg
    import pyfits as pf

    def processArgs(printHelp=False):
        """
        Processes command line arguments.
        """
        parser = OptionParser()

        parser.add_option('-f', '--file', dest='file',
                          help="Input file e.g. 'Q0_00_00stars245ver2.fits'", metavar='string')
        parser.add_option('-s', '--sigma', dest='sigma',
                          help="Size of the Gaussian smoothing kernel", metavar='float')
        parser.add_option('-l', '--largest', dest='large',
                          help="The maximum number of pixels a largest object can cover", metavar='float')
        parser.add_option('-m', '--minimum', dest='minimum',
                          help="The minimum number of pixels a smallest object can cover", metavar='float')
        parser.add_option('-a', '--above_background', dest='above',
                          help="The significance of the detection (x sigma above the background)", metavar='float')
        if printHelp:
            parser.print_help()
        else:
            return parser.parse_args()

    log = lg.setUpLogger('sourceFinding.log')

    opts, args = processArgs()

    if opts.file is None:
        processArgs(True)
        sys.exit(8)

    settings = dict()
    if opts.sigma is not None:
        settings.update({'sigma' : float(opts.sigma)})
    if opts.large is not None:
        settings.update({'clean_size_max' : float(opts.large)})
    if opts.minimum is not None:
        settings.update({'clean_size_min' : float(opts.minimum)})
    if opts.above is not None:
        settings.update({'above_background' : float(opts.above)})

    sf = sourceFinder(pf.getdata(opts.file), log, **settings)
    tmp = sf.runAll()
