"""
Object finding
==============

Simple source finder that can be used to find objects from astronomical images.

:reqiures: NumPy
:requires: SciPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import matplotlib
matplotlib.use('PDF')
import datetime, sys
from itertools import groupby, izip, count
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


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
        self.image = image
        self.log = log
        #set default parameter values and then update using kwargs
        self.settings = dict(above_background=10.0,
                             clean_size_min=9,
                             clean_size_max=110,
                             sigma=1.5,
                             disk_struct=3,
                             output='objects.txt')
        self.settings.update(kwargs)


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
        """
        #smooth the image
        img = ndimage.gaussian_filter(self.image, sigma=self.settings['sigma'])
        med = np.median(img)
        self.log.info('Median of the gaussian filtered image = %f' % med)

        #find pixels above the median
        msk = self.image > med
        #get background image and calculate statistics
        backgrd = self.image[~msk]
        #only take values greater than zero
        backgrd = backgrd[backgrd > 0.0]

        if len(backgrd) < 1:
            #no real background in the image, a special case, a bit of a hack for now
            mean = 0.0
            rms = 1.0
            #find objects above the background
            self.mask = self.image > rms * self.settings['above_background'] + mean
            self.settings['clean_size_min'] = 1.0
            self.settings['clean_size_max'] = min(self.image.shape) / 1.5
        else:
            std = np.std(backgrd).item() #items required if image was memmap'ed by pyfits
            mean = np.mean(backgrd).item() #items required if image was memmap'ed by pyfits
            rms = np.sqrt(std ** 2 + mean ** 2)

            #find objects above the background
            filtered = ndimage.median_filter(self.image, self.settings['sigma'])
            self.mask = filtered > rms * self.settings['above_background'] + mean

        self.log.info('Background: average={0:.4f} and rms={1:.4f}'.format(mean, rms))

        #get labels
        self.label_im, self.nb_labels = ndimage.label(self.mask)

        self.log.info('Finished the initial run and found {0:d} objects...'.format(self.nb_labels))

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

        :return: xposition, yposition, center-of-masses
        :rtype: list
        """
        if not hasattr(self, 'label_clean'):
            self.cleanSample()

        self.cms = ndimage.center_of_mass(self.image,
                                          labels=self.label_clean,
                                          index=np.unique(self.label_clean))
        self.xcms = [c[1] for c in self.cms]
        self.ycms = [c[0] for c in self.cms]

        self.log.info('After cleaning found {0:d} objects'.format(len(self.xcms)))

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
        for x, y in zip(self.xcms, self.ycms):
            fh.write('%10.3f %10.3f\n' % (x + 1, y + 1))
            rg.write('circle({0:.3f},{1:.3f},5)\n'.format(x + 1, y + 1))
        fh.close()
        rg.close()


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

        results = dict(xcms=self.xcms, ycms=self.ycms, cms=self.cms,
                       sizes=self.sizes, fluxes=self.fluxes)

        return results

