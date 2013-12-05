"""
This script can be used to derive statistics of background pixels.
"""
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf
from astropy.stats import sigma_clip
from support import files as fileIO
from statsmodels.nonparametric.kde import KDEUnivariate
from astropy.modeling import models, fitting
from skimage.morphology import reconstruction, disk, binary_opening
from scipy.ndimage import gaussian_filter
from skimage.filter.rank import entropy


def plotStatistcs(data):
    """
    Plot statistics of the background pixels. Assumes that the input contains only
    the background pixel values as a 1D array.

    :return: None
    """
    #derive KDE
    kd = KDEUnivariate(data.copy().astype(np.float64))
    kd.fit(adjust=3)
    #kd.fit(kernel='biw', fft=False)

    #plot data
    fig, axarr = plt.subplots(1, 2, sharey=True)
    ax1 = axarr[0]
    ax2 = axarr[1]
    fig.subplots_adjust(wspace=0)

    ax1.set_title('Background Pixels')
    ax2.set_title('Models')

    d = ax1.hist(data, bins=np.linspace(data.min(), data.max()), normed=True, alpha=0.7)
    ax1.plot(kd.support, kd.density, 'r-', label='Gaussian KDE')

    # Gaussian fit
    x = [0.5 * (d[1][i] + d[1][i+1]) for i in xrange(len(d[1])-1)]
    y = d[0]
    g_init = models.Gaussian1D(amplitude=1., mean=np.mean(data), stddev=np.std(data))
    f2 = fitting.NonLinearLSQFitter()
    g = f2(g_init, x, y)

    ax2.plot(x, g(x), 'b-', label='Gaussian Fit')
    ax2.plot(kd.support, kd.density, 'r-', label='Gaussian KDE', alpha=0.7)

    ax1.set_ylabel('PDF')

    ax1.set_xticks(ax1.get_xticks()[:-1])
    ax2.set_xticks(ax2.get_xticks()[1:])

    plt.legend(shadow=True, fancybox=True)
    plt.savefig('backgroundStatistics.pdf')
    plt.close()


def maskObjects(data, sigma=4., iterations=None):
    """
    Mask objects using sigma clipping around the median.

    Will also plot the data and the mask.

    :return: masked numpy array
    """
    #sigma clip
    masked = sigma_clip(data.copy(), sig=sigma, iters=iterations)
    print masked.min(), masked.max(), masked.mean(), masked.std()

    mask = ~masked.mask
    d = masked.data

    #plot the image
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Data')
    ax2.set_title('Mask')

    im1 = ax1.imshow(np.log10(data), origin='lower', vmin=2., vmax=3.5, interpolation='none')
    im2 = ax2.imshow(mask, origin='lower', interpolation='none')
    c1 = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    c2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', ticks=[0, 1])
    c2.ax.set_xticklabels(['False', 'True'])
    c1.set_label('$\log_{10}$(Counts [ADU])')
    c2.set_label('Mask')
    plt.savefig('masking.png')
    plt.close()

    out = d*mask
    fileIO.writeFITS(out, 'masking.fits', int=False)

    o = out.ravel()
    o = o[o > 0.]

    return o


def dilation(data):
    """
    Use dilation to define the background. Not working too well...
    """
    image = gaussian_filter(data, 1)

    h = 1
    seed = np.copy(image) - h
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    dilated = data - dilated

    fileIO.writeFITS(dilated, 'dilation.fits', int=False)

    #plot the image
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Data')
    ax2.set_title('Background')

    im1 = ax1.imshow(np.log10(data), origin='lower', vmin=2., vmax=3.5, interpolation='none')
    im2 = ax2.imshow(dilated, origin='lower', interpolation='none')
    c1 = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    c2 = plt.colorbar(im2, ax=ax2, orientation='horizontal')
    c1.set_label('$\log_{10}$(Counts [ADU])')
    c2.set_label('Dilation')
    plt.savefig('dilation.png')
    plt.close()

    return dilated.ravel()


def sigmaClippedOpening(data):
    """
    Perform Gaussian filtering, sigma clipping and binary opening to define
    a mask for the background pixels.
    """
    image = gaussian_filter(data, 0.5)

    #sigma clip
    masked = sigma_clip(image, sig=5., iters=None)

    mask = ~masked.mask
    d = masked.data

    #update mask with opening
    mask = binary_opening(mask, disk(8))

    #plot the image
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Data, Single Quadrant')
    ax2.set_title('Background Mask')

    im1 = ax1.imshow(np.log10(data), origin='lower', vmin=2., vmax=3.5, interpolation='none')
    im2 = ax2.imshow(mask, origin='lower', interpolation='none')
    c1 = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    c2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', ticks=[0, 1])
    c2.ax.set_xticklabels(['False', 'True'])
    c1.set_label('$\log_{10}$(Counts [ADU])')
    c2.set_label('Mask')
    plt.savefig('opening.png')
    plt.close()

    out = data*mask
    fileIO.writeFITS(out, 'opening.fits', int=False)

    o = out.ravel()
    o = o[o > 0.]

    print o.min(), o.max(), o.mean(), o.std(), len(o)

    return o


def loadData(filename):
    """
    Load data, exclude pre- and over scan regions, and subtract the bias level found from the header.
    """
    fh = pf.open(filename)
    bias = float(fh[1].header['BIAS'])
    gain = float(fh[1].header['GAIN'])
    scatter = float(fh[1].header['SCATTER'])
    dark = float(fh[1].header['DARK'])
    bcgr = float(fh[1].header['COSMIC_'])
    exptime = float(fh[1].header['EXPTIME'])
    fh.close()

    #load data without the pre- and overscan regions
    data = fileIO.readFITSDataExcludeScanRegions([filename,])[0]
    data.astype(np.float64)

    #subtract bias
    data -= bias

    #multiply by gain
    data *= gain

    #background
    print 'Total Background:', exptime * (scatter + dark + bcgr)

    return data


def plotEntropy(filename):
    """
    Plots the entropy in the data. No really suitable as the input data
    not in a useful format.
    """
    data = pf.getdata(filename).astype(np.uint32)

    #plot the image
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Data, Single Qaudrant')
    ax2.set_title('Background Mask')

    im1 = ax1.imshow(np.log10(data), origin='lower', vmin=2., vmax=3.5, interpolation='none')
    im2 = ax2.imshow(entropy(data, disk(5)), origin='lower', interpolation='none')
    c1 = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    c2 = plt.colorbar(im2, ax=ax2, orientation='horizontal')
    c1.set_label('$\log_{10}$(Counts [ADU])')
    c2.set_label('Entropy')
    plt.savefig('entropy.png')
    plt.close()


def doAll(filename):
    d = loadData(filename)

    #plotEntropy(filename)

    #masked = maskObjects(d)
    #masked = dilation(d)
    masked = sigmaClippedOpening(d)

    plotStatistcs(masked)


if __name__ == '__main__':
    #filename = 'Q0_00_00testscience.fits'
    #filename = 'Q0_00_00test2.fits'
    #filename = 'Q0_00_00test3.fits'
    #filename = 'Q0_00_00test4.fits'
    filename = 'Q0_00_00test5.fits'

    doAll(filename)
