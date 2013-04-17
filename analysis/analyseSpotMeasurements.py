"""
CCD Spot Measurements
=====================

This scripts can be used to study CCD spot measurements.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.1

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pyfits as pf
import numpy as np
import scipy as sp
import glob as g
from scipy import fftpack
from scipy import ndimage
from scipy import signal
from analysis import shape
from support import logger as lg
from support import files as fileIO
import SamPy.fitting.fits as fit


def weinerFilter(data, kernel, reqularization=0.01, normalize=True):
    """
    Performs Wiener deconvolution on data using a given kernel. The input can be either 1D or 2D array.

    For further information:
    http://en.wikipedia.org/wiki/Wiener_deconvolution

    :param data: data to be deconvolved (either 1D or 2D array)
    :param kernel: kernel that is used for the deconvolution
    :param reqularization: reqularization parameter
    :param normalize: whether or not the peak value should be normalized to unity

    :return: deconvolved data
    """
    if len(data.shape) == 1:
        #1D
        H = np.fft.fft(kernel)
        decon = np.fft.ifftshift(np.fft.ifft(np.fft.fft(data) * np.conj(H) / (H * np.conj(H) + reqularization ** 2)))
    else:
        #2D
        H = np.fft.fft2(kernel)
        decon = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(data) * np.conj(H) / (H * np.conj(H) + reqularization ** 2)))

    decon = np.abs(decon)

    if normalize:
        decon /= float(np.max(decon))

    return decon


def gaussianFit(ydata, xdata=None, initials=None):
    """
    Fits a single Gaussian to a given data.
    Uses scipy.optimize.leastsq for fitting.

    :param ydata: to which a Gaussian will be fitted to.
    :param xdata: if not given uses np.arange
    :param initials: initial guess for Gaussian parameters in order [amplitude, mean, sigma, floor]

    :return: coefficients, best fit params, success
    :rtype: dictionary
    """
    # define a gaussian fitting function where
    # p[0] = amplitude
    # p[1] = mean
    # p[2] = sigma
    # p[3] = floor
    fitfunc = lambda p, x: p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] ** 2)) + p[3]
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    if initials is None:
        initials = sp.c_[np.max(ydata), np.argmax(ydata), 5, 0][0]

    if xdata is None:
        xdata = np.arange(len(ydata))

    # fit a gaussian to the correlation function
    p1, success = sp.optimize.leastsq(errfunc, initials[:], args=(xdata, ydata))

    # compute the best fit function from the best fit parameters
    corrfit = fitfunc(p1, xdata)

    out = {}
    out['fit'] = corrfit
    out['parameters'] = p1
    out['amplitude'] = p1[0]
    out['mean'] = p1[1]
    out['sigma'] = p1[2]
    out['floor'] = p1[3]
    out['fwhm'] = 2 * np.sqrt(2 * np.log(2)) * out['sigma']
    out['success'] = success

    return out


def generateBessel(radius=1.5, oversample=500, size=1000, cx=None, cy=None, debug=False):
    """
    Generates a 2D Bessel function by taking a Fourier transform of a disk with a given radius. The real image
    and the subsequent power spectrum is oversampled with a given factor. The peak value of the generated
    Bessel function is normalized to unity.


    :param radius: radius of the disc [default=1.5]
    :param oversample: oversampling factor [default=500]
    :param size: size of the output array
    :param cx: centre of the disc in x direction
    :param cy: centre of the disc in y direction
    :param debug: whether or not to generate FITS files

    :return:
    """
    pupil = np.zeros((size, size))

    #centroid of the disc
    if cx is None:
        cx = np.shape(pupil)[1] / 2
    if cy is None:
        cy = np.shape(pupil)[0] / 2

    y, x = np.indices(pupil.shape)
    xc = x - cx
    yc = y - cy
    rad = np.sqrt(xc**2 + yc**2)
    mask = rad < (radius*oversample)
    pupil[mask] = 1.

    if debug:
        fileIO.writeFITS(pupil, 'disc.fits', int=False)

    F1 = fftpack.fft2(pupil)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)
    # Calculate a 2D power spectrum
    psd2D = np.abs(F2)**2
    #normalize it
    psd2D /= np.max(psd2D)

    if debug:
        fileIO.writeFITS(psd2D, 'besselOversampled.fits', int=False)

    return psd2D


def readData(filename, crop=True):
    """

    :param filename:
    :param crop:
    :return:
    """
    if crop:
        return pf.getdata(filename)[42:58, 42:58]
    else:
        return pf.getdata(filename)


def stackData(data):
    avg = np.average(data.copy(), axis=0)

    fig = plt.figure(figsize=(14, 8))
    stopy, stopx = avg.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
    dx = np.ones(stopy*stopx)*0.95
    dy = dx.copy()
    dz = np.zeros(stopy*stopx)

    #add subplot
    ax = Axes3D(fig)
    plt.title('Averaged Spot')
    #ax.plot_surface(X, Y, avg, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.bar3d(X.flatten(), Y.flatten(), dz, dx, dy, avg.flatten(), color='b')
    #ax.set_zlim(-0.05, 0.05)
    ax.set_xlabel('X [pixel]')
    ax.set_ylabel('Y [pixel]')
    #ax.set_zlabel(r'$\log_{10}$(Counts [ADU])')
    ax.set_zlabel('Counts [ADU]')
    plt.savefig('AveragedSpot.pdf')
    plt.close()

    fig = plt.figure(figsize=(14, 8))
    stopy, stopx = avg.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

    #add subplot
    ax = Axes3D(fig)
    plt.title('Spot')
    ax.plot_surface(X, Y, data[0], rstride=1, cstride=1, edgecolors='w')
    #ax.bar3d(X.flatten(), Y.flatten(), dz, dx, dy, data[0].flatten(), color='b')
    #ax.set_zlim(-0.05, 0.05)
    ax.set_xlabel('X [pixel]')
    ax.set_ylabel('Y [pixel]')
    #ax.set_zlabel(r'$\log_{10}$(Counts [ADU])')
    ax.set_zlabel(r'Counts [ADU]')
    plt.savefig('SingleSpot.pdf')
    plt.close()


def analyseSpotsDeconvolution(files):
    """
    Analyse spot measurements using deconvolutions.

    Note: does not really work... perhaps an issue with sizes.

    :param files: a list of input files
    :type files: list

    :return: None
    """
    d = {}
    data = []
    for filename in files:
        tmp = readData(filename, crop=False)
        f = filename.replace('.fits', '')
        d[f] = tmp
        data.append(tmp)
    data = np.asarray(data)

    #sanity check plots
    #stackData(data)

    #deconvolve with top hat
    dec1 = {}
    y, x = data[0].shape
    top = np.zeros((y, x))
    top[y/2, x/2] = 1.
    fileIO.writeFITS(top, 'tophat.fits', int=False)
    for filename, im in zip(files, data):
        deconv = weinerFilter(im, top, normalize=False)
        f = filename.replace('.fits', 'deconv1.fits')
        fileIO.writeFITS(deconv, f, int=False)
        dec1[f] = deconv

    print "Tophat deconvolution done"

    #deconvolve with a Besssel
    dec2 = {}
    bes = generateBessel(radius=0.13)
    bes = ndimage.zoom(bes, 1./2.5, order=0)
    bes /= np.max(bes)
    fileIO.writeFITS(bes, 'bessel.fits', int=False)
    for key, value in dec1.iteritems():
        value = ndimage.zoom(value, 4., order=0)
        value -= np.median(value)
        deconv = weinerFilter(value, bes, reqularization=2.0, normalize=False)
        f = key.replace('deconv1.fits', 'deconv2.fits')
        fileIO.writeFITS(deconv, f, int=False)
        dec2[f] = deconv

    print 'Bessel deconvolution done'


def plotLineFits(data, fit, output, horizontal=True):
    p0 = fit['amplitude']
    p1 = fit['mean']
    p2 = fit['sigma']
    p3 = fit['floor']
    x = np.linspace(0, len(data), num=200)
    y = p0 * np.exp(-(x - p1) ** 2 / (2.0 * p2 ** 2)) + p3

    fig = plt.figure()
    if horizontal:
        plt.suptitle('Horizontal Fitting')
    else:
        plt.suptitle('Vertical Fitting')
    ax1 = fig.add_subplot(111)
    ax1.plot(data, 'bo', label='Data')
    ax1.plot(x, y, 'r-', label='Gaussian Fit')
    if horizontal:
        plt.savefig(output + 'H.pdf')
    else:
        plt.savefig(output + 'V.pdf')
    plt.close()


def plotPixelValues(data, ids):
    grid1 = np.zeros((7, 7))
    grid2 = np.zeros((7, 7))
    grid3 = np.zeros((7, 7))
    grid4 = np.zeros((7, 7))
    for f, im in data.iteritems():
        xpos = ids[f][0]
        ypos = ids[f][1]
        grid1[ypos, xpos] = im[7, 7]
        grid2[ypos, xpos] = im[7, 8]
        grid3[ypos, xpos] = im[8, 7]
        grid4[ypos, xpos] = im[8, 8]

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax3.set_title('pixel (0,0)')
    ax1.set_title('pixel (0,1)')
    ax4.set_title('pixel (1,0)')
    ax2.set_title('pixel (1,1)')

    ax3.imshow(grid1, origin='lower', interpolation='none', rasterized=True)
    ax1.imshow(grid2, origin='lower', interpolation='none', rasterized=True)
    ax4.imshow(grid3, origin='lower', interpolation='none', rasterized=True)
    ax2.imshow(grid4, origin='lower', interpolation='none', rasterized=True)
    plt.savefig('PixelValues.pdf')
    plt.close()


def plotGaussianResults(data, ids, output, vals=[0, 2]):
    """

    :param res: results
    :param ids: file identifiers
    :return: None
    """
    grid1 = np.zeros((7, 7))
    grid2 = np.zeros((7, 7))
    for f, im in data.iteritems():
        xpos = ids[f][0]
        ypos = ids[f][1]
        grid1[ypos, xpos] = data[f][vals[0]]
        grid2[ypos, xpos] = data[f][vals[1]]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title('Horizontal')
    ax2.set_title('Vertical')

    ax1.imshow(grid1, origin='lower', interpolation='none', rasterized=True)
    ax2.imshow(grid2, origin='lower', interpolation='none', rasterized=True)
    plt.savefig(output + 'GaussianValues.pdf')
    plt.close()

    #3D plots
    stopy, stopx = grid1.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

    #add subplots
    fig = plt.figure(figsize=(12, 8))
    ax1 = Axes3D(fig, fig.add_subplot(1, 2, 1, frame_on=False, visible=False).get_position())
    ax2 = Axes3D(fig, fig.add_subplot(1, 2, 2, frame_on=False, visible=False).get_position())

    ax1.set_title('Horizontal (channel-stop between pixels)')
    ax2.set_title('Vertical (potentials set by electrodes)')

    ax1.set_xlabel('Grid Position')
    ax1.set_ylabel('Grid Position')
    ax1.set_zlabel(r'$\sigma_{pix}$')
    ax2.set_xlabel('Grid Position')
    ax2.set_ylabel('Grid Position')
    ax2.set_zlabel(r'$\sigma_{pix}$')

    #grid1 /= np.max(grid1)
    #grid2 /= np.max(grid2)

    ax1.plot_surface(X, Y, grid1, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    ax2.plot_surface(X, Y, grid2, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    #ax1.set_zlim3d(0.38, 0.48)
    #ax2.set_zlim3d(0.38, 0.48)

    plt.savefig(output + 'GaussianValues3D.pdf')
    plt.close()


def fitf(height, center_x, center_y, width_x, width_y):
    """
    Fitting function: 2D gaussian
    """
    width_x = float(width_x)
    width_y = float(width_y)
    #return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2.)
    return lambda x, y: 1. * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2.)


def analyseSpotsFitting(files, gaussian=False, pixelvalues=False, bessel=True, maxfev=10000):
    """
    Analyse spot measurements using different fitting methods.

    :param files: names of the FITS files to analyse (should match the IDs)
    :param gaussian: whether or not to do a simple Gaussian fitting analysis
    :param pixelvalues: whether or not to plot pixel values on a grid
    :param bessel: whether or not to do a Bessel + Gaussian convolution analysis
    :param maxfev: maximum number of iterations in the least squares fitting

    :return: None
    """
    log = lg.setUpLogger('spots.log')
    log.info('Starting...')
    over = 24
    settings = dict(itereations=8)
    ids = fileIDs()

    d = {}
    for filename in files:
        tmp = readData(filename, crop=False)
        f = filename.replace('small.fits', '')
        d[f] = tmp

    if pixelvalues:
        #plot differrent pixel values
        plotPixelValues(d, ids)

    if gaussian:
        #fit simple Gaussians
        Gaussians = {}
        for f, im in d.iteritems():
            #horizontal direction
            sumH = np.sum(im, axis=0)
            Hfit = gaussianFit(sumH, initials=[np.max(sumH) - np.median(sumH), 8., 0.4, np.median(sumH)])
            plotLineFits(sumH, Hfit, f)

            #vertical direction
            sumV = np.sum(im, axis=1)
            Vfit = gaussianFit(sumV, initials=[np.max(sumV) - np.median(sumV), 8., 0.4, np.median(sumV)])
            plotLineFits(sumH, Hfit, f, horizontal=False)

            #2D gaussian
            tmp = im.copy() - np.median(im)
            twoD = fit.Gaussian2D(tmp, intials=[np.max(tmp), 7, 7, 0.4, 0.4])

            print f, Hfit['sigma'], twoD[4], Vfit['sigma'], twoD[3], int(np.max(im))
            Gaussians[f] = [Hfit['sigma'], twoD[4], Vfit['sigma'], twoD[3]]

        fileIO.cPickleDumpDictionary(Gaussians, 'SpotmeasurementsGaussian.pk')

        plotGaussianResults(Gaussians, ids, output='line')
        plotGaussianResults(Gaussians, ids, output='twoD', vals=[1, 3])

    if bessel:
        Gaussians = {}
        #Bessel + Gaussian
        hf = 8 * over
        for f, im in d.iteritems():
            #if '21_59_31s' not in f:
            #    continue

            #over sample the data, needed for convolution
            oversampled = ndimage.zoom(im.copy(), over, order=0)
            fileIO.writeFITS(oversampled, f+'block.fits', int=False)

            #find the centre in oversampled frame, needed for bessel and gives a starting point for fitting
            tmp = oversampled.copy() - np.median(oversampled)
            sh = shape.shapeMeasurement(tmp, log, **settings)
            results = sh.measureRefinedEllipticity()
            midx = results['centreX'] - 1.
            midy = results['centreY'] - 1.

            #generate 2D bessel and re-centre using the above centroid, normalize to the maximum image value and
            #save to a FITS file.
            bes = generateBessel(radius=0.45, oversample=over, size=16*over)
            shiftx = -midx + hf
            shifty = -midy + hf
            bes = ndimage.interpolation.shift(bes, [-shifty, -shiftx], order=0)
            bes /= np.max(bes)
            fileIO.writeFITS(bes, f+'bessel.fits', int=False)

            #check the residual with only the bessel and save to a FITS file
            t = ndimage.zoom(bes.copy(), 1./over, order=0)
            t /= np.max(t)
            fileIO.writeFITS(im.copy() - np.median(oversampled) - t*np.max(tmp), f+'residual.fits', int=False)
            fileIO.writeFITS(oversampled - bes.copy()*np.max(tmp), f+'residualOversampled.fits', int=False)

            #best guesses for fitting parameters
            params = [1., results['centreX'], results['centreY'], 0.5, 0.5]

            biassubtracted = im.copy() - np.median(oversampled)
            #error function is a convolution between a bessel function and 2D gaussian - data
            #note that the error function must be on low-res grid because it is the pixel values we try to match
            errfunc = lambda p: np.ravel(ndimage.zoom(signal.fftconvolve(fitf(*p)(*np.indices(tmp.shape)), bes.copy(), mode='same'), 1./over, order=0)*np.max(tmp) - biassubtracted.copy())

            #fit
            res = sp.optimize.leastsq(errfunc, params, full_output=True, maxfev=maxfev)

            #save the fitted residuals
            t = signal.fftconvolve(fitf(*res[0])(*np.indices(tmp.shape)), bes.copy(), mode='same')
            fileIO.writeFITS(res[2]['fvec'].reshape(im.shape), f+'residualFit.fits', int=False)
            fileIO.writeFITS(fitf(*res[0])(*np.indices(tmp.shape)), f+'gaussian.fits', int=False)
            fileIO.writeFITS(t, f+'BesselGausOversampled.fits', int=False)
            fileIO.writeFITS(ndimage.zoom(t, 1./over, order=0), f+'BesselGaus.fits', int=False)

            #print out the results and save to a dictionary
            print results['centreX'], results['centreY'], res[2]['nfev'], res[0]

            #sigmas are symmetric as the width of the fitting function is later squared...
            sigma1 = np.abs(res[0][3])
            sigma2 = np.abs(res[0][4])
            Gaussians[f] = [sigma1, sigma2]

        fileIO.cPickleDumpDictionary(Gaussians, 'SpotmeasurementsBesselGaussian.pk')

        #plot the findings
        plotGaussianResults(Gaussians, ids, output='Bessel', vals=[0, 1])


def fileIDs():
    d = {'30Apr_21_59_31s_Euclid' : [0, 0],
         '30Apr_21_59_47s_Euclid' : [0, 1],
         '30Apr_22_00_03s_Euclid' : [0, 2],
         '30Apr_22_00_20s_Euclid' : [0, 3],
         '30Apr_22_00_36s_Euclid' : [0, 4],
         '30Apr_22_00_53s_Euclid' : [0, 5],
         '30Apr_22_01_09s_Euclid' : [0, 6],
         '30Apr_22_01_27s_Euclid' : [1, 0],
         '30Apr_22_01_44s_Euclid' : [1, 1],
         '30Apr_22_02_00s_Euclid' : [1, 2],
         '30Apr_22_02_17s_Euclid' : [1, 3],
         '30Apr_22_02_33s_Euclid' : [1, 4],
         '30Apr_22_02_50s_Euclid' : [1, 5],
         '30Apr_22_03_06s_Euclid' : [1, 6],
         '30Apr_22_03_24s_Euclid' : [2, 0],
         '30Apr_22_03_40s_Euclid' : [2, 1],
         '30Apr_22_03_57s_Euclid' : [2, 2],
         '30Apr_22_04_13s_Euclid' : [2, 3],
         '30Apr_22_04_30s_Euclid' : [2, 4],
         '30Apr_22_04_46s_Euclid' : [2, 5],
         '30Apr_22_05_03s_Euclid' : [2, 6],
         '30Apr_22_05_21s_Euclid' : [3, 0],
         '30Apr_22_05_37s_Euclid' : [3, 1],
         '30Apr_22_05_53s_Euclid' : [3, 2],
         '30Apr_22_06_10s_Euclid' : [3, 3],
         '30Apr_22_06_26s_Euclid' : [3, 4],
         '30Apr_22_06_43s_Euclid' : [3, 5],
         '30Apr_22_06_59s_Euclid' : [3, 6],
         '30Apr_22_07_17s_Euclid' : [4, 0],
         '30Apr_22_07_34s_Euclid' : [4, 1],
         '30Apr_22_07_50s_Euclid' : [4, 2],
         '30Apr_22_08_06s_Euclid' : [4, 3],
         '30Apr_22_08_23s_Euclid' : [4, 4],
         '30Apr_22_08_39s_Euclid' : [4, 5],
         '30Apr_22_08_56s_Euclid' : [4, 6],
         '30Apr_22_09_14s_Euclid' : [5, 0],
         '30Apr_22_09_30s_Euclid' : [5, 1],
         '30Apr_22_09_47s_Euclid' : [5, 2],
         '30Apr_22_10_03s_Euclid' : [5, 3],
         '30Apr_22_10_20s_Euclid' : [5, 4],
         '30Apr_22_10_36s_Euclid' : [5, 5],
         '30Apr_22_10_53s_Euclid' : [5, 6],
         '30Apr_22_11_11s_Euclid' : [6, 0],
         '30Apr_22_11_27s_Euclid' : [6, 1],
         '30Apr_22_11_43s_Euclid' : [6, 2],
         '30Apr_22_12_00s_Euclid' : [6, 3],
         '30Apr_22_12_16s_Euclid' : [6, 4],
         '30Apr_22_12_33s_Euclid' : [6, 5],
         '30Apr_22_12_49s_Euclid' : [6, 6]}
    return d


def examples():
    """
    Simple convolution-deconvolution examples.

    :return: None
    """
    from scipy import signal

    tri = np.zeros(100)
    tri[20:50] = 1/29.*np.arange(30)
    tri[49:79] = 1/29.*np.arange(30)[::-1]

    #gaussian = np.exp(-(30 - np.arange(60))**2/20.)

    top = np.zeros(100)
    top[30:71] = 1.

    #conv = signal.convolve(gaussian, top)
    conv = signal.convolve(tri, top, mode='same')
    conv /= float(np.max(conv))

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle('Convolution Example')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(tri, 'b-')
    ax2.plot(top, 'b-')
    ax3.plot(conv, 'b-')
    ax1.set_ylim(-0.01, 1.1)
    ax2.set_ylim(-0.01, 1.1)
    ax3.set_ylim(-0.01, 1.1)
    plt.savefig('ExampleConvolution.pdf')
    plt.close()

    #deconvolution -- Weiner filter type
    #lambda is your regularisation parameter
    decon = weinerFilter(conv, top)
    decon = np.concatenate((decon, np.zeros(1)))[1:]  #for some reason the output is shifted by one

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle('Deconvolution Example')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(conv, 'b-')
    ax2.plot(top, 'b-')
    ax3.plot(decon, 'b-')
    ax3.plot(tri, 'r--')
    ax1.set_ylim(-0.01, 1.1)
    ax2.set_ylim(-0.01, 1.1)
    ax3.set_ylim(-0.01, 1.1)
    plt.savefig('ExampleDeconvolution.pdf')
    plt.close()

    #2D examples
    tri = np.zeros((100, 100))
    tri[30:70, 20:50] = 1/29.*np.arange(30)
    tri[30:70, 49:79] = 1/29.*np.arange(30)[::-1]

    top = np.zeros((100, 100))
    top[30:71, 30:71] = 1.

    conv = signal.fftconvolve(tri, top, mode='same')
    conv /= float(np.max(conv))

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle('Convolution Example')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    i1 = ax1.imshow(tri, origin='lower', vmin=-.01, vmax=1.1)
    i2 = ax2.imshow(top, origin='lower', vmin=-.01, vmax=1.1)
    i3 = ax3.imshow(conv, origin='lower', vmin=-.01, vmax=1.1)
    plt.colorbar(i1, ax=ax1, orientation='horizontal')
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    plt.colorbar(i3, ax=ax3, orientation='horizontal')
    plt.savefig('ExampleConvolution2D.pdf')
    plt.close()

    #deconv
    decon = weinerFilter(conv, top)
    decon /= float(np.max(decon))
    decon = np.vstack((decon, np.zeros((1, decon.shape[1]))))   #again there is a shift...
    decon = np.hstack((decon, np.zeros((decon.shape[0], 1))))   #will append zeroes
    decon = decon[1:, 1:]                                       #and then cut out the first row and column
    print np.argmax(decon), np.argmax(tri)

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle('Convolution Example')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    i1 = ax1.imshow(conv.copy(), origin='lower', vmin=-.01, vmax=1.1)
    i2 = ax2.imshow(top.copy(), origin='lower', vmin=-.01, vmax=1.1)
    i3 = ax3.imshow(decon.copy(), origin='lower', vmin=-.01, vmax=1.1)
    plt.colorbar(i1, ax=ax1, orientation='horizontal')
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    plt.colorbar(i3, ax=ax3, orientation='horizontal')
    plt.savefig('ExampleDeconvolution2D.pdf')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(tri.copy() / decon.copy(), origin='lower', vmin=0.99, vmax=1.01)
    i2 = ax2.imshow(tri.copy() - decon.copy(), origin='lower', vmin=-.1, vmax=.1)
    plt.colorbar(i1, ax=ax1, orientation='horizontal', ticks=[0.99, 1, 1.01])
    plt.colorbar(i2, ax=ax2, orientation='horizontal', ticks=[-0.1, -0.05, 0, 0.05, 0.1])
    plt.savefig('ExampleResidual2D.pdf')
    plt.close()


if __name__ == '__main__':
    #examples()

    allFITS = g.glob('*Euclidsmall.fits')
    analyseSpotsFitting(allFITS)

    #allFITS = g.glob('*Euclid.fits')
    #analyseSpotsDeconvolution(allFITS)