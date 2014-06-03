"""
A simple script to analyse ground/lab flat fields.

This script has been written to analyse the wavelength dependency of the PRNU.

:author: Sami-Matias Niemi
:version: 0.3
"""
import matplotlib
#matplotlib.use('pdf')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['image.interpolation'] = 'none'
import matplotlib.pyplot as plt
import pyfits as pf
import numpy as np
import glob as g
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.linalg import norm
from scipy import fftpack
from scipy import ndimage
from support import files as fileIO
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from multiprocessing import Pool
import math
from PIL import Image
from scipy.interpolate import interp1d
from skimage.measure import structural_similarity as ssim
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcess
from matplotlib.ticker import FixedFormatter
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.neighbors import KernelDensity


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)



def subtractBias(data, biasfile):
    """
    Subtract ADC offset using the pre- and overscan information for each quadrant.
    """
    if biasfile:
        b = pf.getdata('bias.fits')
        data -= b
    else:
        prescanL = data[3:2060, 3:51].mean()
        prescanH = data[2076:4125, 3:51].mean()
        overscanL = data[3:2060, 4150:4192].mean()
        overscanH = data[2076:4125, 4150:4192].mean()

        Q0 = data[:2060, :2098]
        Q2 = data[2076:, :2098]
        Q1 = data[:2060, 2098:]
        Q3 = data[2076:, 2098:]

        #subtract the bias levels
        Q0 -= prescanL
        Q2 -= prescanH
        Q1 -= overscanL
        Q3 -= overscanH

        data[:2060, :2098] = Q0
        data[2076:, :2098] = Q2
        data[:2060, 2098:] = Q1
        data[2076:, 2098:] = Q3

    return data


def makeBias(files):
    """
    Generate a median combined bias frame from the input files
    """
    d = np.asarray([pf.getdata(file) for file in files])

    #write out FITS file
    med = np.median(d, axis=0)
    fileIO.writeFITS(med, 'bias.fits', int=False)


def makeFlat(files, output, gain=3.09, biasfile=True):
    """
    Combine flat fields
    """
    d = []

    for file in files:
        data = subtractBias(pf.getdata(file), biasfile) * gain
        fileIO.writeFITS(data, file.replace('.fits', 'biasremoved.fits'), int=False)

        d.append(data)

    d = np.asarray(d)

    #write out FITS file
    avg = np.average(d, axis=0)
    fileIO.writeFITS(avg, output+'averaged.fits', int=False)

    med = np.median(d, axis=0)
    fileIO.writeFITS(med, output+'median.fits', int=False)

    return avg, med


def normaliseFlat(data, output, limit=8.e4, order=5):
    """
    Normalise each quadrant separately. If limit set use to to generate a mask.
    """
    #split to quadrants
    Q0 = data[3:2060, 52:2098].copy()
    Q2 = data[2076:, 52:2098].copy()
    Q1 = data[3:2060, 2098:4145].copy()
    Q3 = data[2076:, 2098:4145].copy()
    Qs = [Q0, Q1, Q2, Q3]

    res = []
    for tmp in Qs:
        if limit is not None:
            print 'Using masked arrays in the fitting...'
            t = np.ma.MaskedArray(tmp, mask=(tmp > limit))

        #meshgrid representing data
        x, y = np.mgrid[:t.shape[0], :t.shape[1]]

        #fit a polynomial 2d surface to remove the illumination profile
        p_init = models.Polynomial2D(degree=order)
        f = fitting.NonLinearLSQFitter()
        p = f(p_init, x, y, t)

        #normalize data and save it to res list
        tmp /= p(x, y)
        res.append(tmp)
        print np.mean(tmp)

    #save out
    out = np.zeros_like(data)
    out[3:2060, 52:2098] = res[0]
    out[3:2060, 2098:4145] = res[1]
    out[2076:, 52:2098] = res[2]
    out[2076:, 2098:4145] = res[3]

    fileIO.writeFITS(out, output+'FlatField.fits', int=False)

    return out


def findFiles():
    """
    Find files for each wavelength
    """
    #pre-process: 28th
    files = g.glob('28Apr/*Euclid.fits')
    f = [file.replace('28Apr/', '') for file in files]
    f = [file.replace('_', '.') for file in f]
    times = [float(file[:5]) for file in f]
    files = np.asarray(files)
    times = np.asarray(times)

    #545: 15_42 _13sEuclid - 16_25 _37sEuclid
    msk545 = (times >= 15.42) & (times <= 16.25)
    f545 = files[msk545]

    #570: 16_36 _58sEuclid - 17_00 _12sEuclid
    msk570 = (times >= 16.36) & (times <= 17.00)
    f570 = files[msk570]

    #bias: 14_48 _49sEuclid - 15_00 _25sEuclid
    mskbias = (times >= 14.48) & (times <= 15.00)
    bias = files[mskbias]

    #pre-process: 29th
    files = g.glob('29Apr/*Euclid.fits')
    f = [file.replace('29Apr/', '') for file in files]
    f = [file.replace('_', '.') for file in f]
    times = [float(file[:5]) for file in f]
    files = np.asarray(files)
    times = np.asarray(times)

    #660: 13_31 _55sEuclid - 14_10 _49sEuclid
    msk660 = (times >= 13.31) & (times <= 14.10)
    f660 = files[msk660]

    #700: 14_32 _24sEuclid - 15_08 _02sEuclid
    msk700 = (times >= 14.32) & (times <= 15.08)
    f700 = files[msk700]

    #800: 15_22 _34sEuclid - 15_59 _06sEuclid
    msk800 = (times >= 15.22) & (times <= 15.59)
    f800 = files[msk800]

    #850: 16_24 _37sEuclid - 17_04 _03sEuclid
    msk850 = (times >= 16.24) & (times <= 17.04)
    f850 = files[msk850]

    #pre-process: 30th
    files = g.glob('30Apr/*Euclid.fits')
    f = [file.replace('30Apr/', '') for file in files]
    f = [file.replace('_', '.') for file in f]
    times = [float(file[:5]) for file in f]
    files = np.asarray(files)
    times = np.asarray(times)

    #600: 16_12 _49sEuclid-16_50 _22sEuclid
    msk600 = (times >= 16.12) & (times <= 16.50)
    f600 = files[msk600]

    #940: 17_09 _37sEuclid - 17_48 _13sEuclid
    msk940 = (times >= 17.09) & (times <= 17.48)
    f940 = files[msk940]

    #dictionary
    out = dict(f545=f545, f570=f570, f600=f600, f660=f660, f700=f700, f800=f800, f850=f850, f940=f940, bias=bias)

    return out


def _generateFlats(key, files):
    """
    Actual calls to generate flat fields.
    """
    print key, files, files.shape
    avg, med = makeFlat(files, key)
    normed = normaliseFlat(med, key)
    return normed


def generateFlats(args):
    """
    A wrapper to generate flat fields simultaneously at different wavelengths.
    A hack required as Pool does not accept multiple arguments.
    """
    return _generateFlats(*args)


def flats():
    #search for the right files
    files = findFiles()

    #generate bias
    makeBias(files['bias'])
    files.pop('bias', None)

    #generate flats using multiprocessing
    pool = Pool(processes=6)
    pool.map(generateFlats, [(key, files[key]) for key in files.keys()])


def plot(xmin=300, xmax=3500, ymin=200, ymax=1600, smooth=2.):
    #load data
    data = {}
    for file in g.glob('*FlatField.fits'):
        fh = pf.open(file)
        wave = file[1:4]
        data[wave] = fh[1].data
        fh.close()

    #simple plot showing some structures
    number_of_subplots = math.ceil(len(data.keys())/2.)

    fig = plt.figure(figsize=(13, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

    #loop over data from shortest wavelength to the longest
    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][ymin:ymax, xmin:xmax].copy()

        #Gaussian smooth to enhance structures for plotting
        if smooth > 1:
            tmp = gaussian_filter(tmp, smooth)

        if 690 < int(wave) < 710:
            norm = tmp

        ax = plt.subplot(number_of_subplots, 2, i+1)
        im = ax.imshow(tmp, interpolation='none', origin='lower', vmin=0.997, vmax=1.003)

        ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm')

        plt.axis('off')

    cbar = plt.colorbar(im, cax=fig.add_axes([0.65, 0.14, 0.25, 0.03], frameon=False),
                        ticks=[0.997, 1, 1.003], format='%.3f', orientation='horizontal')
    cbar.set_label('Normalised Pixel Values')

    plt.savefig('PRNUmaps.png')
    plt.close()

    #normalise to 700nm
    fig = plt.figure(figsize=(13, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][ymin:ymax, xmin:xmax].copy()

        #Gaussian smooth to enhance structures for plotting
        if smooth > 1:
            tmp = gaussian_filter(tmp, smooth)

        tmp /= norm

        ax = plt.subplot(number_of_subplots, 2, i+1)
        im = ax.imshow(tmp, interpolation='none', origin='lower', vmin=0.997, vmax=1.003)

        ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm')

        plt.axis('off')

    cbar = plt.colorbar(im, cax=fig.add_axes([0.65, 0.14, 0.25, 0.03], frameon=False),
                        ticks=[0.997, 1, 1.003], format='%.3f', orientation='horizontal')
    cbar.set_label('Normalised Pixel Values')

    plt.savefig('PRNUmapsNormed.png')
    plt.close()

    #loop over data from shortest wavelength to the longest
    dat = {}
    ylen = 100
    xlen = 100
    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][ymin:ymax, xmin:xmax].copy()
        #select sub regions to calculate the PRNU in
        prnu = []
        ydim, xdim = tmp.shape
        samplesx = xdim / xlen
        samplesy = ydim / ylen
        print samplesx, samplesy
        for a in range(samplesy):
            for b in range(samplesx):
                area = tmp[a*ylen:(a+1)*ylen, b*xlen:(b+1)*xlen]
                prn = np.std(sigma_clip(area, 6.)) * 100.
                prnu.append(prn)

        dat[int(wave)] = prnu

    #calculate the mean for each wavelength and std
    w = []
    mean = []
    std = []
    for wave in sorted(dat.keys()):
        m = np.mean(dat[wave])
        s = np.std(dat[wave])
        w.append(wave)
        mean.append(m)
        std.append(s)
        print wave, m, s

    #polynomial fit to PRNU data
    z2 = np.polyfit(w, mean, 1)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(w, mean, 3)
    p3 = np.poly1d(z3)
    x = np.linspace(500, 900)

    #using a gaussian process
    X = np.atleast_2d(w).T
    y = np.asarray(mean)
    ev = np.atleast_2d(np.linspace(500, 900, 1000)).T #points at which to evaluate
    # Instanciate a Gaussian Process model
    gp = GaussianProcess(corr='squared_exponential', theta0=1e-1, thetaL=1e-3, thetaU=1,
                         nugget=(np.asarray(std)*3/y)**2, random_start=100)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(ev, eval_MSE=True)
    sigma = np.sqrt(MSE)

    #standard error of the mean
    sigma3 = 3.*np.asarray(std)/np.sqrt(len(std))

    #wavelength dependency plot
    plt.title('Wavelength Dependency of the PRNU')
    #plt.plot(x, p3(x), 'g--', label='3rd order fit')
    plt.plot(ev, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([ev, ev[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label=u'95\% confidence interval')
    plt.errorbar(w, mean, yerr=sigma3, c='r', fmt='o', label='data, $3\sigma$ errors')
    plt.plot(x, p2(x), 'g-', lw=2, label='linear fit')
    plt.xlabel(r'Wavelength $\lambda$ [nm]')
    plt.ylabel(r'PRNU $[\%]$')
    plt.xlim(500, 900)
    plt.ylim(0.5, 1.)
    plt.legend(shadow=True, fancybox=True, scatterpoints=1, numpoints=1)
    plt.savefig('PRNUwave.pdf')
    plt.close()

    #residual variance after normalization
    #loop over data from shortest wavelength to the longest
    for j, refwave in enumerate(sorted(data.keys())):
        dat = {}
        ylen = 100
        xlen = 100
        for i, wave in enumerate(sorted(data.keys())):
            tmp = data[wave][ymin:ymax, xmin:xmax].copy()
            ref = data[refwave][ymin:ymax, xmin:xmax].copy()
            #select sub regions to calculate the PRNU in
            prnu = []
            ydim, xdim = tmp.shape
            samplesx = xdim / xlen
            samplesy = ydim / ylen
            for a in range(samplesy):
                for b in range(samplesx):
                    area = tmp[a*ylen:(a+1)*ylen, b*xlen:(b+1)*xlen]
                    arearef = ref[a*ylen:(a+1)*ylen, b*xlen:(b+1)*xlen]
                    area /= arearef
                    prn = np.std(sigma_clip(area, 6.)) * 100.
                    prnu.append(prn)

            dat[int(wave)] = prnu

        #calculate the mean for each wavelength and std
        w = []
        mean = []
        std = []
        for wave in sorted(dat.keys()):
            m = np.mean(dat[wave])
            s = np.std(dat[wave])
            w.append(wave)
            mean.append(m)
            std.append(s)
            print wave, m, s

        #standard error of the mean
        sigma3 = 3.*np.asarray(std)/np.sqrt(len(std))

        #wavelength dependency plot
        plt.subplots_adjust(left=0.13)
        plt.title('Residual Dispersion')
        plt.errorbar(w, mean, yerr=sigma3, c='r', fmt='o', label='data, $3\sigma$ errors')
        plt.xlabel(r'Wavelength $\lambda$ [nm]')
        plt.ylabel(r'$\sigma \left ( \frac{{M_{{i}}}}{{M_{{{0:s}}}}}  \right )$ $[\%]$'.format(refwave))
        plt.xlim(500, 900)
        #plt.ylim(-0.05, 0.3)
        plt.legend(shadow=True, fancybox=True, scatterpoints=1, numpoints=1)
        plt.savefig('PRNUwaveResidual%s.pdf' % refwave)
        plt.close()


def spatialAutocorrelation(interpolation='none', smooth=2):
    #load data
    data = {}
    for file in g.glob('*FlatField.fits'):
        fh = pf.open(file)
        wave = file[1:4]
        data[wave] = fh[1].data
        fh.close()

    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][500:1524, 500:1524].copy()
        tmp = gaussian_filter(tmp, smooth)

        autoc = signal.fftconvolve(tmp, np.flipud(np.fliplr(tmp)), mode='full')
        autoc /= np.max(autoc)
        autoc *= 100.
        fileIO.writeFITS(autoc, 'autocorrelationRealdata%s.fits' % wave, int=False)

        #plot images
        fig = plt.figure(figsize=(14.5, 6.5))
        plt.suptitle(r'Autocorrelation of Flat Field Data $\lambda = %i$ \AA' % int(wave))
        plt.suptitle(r'Autocorrelation Interaction $[\%]$', x=0.7, y=0.26)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        i1 = ax1.imshow(tmp, origin='lower', interpolation=interpolation, vmin=0.997, vmax=1.003)
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.3f')
        i2 = ax2.imshow(autoc, interpolation=interpolation, origin='lower', rasterized=True, vmin=0, vmax=100)
        plt.colorbar(i2, ax=ax2, orientation='horizontal')
        ax1.set_xlabel('X [pixel]')
        ax1.set_ylabel('Y [pixel]')
        plt.savefig('SpatialAutocorrelation%s.pdf' % wave)
        plt.close()


def powerSpectrum(interpolation='none'):
    """

    """
    x, y = np.mgrid[0:32, 0:32]
    img = 100 * np.cos(x*np.pi/4.) * np.cos(y*np.pi/4.)
    kernel = np.array([[0.0025, 0.01, 0.0025], [0.01, 0.95, 0.01], [0.0025, 0.01, 0.0025]])
    img = ndimage.convolve(img.copy(), kernel)

    fourierSpectrum2 = np.abs(fftpack.fft2(img))
    print np.mean(fourierSpectrum2), np.median(fourierSpectrum2), np.std(fourierSpectrum2), np.max(fourierSpectrum2), np.min(fourierSpectrum2)

    fig = plt.figure(figsize=(14.5, 6.5))
    plt.suptitle('Fourier Analysis of Flat-field Data')
    plt.suptitle('Original Image', x=0.32, y=0.26)
    plt.suptitle(r'$\log_{10}$(2D Power Spectrum)', x=0.72, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    i1 = ax1.imshow(img, origin='lower', interpolation=interpolation, rasterized=True)
    plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.1e')
    i2 = ax2.imshow(fourierSpectrum2[0:512, 0:512], interpolation=interpolation, origin='lower',
                    rasterized=True)
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    ax1.set_xlabel('X [pixel]')
    ax2.set_xlabel('$l_{x}$')
    ax2.set_ylim(0, 16)
    ax2.set_xlim(0, 16)
    ax1.set_ylabel('Y [pixel]')
    plt.savefig('FourierSin.pdf')
    plt.close()

    #load data
    data = {}
    for file in g.glob('*FlatField.fits'):
        fh = pf.open(file)
        wave = file[1:4]
        data[wave] = fh[1].data
        fh.close()

    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][500:1524, 500:1524].copy()
        #tmp = gaussian_filter(tmp, 3.)

        fourierSpectrum = np.abs(fftpack.fft2(tmp.astype(np.float32)))
        fp = np.log10(fourierSpectrum[0:512, 0:512])

        fig = plt.figure(figsize=(14.5, 6.5))
        plt.suptitle('Fourier Analysis of Flat-field Data')
        plt.suptitle('Original Image', x=0.32, y=0.26)
        plt.suptitle(r'$\log_{10}$(2D Power Spectrum)', x=0.72, y=0.26)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        i1 = ax1.imshow(tmp, origin='lower', interpolation=interpolation, rasterized=True,
                        vmin=0.96, vmax=1.04)
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.2f')
        i2 = ax2.imshow(fp, interpolation=interpolation, origin='lower', rasterized=True,
                        vmin=-1., vmax=3.)
        plt.colorbar(i2, ax=ax2, orientation='horizontal')
        ax1.set_xlabel('X [pixel]')
        ax2.set_xlabel('$l_{x}$')
        ax2.set_ylim(0, 16)
        ax2.set_xlim(0, 16)
        ax1.set_ylabel('Y [pixel]')
        plt.savefig('PowerSpectrum%s.pdf' % wave)
        plt.close()


def normalisedCrosscorrelation(image1, image2):
    dist_ncc = np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2)) ) / \
                       ((image1.size - 1) * np.std(image1) * np.std(image2) )
    return dist_ncc


def correlate(xmin=300, xmax=3500, ymin=200, ymax=1600):
    #load data
    data = {}
    for file in g.glob('*FlatField.fits'):
        fh = pf.open(file)
        wave = file[1:4]
        data[wave] = fh[1].data
        fh.close()

    cr = []
    crRandom = []
    for i, wave1 in enumerate(sorted(data.keys())):
        for j, wave2 in enumerate(sorted(data.keys())):
            tmp1 = data[wave1][ymin:ymax, xmin:xmax].copy()
            tmp2 = data[wave2][ymin:ymax, xmin:xmax].copy()

            # st1 = np.std(tmp1)
            # av1 = np.mean(tmp1)
            # msk = (tmp1 > st1*1 + av1) & (tmp1 < av1 - st1*1)
            # tmp1 = tmp1[~msk]
            # tmp2 = tmp2[~msk]

             # calculate the difference and its norms
            diff = tmp1 - tmp2  # elementwise for scipy arrays
            m_norm = np.sum(np.abs(diff))  # Manhattan norm
            z_norm = norm(diff.ravel(), 0)  # Zero norm

            #cor = np.corrcoef(tmp1, tmp2)
            dist_ncc = normalisedCrosscorrelation(tmp1, tmp2)

            print wave1, wave2
            print "Manhattan norm:", m_norm, "/ per pixel:", m_norm/tmp1.size
            print "Zero norm:", z_norm, "/ per pixel:", z_norm*1./tmp1.size
            print "Normalized cross-correlation:", dist_ncc

            cr.append(dist_ncc)
            crRandom.append(normalisedCrosscorrelation(np.random.random(tmp1.shape), np.random.random(tmp1.shape)))

    #data containers, make a 2D array of the cross-correlations
    wx = [x for x in sorted(data.keys())]
    wy = [y for y in sorted(data.keys())]
    cr = np.asarray(cr).reshape(len(wx), len(wy))
    crRandom = np.asarray(crRandom).reshape(len(wx), len(wy))

    fig = plt.figure()
    plt.title('Normalized Cross-Correlation')
    ax = fig.add_subplot(111)

    plt.pcolor(cr, cmap='Greys', vmin=0.9, vmax=1.)
    plt.colorbar()

    #change the labels and move ticks to centre
    ticks = np.arange(len(wx)) + 0.5
    plt.xticks(ticks, wx)
    plt.yticks(ticks, wy)
    ax.xaxis.set_ticks_position('none') #remove the tick marks
    ax.yaxis.set_ticks_position('none') #remove the tick marks

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Wavelength [nm]')

    plt.savefig('Crosscorrelation.pdf')
    plt.close()

    #plot the data from random arrays
    fig = plt.figure()
    plt.title('Normalized Cross-Correlation (Random)')
    ax = fig.add_subplot(111)

    plt.pcolor(crRandom, cmap='Greys')#, vmin=0.9, vmax=1.)
    plt.colorbar()

    #change the labels and move ticks to centre
    ticks = np.arange(len(wx)) + 0.5
    plt.xticks(ticks, wx)
    plt.yticks(ticks, wy)
    ax.xaxis.set_ticks_position('none') #remove the tick marks
    ax.yaxis.set_ticks_position('none') #remove the tick marks

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Wavelength [nm]')

    plt.savefig('CrosscorrelationRandom.pdf')
    plt.close()


def morphPRNUmap():
    data = _loadPRNUmaps()

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blended = Image.blend(data['570'], data['660'], alpha)

        fig = plt.figure()
        plt.title('Image Blending')
        ax = fig.add_subplot(111)

        i1 = ax.imshow(blended, origin='lower', interpolation='none', rasterized=True)
        plt.colorbar(i1, ax=ax, orientation='horizontal', format='%.2f')

        plt.savefig('Blending%f.pdf' %alpha)
        plt.close()


def mse(x, y):
    """
    Mean Square Error (MSE)
    """
    return np.linalg.norm(x - y)


def structuralSimilarity(xmin=300, xmax=3500, ymin=200, ymax=1600, smooth=0.):
    """
    Adapted from:
    http://scikit-image.org/docs/0.9.x/auto_examples/plot_ssim.html#example-plot-ssim-py
    """
    data = _loadPRNUmaps()

    ref = data['700'][ymin:ymax, xmin:xmax].copy()
    if smooth > 1:
        ref = gaussian_filter(ref, smooth)

    number_of_subplots = math.ceil(len(data.keys())/2.)

    fig = plt.figure(figsize=(13, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

    #loop over data from shortest wavelength to the longest
    wavearray = []
    msearray = []
    ssiarray = []
    for i, wave in enumerate(sorted(data.keys())):
        tmp = data[wave][ymin:ymax, xmin:xmax].copy()
        rows, cols = tmp.shape

        #Gaussian smooth to enhance structures for plotting
        if smooth > 1:
            tmp = gaussian_filter(tmp, smooth)

        ms = mse(tmp, ref)
        #careful with the win_size, can take up to 16G of memory if set to e.g. 19
        ssi = ssim(tmp, ref, dynamic_range=tmp.max() - tmp.min(), win_size=9)

        ax = plt.subplot(number_of_subplots, 2, i+1)
        im = ax.imshow(tmp, interpolation='none', origin='lower', vmin=0.997, vmax=1.003)

        ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm;' + ' MSE: %.2f, SSIM: %.5f' % (ms, ssi))

        plt.axis('off')

        print wave, ms, ssi
        wavearray.append(int(wave))
        msearray.append(ms)
        ssiarray.append(ssi)

    cbar = plt.colorbar(im, cax=fig.add_axes([0.65, 0.14, 0.25, 0.03], frameon=False),
                        ticks=[0.997, 1, 1.003], format='%.3f', orientation='horizontal')
    cbar.set_label('Normalised Pixel Values')

    plt.savefig('StructuralSimilarities.png')
    plt.close()

    fig = plt.figure()
    plt.title(r'Mean Squared Error Wrt. $\lambda = 700$nm')
    ax = fig.add_subplot(111)
    ax.plot(wavearray, msearray, 'bo')
    ax.set_xlim(500, 900)
    ax.set_ylim(-0.03, 6.)
    ax.set_ylabel('MSE')
    ax.set_xlabel('Wavelength [nm]')
    plt.savefig('MSEwave.pdf')
    plt.close()


def interpolateMissing(hide=600, kind='linear', three=False):
    """
    Interpolates the hidden PRNU map by using the other PRNU maps and interpolating between them.
    The current implementations is really slow because it relies in nested 1D interpolation.

    :param hide: which wavelength is used as the target to recover
    :type hide: int
    :param kind: interpolation type e.g. linear or qaudratic, see scipy interp1d for information
    :type kind: str
    :param three: whether only three wavelengths were available
    :type three: bool

    :return: derived PRNU map, target PRNU map
    :rtype: list of ndarrays
    """
    shide = str(hide)

    data = _loadPRNUmaps()

    #these are all the other data
    rest = []
    w  = []
    for i, wave in enumerate(sorted(data.keys())):
        if shide not in wave:
            rest.append(data[wave][1150:1300, 1200:1400])
            #rest.append(data[wave][200:1600, 300:3500])
            w.append(int(wave))
        else:
            #this is the one we try to recover
            hidden = data[shide][1150:1300, 1200:1400]
            #hidden = data[shide][200:1600, 300:3500]

    if three:
        #pick first and last and one from the middle
        rest = [rest[0], rest[2], rest[-1]]
        w = [w[0], w[2], w[-1]]

    rest = np.asarray(rest)
    ww, jj, ii = rest.shape
    print w

    out = np.zeros(hidden.shape)
    #this is really slow way of doing this...
    for j in range(jj):
        for i in range(ii):
            f = interp1d(w, rest[:, j, i], kind=kind)
            out[j, i] = f(hide)

    ratio = out / hidden
    print 'Mean, STD, MSE:'
    print ratio.mean(), ratio.std(), mse(out, hidden)

    #save FITS files
    if three:
        fileIO.writeFITS(out, 'recoveredThree%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hiddenThree%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residualThree%i.fits' % hide, int=False)
    else:
        fileIO.writeFITS(out, 'recovered%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hidden%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residual%i.fits' % hide, int=False)

    return out, hidden


def predictMissingLinearRegression(hide=600, three=False):
    shide = str(hide)

    data = _loadPRNUmaps()

    #these are all the other data
    rest = []
    w  = []
    for i, wave in enumerate(sorted(data.keys())):
        if shide not in wave:
            #rest.append(data[wave][1150:1300, 1200:1400])
            rest.append(data[wave][200:1600, 300:3500])
            w.append(int(wave))
        else:
            #this is the one we try to recover
            #hidden = data[shide][1150:1300, 1200:1400]
            hidden = data[shide][200:1600, 300:3500]

    if three:
        #pick first and last and one from the middle
        rest = [rest[0], rest[2], rest[-1]]
        w = [w[0], w[2], w[-1]]

    rest = np.asarray(rest)
    print w
    w = np.atleast_2d(w).T  #needed for the linear regression

    #Create linear regression object
    ww, jj, ii = rest.shape
    out = np.zeros(hidden.shape)
    #this is really slow way of doing this...
    for j in range(jj):
        for i in range(ii):
            regr = linear_model.LinearRegression()
            regr.fit(w, rest[:, j, i])
            out[j, i] = regr.predict(hide)

    ratio = out / hidden
    print 'Mean, STD, MSE:'
    print ratio.mean(), ratio.std(), mse(out, hidden)

    #save FITS files
    if three:
        fileIO.writeFITS(out, 'recoveredThreeLR%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hiddenThreeLR%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residualThreeLR%i.fits' % hide, int=False)
    else:
        fileIO.writeFITS(out, 'recoveredLR%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hiddenLR%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residualLR%i.fits' % hide, int=False)

    return out, hidden


def predictMissingGaussianProcessRegression(hide=600, three=False):
    shide = str(hide)

    data = _loadPRNUmaps()

    #these are all the other data
    rest = []
    w  = []
    for i, wave in enumerate(sorted(data.keys())):
        if shide not in wave:
            rest.append(data[wave][1150:1300, 1200:1400])
            #rest.append(data[wave][200:1600, 300:3500])
            w.append(int(wave))
        else:
            #this is the one we try to recover
            hidden = data[shide][1150:1300, 1200:1400]
            #hidden = data[shide][200:1600, 300:3500]

    if three:
        #pick first and last and one from the middle
        rest = [rest[0], rest[2], rest[-1]]
        w = [w[0], w[2], w[-1]]

    rest = np.asarray(rest)
    print w
    w = np.atleast_2d(w).T  #needed for the gp

    #Create linear regression object
    ww, jj, ii = rest.shape
    out = np.zeros(hidden.shape)
    #this is really slow way of doing this...
    for j in range(jj):
        for i in range(ii):
            # Instanciate a Gaussian Process model
            gp = GaussianProcess(regr='linear', corr='linear')
            gp.fit(w, rest[:, j, i])
            out[j, i] = gp.predict(hide)

    ratio = out / hidden
    print 'Mean, STD, MSE:'
    print ratio.mean(), ratio.std(), mse(out, hidden)

    #save FITS files
    if three:
        fileIO.writeFITS(out, 'recoveredThreeGP%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hiddenThreeGP%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residualThreeGP%i.fits' % hide, int=False)
    else:
        fileIO.writeFITS(out, 'recoveredGP%i.fits' % hide, int=False)
        fileIO.writeFITS(hidden, 'hiddenGP%i.fits' % hide, int=False)
        fileIO.writeFITS(ratio, 'residualGP%i.fits' % hide, int=False)

    return out, hidden


def plotMissing(target, derived, hide, out='', smooth=2, ratio=True):
    """
    Generate plots showing the target and derived PRNU maps and the residual of the two.
    """
    #plot simple images
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_title('Target')
    ax2.set_title('Derived')
    if ratio:
        ax3.set_title(r'Residual: (D/T)')
    else:
        ax3.set_title(r'Residual: $|D - T|$')

    i1 = ax1.imshow(gaussian_filter(target, smooth),
                    origin='lower', interpolation='none', rasterized=True, vmin=0.995, vmax=1.005)
    plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.3f', ticks=[0.995, 1, 1.005])
    i2 = ax2.imshow(gaussian_filter(derived, smooth),
                    interpolation='none', origin='lower', rasterized=True, vmin=0.995, vmax=1.005)
    plt.colorbar(i2, ax=ax2, orientation='horizontal', format='%.3f', ticks=[0.995, 1, 1.005])
    if ratio:
        i3 = ax3.imshow(derived/target,
                        interpolation='none', origin='lower', rasterized=True, vmin=0.995, vmax=1.005)
        plt.colorbar(i3, ax=ax3, orientation='horizontal', format='%.3f', ticks=[0.995, 1, 1.005])
    else:
        i3 = ax3.imshow(np.absolute(derived - target),
                        interpolation='none', origin='lower', rasterized=True, vmin=0., vmax=1e-3)
        plt.colorbar(i3, ax=ax3, orientation='horizontal', format='%.1e', ticks=[0., 5e-4, 1e-3])
    plt.savefig('PRNUmapRecovery%s%i.pdf' % (out, hide))
    plt.close()


def calculateAreaStatistcs(data, ylen=100, xlen=100):
    """
    Calculates a standard deviation in regions of a given size.
    """
    st = []
    ydim, xdim = data.shape
    samplesx = xdim / xlen
    samplesy = ydim / ylen
    #print samplesx, samplesy
    for a in range(samplesy):
        for b in range(samplesx):
            area = data[a*ylen:(a+1)*ylen, b*xlen:(b+1)*xlen]
            s = np.std(sigma_clip(area, 6.)) * 100.
            st.append(s)

    return st


def plotRecoveredResiduals(xmin=-0.0035, xmax=0.0035, ymax=0.065, nbins=60):

    six = g.glob('residualLR*.fits')
    three = g.glob('residualThreeLR*.fits')

    sixdata = []
    #data format: [(wave1, data1), (wave2, data2)...]
    for file in six:
        #sixdata.append((int(file[10:13]), pf.getdata(file))) #ratio data
        sixdata.append((int(file[10:13]),
                        pf.getdata(file.replace('residual', 'hidden')) -
                        pf.getdata(file.replace('residual', 'recovered'))))

    threedata = []
    for file in three:
        #threedata.append((int(file[15:18]), pf.getdata(file))) #ratio data
        threedata.append((int(file[15:18]),
                          pf.getdata(file.replace('residual', 'hidden')) -
                          pf.getdata(file.replace('residual', 'recovered'))))

    #simple plot showing some structures
    number_of_subplots = math.ceil(len(threedata))

    plt.figure(figsize=(13, 13))
    plt.subplots_adjust(wspace=0.01, hspace=0.2, left=0.05, right=0.99, top=0.95, bottom=0.05)

    bins = np.linspace(xmin, xmax, nbins)

    #loop over data from shortest wavelength to the longest
    i = 1
    pdfs = []
    for wave, data in sixdata:
        d = data.ravel()
        pdf = kde_statsmodels_u(d.astype(np.float64), bins, bandwidth=1.e-4) * 1.e-4 * 1.2
        pdfs.append(pdf)
        txt = r'Six LEDs: $\lambda =$ ' + str(wave) + 'nm'
        ax = plt.subplot(number_of_subplots, 2, i)
        ax.set_title(txt)
        ax.hist(d, bins=bins, weights=np.ones_like(d)/len(d))
        ax.plot(bins, pdf, 'r-', lw=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0., ymax)
        ax.axvline(x=0, color='g', lw=1.5)
        #ax.set_xticks([xmin+2e-3, 1, xmax-2e-3])
        plt.setp(ax.get_xticklabels(), visible=False)
        xx, locs = plt.xticks()
        ll = ['%.3f' % a for a in xx]
        plt.gca().xaxis.set_major_formatter(FixedFormatter(ll))
        ax.annotate(r'$\sigma(T-D) \sim %.2e$' % d.std(), xycoords='axes fraction', xy=(0.05, 0.85))

        i += 2

    plt.setp(ax.get_xticklabels(), visible=True)

    i = 2
    j = 0
    for wave, data in threedata:
        d = data.ravel()
        txt = r'Three LEDs: $\lambda =$ ' + str(wave) + 'nm'
        ax = plt.subplot(number_of_subplots, 2, i)
        ax.set_title(txt)
        ax.hist(d, bins=bins, weights=np.ones_like(d)/len(d))
        ax.plot(bins, pdfs[j], 'r-', lw=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0., ymax)
        ax.axvline(x=0, color='g', lw=1.5)
        #ax.set_xticks([xmin+2e-3, 1, xmax-2e-3])
        plt.setp(ax.get_xticklabels(), visible=False)
        xx, locs = plt.xticks()
        ll = ['%.3f' % a for a in xx]
        plt.gca().xaxis.set_major_formatter(FixedFormatter(ll))
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.annotate(r'$\sigma(T-D) \sim %.2e$' % d.std(), xycoords='axes fraction', xy=(0.05, 0.85))

        i += 2
        j += 1

    plt.setp(ax.get_xticklabels(), visible=True)

    plt.savefig('RecoveredResidualPixelValues.pdf')
    plt.close()


def _loadPRNUmaps(id='*FlatField.fits'):
    #load data
    data = {}
    for file in g.glob(id):
        fh = pf.open(file)
        wave = file[1:4]
        data[wave] = fh[1].data
        fh.close()
    return data


if __name__ == '__main__':
    #generate flats
    #flats()

    #plot generated flats
    #plot()

    #spatial autocorrelations
    #spatialAutocorrelation()

    #power spectrum analysis
    #powerSpectrum()

    #normalised cross-correlation of the PRNU maps
    #correlate()

    #try morphing flats to other wavelengths
    #morphPRNUmap()

    #structural parameters
    #structuralSimilarity()

    # #try to recover a PRNU map using information at other wavelengths
    # #using linear interpolation
    # for hide in [570, 600, 660, 700, 800]:
    #     print 'Hiding %inm, trying to recover using linear interpolation with' % hide
    #     d, t = interpolateMissing(hide=hide)
    #     plotMissing(t, d, hide)
    #     d, t = interpolateMissing(hide=hide, three=True)
    #     plotMissing(t, d, hide, out='Three')

    #try to recover a PRNU map using information at other wavelengths
    #using linear regression
    # for hide in [570, 600, 660, 700, 800]:
    #     print 'Hiding %inm, trying to recover using linear regression with' % hide
    #     d, t = predictMissingLinearRegression(hide=hide)
    #     plotMissing(t, d, hide, out='LR')
    #     d, t = predictMissingLinearRegression(hide=hide, three=True)
    #     plotMissing(t, d, hide, out='ThreeLR')

    # #try to recover a PRNU map using information at other wavelengths
    # #using gaussian process
    # for hide in [570, 600, 660, 700, 800]:
    #     print 'Hiding %inm, trying to recover using gaussian process with' % hide
    #     d, t = predictMissingGaussianProcessRegression(hide=hide)
    #     plotMissing(t, d, hide, out='GP')
    #     d, t = predictMissingGaussianProcessRegression(hide=hide, three=True)
    #     plotMissing(t, d, hide, out='ThreeGP')


    #plotMissing(pf.getdata('small/recovered700.fits'), pf.getdata('small/hidden700.fits'), 700, out='TEST')

    plotRecoveredResiduals()