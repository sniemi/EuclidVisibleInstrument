"""
A simple script to analyse lab PRNU data.

Currently uses the ESA CCD273 data.
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
import pyfits as pf
import numpy as np
import math
import glob as g
from support import files as fileIO
from astropy.modeling import models, fitting
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import sigma_clip


def loadData(wavelow=500, wavehigh=910):
    """
    Load data from flat1*.fits.
    Picks the wavelengths within the given wavelength range.
    Chooses files with mean < 35k to avoid images with saturated areas.

    :return: dictionary of data with wavelengths as keys
    :rtype: dict
    """
    out = {}

    for file in g.glob('flat1*.fits'):
        fh = pf.open(file)
        hdr = fh[0].header
        wave = float(hdr['WAVELEN'])

        if  wave > wavelow and wave < wavehigh and float(hdr['MEAN']) < 35000:
            print file, wave
            out[wave] = fh[0].data

    return out


def plotPRNU(data, xmin=700, xmax=1700, ymin=3500, ymax=4500, order=3, smooth=3.):
    """
    Generate and plot maps of PRNU, ratios of PRNU maps, and show the PRNU as a function of wavelength.
    """
    number_of_subplots = math.ceil(len(data.keys())/2.)

    plt.figure(figsize=(8, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

    #loop over data from shortest wavelength to the longest
    w = []
    prnu = []
    dat = {}
    for i, wave in enumerate(sorted(data.keys())):

        tmp = data[wave][ymin:ymax, xmin:xmax]
        tmp /= np.median(tmp)

        #meshgrid representing data
        x, y = np.mgrid[:tmp.shape[1], :tmp.shape[0]]

        #fit a polynomial 2d surface to remove the illumination profile
        p_init = models.Polynomial2D(degree=order)
        f = fitting.NonLinearLSQFitter()
        p = f(p_init, x, y, tmp)

        #normalize data
        tmp /= p(x, y)

        print tmp.max(), tmp.min()

        #sigma clipped std to reject dead pixels etc.
        prn = np.std(sigma_clip(tmp, 6.)) * 100.

        print 'PRNU:', prn, 'per cent at lambda =', wave, 'nm'
        w.append(int(wave))
        prnu.append(prn)

        if int(wave) > 750 and int(wave) < 850:
            reference = tmp.copy()
        dat[wave] = tmp.copy()

        #Gaussian smooth to enhance structures for plotting
        tmp = gaussian_filter(tmp, smooth)

        ax = plt.subplot(number_of_subplots, 2, i+1)
        ax.imshow(tmp, interpolation='none', origin='lower',
                  vmin=0.995, vmax=1.003)

        ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm')

        plt.axis('off')

        #write to file
        fileIO.writeFITS(tmp, str(int(wave))+'.fits', int=False)

    plt.savefig('PRNUmaps.png')
    plt.close()

    #ratio images
    plt.figure(figsize=(8, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

    for i, wave in enumerate(sorted(dat.keys())):
        tmp = dat[wave]

        #divide by the reference
        tmp /= reference

        #Gaussian smooth to enhance structures for plotting
        tmp = gaussian_filter(tmp, smooth)

        ax = plt.subplot(number_of_subplots, 2, i+1)
        ax.imshow(tmp, interpolation='none', origin='lower',
                  vmin=0.995, vmax=1.003)

        ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm')

        plt.axis('off')

    plt.savefig('PRNURationmaps.png')
    plt.close()


def plotWavelengthDependency(data, xmin=700, xmax=1700, ymin=3500, ymax=4500, order=3, samples=10):

    #loop over data from shortest wavelength to the longest
    dat = {}
    for i, wave in enumerate(sorted(data.keys())):

        tmp = data[wave][ymin:ymax, xmin:xmax]
        tmp /= np.median(tmp)

        #meshgrid representing data
        x, y = np.mgrid[:tmp.shape[1], :tmp.shape[0]]

        #fit a polynomial 2d surface to remove the illumination profile
        p_init = models.Polynomial2D(degree=order)
        f = fitting.NonLinearLSQFitter()
        p = f(p_init, x, y, tmp)

        #normalize data
        tmp /= p(x, y)

        #select sub regions to calculate the PRNU in
        prnu = []
        ydim, xdim = tmp.shape
        ylen = ydim / samples
        xlen = xdim / samples
        for a in range(samples):
            for b in range(samples):
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

    #polynomial fit to PRNU datra
    z2 = np.polyfit(w, mean, 2)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(w, mean, 3)
    p3 = np.poly1d(z3)
    x = np.linspace(500, 950)

    #wavelength dependency plot
    plt.title('Wavelength Dependency of the PRNU')
    plt.plot(x, p2(x), 'b-', label='2nd order fit')
    plt.plot(x, p3(x), 'g--', label='3rd order fit')
    plt.errorbar(w, mean, yerr=3*np.asarray(std), c='r', fmt='o', label='data, $3\sigma$ errors')
    plt.xlabel(r'Wavelength $\lambda$ [nm]')
    plt.ylabel(r'PRNU $[\%]$')
    plt.xlim(500, 950)
    plt.ylim(0.4, 0.8)
    plt.legend(shadow=True, fancybox=True, scatterpoints=1, numpoints=1)
    plt.savefig('PRNUwave.pdf')
    plt.close()


if __name__ == '__main__':
        data = loadData()

        #plotPRNU(data)
        plotWavelengthDependency(data)