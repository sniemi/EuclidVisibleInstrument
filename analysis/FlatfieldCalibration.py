"""
Flat Field Calibration
======================

This simple script can be used to study the number of flat fields required to meet the VIS calibration requirements.

The following requirements related to the bias calibration has been taken from GDPRD.

R-GDP-CAL-0:


R-GDP-CAL-0:


:requires: PyFITS
:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools, glob
from analysis import shape
from support import logger as lg
from support import surfaceFitting as sf
from support import files as fileIO


def generateResidualFlatField(files='Q0*flatfield*.fits', combine=77, lampfile='data/VIScalibrationUnitflux.fits',
                              reference='data/VISFlatField2percent.fits', plots=False, debug=False):
    """

    """
    #find all FITS files
    files = glob.glob(files)

    #choose randomly the files that should be combined
    if combine < len(files):
        ids = np.random.random_integers(0, len(files)-1, combine)
        files = np.asarray(files)[ids]

    #load data and scale to electrons
    data = fileIO.readFITSDataExcludeScanRegions(files)
    data *= 3.5

    #check that the sizes match and median combine
    if len(set(x.shape for x in data))  > 1:
        print 'ERROR -- files are not the same shape, cannot median combine'
    else:
        #median combine
        median = np.median(data, axis=0)

    #fit surface to the median and normalize it out
    #m = sf.polyfit2d(xx.ravel(), yy.ravel(), median.ravel(), order=order)
    # Evaluate it on a rectangular grid
    #fitted = sf.polyval2d(xx, yy, m)

    #load the lamp profile that went in and divide the combined image with the profile
    lamp = pf.getdata(lampfile)
    pixvar = median / lamp

    #load the true reference p-flat and divide the derived flat with it
    real = pf.getdata(reference)
    res = pixvar / real

    if debug:
        fileIO.writeFITS(res, 'residualFlat.fits')
        print np.mean(res), np.min(res), np.max(res), np.std(res)

    if plots:
        #generate a mesh grid fo plotting
        ysize, xsize = median.shape
        xx, yy = np.meshgrid(np.linspace(0, xsize, xsize), np.linspace(0, ysize, ysize))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, median, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel('Flat Field Counts [electrons]')
        plt.savefig('MedianFlat.png')
        plt.close()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, pixvar, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel('Flat Field Counts [electrons]')
        plt.savefig('PixelFlat.png')
        plt.close()

        im = plt.imshow(res, origin='lower')
        c1 = plt.colorbar(im)
        c1.set_label('Derived / Input')
        plt.xlabel('Y [pixels]')
        plt.ylabel('X [pixels]')
        plt.savefig('ResidualFlatField2D.png')
        plt.close()

    return res


def testFlatCalibrationSigma(log, flats=100, surfaces=100,
                             file='data/psf1x.fits', psfs=500, psfscalemin=1.e2, psfscalemax=1.e3,
                             sigma=0.75, plot=False, debug=False):
    """
    Derive the PSF ellipticities for a given number of random surfaces with random PSF positions
    and a given number of flat fields median combined.

    This function is to derive the the actual values so that the knowledge (variance) can be studied.

    """
    #set shape measurement settings
    settings = dict(sigma=sigma)

    #read in PSF and renormalize it
    data = pf.getdata(file)
    data /= np.max(data)

    #random positions for the PSFs, these positions are the lower corners
    #assume that this is done on quadrant level thus the xmax and ymax are 2065 and 2047, respectively
    xpositions = np.random.random_integers(0, 2047 - data.shape[1], psfs)
    ypositions = np.random.random_integers(0, 2065 - data.shape[0], psfs)
    #random scalings for the PSFs
    psfscales = np.random.rand(psfs) * (psfscalemax - psfscalemin) + psfscalemin

    out = {}
    #number of biases to median combine
    for a in np.arange(2, flats, 1):
        print 'Number of Flats to combine: %i / %i' % (a, flats)

        #data storage
        de1 = []
        de2 = []
        de = []
        R2 = []
        R2abs = []

        for b in xrange(surfaces):
            print 'Random Realisations: %i / %i' % (b+1, surfaces)

            residual = generateResidualFlatField(combine=a, plots=plot, debug=debug)

            # generate 2D plot
            if b == 0 and plot:
                im = plt.imshow(residual, extent=(0, 2066, 2048, 0))
                plt.scatter(xpositions + (data.shape[1]/2), ypositions + (data.shape[0]/2), color='white')
                c1 = plt.colorbar(im)
                c1.set_label('Residual Flat Field')
                plt.xlim(0, 2066)
                plt.ylim(0, 2048)
                plt.xlabel('Y [pixels]')
                plt.ylabel('X [pixels]')
                plt.savefig('residualFlat2D%i.png' % a)
                plt.close()

            #loop over the PSFs
            for xpos, ypos, scale in zip(xpositions, ypositions, psfscales):
                #scale the PSF
                tmp = data.copy() * scale

                #get the underlying residual surface ond multiple the PSF with the surface
                small = residual[ypos:ypos+data.shape[0], xpos:xpos+data.shape[1]].copy()
                small *= tmp

                #measure e and R2 from the postage stamp image
                sh = shape.shapeMeasurement(small.copy(), log, **settings)
                results = sh.measureRefinedEllipticity()

                #save values
                de1.append(results['e1'])
                de2.append(results['e2'])
                de.append(math.sqrt(results['e1']*results['e1'] + results['e2']*results['e2']))
                R2.append(results['R2'])
                R2abs.append(results['R2'])

        out[a+1] = [de1, de2, de, R2, R2abs]

    return out


def plotNumberOfFramesSigma(results, reqe=3e-5, reqr2=1e-4, shift=0.1):
    """
    Creates a simple plot to combine and show the results.

    :param results: results to be plotted
    :type results: dict
    :param req: the requirement
    :type req: float
    :param ymax: maximum value to show on the y-axis
    :type ymax: int or float
    :param shift: the amount to shift the e2 results on the abscissa (for clarity)
    :type shift: float
    """
    print '\nSigma results:'
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    plt.title(r'VIS Flat Field Calibration: $\sigma^{2}(e)$')
    ax = fig.add_subplot(111)

    x = 1
    #loop over the number of bias frames combined
    for key in results:
        e1 = np.asarray(results[key][0])
        e2 = np.asarray(results[key][1])
        e = np.asarray(results[key][2])

        var1 = np.var(e1)
        var2 = np.var(e2)
        var = np.var(e)

        ax.scatter(key-shift, var, c='m', marker='*')
        ax.scatter(key, var1, c='b', marker='o')
        ax.scatter(key, var2, c='y', marker='s')

        x += 1

        print key, var, var1, var2


    ax.scatter(key-shift, var, c='m', marker='*', label=r'$\sigma^{2}(e)$')
    ax.scatter(key, var1, c='b', marker='o', label=r'$\sigma^{2}(e_{1})$')
    ax.scatter(key, var2, c='y', marker='s', label=r'$\sigma^{2}(e_{2})$')

    ax.fill_between(np.arange(x+1), np.ones(x+1)*reqe, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqe, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1e-4)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Flat Fields Median Combined')
    ax.set_ylabel(r'$\sigma^{2}(e_{i})\ , \ \ \ i \in [1,2]$')

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
    plt.savefig('FlatCalibrationsigmaE.pdf')
    plt.close()

    #same for R2s
    R4 = 1.44264123086 ** 4

    fig = plt.figure()
    plt.title(r'VIS Flat Field Calibration: $\frac{\sigma^{2}(R^{2})}{R_{ref}^{4}}$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    x = 1
    #loop over the number of bias frames combined
    for key in results:
        dR2 = np.asarray(results[key][4])

        std = np.std(dR2) / (5.06722858929**4) #/ R4
        var = np.var(dR2) / (5.06722858929**4) #/ R4

        print key, var, std

        #ax.scatter(key, std, c='m', marker='*', s=35, zorder=10)
        ax.scatter(key, var, c='b', marker='s', s=35, zorder=10)

        x += 1

    #for the legend
    #ax.scatter(key, std, c='m', marker='*', label=r'$\frac{\sigma(R^{2})}{R_{ref}^{4}}$')
    ax.scatter(key, var, c='b', marker='s', label=r'$\frac{\sigma^{2}(R^{2})}{R_{ref}^{4}}$')

    #show the requirement
    ax.fill_between(np.arange(x+1), np.ones(x+1)*reqr2, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqr2, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1e-3)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Flat Fields Median Combined')
    ax.set_ylabel(r'$\frac{\sigma^{2}(R^{2})}{R_{ref}^{4}}$')

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8    )
    plt.savefig('FlatCalibrationSigmaR2.pdf')
    plt.close()


if __name__ == '__main__':
    run = True
    debug = False

    #start the script
    log = lg.setUpLogger('flatfieldCalibration.log')
    log.info('Testing flat fielding calibration...')

    if run:
        resultsSigma = testFlatCalibrationSigma(log, flats=75, surfaces=100, psfs=1000)
        fileIO.cPickleDumpDictionary(resultsSigma, 'flatfieldResultsSigma.pk')
        if debug:
            residual = generateResidualFlatField()
            fileIO.cPickleDumpDictionary(residual, 'residual.pk')
    else:
        #resultsDelta = cPickle.load(open('flatfieldResultsDelta.pk'))
        resultsSigma = cPickle.load(open('flatfieldResultsSigma.pk'))

    plotNumberOfFramesSigma(resultsSigma)
    #plotNumberOfFramesDelta(resultsDelta)

    log.info('Run finished...\n\n\n')
