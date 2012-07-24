"""
Bias Calibration
================

This simple script can be used to study the number of bias frames required for a given PSF ellipticity knowledge level.

The following requirements related to the bias calibration has been taken from GDPRD.

R-GDP-CAL-052:
The contribution of the residuals of VIS bias subtraction to the *error on the determination of each ellipticity
component* of the local PSF shall not exceed 3x10-5 (one sigma).

R-GDP-CAL-062:
The contribution of the residuals of VIS bias subtraction to the *relative error* \sigma(R2)/R2 on the determination of
the local PSF R2 shall not exceed 1x10-4 (one sigma).

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
import math, datetime, cPickle, itertools, shutil
from analysis import shape
from support import logger as lg
from support import surfaceFitting as sf
from support import bleedingtest as write
from support import files as fileIO


def testBiasCalibration(log, numdata=2066, floor=995, xsize=2048, ysize=2066, order=3, biases=15, surfaces=100,
                        file='psf1x.fits', psfs=500, psfscale=2.e4, escale=1.e5, sigma=0.75, R2scale=1.e4,
                        debug=False, plots=False):
    """
    Derive the PSF ellipticities for a given number of random surfaces with random PSF positions
    and a given number of biases median combined.

    Choices that need to be made and effect the results:

        #. bias surface that is assumed (amplitude, complexity, etc.)
        #. whether the order of the polynomial surface to be fitted is known or not
        #. size of the Gaussian weighting function when calculating the ellipticity components

    There are also other choices such as the number of PSFs and scaling and the random numbers generated for
    the surface that also affect the results, however, to a lesser degree.

    Generates a set of plots that can be used to inspect the simulation.
    """
    log.info('Processing file %s' % file)
    #read in data without noise or bias level and scale it to 20k electrons
    data = pf.getdata(file)
    data /= np.max(data)
    data *= psfscale

    #derive the reference value from the scaled data
    settings = dict(sigma=sigma)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    results = sh.measureRefinedEllipticity()
    #sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    #rescale to not loose numerical accuracy
    reference1 = results['e1'] * escale
    reference2 = results['e2'] * escale
    refR2 = results['R2'] * R2scale #/ 1.44264123086
    reference = math.sqrt(reference1*reference1 + reference2*reference2)

    print 'Reference Ellipticities [in %e] and R2 [in %e]:' % (escale, R2scale)
    print reference1, reference2, reference, refR2

    #generate a random quadrant surface representing BIAS without noise
    #modify zclean if a different order surface is needed
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize),
                         np.linspace(y.min(), y.max(), ysize))
    zclean = yy - xx + 0.78*xx**2 + 15.0*yy**2 - 1.75*xx*yy + 10.0*xx**3 + 0.3*yy**3 + floor

    #random positions for the PSFs, these positions are the lower corners
    xpositions = np.random.random_integers(0, zclean.shape[1] - data.shape[1], psfs)
    ypositions = np.random.random_integers(0, zclean.shape[0] - data.shape[0], psfs)

    # generate 2D plot
    im = plt.imshow(zclean, extent=(0, ysize, xsize, 0))
    plt.scatter(xpositions/data.shape[1]/2, ypositions/data.shape[0]/2)
    c1 = plt.colorbar(im)
    c1.set_label('BIAS [ADUs]')
    plt.xlim(0, ysize)
    plt.ylim(0, xsize)
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('NoNoise2D.png')
    plt.close()
    #and 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx*xsize, yy*ysize, zclean, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('BIAS [ADUs]')
    plt.savefig('NoNoise.png')
    plt.close()

    out = {}
    #number of biases to median combine
    for a in xrange(biases):
        print 'Number of Biases: %i / %i' % (a+1, biases)

        #data storage
        de1 = []
        de2 = []
        de = []
        R2 = []
        R2abs = []

        #number of random readnoised surfaces to loop over
        for b in xrange(surfaces):

            print 'Surface: %i / %i' % (b+1, surfaces)

            #add readout noise based on a+1 median combined biases
            z = addReadoutNoise(zclean.copy(), number=a+1)

            if plots:
                # generate 2D plot
                im = plt.imshow(z, extent=(0, ysize, xsize, 0))
                c1 = plt.colorbar(im)
                c1.set_label('BIAS [ADUs]')
                plt.xlim(0, ysize)
                plt.ylim(0, xsize)
                plt.xlabel('Y [pixels]')
                plt.ylabel('X [pixels]')
                plt.savefig('Readnoised%i%i.png' % (a+1, b+1))
                plt.close()
                #and 3D
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_surface(xx*xsize, yy*ysize, z, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_zlabel('BIAS [ADUs]')
                plt.savefig('Readnoised3D%i%i.png' % (a+1, b+1))
                plt.close()

            # Fit 2d polynomial to the noised data
            m = sf.polyfit2d(xx.ravel(), yy.ravel(), z.ravel(), order=order)
            # Evaluate it on a rectangular grid
            fitted = sf.polyval2d(xx, yy, m)

            if plots:
                # generate 2D plot
                im = plt.imshow(fitted, extent=(0, ysize, xsize, 0))
                c1 = plt.colorbar(im)
                c1.set_label('BIAS [ADUs]')
                plt.xlim(0, ysize)
                plt.ylim(0, xsize)
                plt.xlabel('Y [pixels]')
                plt.ylabel('X [pixels]')
                plt.savefig('Fitted2D%i.png' % (a+1))
                plt.close()
                #and 3D
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_surface(xx*xsize, yy*ysize, fitted, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_zlabel('BIAS [ADUs]')
                plt.savefig('Fitted3D%i.png' % (a+1))
                plt.close()

            #subtract the no noise surface from the fit to get the "knowledge" residual
            fitted -= zclean

            if plots:
                # generate 2D plot
                im = plt.imshow(fitted, extent=(0, ysize, xsize, 0))
                c1 = plt.colorbar(im)
                c1.set_label(r'$\Delta$BIAS [ADUs]')
                plt.xlim(0, ysize)
                plt.ylim(0, xsize)
                plt.xlabel('Y [pixels]')
                plt.ylabel('X [pixels]')
                plt.savefig('Residual2D%i%i.png' % (a+1, b+1))
                plt.close()
                #and 3D
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_surface(xx*xsize, yy*ysize, fitted, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_zlabel(r'$\Delta$BIAS [ADUs]')
                ax.set_zlim(-0.01, 0.01)
                plt.savefig('Residual3D%i%i.png' % (a+1, b+1))
                plt.close()

            #loop over the PSFs
            for xpos, ypos in zip(xpositions, ypositions):
                #measure e and R2 from the postage stamp image
                small = fitted[ypos:ypos+data.shape[0], xpos:xpos+data.shape[1]].copy()
                small += data.copy()
                sh = shape.shapeMeasurement(small.copy(), log, **settings)
                results = sh.measureRefinedEllipticity()

                #rescale the ellipticity components to escale
                e1 = results['e1'] * escale
                e2 = results['e2'] * escale
                R = results['R2'] * R2scale

                #save values
                de1.append(e1 - reference1) #absolute error of e1 component in escale units
                de2.append(e2 - reference2) #absolute error of e2 component in escale units
                de.append(math.sqrt(e1*e1 + e2*e2) - reference) #absolute error of e in escale units
                R2.append((R - refR2) / refR2) #relative error in R2scale units
                R2abs.append((R - refR2)) #absolute error in R2scale units

                if debug:
                    print xpos, ypos
                    write.writeFITSfile(small/data, 'testResidual.fits')
                    print 'DEBUG mode -- exiting now'
                    import sys; sys.exit()
        if plots:
            plotDeltaEs(de1, de2, de, 'MultipleBiases%i.pdf' % (a+1), title='%i Biases median combined' % (a+1))

        out[a+1] = [de1, de2, de, R2, R2abs]

    return out



def plotDeltaEs(deltae1, deltae2, deltae, output, title='', ymax=8, req=3):
    """
    Generates a simple plot showing the errors in the ellipticity components.
    """
    deltae1 = np.asarray(deltae1)
    deltae2 = np.asarray(deltae2)
    deltae = np.asarray(deltae)

    #plot histograms
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)

    bins = np.arange(-6, 6.1, 0.1)
    ax.hist(deltae, bins=bins, label=r'$e$', alpha=0.3, normed=False)
    ax.hist(deltae1, bins=bins, label=r'$e_{1}$', alpha=0.2, normed=False)
    ax.hist(deltae2, bins=bins, label=r'$e_{2}$', alpha=0.1, normed=False)
    ax.axvline(x=req, c='g', ls='--', label='Requirement')
    ax.axvline(x=-req, c='g', ls='--')
    ax.set_xlim(-6, 6)

    ax.set_xlabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')
    ax.set_ylabel('Probability Density')

    plt.legend(shadow=True, fancybox=True)
    plt.savefig('hist' + output)
    plt.close()

    #make scatter plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)

    ax.plot(deltae, 'mD', label=r'$e$')
    ax.plot(deltae2, 'ys', label=r'$e_{2}$')
    ax.plot(deltae1, 'bo', label=r'$e_{1}$')
    ax.fill_between(np.arange(len(deltae1)), np.ones(len(deltae1))*req, ymax, facecolor='red', alpha=0.08)
    ax.fill_between(np.arange(len(deltae1)), -np.ones(len(deltae1))*req, -ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')
    ax.axhline(y=-req, c='g', ls='--')

    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.text(0.5, 0.1,
             r'Average error in $e_{1}=$ %f and $e_{2}=$ %f' % (np.mean(deltae1), np.mean(deltae2)),
             ha='center',
             va='center',
             transform=ax.transAxes)

    plt.legend(shadow=True, fancybox=True, numpoints=1, ncol=2)
    plt.savefig(output)
    plt.close()


def plotNumberOfFrames(results, req=3, ymax=5.1, sigmas=3.0, shift=0.1):
    """
    Creates a simple plot to combine and show the results.

    :param results: results to be plotted
    :type results: dict
    :param req: the requirement
    :type req: float
    :param ymax: maximum value to show on the y-axis
    :type ymax: int or float
    :param sigmas: scaling of the error bars, this many sigmas
    :type sigmas: float
    :param shift: the amount to shift the e2 results on the abscissa (for clarity)
    :type shift: float
    """
    R4 = 1.44264123086 ** 2

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    plt.title(r'VIS Bias Calibration: $\Delta e$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    x = 1
    std1s = []
    std2s = []
    stds = []
    files = []
    #loop over the number of bias frames combined
    for key in results:
        de1 = np.asarray(results[key][0])
        de2 = np.asarray(results[key][1])
        de = np.asarray(results[key][2])

        std1 = np.std(np.abs(de1))
        std2 = np.std(np.abs(de2))
        std = np.std(np.abs(de))

        files.append(key)
        std1s.append(std1**2)
        std2s.append(std2**2)
        stds.append(std**2)

        std1 *= sigmas
        std2 *= sigmas
        std *= sigmas

        avg1 = np.mean(de1)
        avg2 = np.mean(de2)
        avg = np.mean(de)
        max1 = np.max(de1)
        max2 = np.max(de2)
        max = np.max(de)
        min1 = np.min(de1)
        min2 = np.min(de2)
        min = np.min(de)

        print key, avg1**2, avg2**2, avg**2

        ax.errorbar(key, avg, yerr=std, c='m', elinewidth=1.5, marker='D')
        ax.errorbar(key, avg1, yerr=std1, c='b', elinewidth=1.5, marker='o')
        ax.errorbar(key-shift, avg2, yerr=std2, c='y', elinewidth=1.5, marker='s')
        ax.scatter(key, max, c='m', marker='*', s=30, zorder=10)
        ax.scatter(key, max1, c='b', marker='*', s=30, zorder=10)
        ax.scatter(key-shift, max2, c='y', marker='*', s=30, zorder=10)
        ax.scatter(key, min, c='m', marker='*', s=30, zorder=10)
        ax.scatter(key, min1, c='b', marker='*', s=30, zorder=10)
        ax.scatter(key-shift, min2, c='y', marker='*', s=30, zorder=10)

        x += 1

    #for the legend
    ax.scatter(key, max, c='m', marker='*', label=r'Max$(\Delta e)$')
    ax.scatter(key, max1, c='b', marker='*', label=r'Max/Min$(\Delta e_{1})$')
    ax.scatter(key-shift, max2, c='y', marker='*', label=r'Max/Min$(\Delta e_{2})$')
    ax.errorbar(key, avg, yerr=max, c='m', elinewidth=1.5, marker='D', visible=False, label=r'$e %i \sigma$' % sigmas)
    ax.errorbar(key, avg1, yerr=max1, c='b', elinewidth=1.5, marker='o', visible=False, label=r'$e_{1} %i \sigma$' % sigmas)
    ax.errorbar(key-shift, avg2, yerr=max2, c='y', elinewidth=1.5, marker='s', visible=False, label=r'$e_{2} %i \sigma$' % sigmas)

    #show the requirement
    ax.fill_between(np.arange(x+1), np.ones(x+1)*req, ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')
    ax.fill_between(np.arange(x+1), -np.ones(x+1)*req, -ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=-req, c='g', ls='--')

    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
    plt.savefig('BiasCalibrationEtmp.pdf')
    plt.close()

    #Results for Es
    files = np.asarray(files)
    stds = np.asarray(stds)**2
    std1s = np.asarray(std1s)**2
    std2s = np.asarray(std2s)**2

    fig = plt.figure()
    plt.title(r'VIS Bias Calibration: $\sigma^{2}(e)$')
    ax = fig.add_subplot(111)

    ax.scatter(files-shift, stds, c='m', marker='*', label=r'$\sigma^{2}(e)$')
    ax.scatter(files, std1s, c='b', marker='o', label=r'$\sigma^{2}(e_{1})$')
    ax.scatter(files, std2s, c='y', marker='s', label=r'$\sigma^{2}(e_{2})$')


    ax.fill_between(np.arange(x+1), np.ones(x+1)*req, ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')

    ax.set_ylim(-0.01, 4.0)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\sigma^{2}(e_{i})\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
    plt.savefig('BiasCalibrationsigmaE.pdf')
    plt.close()


    #same for R2s
    fig = plt.figure()
    plt.title(r'VIS Bias Calibration: $\frac{\sigma(R^{2})}{R_{ref}^{4}}$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    x = 1
    #loop over the number of bias frames combined
    for key in results:
        dR2 = np.asarray(results[key][4])

        std = np.std(np.abs(dR2))**2 / np.mean(np.abs(dR2))**4

        ax.scatter(key, std, c='m', marker='*', s=35, zorder=10)

        x += 1

    #for the legend
    ax.scatter(key, std, c='m', marker='*', label=r'$\sigma^{2} (R^{2})$')

    #show the requirement
    ax.fill_between(np.arange(x+1), np.ones(x+1), ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=1, c='g', ls='--', label='Requirement')

    ax.set_ylim(-0.01, ymax)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\frac{\sigma(R^{2})}{R_{ref}^{4}} \ \ \ \ [10^{-4}]$')

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8    )
    plt.savefig('BiasCalibrationSigmaR2.pdf')
    plt.close()


def addReadoutNoise(data, readnoise=4.5, number=1):
    """
    Add readout noise to the input data. The readout noise is the median of the number of frames.

    :param data: input data to which the readout noise will be added to
    :type data: ndarray
    :param readnoise: standard deviation of the read out noise [electrons]
    :type readnoise: float
    :param number: number of read outs to median combine before adding to the data
    :type number: int

    :return: data + read out noise
    :rtype: ndarray [same as input data]
    """
    shape = data.shape
    biases = np.random.normal(loc=0.0, scale=math.sqrt(readnoise), size=(shape[0], shape[1], number))
    bias = np.median(biases.astype(np.int), axis=2, overwrite_input=True)
    return data + bias


if __name__ == '__main__':
    run = True

    #start the script
    log = lg.setUpLogger('biasCalibration.log')
    log.info('Testing bias level calibration...')

    if run:
        results = testBiasCalibration(log, biases=10, psfs=100, surfaces=100, plots=False, file='psf1xhighe.fits')
        #results = testBiasCalibration(log, biases=5, psfs=10, surfaces=10, plots=False, file='psf1xsamees.fits')
        fileIO.cPickleDumpDictionary(results, 'biasResults.pk')
    else:
        results = cPickle.load(open('biasResults.pk'))

    plotNumberOfFrames(results)

    log.info('Run finished...\n\n\n')
