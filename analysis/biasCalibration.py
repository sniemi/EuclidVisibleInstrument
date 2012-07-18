"""
This simple script can be used to study the number of bias frames required for a given PSF ellipticity knowledge level.

:requires: PyFITS
:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools
from analysis import shape
from support import logger as lg
from support import surfaceFitting as sf
from support import bleedingtest as write


def testBiasCalibration(log, numdata=2066, floor=995, xsize=2048, ysize=2066, order=3, biases=10,
                        file='psf1x.fits', psfs=100, psfscale=2e4, sigma=0.75):
    """
    Derive the PSF ellipticities for a random surface with random PSF positions
    and a given number of biases median combined.

    Choices that need to be made and effect the results:

        #. bias surface that is assumed (amplitude, complexity, etc.)
        #. whether the order of the polynomial surface to be fitted is known or not
        #. the scaling of the PSF that is being placed on the detector
        #. size of the Gaussian weighting funtion when calculating the ellipticity components

    There are also other choiced such as the number of PSFs and the random numbers generated for
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
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    reference1 = results['e1']
    reference2 = results['e2']

    print 'Reference Ellipticities'
    print reference1, reference2

    #generate a random quadrant surface representing BIAS without noise
    #modify zclean if a different order surface is needed
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize),
                         np.linspace(y.min(), y.max(), ysize))
    zclean = yy - xx + 0.78*xx**2 + 15.0*yy**2 - 1.75*xx*yy + 10.0*xx**3 + 0.3*yy**3 + floor

    # generate 2D plot
    im = plt.imshow(zclean, extent=(0, ysize, xsize, 0))
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
    for a in xrange(biases):
        print '%i / %i' % (a+1, biases)

        #add readout noise based on a+1 median combined biases
        z = addReadoutNoise(zclean.copy(), number=a+1)

        # generate 2D plot
        im = plt.imshow(z, extent=(0, ysize, xsize, 0))
        c1 = plt.colorbar(im)
        c1.set_label('BIAS [ADUs]')
        plt.xlim(0, ysize)
        plt.ylim(0, xsize)
        plt.xlabel('Y [pixels]')
        plt.ylabel('X [pixels]')
        plt.savefig('Readnoised%i.png' % (a+1))
        plt.close()
        #and 3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx*xsize, yy*ysize, z, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel('BIAS [ADUs]')
        plt.savefig('Readnoised3D%i.png' % (a+1))
        plt.close()

        # Fit 2d polynomial to the noised data
        m = sf.polyfit2d(xx.ravel(), yy.ravel(), z.ravel(), order=order)
        # Evaluate it on a rectangular grid
        fitted = sf.polyval2d(xx, yy, m)

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

        #random positions for the PSFs
        xpositions = np.random.random_integers(0, np.min(fitted.shape)-np.max(data.shape), psfs)
        ypositions = np.random.random_integers(0, np.min(fitted.shape)-np.max(data.shape), psfs)

        #subtract the no noise surface from the fit to get the "knowledge" residual
        fitted -= zclean

        # generate 2D plot
        im = plt.imshow(fitted, extent=(0, ysize, xsize, 0))
        c1 = plt.colorbar(im)
        c1.set_label(r'$\Delta$BIAS [ADUs]')
        plt.xlim(0, ysize)
        plt.ylim(0, xsize)
        plt.xlabel('Y [pixels]')
        plt.ylabel('X [pixels]')
        plt.savefig('Residual2D%i.png' % (a+1))
        plt.close()
        #and 3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx*xsize, yy*ysize, fitted, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel(r'$\Delta$BIAS [ADUs]')
        ax.set_zlim(-0.01, 0.01)
        plt.savefig('Residual3D%i.png' % (a+1))
        plt.close()

        #loop over the PSFs
        de1 = []
        de2 = []
        for xpos, ypos in zip(xpositions, ypositions):
            small = fitted[ypos:ypos+data.shape[0], xpos:xpos+data.shape[1]].copy()
            small += data.copy()
            sh = shape.shapeMeasurement(small.copy(), log, **settings)
            results = sh.measureRefinedEllipticity()
            de1.append(results['e1'] - reference1)
            de2.append(results['e2'] - reference2)

            #print xpos, ypos
            #write.writeFITSfile(small/data, 'testResidual.fits')
            #import sys; sys.exit()

        plotDeltaEs(de1, de2, 'MultipleBiases%i.pdf' % (a+1), title='%i Biases median combined' % (a+1))
        out[a+1] = [de1, de2]

    return out


def plotDeltaEs(deltae1, deltae2, output, title='', ymax=15, req=3):
    """
    Generates a simple plot showing the errors in the ellipticity components.
    """
    deltae1 = np.abs(np.asarray(deltae1) * 1e5)
    deltae2 = np.abs(np.asarray(deltae2) * 1e5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)

    ax.plot(deltae1, 'bo', label=r'$e_{1}$')
    ax.plot(deltae2, 'ys', label=r'$e_{2}$')
    ax.fill_between(np.arange(len(deltae1)), np.ones(len(deltae1))*req, ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')

    ax.set_ylim(0, ymax)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.text(0.5, 0.4,
             r'Average error in $e_{1}=$ %f and $e_{2}=$ %f' % (np.mean(deltae1), np.mean(deltae2)),
             ha='center',
             va='center',
             transform=ax.transAxes)

    plt.legend(shadow=True, fancybox=True, numpoints=1)
    plt.savefig(output)
    plt.close()


def plotNumberOfFrames(results, req=3, ymax=7, sigmas=5.0, scaling=1e5, shift=0.1):
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
    :param scaling: the factor to be used to scale the input data
    :type scaling: float
    :param shift: the amount to shift the e2 results on the abscissa (for clarity)
    :type shift: float
    """
    fig = plt.figure()
    plt.title('VIS Bias Calibration (%s)' % datetime.datetime.isoformat(datetime.datetime.now()))
    ax = fig.add_subplot(111)

    x = 1
    #loop over the number of bias frames combined
    for key in results:
        de1 = np.abs(np.asarray(results[key][0]) * scaling)
        de2 = np.abs(np.asarray(results[key][1]) * scaling)

        avg1 = np.mean(de1)
        avg2 = np.mean(de2)
        std1 = np.std(de1) * sigmas
        std2 = np.std(de2) * sigmas
        max1 = np.max(de1)
        max2 = np.max(de2)

        ax.errorbar(key, avg1, yerr=std1, c='b', elinewidth=1.5, marker='o')
        ax.errorbar(key-shift, avg2, yerr=std2, c='y', elinewidth=1.5, marker='s')
        ax.scatter(key, max1, c='b', marker='*', s=30, zorder=10)
        ax.scatter(key-shift, max2, c='y', marker='*', s=30, zorder=10)

        x += 1

    #for the legend
    ax.scatter(key, max1, c='b', marker='*', label=r'Max$(\Delta e_{1})$')
    ax.scatter(key-shift,max2, c='y', marker='*', label=r'Max$(\Delta e_{2})$')
    ax.errorbar(key, avg1, yerr=max1, c='b', elinewidth=1.5, marker='o', visible=False, label=r'$e_{1} %i \sigma$' % sigmas)
    ax.errorbar(key-shift, avg2, yerr=max2, c='y', elinewidth=1.5, marker='s', visible=False, label=r'$e_{2} %i \sigma$' % sigmas)

    #show the requirement
    ax.fill_between(np.arange(x+1), np.ones(x+1)*req, ymax, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')

    ax.set_ylim(-1e-7, ymax)
    ax.set_xlim(0, x)
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0)
    plt.savefig('BiasCalibration.pdf')


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


def cPickleDumpDictionary(dictionary, output):
    """
    Dumps a dictionary of data to a cPickled file.

    :param dictionary: a Python data container does not have to be a dictionary
    :param output: name of the output file

    :return: None
    """
    out = open(output, 'wb')
    cPickle.dump(dictionary, out)
    out.close()


if __name__ == '__main__':
    #start the script
    log = lg.setUpLogger('biasCalibration.log')
    log.info('Testing bias level calibration...')

    #multiple Biases, single surface
    results = testBiasCalibration(log, biases=15, psfs=500)
    cPickleDumpDictionary(results, 'biasResults.pk')
    results = cPickle.load(open('biasResults.pk'))
    plotNumberOfFrames(results)