"""
Impact of Ghost Images
======================

This scripts can be used to study the impact of ghost images on the weak lensing measurements.

:requires: PyFITS
:requires: NumPy (1.7.0 or newer for numpy.pad)
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.2

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
import pyfits as pf
import numpy as np
import pprint, cPickle, os, glob, shutil
from scipy import interpolate
from analysis import shape
from support import logger as lg
from support import files as fileIO


def drawDoughnut(inner, outer, oversample=1, plot=False):
    """
    Draws a doughnut shape with a given inner and outer radius.

    :param inner: inner radius in pixels
    :type inner: float
    :param outer: outer radius in pixels
    :type outer: float
    :param oversample: oversampling factor [default=1]
    :type oversample: int
    :param plot: whether to generate a plot showing the doughnut or not
    :type plot: bool

    :return: image, xcentre, ycentre
    :rtype: list
    """
    canvas = np.zeros((int(np.ceil(outer*2.05)*oversample), int(np.ceil(outer*2.05)*oversample)))
    indy, indx = np.indices(canvas.shape)
    y, x = canvas.shape
    indx -= x/2
    indy -= y/2
    incut = np.sqrt(indx**2 + indy**2) <= inner*oversample
    largecircle = np.sqrt(indx**2 + indy**2) <= outer*oversample
    canvas[largecircle] = 1.
    canvas[incut] = 0.0

    if plot:
        fig = plt.figure()
        plt.title('Doughnut Drawing')
        ax = fig.add_subplot(111)
        ax.imshow(canvas, origin='lower', interpolation='none', rasterized=True, vmin=0, vmax=1)
        plt.savefig('doughnut.pdf')
        plt.close()

    return canvas, x/2, y/2


def analyseInFocusImpact(log, filename='data/psf4x.fits', psfscale=100000, maxdistance=100,
                         oversample=4.0, psfs=1000, iterations=6, sigma=0.75):
    """
    Calculates PSF size and ellipticity when including another PSF scaled to a given level (requirement = 5e-5)

    :param log:
    :param filename: name of the PSF file to analyse
    :param psfscale: level to which the original PSF is scaled to
    :param maxdistance: maximum distance the ghost image can be from the original PSF (centre to centre)
    :param oversample: oversampling factor
    :param psfs: number of PSFs to analyse (number of ghosts in random locations)
    :param iterations: number of iterations in the shape measurement
    :param sigma: size of the Gaussian weighting function

    :return: results
    :rtype: dict
    """
    #read in PSF and renormalize it
    data = pf.getdata(filename)
    data /= np.max(data)

    #place it a larger canvas with zero padding around
    ys, xs = data.shape
    yd = int(np.round(ys/2., 0))
    xd = int(np.round(xs/2., 0))
    canvas = np.pad(data, xs+maxdistance, mode='constant', constant_values=0)  #requires numpy >= 1.7.0
    ys, xs = canvas.shape
    xcen = int(np.round(xs/2., 0))
    ycen = int(np.round(ys/2., 0))
    #print canvas.shape
    #print canvas.flags

    canvas /= np.max(canvas)
    canvas *= float(psfscale)

    #set sampling etc. for shape measurement
    settings = dict(sampling=1.0 / oversample, itereations=iterations, sigma=sigma)

    #positions
    x = np.round((np.random.rand(psfs)-0.5)*maxdistance, 0).astype(np.int)
    y = np.round((np.random.rand(psfs)-0.5)*maxdistance, 0).astype(np.int)

    #ghost level
    ghosts = np.logspace(-7, -4, 10)[::-1] #largest first
    tot = ghosts.size
    res = {}
    for i, scale in enumerate(ghosts):
        print'ghost levels: %i / %i' % (i + 1, tot)
        R2 = []
        e1 = []
        e2 = []
        e = []
        scaled = data.copy() * (scale * psfscale)

        #loop over the ghost positions
        for xc, yc in zip(x, y):
            tmp = canvas.copy()
            xm = xcen + xc
            ym = ycen + yc
            try:
                tmp[ym-yd:ym+yd+1, xm-xd:xm+xd+1] += scaled
            except:
                try:
                    tmp[ym-yd:ym+yd, xm-xd:xm+xd] += scaled
                except:
                    print scaled.shape
                    print tmp[ym-yd:ym+yd+1, xm-xd:xm+xd+1].shape
                    print 'ERROR -- cannot place the ghost to the image!!'
                    continue

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(tmp, log, **settings)
            results = sh.measureRefinedEllipticity()

            #save values
            e1.append(results['e1'])
            e2.append(results['e2'])
            e.append(results['ellipticity'])
            R2.append(results['R2'])

        out = dict(e1=np.asarray(e1), e2=np.asarray(e2), e=np.asarray(e), R2=np.asarray(R2))
        res[scale] = out

    return res


def analyseOutofFocusImpact(log, filename='data/psf4x.fits', psfscale=100000, maxdistance=100,
                            inner=8, outer=60, oversample=4.0, psfs=5000, iterations=5, sigma=0.75,
                            lowghost=-7, highghost=-2, samples=9):
    """
    Calculates PSF size and ellipticity when including an out-of-focus doughnut of a given contrast level.
    The dougnut pixel values are all scaled to a given scaling value (requirement 5e-5).

    :param log: logger instance
    :param filename: name of the PSF file to analyse
    :param psfscale: level to which the original PSF is scaled to
    :param maxdistance: maximum distance the ghost image can be from the original PSF (centre to centre)
    :param inner: inner radius of the out-of-focus doughnut
    :param outer: outer radius of the out-of-focus doughnut
    :param oversample: oversampling factor
    :param psfs: number of PSFs to analyse (number of ghosts in random locations)
    :param iterations: number of iterations in the shape measurement
    :param sigma: size of the Gaussian weighting function
    :param lowghost: log of the highest ghost contrast ratio to study
    :param highghost: log of the lowest ghost contrast ratio to study
    :param samples: number of points for the contrast ratio to study

    :return: results
    :rtype: dict
    """
    #read in PSF and renormalize it
    data = pf.getdata(filename)

    #place it a larger canvas with zero padding around
    ys, xs = data.shape
    canvas = np.pad(data, int(maxdistance*oversample + xs + outer),
                    mode='constant', constant_values=0)  #requires numpy >= 1.7.0
    ys, xs = canvas.shape
    xcen = int(np.round(xs/2., 0))
    ycen = int(np.round(ys/2., 0))
    #print canvas.shape
    #print canvas.flags

    canvas /= np.max(canvas)
    canvas *= float(psfscale)

    #make out of focus image, a simple doughnut
    img, xd, yd = drawDoughnut(inner, outer, oversample=oversample)

    #set sampling etc. for shape measurement
    settings = dict(sampling=1.0 / oversample, itereations=iterations, sigma=sigma)

    #positions
    x = np.round((np.random.rand(psfs)-0.5)*maxdistance*oversample, 0).astype(np.int)
    y = np.round((np.random.rand(psfs)-0.5)*maxdistance*oversample, 0).astype(np.int)

    #ghost level
    ghosts = np.logspace(lowghost, highghost, samples)[::-1] #largest first
    tot = ghosts.size
    res = {}
    for i, scale in enumerate(ghosts):
        print'ghost levels: %i / %i' % (i + 1, tot)
        R2 = []
        e1 = []
        e2 = []
        e = []

        #scale the doughtnut pixel values, note that all pixels have the same value...
        scaled = img.copy() * (scale * psfscale)

        #loop over the ghost positions
        for xc, yc in zip(x, y):
            tmp = canvas.copy()
            xm = xcen + xc
            ym = ycen + yc
            try:
                tmp[ym-yd:ym+yd+1, xm-xd:xm+xd+1] += scaled
            except:
                try:
                    tmp[ym-yd:ym+yd, xm-xd:xm+xd] += scaled
                except:
                    print scaled.shape
                    print tmp[ym-yd:ym+yd+1, xm-xd:xm+xd+1].shape
                    print 'ERROR -- cannot place the ghost to the image!!'
                    continue

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(tmp, log, **settings)
            results = sh.measureRefinedEllipticity()

            #save values
            e1.append(results['e1'])
            e2.append(results['e2'])
            e.append(results['ellipticity'])
            R2.append(results['R2'])

        out = dict(e1=np.asarray(e1), e2=np.asarray(e2), e=np.asarray(e), R2=np.asarray(R2))
        res[scale] = out

    return res


def ghostContributionToStar(log, filename='data/psf12x.fits', psfscale=2e5, distance=750,
                            inner=8, outer=60, oversample=12.0, iterations=20, sigma=0.75,
                            scale=5e-5, fixedPosition=True):
    #set sampling etc. for shape measurement
    settings = dict(sampling=1.0 / oversample, itereations=iterations, sigma=sigma, debug=True)

    #read in PSF
    data = pf.getdata(filename)

    #place it a larger canvas with zero padding around
    canvas = np.pad(data, int(distance*oversample + outer + 1),
                    mode='constant', constant_values=0)  #requires numpy >= 1.7.0
    ys, xs = canvas.shape
    xcen = int(np.round(xs / 2., 0))
    ycen = int(np.round(ys / 2., 0))

    #normalize canvas and save it
    canvas /= np.max(canvas)
    canvas *= float(psfscale)
    fileIO.writeFITS(canvas, 'originalPSF.fits', int=False)

    #reference values
    sh = shape.shapeMeasurement(canvas, log, **settings)
    reference = sh.measureRefinedEllipticity()
    fileIO.cPickleDumpDictionary(reference, 'ghostStarContribution.pk')
    print 'Reference:'
    pprint.pprint(reference)

    #make out of focus image, a simple doughnut
    img, xd, yd = drawDoughnut(inner, outer, oversample=oversample)

    #positions (shift respect to the centring of the star)
    xc = 0
    yc = distance * oversample

    #indices range
    xm = xcen + xc
    ym = ycen + yc

    #ghost level
    #scale the doughtnut pixel values, note that all pixels have the same value...
    img /= np.max(img)
    scaled = img.copy() * scale * psfscale
    fileIO.writeFITS(scaled, 'ghostImage.fits', int=False)

    tmp = canvas.copy()

    if oversample % 2 == 0:
        tmp[ym - yd:ym + yd, xm - xd:xm + xd] += scaled
    else:
        tmp[ym - yd:ym + yd + 1, xm - xd:xm + xd + 1] += scaled

    fileIO.writeFITS(tmp, 'originalPlusGhost.fits', int=False)

    #use fixed positions
    if fixedPosition:
        settings['fixedPosition'] = True
        settings['fixedX'] = reference['centreX']
        settings['fixedY'] = reference['centreY']

    #measure e and R2 from the postage stamp image
    sh = shape.shapeMeasurement(tmp, log, **settings)
    results = sh.measureRefinedEllipticity()
    fileIO.cPickleDumpDictionary(results, 'ghostStarContribution.pk')

    #save values
    print '\nWith Doughnut:'
    pprint.pprint(results)

    print '\nDelta: with ghost - reference'
    print 'e1', results['e1'] - reference['e1']
    print 'e2', results['e2'] - reference['e2']
    print 'e', results['ellipticity'] - reference['ellipticity']
    print 'R2', results['R2'] - reference['R2']
    print 'Xcen', results['centreX'] - reference['centreX']
    print 'Ycen', results['centreY'] - reference['centreY']

    return results


def ghostContribution(log, filename='data/psf1x.fits', magnitude=24.5, zp=25.5, exptime=565., exposures=3,
                      iterations=100, sigma=0.75, centered=False, offset=9):
    #set sampling etc. for shape measurement
    settings = dict(itereations=iterations, sigma=sigma, debug=True)

    #read in PSF
    data = pf.getdata(filename)

    #place it a larger canvas with zero padding around
    canvas = np.pad(data, 100, mode='constant', constant_values=0)  #requires numpy >= 1.7.0
    ys, xs = canvas.shape
    xcen = int(np.round(xs / 2., 0))
    ycen = int(np.round(ys / 2., 0))

    #normalize canvas, scale it to magnitude and save it
    canvas /= np.max(canvas)
    intscale = 10.0**(-0.4 * (magnitude-zp)) * exptime * exposures
    canvas *= intscale
    fileIO.writeFITS(canvas, 'originalPSF.fits', int=False)

    #reference values
    sh = shape.shapeMeasurement(canvas, log, **settings)
    reference = sh.measureRefinedEllipticity()
    fileIO.cPickleDumpDictionary(reference, 'ghostStarContribution.pk')
    print 'Reference:'
    pprint.pprint(reference)

    #load ghost
    ghostModel = pf.getdata('data/ghost800nm.fits')[355:423, 70:131]
    ghostModel /= np.max(ghostModel)  #peak is 1 now
    ys, xs = ghostModel.shape
    yd = int(np.round(ys / 2., 0))
    xd = int(np.round(xs / 2., 0))
    fileIO.writeFITS(ghostModel, 'ghostImage.fits', int=False)

    #ghost level
    #scale the doughtnut pixel values, note that all pixels have the same value...
    scales = np.logspace(-8, -3, 21)

    result = {}
    for scale in scales:
        scaled = ghostModel.copy() * intscale * scale
        #fileIO.writeFITS(scaled, 'ghostImage.fits', int=False)

        tmp = canvas.copy()
        if centered:
            tmp[ycen - yd:ycen + yd, xcen - xd:xcen + xd + 1] += scaled
        else:
            tmp[ycen - yd + offset:ycen + yd + offset, xcen - xd + offset:xcen + xd + 1 + offset] += scaled
            #tmp[ycen: ycen + 2*yd, xcen:xcen + 2*xd + 1] += scaled
        #fileIO.writeFITS(tmp, 'originalPlusGhost.fits', int=False)

        #measure e and R2 from the postage stamp image
        sh = shape.shapeMeasurement(tmp, log, **settings)
        results = sh.measureRefinedEllipticity()

        de1 = results['e1'] - reference['e1']
        de2 = results['e2'] - reference['e2']
        de = np.sqrt(de1**2 + de2**2)
        dR2 = (results['R2'] - reference['R2']) / reference['R2']

        print '\n\nscale=', scale
        print 'Delta: with ghost - reference'
        print 'e1', de1
        print 'e2', de2
        print 'e', de
        print 'R2', dR2

        result[scale] = [de1, de2, de, dR2, results['e1'], results['e2'], results['ellipticity'], results['R2']]

    return result


def plotGhostContribution(res,title, output, title2, output2):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)

    for key in res.keys():
        ax.scatter(key, np.abs(res[key][0]), c='m', marker='*', s=35)
        ax.scatter(key, np.abs(res[key][1]), c='y', marker='s', s=35)
        ax.scatter(key, np.abs(res[key][2]), c='r', marker='o', s=35)

    ax.scatter(key, np.abs(res[key][0]), c='m', marker='*', label=r'$\delta e_{1}$')
    ax.scatter(key, np.abs(res[key][1]), c='y', marker='s', label=r'$\delta e_{2}$')
    ax.scatter(key, np.abs(res[key][2]), c='r', marker='o', label=r'$\delta e$')

    ax.axvline(x=5e-5, c='g', ls='--', label='Ghost Requirement: 5e-5')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(1e-8, 1e-3)
    ax.set_xlabel('Ghost Contrast Ratio')
    ax.set_ylabel(r'$| \delta (e_{i})| \ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output)
    plt.close()

    #R2
    fig = plt.figure()
    plt.title(title2)
    ax = fig.add_subplot(111)

    for key in res.keys():
        ax.scatter(key, np.abs(res[key][3]), c='m', marker='*', s=35)

    ax.scatter(key, np.abs(res[key][3]), c='m', marker='*', label=r'$\frac{\delta R^{2}}{R^{2}_{ref}}$')

    ax.axvline(x=5e-5, c='g', ls='--', label='Ghost Requirement: 5e-5')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e0)
    ax.set_xlim(1e-8, 1e-3)
    ax.set_xlabel('Ghost Contrast Ratio')
    ax.set_ylabel(r'$\left | \frac{\delta R^{2}}{R^{2}_{ref}} \right |$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output2)
    plt.close()


def ghostContributionElectrons(log, filename='data/psf1x.fits', magnitude=24.5, zp=25.5, exptime=565., exposures=3,
                      iterations=100, sigma=0.75, centered=False, offset=9, verbose=False):
    #set sampling etc. for shape measurement
    settings = dict(itereations=iterations, sigma=sigma, debug=True)

    #read in PSF
    data = pf.getdata(filename)

    #place it a larger canvas with zero padding around
    canvas = np.pad(data, 100, mode='constant', constant_values=0)  #requires numpy >= 1.7.0
    ys, xs = canvas.shape
    xcen = int(np.round(xs / 2., 0))
    ycen = int(np.round(ys / 2., 0))

    #normalize canvas, scale it to magnitude and save it
    canvas /= np.max(canvas)
    intscale = 10.0**(-0.4 * (magnitude-zp)) * exptime * exposures
    canvas *= intscale
    fileIO.writeFITS(canvas, 'originalPSF.fits', int=False)

    #reference values
    sh = shape.shapeMeasurement(canvas, log, **settings)
    reference = sh.measureRefinedEllipticity()
    fileIO.cPickleDumpDictionary(reference, 'ghostStarContribution.pk')

    if verbose:
        print 'Reference:'
        pprint.pprint(reference)

    #load ghost
    ghostModel = pf.getdata('data/ghost800nm.fits')[355:423, 70:131]
    ghostModel /= np.max(ghostModel)  #peak is 1 now
    ys, xs = ghostModel.shape
    yd = int(np.round(ys / 2., 0))
    xd = int(np.round(xs / 2., 0))
    fileIO.writeFITS(ghostModel, 'ghostImage.fits', int=False)

    #ghost level
    #scale the doughtnut pixel values, note that all pixels have the same value...
    scales = np.logspace(-4, 2, 21)

    result = {}
    for scale in scales:
        scaled = ghostModel.copy() * scale
        #fileIO.writeFITS(scaled, 'ghostImage.fits', int=False)

        tmp = canvas.copy()
        if centered:
            tmp[ycen - yd:ycen + yd, xcen - xd:xcen + xd + 1] += scaled
        else:
            tmp[ycen - yd + offset:ycen + yd + offset, xcen - xd + offset:xcen + xd + 1 + offset] += scaled
            #tmp[ycen: ycen + 2*yd, xcen:xcen + 2*xd + 1] += scaled
        #fileIO.writeFITS(tmp, 'originalPlusGhost.fits', int=False)

        #measure e and R2 from the postage stamp image
        sh = shape.shapeMeasurement(tmp, log, **settings)
        results = sh.measureRefinedEllipticity()

        de1 = results['e1'] - reference['e1']
        de2 = results['e2'] - reference['e2']
        de = np.sqrt(de1**2 + de2**2)
        dR2 = (results['R2'] - reference['R2']) / reference['R2']

        if verbose:
            print '\n\nscale=', scale
            print 'Delta: with ghost - reference'
            print 'e1', de1
            print 'e2', de2
            print 'e', de
            print 'R2', dR2

        result[scale] = [de1, de2, de, dR2, results['e1'], results['e2'], results['ellipticity'], results['R2']]

    return result


def plotGhostContributionElectrons(res,title, output, title2, output2):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)

    levels = []
    e = []
    for key in res.keys():
        ax.scatter(key, np.abs(res[key][0]), c='m', marker='*', s=35)
        ax.scatter(key, np.abs(res[key][1]), c='y', marker='s', s=35)
        ax.scatter(key, np.abs(res[key][2]), c='r', marker='o', s=35)
        levels.append(key)
        e.append(np.abs(res[key][2]))

    ax.scatter(key, np.abs(res[key][0]), c='m', marker='*', label=r'$\delta e_{1}$')
    ax.scatter(key, np.abs(res[key][1]), c='y', marker='s', label=r'$\delta e_{2}$')
    ax.scatter(key, np.abs(res[key][2]), c='r', marker='o', label=r'$\delta e$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e0)
    ax.set_xlim(1e-4, 1e2)
    ax.set_xlabel('Ghost Contribution Peak Pixel [$e^{-}]$')
    ax.set_ylabel(r'$| \delta (e_{i})| \ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output)
    plt.close()

    #find the crossing
    levels = np.asarray(levels)
    srt = np.argsort(levels)
    results = np.asarray(e)
    f = interpolate.interp1d(levels[srt], results[srt], kind='cubic')
    x = np.logspace(np.log10(levels.min()), np.log10(levels.max()), 1000)
    vals = f(x)
    msk1 = vals.copy() <= 1e-2
    msk2 = vals.copy() <= 1e-4
    maxn1 = np.max(x[msk1])
    maxn2 = np.max(x[msk2])
    print 'Ghost electron level <=:'
    print maxn1
    print maxn2
    print 'Corresponds to'
    print 25.5 -5/2.*np.log10((maxn1)/(565*3*0.7*5e-5))
    print 25.5 -5/2.*np.log10((maxn2)/(565*3*0.7*5e-5))

    #R2
    fig = plt.figure()
    plt.title(title2)
    ax = fig.add_subplot(111)

    levels = []
    r2 = []
    for key in res.keys():
        ax.scatter(key, np.abs(res[key][3]), c='m', marker='*', s=35)
        levels.append(key)
        r2.append(np.abs(res[key][3]))

    ax.scatter(key, np.abs(res[key][3]), c='m', marker='*', label=r'$\frac{\delta R^{2}}{R^{2}_{ref}}$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e1)
    ax.set_xlim(1e-4, 1e2)
    ax.set_xlabel('Ghost Contribution Peak Pixel [$e^{-}]$')
    ax.set_ylabel(r'$\left | \frac{\delta R^{2}}{R^{2}_{ref}} \right |$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output2)
    plt.close()

    #find the crossing
    levels = np.asarray(levels)
    srt = np.argsort(levels)
    results = np.asarray(r2)
    f = interpolate.interp1d(levels[srt], results[srt], kind='cubic')
    x = np.logspace(np.log10(levels.min()), np.log10(levels.max()), 1000)
    vals = f(x)
    msk1 = vals.copy() <= 1e-2
    msk2 = vals.copy() <= 1e-4
    maxn1 = np.max(x[msk1])
    maxn2 = np.max(x[msk2])
    print 'Ghost electron level <=:'
    print maxn1
    print maxn2
    print 'Corresponds to'
    print 25.5 -5/2.*np.log10((maxn1)/(565*3*0.7*5e-5))
    print 25.5 -5/2.*np.log10((maxn2)/(565*3*0.7*5e-5))



def plotResults(res, output, title, reqe=3e-5, reqR2=1e-4, ghostlevel=5e-5):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    vals = []
    for key in res.keys():
        e1 = np.std(res[key]['e1'])
        e2 = np.std(res[key]['e2'])
        e = np.std(res[key]['e'])

        vals.append(e)

        ax.scatter(key, e1, c='m', marker='*', s=35)
        ax.scatter(key, e2, c='y', marker='s', s=35)
        ax.scatter(key, e, c='r', marker='o', s=35)

        print key, e, e1, e2

    #for the legend
    ax.scatter(key, e1, c='m', marker='*', label=r'$\sigma(e_{1})$')
    ax.scatter(key, e2, c='y', marker='s', label=r'$\sigma(e_{2})$')
    ax.scatter(key, e, c='r', marker='o', label=r'$\sigma(e)$')

    #show the requirement
    ks = np.asarray(res.keys())
    ran = np.linspace(ks.min() * 0.99, ks.max() * 1.01)
    ax.fill_between(ran, np.ones(ran.size) * reqe, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqe, c='g', ls='--', label='Level')
    ax.axvline(x=ghostlevel, c='r', ls='--', label='Ghost Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(ks.min() * 0.9, ks.max() * 1.01)
    ax.set_xlabel('Ghost Contrast Ratio')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output + 'e.pdf')
    plt.close()

    #size
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    vals = []
    for key in res.keys():
        dR2 = res[key]['R2']
        normed = np.std(dR2) / np.mean(dR2)

        ax.scatter(key, normed, c='m', marker='*', s=35)
        vals.append(normed)

        print key, normed

    #for the legend
    ax.scatter(key, normed, c='m', marker='*', label=r'$\frac{\sigma(R^{2})}{R_{ref}^{2}}$')

    #show the requirement
    ks = np.asarray(res.keys())
    ran = np.linspace(ks.min() * 0.99, ks.max() * 1.01)
    ax.fill_between(ran, np.ones(ran.size) * reqR2, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqR2, c='g', ls='--', label='Level')
    ax.axvline(x=ghostlevel, c='r', ls='--', label='Ghost Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(ks.min() * 0.9, ks.max() * 1.01)
    ax.set_xlabel('Ghost Contrast Ratio')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output + 'size.pdf')
    plt.close()


def deleteAndMove(dir, files='*.fits'):
    """

    :param dir:
    :param files:
    :return:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        for f in glob.glob(dir + '/' + files):
            os.remove(f)

    for f in glob.glob(files):
        shutil.move(f, './'+dir+'/'+f)


if __name__ == '__main__':
    run = False
    debug = False
    focus = False
    star = True

    #start the script
    log = lg.setUpLogger('ghosts.log')
    log.info('Analysing the impact of ghost images...')

    res = ghostContributionElectrons(log, sigma=0.2)
    fileIO.cPickleDumpDictionary(res, 'ghostContributionToStarElectrons.pk')
    res = cPickle.load(open('ghostContributionToStarElectrons.pk'))
    plotGhostContributionElectrons(res, r'Shape Bias: 24.5 mag$_{AB}$ Point Source', 'shapeBiasElectrons.pdf',
                                   r'Size Bias: 24.5 mag$_{AB}$ Point Source', 'sizeBiasElectrons.pdf')

    # res = ghostContributionElectrons(log, centered=True)
    # fileIO.cPickleDumpDictionary(res, 'ghostContributionToStarCenteredElectrons.pk')
    # res = cPickle.load(open('ghostContributionToStarCenteredElectrons.pk'))
    # plotGhostContributionElectrons(res, r'Shape Bias: 24.5 mag$_{AB}$ Point Source', 'shapeBiasCentredElectrons.pdf',
    #                                r'Size Bias: 24.5 mag$_{AB}$ Point Source', 'sizeBiasCentredElectrons.pdf')

    # res = ghostContribution(log)
    # fileIO.cPickleDumpDictionary(res, 'ghostContributionToStar.pk')
    # res = cPickle.load(open('ghostContributionToStar.pk'))
    # plotGhostContribution(res, r'Shape Bias: 24.5 mag$_{AB}$ Point Source', 'shapeBias.pdf',
    #                       r'Size Bias: 24.5 mag$_{AB}$ Point Source', 'sizeBias.pdf')
    #
    # res = ghostContribution(log, centered=True)
    # fileIO.cPickleDumpDictionary(res, 'ghostContributionToStarCentered.pk')
    # res = cPickle.load(open('ghostContributionToStarCentered.pk'))
    # plotGhostContribution(res, r'Shape Bias: 24.5 mag$_{AB}$ Point Source', 'shapeBiasCentred.pdf',
    #                       r'Size Bias: 24.5 mag$_{AB}$ Point Source', 'sizeBiasCentred.pdf')


    # if debug:
    #     #out of focus ghosts
    #     res = analyseOutofFocusImpact(log, filename='data/psf1x.fits', maxdistance=100, samples=7,
    #                                   inner=8, outer=60, oversample=1.0, psfs=1000, iterations=5, sigma=0.75)
    #     fileIO.cPickleDumpDictionary(res, 'OutofFocusResultsDebug.pk')
    #     res = cPickle.load(open('OutofFocusResultsDebug.pk'))
    #     plotResults(res, 'OutofFocusGhostsDebug', 'VIS Ghosts: PSF Knowledge')
    #
    #
    # if focus:
    #     #if the ghosts were in focus
    #     res = analyseInFocusImpact(log, filename='data/psf2x.fits', psfscale=100000, maxdistance=100,
    #                                oversample=2.0, psfs=200, iterations=4, sigma=0.75)
    #     fileIO.cPickleDumpDictionary(res, 'InfocusResultsDebug.pk')
    #     plotResults(res, 'InfocusGhosts', 'VIS Ghosts: In Focus Analysis')
    #
    # if run:
    #     #real run
    #     res = analyseOutofFocusImpact(log)
    #     fileIO.cPickleDumpDictionary(res, 'OutofFocusResults.pk')
    #     res = cPickle.load(open('OutofFocusResults.pk'))
    #     plotResults(res, 'OutofFocusGhosts', 'Weak Lensing Channel Ghost: PSF Knowledge')
    #
    # if star:
    #     print '\n\n\n\nWith Fixed Position:'
    #     ghostContributionToStar(log, filename='data/psf1x.fits', oversample=1.0)
    #     deleteAndMove('psf1xFixed')
    #
    #     print '\n\n\n\nNo Fixed Position:'
    #     ghostContributionToStar(log, filename='data/psf1x.fits', oversample=1.0, fixedPosition=False)
    #     deleteAndMove('psf1x')
    #
    #     print '\n\n\n\n2 Times Oversampled Fixed Position:'
    #     ghostContributionToStar(log, filename='data/psf2x.fits', oversample=2.0)
    #     deleteAndMove('psf2xFixed')


    log.info('Run finished...\n\n\n')