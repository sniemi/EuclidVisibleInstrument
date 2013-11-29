"""
Flat Field Calibration
======================

This simple script can be used to study the number of flat fields required to meet the VIS calibration requirements.

The following requirements related to the flat field calibration has been taken from GDPRD.

R-GDP-CAL-054:
The contribution of the residuals of VIS flat-field correction to the error on the determination of each
ellipticity component of the local PSF shall not exceed 3x10-5 (one sigma).

R-GDP-CAL-064:
The contribution of the residuals of VIS flat-field correction to the relative error on the determination
of the local PSF R2 shall not exceed 1x10-4 (one sigma).

.. Note:: The amount of cosmic rays in the simulated input images might be too low, because the exposure was
          set to 10 seconds and cosmic rays were calculated based on this. However, in reality the readout
          takes about 80 seconds. Thus, the last row is effected by cosmic a lot more than by assuming a single
          10 second exposure.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.97

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyfits as pf
import numpy as np
import datetime, cPickle, glob, os, sys
from scipy.ndimage.interpolation import zoom
from scipy import interpolate
from analysis import shape
from support import logger as lg
from support import surfaceFitting as sf
from support import files as fileIO


def generateResidualFlatField(files='Q0*flatfield*.fits', combine=77, lampfile='data/VIScalibrationUnitflux.fits',
                              reference='data/VISFlatField1percent.fits', gain=3.5, plots=False, debug=False):
    """
    Generate a median combined flat field residual from given input files.

    Randomly draws a given number (kw combine) of files from the file list identified using the files kw.
    Median combine all files before the lamp profile given by lampfile kw is being divided out. This
    will produce a derived flat field. This flat can be compared against the reference that was used
    to produce the initial data to derive a residual flat that describes the error in the flat field
    that was derived.

    :param files: wildcard flagged name identifier for the FITS files to be used for generating a flat
    :type files: str
    :param combine: number of files to median combine
    :type combine: int
    :param lampfile: name of the calibration unit flux profile FITS file
    :type lampfile: str
    :param reference: name of the reference pixel-to-pixel flat field FITS file
    :type reference: str
    :param gain: gain factor [e/ADU]
    :type gain: float
    :param plots: whether or not to generate plots
    :type plots: boolean
    :param debug: whether or not to produce output FITS files
    :type debug: boolean

    .. Warning:: Remember to use an appropriate lamp and reference files so that the error in the derived
                 flat field can be correctly calculated.

    :return: residual flat field (difference between the generated flat and the reference)
    :rtype: ndarray
    """
    #find all FITS files
    files = glob.glob(files)

    #choose randomly the files that should be combined
    if combine < len(files):
        ids = np.random.random_integers(0, len(files)-1, combine)
        files = np.asarray(files)[ids]

    #load data and scale to electrons
    data = fileIO.readFITSDataExcludeScanRegions(files)
    data *= gain

    #check that the sizes match and median combine
    if len(set(x.shape for x in data)) > 1:
        sys.exit('ERROR -- files are not the same shape, cannot median combine!')
    else:
        medianCombined = np.median(data, axis=0)

    #fit surface to the median and normalize it out
    #m = sf.polyfit2d(xx.ravel(), yy.ravel(), median.ravel(), order=order)
    # Evaluate it on a rectangular grid
    #fitted = sf.polyval2d(xx, yy, m)

    #load the lamp profile that went in and divide the combined image with the profile
    lamp = pf.getdata(lampfile)
    pixvar = medianCombined.astype(np.float64).copy() / lamp

    #load the true reference p-flat and calculate the error in the derived flat field (i.e. residual)
    real = pf.getdata(reference).astype(np.float64)
    #res = np.abs(real - pixvar) / (real*pixvar) + 1.  #old: maybe incorrect?
    res = 1. - pixvar / real

    if debug:
        print np.mean(res), np.min(res), np.max(res), np.std(res)

        if not os.path.exists('debug'):
            os.makedirs('debug')

        fileIO.writeFITS(medianCombined, 'debug/medianFlat.fits')
        fileIO.writeFITS(pixvar, 'debug/derivedFlat.fits')
        fileIO.writeFITS(res, 'debug/residualFlat.fits')

    if plots:
        if not os.path.exists('plots'):
            os.makedirs('plots')

        #generate a mesh grid fo plotting
        ysize, xsize = medianCombined.shape
        xx, yy = np.meshgrid(np.linspace(0, xsize, xsize), np.linspace(0, ysize, ysize))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, medianCombined, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel('Counts [electrons]')
        ax.set_zlim(8.9e4, 1.05e5)
        plt.savefig('plots/MedianFlat.png')
        plt.close()

        im = plt.imshow(medianCombined, origin='lower', vmin=8.9e4, vmax=1.05e5)
        c1 = plt.colorbar(im)
        c1.set_label('Counts [electrons]')
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.savefig('plots/Mediand2D.png')
        plt.close()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, pixvar, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlim(0.95, 1.05)
        ax.set_zlabel('Counts [electrons]')
        plt.savefig('plots/PixelFlat.png')
        plt.close()

        im = plt.imshow(pixvar, origin='lower', vmin=0.95, vmax=1.05)
        c1 = plt.colorbar(im)
        c1.set_label('Counts [electrons]')
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.savefig('plots/PixelFlat2D.png')
        plt.close()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, res, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlim(0.95, 1.05)
        ax.set_zlabel('Residual Flat Field')
        plt.savefig('plots/ResidualFlatField.png')
        plt.close()

        im = plt.imshow(res, origin='lower', vmin=0.95, vmax=1.05)
        c1 = plt.colorbar(im)
        c1.set_label('Residual Flat Field')
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.savefig('plots/ResidualFlatField2D.png')
        plt.close()

    return res


def testFlatCalibration(log, flats, surfaces=10, file='data/psf1x.fits', psfs=5000,
                        sigma=0.75, iterations=7, weighting=True, plot=False, debug=False):
    """
    Derive the PSF ellipticities for a given number of random surfaces with random PSF positions
    and a given number of flat fields median combined.

    This function is to derive the the actual values so that the knowledge (variance) can be studied.

    """
    #read in PSF and rescale to avoid rounding or truncation errors
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 300. #SNR about 10 for star...

    #derive reference values
    settings = dict(sigma=sigma, iterations=iterations, weighted=weighting)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()

    #random positions for the PSFs, these positions are the lower corners
    #assume that this is done on quadrant level thus the xmax and ymax are 2065 and 2047, respectively
    xpositions = np.random.random_integers(0, 2047 - data.shape[1], psfs)
    ypositions = np.random.random_integers(0, 2065 - data.shape[0], psfs)

    out = {}
    #number of biases to median combine
    for a in flats:
        print 'Number of Flats to combine: %i / %i' % (a, flats[-1])

        #data storage
        de1 = []
        de2 = []
        de = []
        R2 = []
        dR2 = []
        e1 = []
        e2 = []
        e = []

        for b in xrange(surfaces):
            print 'Random Realisations: %i / %i' % (b+1, surfaces)

            residual = generateResidualFlatField(combine=a, plots=plot, debug=debug)

            print 'Average residual = %e' % (np.mean(residual) - 1.)

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
            for xpos, ypos in zip(xpositions, ypositions):
                tmp = data.copy()

                #get the underlying residual surface and multiple with the PSF
                small = residual[ypos:ypos+data.shape[0], xpos:xpos+data.shape[1]].copy()
                small += 1.
                small *= tmp
                #small *= tmp   # depends on the residual geenration

                #measure e and R2 from the postage stamp image
                sh = shape.shapeMeasurement(small.copy(), log, **settings)
                results = sh.measureRefinedEllipticity()

                #save values
                e1.append(results['e1'])
                e2.append(results['e2'])
                e.append(results['ellipticity'])
                R2.append(results['R2'])
                de1.append(results['e1'] - reference['e1'])
                de2.append(results['e2'] - reference['e2'])
                de.append(results['ellipticity'] - reference['ellipticity'])
                dR2.append(results['R2'] - reference['R2'])

        out[a+1] = [e1, e2, e, R2, de1, de2, de, dR2]

    return out, reference


def plotNumberOfFrames(results, reqe=3e-5, reqr2=1e-4, shift=0.1, outdir='results', timeStamp=False):
    """
    Creates a simple plot to combine and show the results.

    :param res: results to be plotted [results dictionary, reference values]
    :type res: list
    :param reqe: the requirement for ellipticity [default=3e-5]
    :type reqe: float
    :param reqr2: the requirement for size R2 [default=1e-4]
    :type reqr2: float
    :param shift: the amount to shift the e2 results on the abscissa (for clarity)
    :type shift: float
    :param outdir: output directory to which the plots will be saved to
    :type outdir: str
    :param timeStamp: whether or not to include a time stamp to the output image
    :type timeStamp: bool

    :return: None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #rename
    ref = results[1]
    res = results[0]

    print '\nSigma results:'
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    plt.title(r'VIS Flat Field Calibration: $\sigma(e)$')
    ax = fig.add_subplot(111)

    maxx = 0
    frames = []
    values = []
    #loop over the number of frames combined
    for key in res:
        e1 = np.asarray(res[key][0])
        e2 = np.asarray(res[key][1])
        e = np.asarray(res[key][2])

        std1 = np.std(e1)
        std2 = np.std(e2)
        std = np.std(e)

        frames.append(key)
        values.append(std)

        ax.scatter(key-shift, std, c='m', marker='*')
        ax.scatter(key, std1, c='b', marker='o')
        ax.scatter(key, std2, c='y', marker='s')

        if key > maxx:
            maxx = key
        print key, std, std1, std2

    #label
    ax.scatter(key-shift, std, c='m', marker='*', label=r'$\sigma (e)$')
    ax.scatter(key, std1, c='b', marker='o', label=r'$\sigma (e_{1})$')
    ax.scatter(key, std2, c='y', marker='s', label=r'$\sigma (e_{2})$')

    #sort and interpolate
    values = np.asarray(values)
    frames = np.asarray(frames)
    srt = np.argsort(frames)
    x = np.arange(frames.min(), frames.max()+1)
    f = interpolate.interp1d(frames[srt], values[srt], kind='cubic')
    vals = f(x)
    ax.plot(x, vals, ':', c='0.2', zorder=20)
    try:
        msk = vals < reqe
        minn = np.min(x[msk])
        plt.text(np.mean(frames), 8e-6, r'Flats Required $\raise-.5ex\hbox{$\buildrel>\over\sim$}$ %i' % np.ceil(minn),
                 ha='center', va='center', fontsize=11)
    except:
        pass

    ax.fill_between(np.arange(maxx+10), np.ones(maxx+10)*reqe, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqe, c='g', ls='--', label='Requirement')
    plt.text(1, 0.9*reqe, '%.1e' % reqe, ha='left', va='top', fontsize=11)

    ax.set_yscale('log')
    ax.set_ylim(5e-6, 1e-4)
    ax.set_xlim(0, maxx+1)
    ax.set_xlabel('Number of Flat Fields Median Combined')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
    plt.savefig(outdir+'/FlatCalibrationsigmaE.pdf')
    plt.close()

    #same for R2s
    fig = plt.figure()
    plt.title(r'VIS Flat Field Calibration: $\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    maxx = 0
    frames = []
    values = []
    #loop over the number of frames combined
    for key in res:
        dR2 = np.asarray(res[key][3])

        #std = np.std(dR2) / ref['R2']
        std = np.std(dR2) / np.mean(dR2)

        frames.append(key)
        values.append(std)
        print key, std

        ax.scatter(key, std, c='b', marker='s', s=35, zorder=10)

        if key > maxx:
            maxx = key

    #for the legend
    ax.scatter(key, std, c='b', marker='s', label=r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    #sort and interpolate
    values = np.asarray(values)
    frames = np.asarray(frames)
    srt = np.argsort(frames)
    x = np.arange(frames.min(), frames.max())
    f = interpolate.interp1d(frames[srt], values[srt], kind='cubic')
    vals = f(x)
    ax.plot(x, vals, ':', c='0.2', zorder=10)
    try:
        msk = vals < reqr2
        minn = np.min(x[msk])
        plt.text(np.mean(frames), 2e-5, r'Flats Required $\raise-.5ex\hbox{$\buildrel>\over\sim$}$ %i' % np.ceil(minn),
                 fontsize=11, ha='center', va='center')
    except:
        pass

    #show the requirement
    ax.fill_between(np.arange(maxx+10), np.ones(maxx+10)*reqr2, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=reqr2, c='g', ls='--', label='Requirement')
    plt.text(1, 0.9*reqr2, '%.1e' % reqr2, ha='left', va='top', fontsize=11)

    ax.set_yscale('log')
    ax.set_ylim(5e-6, 1e-3)
    ax.set_xlim(0, maxx+1)
    ax.set_xlabel('Number of Flat Fields Median Combined')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8    )
    plt.savefig(outdir+'/FlatCalibrationSigmaR2.pdf')
    plt.close()

    print '\nDelta results:'
    #loop over the number of frames combined
    for key in res:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.title(r'VIS Flat Field Calibration (%i exposures): $\delta e$' % key)

        de1 = np.asarray(res[key][4])
        de2 = np.asarray(res[key][5])
        de = np.asarray(res[key][6])

        avg1 = np.mean(de1)**2
        avg2 = np.mean(de2)**2
        avg = np.mean(de)**2

        #write down the values
        print key, avg, avg1, avg2
        plt.text(0.08, 0.9, r'$\left< \delta e_{1} \right>^{2} = %e$' %avg1, fontsize=10, transform=ax.transAxes)
        plt.text(0.08, 0.85, r'$\left< \delta e_{2}\right>^{2} = %e$' %avg2, fontsize=10, transform=ax.transAxes)
        plt.text(0.08, 0.8, r'$\left< \delta | \bar{e} |\right>^{2} = %e$' %avg, fontsize=10, transform=ax.transAxes)

        ax.hist(de, bins=15, color='y', alpha=0.2, label=r'$\delta | \bar{e} |$', normed=True, log=True)
        ax.hist(de1, bins=15, color='b', alpha=0.5, label=r'$\delta e_{1}$', normed=True, log=True)
        ax.hist(de2, bins=15, color='g', alpha=0.3, label=r'$\delta e_{2}$', normed=True, log=True)

        ax.axvline(x=0, ls=':', c='k')

        ax.set_ylabel('Probability Density')
        ax.set_xlabel(r'$\delta e_{i}\ , \ \ \ i \in [1,2]$')

        if timeStamp:
            plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

        plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
        plt.savefig(outdir+'/FlatCalibrationEDelta%i.pdf' % key)
        plt.close()

    #same for R2s
    for key in res:
        fig = plt.figure()
        plt.title(r'VIS Flat Field Calibration (%i exposures): $\frac{\delta R^{2}}{R_{ref}^{2}}$' % key)
        ax = fig.add_subplot(111)

        dR2 = np.asarray(res[key][7])
        avg = np.mean(dR2/ref['R2'])**2

        ax.hist(dR2, bins=15, color='y', label=r'$\frac{\delta R^{2}}{R_{ref}^{2}}$', normed=True, log=True)

        print key, avg
        plt.text(0.1, 0.9, r'$\left<\frac{\delta R^{2}}{R^{2}_{ref}}\right>^{2} = %e$' %avg,
                 fontsize=10, transform=ax.transAxes)

        ax.axvline(x=0, ls=':', c='k')

        ax.set_ylabel('Probability Density')
        ax.set_xlabel(r'$\frac{\delta R^{2}}{R_{ref}^{2}}$')

        if timeStamp:
            plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

        plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8)
        plt.savefig(outdir+'/FlatCalibrationDeltaSize%i.pdf' % key)
        plt.close()


def findTolerableError(log, file='data/psf4x.fits', oversample=4.0, psfs=10000, iterations=7, sigma=0.75):
    """
    Calculate ellipticity and size for PSFs of different scaling when there is a residual
    pixel-to-pixel variations.
    """
    #read in PSF and renormalize it
    data = pf.getdata(file)
    data /= np.max(data)

    #PSF scalings for the peak pixel, in electrons
    scales = np.random.random_integers(1e2, 2e5, psfs)

    #set the scale for shape measurement
    settings = dict(sampling=1.0/oversample, itereations=iterations, sigma=sigma)

    #residual from a perfect no pixel-to-pixel non-uniformity
    residuals = np.logspace(-7, -1.6, 9)[::-1] #largest first
    tot = residuals.size
    res = {}
    for i, residual in enumerate(residuals):
        print'%i / %i' % (i+1, tot)
        R2 = []
        e1 = []
        e2 = []
        e = []

        #loop over the PSFs
        for scale in scales:
            #random residual pixel-to-pixel variations
            if oversample < 1.1:
                residualSurface = np.random.normal(loc=1.0, scale=residual, size=data.shape)
            elif oversample == 4.0:
                tmp = np.random.normal(loc=1.0, scale=residual, size=(170, 170))
                residualSurface = zoom(tmp, 4.013, order=0)
            else:
                sys.exit('ERROR when trying to generate a blocky pixel-to-pixel non-uniformity map...')

            #make a copy of the PSF and scale it with the given scaling
            #and then multiply with a residual pixel-to-pixel variation
            tmp = data.copy() * scale * residualSurface

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(tmp.copy(), log, **settings)
            results = sh.measureRefinedEllipticity()

            #save values
            e1.append(results['e1'])
            e2.append(results['e2'])
            e.append(results['ellipticity'])
            R2.append(results['R2'])

        out = dict(e1=np.asarray(e1), e2=np.asarray(e2), e=np.asarray(e), R2=np.asarray(R2))
        res[residual] = out

    return res


def plotTolerableErrorR2(res, output, req=1e-4):
    fig = plt.figure()
    plt.title(r'VIS Flat Fielding')
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
    ax.fill_between(ran, np.ones(ran.size) * req, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')

    #find the crossing
    srt = np.argsort(ks)
    values = np.asarray(vals)
    f = interpolate.interp1d(ks[srt], values[srt], kind='cubic')
    x = np.logspace(np.log10(ks.min()), np.log10(ks.max()), 100)
    vals = f(x)
    ax.plot(x, vals, ':', c='0.2', zorder=10)
    msk = vals < req
    maxn = np.max(x[msk])
    plt.text(1e-5, 2e-5, r'Error must be $\leq %.2e$ per cent' % (maxn*100),
             fontsize=11, ha='center', va='center')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-7, 1e-2)
    ax.set_xlim(ks.min() * 0.99, ks.max() * 1.01)
    ax.set_xlabel('Error in the Flat Field Map')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output)
    plt.close()


def plotTolerableErrorE(res, output, req=3e-5):
    fig = plt.figure()
    plt.title(r'VIS Flat Fielding')
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    vals = []
    for key in res.keys():
        e1 = np.std(res[key]['e1'])
        e2 = np.std(res[key]['e'])
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
    ax.fill_between(ran, np.ones(ran.size) * req, 1.0, facecolor='red', alpha=0.08)
    ax.axhline(y=req, c='g', ls='--', label='Requirement')

    #find the crossing
    srt = np.argsort(ks)
    values = np.asarray(vals)
    f = interpolate.interp1d(ks[srt], values[srt], kind='cubic')
    x = np.logspace(np.log10(ks.min()), np.log10(ks.max()), 100)
    vals = f(x)
    ax.plot(x, vals, ':', c='0.2', zorder=10)
    msk = vals < req
    maxn = np.max(x[msk])
    plt.text(1e-5, 2e-5, r'Error for $e$ must be $\leq %.2e$ per cent' % (maxn*100),
             fontsize=11, ha='center', va='center')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-7, 1e-2)
    ax.set_xlim(ks.min() * 0.99, ks.max() * 1.01)
    ax.set_xlabel('Error in the Flat Field Map')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(output)
    plt.close()


def testNoFlatfieldingEffects(log, file='data/psf1x.fits', oversample=1.0, psfs=500):
    """
    Calculate ellipticity and size variance and error in case of no pixel-to-pixel flat field correction.
    """
    #read in PSF and renormalize it
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 1e5

    #derive reference values
    settings = dict(sampling=1.0/oversample)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()
    print reference

    #residual
    residual = pf.getdata('data/VISFlatField2percent.fits') #'data/VISFlatField1percent.fits'

    if oversample == 4.0:
        residual = zoom(zoom(residual, 2, order=0), 2, order=0)
    elif oversample == 1.0:
        pass
    else:
        print 'ERROR--cannot do arbitrary oversampling...'

    #random positions for the PSFs, these positions are the lower corners
    xpositions = np.random.random_integers(0, residual.shape[1] - data.shape[1], psfs)
    ypositions = np.random.random_integers(0, residual.shape[0] - data.shape[0], psfs)

    #data storage
    out = {}
    de1 = []
    de2 = []
    de = []
    R2 = []
    dR2 = []
    e1 = []
    e2 = []
    e = []
    rnd = 1
    tot = xpositions.size
    #loop over the PSFs
    for xpos, ypos in zip(xpositions, ypositions):
        print'%i / %i' % (rnd, tot)
        rnd += 1

        #make a copy of the PSF
        tmp = data.copy()

        #get the underlying residual surface ond multiple the PSF with the surface
        small = residual[ypos:ypos+data.shape[0], xpos:xpos+data.shape[1]].copy()
        small *= tmp

        #measure e and R2 from the postage stamp image
        sh = shape.shapeMeasurement(small.copy(), log, **settings)
        results = sh.measureRefinedEllipticity()

        #save values
        e1.append(results['e1'])
        e2.append(results['e2'])
        e.append(results['ellipticity'])
        R2.append(results['R2'])
        de1.append(results['e1'] - reference['e1'])
        de2.append(results['e2'] - reference['e2'])
        de.append(results['ellipticity'] - reference['ellipticity'])
        dR2.append(results['R2'] - reference['R2'])

    out[1] = [e1, e2, e, R2, de1, de2, de, dR2]

    return out, reference


if __name__ == '__main__':
    run = True
    debug = False
    plots = True
    error = False

    #start the script
    log = lg.setUpLogger('flatfieldCalibration.log')
    log.info('Testing flat fielding calibration...')

    if error:
        res = findTolerableError(log)

        fileIO.cPickleDumpDictionary(res, 'errors/residuals.pk')
        res = cPickle.load(open('errors/residuals.pk'))

        plotTolerableErrorE(res, output='errors/FlatFieldingTolerableErrorE.pdf')
        plotTolerableErrorR2(res, output='errors/FlatFieldingTolerableErrorR2.pdf')

    if run:
        results = testFlatCalibration(log, flats=np.arange(5, 100, 9))
        fileIO.cPickleDumpDictionary(results, 'flatfieldResults.pk')

    if debug:
        #calculate RMS on image with x frames combined together
        combined = generateResidualFlatField(combine=30, plots=True, debug=True)
        print np.std(combined), np.std(combined[500:561, 500:561]), np.std(combined[300:361, 300:361])

        results = testNoFlatfieldingEffects(log, oversample=4.0, file='data/psf4x.fits', psfs=400)

    if plots:
        if not run:
            results = cPickle.load(open('flatfieldResults.pk'))
        plotNumberOfFrames(results)

    log.info('Run finished...\n\n\n')
