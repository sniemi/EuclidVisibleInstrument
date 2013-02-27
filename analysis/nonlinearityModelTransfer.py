"""
Non-linearity II: Model Transfer
================================

Most PSF stars are bright while most weak lensing galaxies are faint,
thus the PSF model needs to be transferable to an appropriate object luminosity (and colour, position, etc.).
This simple script can be used to study the error in the model transfer due to non-linearity that can be
tolerated given the requirements.

The following requirements related to the model transfer have been taken from CalCD-B.

:requires: PyFITS
:requires: NumPy
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
import datetime, cPickle, os
from analysis import shape
from scipy import interpolate
from support import logger as lg
from support import files as fileIO
from support import VISinstrumentModel


def testNonlinearityModelTransfer(log, file='data/psf12x.fits', oversample=12.0, sigma=0.75,
                                  psfs=5000, amps=12, multiplier=1.5, minerror=-5., maxerror=-1.,
                                  lowflux=500, highflux=180000, linspace=False):
    """
    Function to study the error in the non-linearity correction on the knowledge of the PSF ellipticity and size.

    The error has been assumed to follow a sinusoidal curve with random phase and a given number of angular
    frequencies (defined by the multiplier). The amplitudes being studied, i.e. the size of the maximum deviation,
    can be spaced either linearly or logarithmically.

    :param log: logger instance
    :type log: instance
    :param file: name of the PSF FITS files to use [default=data/psf12x.fits]
    :type file: str
    :param oversample: the PSF oversampling factor, which needs to match the input file [default=12]
    :type ovesample: float
    :param sigma: 1sigma radius of the Gaussian weighting function for shape measurements
    :type sigma: float
    :param phs: phase in case phases = None
    :type phs: float
    :param phases: if None then a single fixed phase will be applied, if an int then a given number of random
                   phases will be used
    :type phases: None or int
    :param psfs: the number of PSFs to use.
    :type psfs: int
    :param amps: the number of individual samplings used when covering the error space
    :type amps: int
    :param multiplier: the number of angular frequencies to be used
    :type multiplier: int or float
    :param minerror: the minimum error to be covered, given in log10(min_error) [default=-5 i.e. 0.001%]
    :type minerror: float
    :param maxerror: the maximum error to be covered, given in log10(max_error) [default=-1 i.e. 10%]
    :type maxerror: float
    :param linspace: whether the amplitudes of the error curves should be linearly or logarithmically spaced.
    :type linspace: boolean

    :return: reference value and results dictionaries
    :rtype: list
    """
    #read in PSF and renormalize it to norm
    data = pf.getdata(file)
    data /= np.max(data)

    #derive reference values from clean PSF
    settings = dict(sampling=1.0/oversample, sigma=sigma)
    sh = shape.shapeMeasurement(data.copy()*175000., log, **settings)
    reference = sh.measureRefinedEllipticity()

    #range of amplitude to study
    if linspace:
        amplitudes = np.linspace(10**minerror, 1**maxerror, amps)[::-1] #flip so that the largest is first
    else:
        amplitudes = np.logspace(minerror, maxerror, amps)[::-1]

    out = {}
    #loop over all the amplitudes to be studied
    for i, amp in enumerate(amplitudes):
        print'Run %i / %i with amplitude=%e' % (i+1, amps, amp)
        R2 = []
        e1 = []
        e2 = []
        e = []

        #random phases to Monte Carlo
        ph = np.random.random(psfs)

        for phase in ph:
            #apply nonlinearity model to the scaled PSF
            scaled = data.copy() * highflux
            newdata = VISinstrumentModel.CCDnonLinearityModelSinusoidal(scaled, amp, phase=phase, multi=multiplier)

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(newdata, log, **settings)
            resultsH = sh.measureRefinedEllipticity()

            #apply nonlinearity model to the scaled PSF
            scaled = data.copy() * lowflux
            newdata = VISinstrumentModel.CCDnonLinearityModelSinusoidal(scaled, amp, phase=phase, multi=multiplier)

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(newdata, log, **settings)
            resultsL = sh.measureRefinedEllipticity()

            #save values
            e1.append(resultsH['e1'] - resultsL['e1'])
            e2.append(resultsH['e2'] - resultsL['e2'])
            e.append(resultsH['ellipticity'] - resultsL['ellipticity'])
            R2.append(resultsH['R2'] - resultsL['R2'])

        out[amp] = [e1, e2, e, R2]

    return reference, out


def plotResults(results, reqe=3.5e-5, reqr2=3.5e-4, outdir='results', timeStamp=False):
    """
    Creates a simple plot to combine and show the results.

    :param res: results to be plotted [reference values, results dictionary]
    :type res: list
    :param reqe: the requirement for ellipticity [default=3.5e-5]
    :type reqe: float
    :param reqr2: the requirement for size R2 [default=3.5e-4]
    :type reqr2: float
    :param outdir: output directory to which the plots will be saved to
    :type outdir: str

    :return: None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ref = results[0]
    res = results[1]

    print '\nSigma results:'
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    plt.title(r'VIS Non-linearity: $\sigma(e)$')
    ax = fig.add_subplot(111)

    keys = res.keys()
    keys.sort()
    vals = []
    for key in keys:
        e1 = np.asarray(res[key][0])
        e2 = np.asarray(res[key][1])
        e = np.asarray(res[key][2])

        std1 = np.std(e1)
        std2 = np.std(e2)
        std = np.std(e)
        vals.append(std)

        ax.scatter(key*0.9, std, c='m', marker='*')
        ax.scatter(key, std1, c='b', marker='o')
        ax.scatter(key, std2, c='y', marker='s')

        print key, std, std1, std2

    #find the crossing
    ks = np.asarray(keys)
    values = np.asarray(vals)
    f = interpolate.interp1d(ks, values, kind='cubic')
    x = np.logspace(np.log10(ks.min()*1.05), np.log10(ks.max()*0.95), 1000)
    vals = f(x)
    ax.plot(x, vals, '--', c='0.2', zorder=10)
    msk = vals < reqe
    maxn = np.max(x[msk])
    plt.text(1e-3, 2e-5, r'Error for $e$ must be $\leq %.2e$ per cent' % (maxn * 100),
             fontsize=11, ha='center', va='center')

    #label
    ax.scatter(key, std, c='m', marker='*', label=r'$\sigma (e)$')
    ax.scatter(key*1.1, std1, c='b', marker='o', label=r'$\sigma (e_{1})$')
    ax.scatter(key, std2, c='y', marker='s', label=r'$\sigma (e_{2})$')

    ax.axhline(y=reqe, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-3)
    ax.set_xlabel('Error in the Model Transfer Non-linearity')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    xmin, xmax = ax.get_xlim()
    ax.fill_between(np.linspace(xmin, xmax, 10), np.ones(10)*reqe, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*1.05, 0.9*reqe, '%.1e' % reqe, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2, loc='upper left')
    plt.savefig(outdir+'/NonLinModelTransferSigmaE.pdf')
    plt.close()

    #same for R2s
    fig = plt.figure()
    plt.title(r'VIS Non-linearity Calibration: $\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    #loop over
    keys = res.keys()
    keys.sort()
    vals = []
    for key in keys:
        dR2 = np.asarray(res[key][3])
        std = np.std(dR2) / ref['R2']
        vals.append(std)
        print key, std
        ax.scatter(key, std, c='b', marker='s', s=35, zorder=10)

    #find the crossing
    ks = np.asarray(keys)
    values = np.asarray(vals)
    f = interpolate.interp1d(ks, values, kind='cubic')
    x = np.logspace(np.log10(ks.min()*1.05), np.log10(ks.max()*0.95), 1000)
    vals = f(x)
    ax.plot(x, vals, '--', c='0.2', zorder=10)
    msk = vals < reqr2
    maxn = np.max(x[msk])
    plt.text(1e-3, 7e-5, r'Error for $e$ must be $\leq %.2e$ per cent' % (maxn * 100),
             fontsize=11, ha='center', va='center')

    ax.scatter(key, std, c='b', marker='s', label=r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax.axhline(y=reqr2, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(6e-6, 4e-3)
    ax.set_xlabel('Error in the Model Transfer Non-linearity')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    ax.fill_between(np.linspace(xmin, xmax, 10), np.ones(10)*reqr2, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*1.05, 0.9*reqr2, '%.1e' % reqr2, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(outdir+'/NonLinModelTransferSigmaR2.pdf')
    plt.close()

    print '\nDelta results:'
    for i, key in enumerate(res):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.title(r'VIS Non-linearity Correction (%f): $\delta e$' % key)

        de1 = np.asarray(res[key][0])
        de2 = np.asarray(res[key][1])
        de = np.asarray(res[key][2])

        avg1 = np.mean(de1) ** 2
        avg2 = np.mean(de2) ** 2
        avg = np.mean(de) ** 2

        #write down the values
        print i, key, avg, avg1, avg2
        plt.text(0.08, 0.9, r'$\left< \delta e_{1} \right>^{2} = %e$' % avg1, fontsize=10, transform=ax.transAxes)
        plt.text(0.08, 0.85, r'$\left< \delta e_{2}\right>^{2} = %e$' % avg2, fontsize=10, transform=ax.transAxes)
        plt.text(0.08, 0.8, r'$\left< \delta | \bar{e} |\right>^{2} = %e$' % avg, fontsize=10, transform=ax.transAxes)

        ax.hist(de, bins=15, color='y', alpha=0.2, label=r'$\delta | \bar{e} |$', normed=True, log=True)
        ax.hist(de1, bins=15, color='b', alpha=0.5, label=r'$\delta e_{1}$', normed=True, log=True)
        ax.hist(de2, bins=15, color='g', alpha=0.3, label=r'$\delta e_{2}$', normed=True, log=True)

        ax.axvline(x=0, ls=':', c='k')

        ax.set_ylabel('Probability Density')
        ax.set_xlabel(r'$\delta e_{i}\ , \ \ \ i \in [1,2]$')

        if timeStamp:
            plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

        plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
        plt.savefig(outdir + '/NonlinearityModelTransferEDelta%i.pdf' % i)
        plt.close()

    #same for R2s
    for i, key in enumerate(res):
        fig = plt.figure()
        plt.title(r'VIS Non-linearity Correction (%f): $\frac{\delta R^{2}}{R_{ref}^{2}}$' % key)
        ax = fig.add_subplot(111)

        dR2 = np.asarray(res[key][3])
        avg = np.mean(dR2 / ref['R2']) ** 2

        ax.hist(dR2, bins=15, color='y', label=r'$\frac{\delta R^{2}}{R_{ref}^{2}}$', normed=True, log=True)

        print i, key, avg

        plt.text(0.1, 0.9, r'$\left<\frac{\delta R^{2}}{R^{2}_{ref}}\right>^{2} = %e$' % avg,
                 fontsize=10, transform=ax.transAxes)

        ax.axvline(x=0, ls=':', c='k')

        ax.set_ylabel('Probability Density')
        ax.set_xlabel(r'$\frac{\delta R^{2}}{R_{ref}^{2}}$')

        if timeStamp:
            plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

        plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8)
        plt.savefig(outdir + '/NonlinearityModelTransferDeltaSize%i.pdf' % i)
        plt.close()

if __name__ == '__main__':
    run = True
    debug = True
    plot = True

    #different runs
    runs = {'run1': dict(multiplier=1.5),
            'run2': dict(multiplier=0.5),
            'run3': dict(multiplier=2.0),
            'run4': dict(multiplier=3.0),
            'run5': dict(multiplier=4.0)}

    for key, value in runs.iteritems():
        if not os.path.exists(key):
            os.makedirs(key)

        #start a logger
        log = lg.setUpLogger(key+'/nonlinearityModelTransfer.log')
        log.info('Testing non-linearity model transfer...')
        log.info('Multiplier = %f' % value['multiplier'])

        if run:
            if debug:
                res = testNonlinearityModelTransfer(log, psfs=2000, file='data/psf1x.fits', oversample=1.0)
            else:
                res = testNonlinearityModelTransfer(log)

            fileIO.cPickleDumpDictionary(res, key+'/nonlinModelResults.pk')

        if plot:
            if not run:
                res = cPickle.load(open(key+'/nonlinModelResults.pk'))

            plotResults(res, outdir=key)

        log.info('Run finished...\n\n\n')
