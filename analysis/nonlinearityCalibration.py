"""
Non-linearity Calibration
=========================

This simple script can be used to study the error in the non-linearity correction that can be tolerated given the
requirements.

The following requirements related to the non-linearity has been taken from GDPRD.

R-GDP-CAL-058: The contribution of the residuals of the non-linearity correction on the error on the determination
of each ellipticity component of the local PSF shall not exceed 3x10**-5 (one sigma).

R-GDP-CAL-068: The contribution of the residuals of the non-linearity correction on the error on the relative
error sigma(R**2)/R**2 on the determination of the local PSF R**2 shall not exceed 1x10**-4 (one sigma).

:requires: PyFITS
:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.8

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
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools, os, sys
from analysis import shape
from support import logger as lg
from support import files as fileIO


def testNonlinearity(log, file='data/psf12x.fits', oversample=12.0, norm=1.8e5,
                     phases=500, amps=30, multiplier=12, minerror=-4., maxerror=-1.6, linspace=False):
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
    :param norm: PSF normalization constant that the peak pixel will take.
    :type norm: float
    :param phases: the number of error curves to draw with random phases.
    :type phases: int
    :param amps: the number of individual samplings used when covering the error space
    :type amps: int
    :param multiplier: the number of angular frequencies to be used
    :type multiplier: int
    :param minerror: the minimum error to be covered, given in log10(min_error) [default=-4 i.e. 0.01%]
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
    data *= norm

    #derive reference values from clean PSF
    settings = dict(sampling=1.0/oversample)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()
    print reference

    #range of amplitude to study
    if linspace:
        amplitudes = np.linspace(10**minerror, 1**maxerror, amps)
    else:
        amplitudes = np.logspace(minerror, maxerror, amps)

    #PSF range for sin function that will be phased
    var = data.copy()/data.max() * multiplier * math.pi

    out = {}
    #loop over all the amplitudes to be studied
    for i, amp in enumerate(amplitudes):
        de1 = []
        de2 = []
        de = []
        R2 = []
        dR2 = []
        e1 = []
        e2 = []
        e = []
        #random phases to Monte Carlo
        ph = np.random.random(phases)
        print'%i / %i: %e' % (i+1, amps, amp)
        for phase in ph:
            #derive the non-linearity error curve, scaled to the amplitude being studied
            non_line = amp*np.sin(var.copy() + phase*2.*math.pi)

            #multiply the original PSF with the non-linearity error and add to the PSF
            newdata = data.copy() * non_line + data.copy()

            #measure e and R2 from the postage stamp image
            sh = shape.shapeMeasurement(newdata.copy(), log, **settings)
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

        out[amp] = [e1, e2, e, R2, de1, de2, de, dR2]

    return reference, out


def plotResults(results, reqe=3e-5, reqr2=1e-4, outdir='results'):
    """
    Creates a simple plot to combine and show the results.

    :param res: results to be plotted [reference values, results dictionary]
    :type res: list
    :param reqe: the requirement for ellipticity [default=3e-5]
    :type reqe: float
    :param reqr2: the requirement for size R2 [default=1e-4]
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
    for key in keys:
        e1 = np.asarray(res[key][0])
        e2 = np.asarray(res[key][1])
        e = np.asarray(res[key][2])

        std1 = np.std(e1)
        std2 = np.std(e2)
        std = np.std(e)

        ax.scatter(key*0.9, std, c='m', marker='*')
        ax.scatter(key, std1, c='b', marker='o')
        ax.scatter(key, std2, c='y', marker='s')

        print key, std, std1, std2

    #label
    ax.scatter(key*0.9, std, c='m', marker='*', label=r'$\sigma (e)$')
    ax.scatter(key, std1, c='b', marker='o', label=r'$\sigma (e_{1})$')
    ax.scatter(key, std2, c='y', marker='s', label=r'$\sigma (e_{2})$')

    ax.axhline(y=reqe, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-3)
    ax.set_xlabel('Error in the Non-linearity Correction')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    xmin, xmax = ax.get_xlim()
    ax.fill_between(np.linspace(xmin, xmax, 10), np.ones(10)*reqe, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*1.05, 0.9*reqe, '%.1e' % reqe, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2)
    plt.savefig(outdir+'/NonLinCalibrationsigmaE.pdf')
    plt.close()

    #same for R2s
    fig = plt.figure()
    plt.title(r'VIS Non-linearity Calibration: $\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax = fig.add_subplot(111)

    ax.axhline(y=0, c='k', ls=':')

    #loop over
    keys = res.keys()
    keys.sort()
    for key in keys:
        dR2 = np.asarray(res[key][3])
        std = np.std(dR2) / (ref['R2']**2)
        print key, std
        ax.scatter(key, std, c='b', marker='s', s=35, zorder=10)

    ax.scatter(key, std, c='b', marker='s', label=r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax.axhline(y=reqr2, c='g', ls='--', label='Requirement')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-6, 1e-3)
    ax.set_xlabel('Error in the Non-linearity Correction')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    ax.fill_between(np.linspace(xmin, xmax, 10), np.ones(10)*reqr2, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*1.05, 0.9*reqr2, '%.1e' % reqr2, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8    )
    plt.savefig(outdir+'/NonLinCalibrationSigmaR2.pdf')
    plt.close()


if __name__ == '__main__':
    run = True

    #start the script
    log = lg.setUpLogger('nonlinearityCalibration.log')
    log.info('Testing non-linearity calibration...')

    if run:
        #res = testNonlinearity(log, phases=300, amps=15, file='data/psf3x.fits', oversample=3.0, multiplier=48)
        res = testNonlinearity(log, phases=600, amps=15)
        fileIO.cPickleDumpDictionary(res, 'nonlinResults.pk')
    else:
        res = cPickle.load(open('nonlinResults.pk'))

    plotResults(res)

    log.info('Run finished...\n\n\n')
