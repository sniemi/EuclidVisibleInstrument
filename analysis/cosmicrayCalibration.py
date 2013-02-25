"""
Cosmic Ray Rejection
====================

This simple script can be used to study the error in the cosmic ray rejection that can be tolerated given the
requirements.

The following requirements related to the cosmic rays have been taken from CalCD-B.

Note that the analysis is for a single star. Thus, if we consider a single exposure we can
relax the requirements given that there will be on average about 1850 stars in each field that
are usable for PSF modelling. Furthermore it shuold be noted that the presence of cosmic rays
will always increase the size. Because of this one cannot combine the contribution from cosmic
rays with other effects by adding each individual contribution in quadrature. It is more
appropriate to add the impact of cosmic rays to the size of the PSF linearly given the preferred
direction.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.3

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
import datetime, cPickle, os, pprint
from analysis import shape
from scipy import interpolate
from support import logger as lg
from support import files as fileIO
from support import cosmicrays


def testCosmicrayRejection(log, file='data/psf1x.fits', oversample=1.0, sigma=0.75, psfs=20000, scale=1e3,
                           min=1e-5, max=50, levels=15, covering=1.4, single=False):
    #read in PSF and renormalize it to norm
    data = pf.getdata(file)
    data /= np.max(data)

    #derive reference values from clean PSF
    settings = dict(sampling=1.0/oversample, sigma=sigma, iterations=6)
    scaled = data.copy() * scale
    sh = shape.shapeMeasurement(scaled.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()

    cosmics = cosmicrays.cosmicrays(log, np.zeros((2,2)))
    crInfo = cosmics._readCosmicrayInformation()

    out = {}
    #loop over all the amplitudes to be studied
    for level in np.logspace(np.log10(min), np.log10(max), levels):
        print 'Deposited Energy of Cosmic Rays: %i electrons' % level

        de1 = []
        de2 = []
        de = []
        R2 = []
        dR2 = []
        e1 = []
        e2 = []
        e = []

        for i in range(psfs):
            print'Run %i / %i' % (i + 1, psfs)

            #add cosmic rays to the scaled image
            cosmics = cosmicrays.cosmicrays(log, scaled, crInfo=crInfo)
            #newdata = cosmics.addCosmicRays(limit=level)
            if single:
                newdata = cosmics.addSingleEvent(limit=level)
            else:
                newdata = cosmics.addUpToFraction(covering, limit=level)

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

        out[level] = [e1, e2, e, R2, de1, de2, de, dR2]

    return reference, out


def test(log, file='data/psf1x.fits', oversample=1.0, sigma=0.75, scale=1e2, level=10, covering=1.4, single=False):
    #read in PSF and renormalize it to norm
    data = pf.getdata(file)
    data /= np.max(data)

    #derive reference values from clean PSF
    settings = dict(sampling=1.0 / oversample, sigma=sigma, iterations=10)
    scaled = data.copy() * scale
    sh = shape.shapeMeasurement(scaled.copy(), log, **settings)
    reference = sh.measureRefinedEllipticity()
    print 'Reference:'
    pprint.pprint(reference)

    cosmics = cosmicrays.cosmicrays(log, np.zeros((2, 2)))
    crInfo = cosmics._readCosmicrayInformation()

    print 'Deposited Energy of Cosmic Rays: %i electrons' % level
    #add cosmic rays to the scaled image
    cosmics = cosmicrays.cosmicrays(log, scaled, crInfo=crInfo)
    if single:
        #only one cosmic with a given energy level, length drawn from a distribution
        newdata = cosmics.addSingleEvent(limit=level)
    else:
        #x cosmic ray events to reach a covering fraction, say 1.4 per cent
        newdata = cosmics.addUpToFraction(covering, limit=level)

    #write out new data for inspection
    fileIO.writeFITS(newdata, 'example.fits', int=False)

    #measure e and R2 from the postage stamp image
    sh = shape.shapeMeasurement(newdata.copy(), log, **settings)
    results = sh.measureRefinedEllipticity()
    print 'Results:'
    pprint.pprint(results)

    print 'delta e_1: ', results['e1'] - reference['e1']
    print 'delta e_2: ', results['e2'] - reference['e2']
    print 'delta e: ', results['ellipticity'] - reference['ellipticity']
    print 'delta R**2: ', results['R2'] - reference['R2']


def plotResults(results, reqe=3e-5, reqr2=1e-4, outdir='results', timeStamp=False):
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
    plt.title(r'VIS Cosmic Ray Rejection: $\sigma(e)$')
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
    x = np.logspace(np.log10(ks.min()*1.001), np.log10(ks.max()*0.995), 5000)
    vals = f(x)
    ax.plot(x, vals, '--', c='0.2', zorder=10)
    msk = vals < reqe
    maxn = np.max(x[msk])
    plt.text(1, 2e-5, r'Error for $e$ must be $\leq %.2e$ electrons' % (maxn),
             fontsize=11, ha='center', va='center')

    #label
    ax.scatter(key, std, c='m', marker='*', label=r'$\sigma (e)$')
    ax.scatter(key*1.1, std1, c='b', marker='o', label=r'$\sigma (e_{1})$')
    ax.scatter(key, std2, c='y', marker='s', label=r'$\sigma (e_{2})$')

    ax.axhline(y=reqe, c='g', ls='--', label='Requirement')

    ax.set_xlabel('Total Deposited Energy of the undetected Cosmic Rays')
    ax.set_ylabel(r'$\sigma (e_{i})\ , \ \ \ i \in [1,2]$')

    xmin, xmax = 1e-4, 1e2 # ax.get_xlim()
    ax.fill_between(np.linspace(xmin, xmax, 5), np.ones(5)*reqe, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*3.0, 0.9*reqe, '%.1e' % reqe, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1e-2)
    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=2.0, ncol=2, loc='upper left')
    plt.savefig(outdir+'/CosmicraySigmaE.pdf')
    plt.close()

    #same for R2s
    fig = plt.figure()
    plt.title(r'VIS Cosmic Ray Rejection: $\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
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
    x = np.logspace(np.log10(ks.min()*1.001), np.log10(ks.max()*0.995), 5000)
    vals = f(x)
    ax.plot(x, vals, '--', c='0.2', zorder=10)
    msk = vals < reqr2
    maxn = np.max(x[msk])
    plt.text(1, 7e-5, r'Error for $e$ must be $\leq %.2e$ electrons' % (maxn),
             fontsize=11, ha='center', va='center')

    ax.scatter(key, std, c='b', marker='s', label=r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')
    ax.axhline(y=reqr2, c='g', ls='--', label='Requirement')

    ax.set_xlabel('Total Deposited Energy of the undetected Cosmic Rays')
    ax.set_ylabel(r'$\frac{\sigma (R^{2})}{R_{ref}^{2}}$')

    ax.fill_between(np.linspace(xmin, xmax, 10), np.ones(10)*reqr2, 1.0, facecolor='red', alpha=0.08)
    plt.text(xmin*3., 0.9*reqr2, '%.1e' % reqr2, ha='left', va='top', fontsize=11)
    ax.set_xlim(xmin, xmax)

    if timeStamp:
        plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(6e-6, 1e-2)
    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig(outdir+'/CosmicraySigmaR2.pdf')
    plt.close()

    print '\nDelta results:'
    for i, key in enumerate(res):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.title(r'VIS Cosmic Ray Rejection (%f): $\delta e$' % key)

        de1 = np.asarray(res[key][4])
        de2 = np.asarray(res[key][5])
        de = np.asarray(res[key][6])

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
        plt.savefig(outdir + '/CosmicrayDeltaE%i.pdf' % i)
        plt.close()

    #same for R2s
    for i, key in enumerate(res):
        fig = plt.figure()
        plt.title(r'VIS Cosmic Ray Rejection (%f): $\frac{\delta R^{2}}{R_{ref}^{2}}$' % key)
        ax = fig.add_subplot(111)

        dR2 = np.asarray(res[key][7])
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
        plt.savefig(outdir + '/CosmicrayDeltaSize%i.pdf' % i)
        plt.close()


if __name__ == '__main__':
    run = True
    plot = True
    debug = False
    file = 'CosmicrayResults.pk'

    #start a logger
    log = lg.setUpLogger('CosmicrayRejection.log')
    log.info('Testing Cosmic Ray Rejection...')

    if debug:
        test(log)

    if run:
        res = testCosmicrayRejection(log, single=True)
        fileIO.cPickleDumpDictionary(res, file)

    if plot:
        if not run:
            res = cPickle.load(open(file))

        plotResults(res, outdir='results')

    log.info('Run finished...\n\n\n')
