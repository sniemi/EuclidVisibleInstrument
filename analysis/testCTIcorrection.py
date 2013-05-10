"""
Testing the CTI Correction Algorithm
====================================

This script can be used to test the CTI correction algorithm performance.

:requires: NumPy
:requires: PyFITS
:requires: matplotlib

:version: 0.1

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
import glob as g
import pyfits as pf
import numpy as np
from analysis import shape
from support import logger as lg
from support import files as fileIO


def testCTIcorrection(log, files, sigma=0.75, iterations=4, xcen=1900, ycen=1900, side=20):
    """
    Calculates PSF properties such as ellipticity and size from data without CTI and from
    reduced data.

    :param log: python logger instance
    :type log: instance
    :param files: a list of files to be processed
    :type files: list
    :param sigma: size of the Gaussian weighting function
    :type sigma: float
    :param iterations: the number of iterations for the moment based shape estimator
    :type iterations: int
    :param xcen: x-coordinate of the object centre
    :type xcen: int
    :param ycen: y-coordinate of the object centre
    :type ycen: int
    :param side: size of the cutout around the centre (+/- side)
    :type side: int

    :return: ellipticity and size
    :rtype: dict
    """
    settings = dict(sigma=sigma, iterations=iterations)

    eclean = []
    e1clean = []
    e2clean = []
    R2clean = []
    ereduced = []
    e1reduced = []
    e2reduced = []
    R2reduced = []
    for file in files:
        #load no cti data
        nocti = pf.getdata(file.replace('reduced', 'nocti'))[ycen-side:ycen+side, xcen-side:xcen+side]
        #subtract background
        nocti -= 27.765714285714285
        nocti[nocti < 0.] = 0.  #remove negative numbers

        #load reduced data
        reduced = pf.getdata(file)[ycen-side:ycen+side, xcen-side:xcen+side]
        reduced[reduced < 0.] = 0. #remove negative numbers

        sh = shape.shapeMeasurement(nocti, log, **settings)
        results = sh.measureRefinedEllipticity()

        eclean.append(results['ellipticity'])
        e1clean.append(results['e1'])
        e2clean.append(results['e2'])
        R2clean.append(results['R2'])

        sh = shape.shapeMeasurement(reduced, log, **settings)
        results = sh.measureRefinedEllipticity()

        ereduced.append(results['ellipticity'])
        e1reduced.append(results['e1'])
        e2reduced.append(results['e2'])
        R2reduced.append(results['R2'])

    results = {'eclean' : np.asarray(eclean),
               'e1clean' : np.asarray(e1clean),
               'e2clean' : np.asarray(e2clean),
               'R2clean' : np.asarray(R2clean),
               'ereduced' : np.asarray(ereduced),
               'e1reduced' : np.asarray(e1reduced),
               'e2reduced' : np.asarray(e2reduced),
               'R2reduced' : np.asarray(R2reduced)}

    #save to a file
    fileIO.cPickleDumpDictionary(results, 'results.pk')

    return results


def plotResults(results):
    """
    Plot the CTI correction algorithm results.

    :param results: CTI test results
    :return: None
    """
    e = results['eclean'] - results['ereduced']
    e1 = results['e1clean'] - results['e1reduced']
    e2 = results['e2clean'] - results['e2reduced']

    print 'Delta e, e_1, e_2:', np.mean(e), np.mean(e1), np.mean(e2)
    print 'std e, e_1, e_2:', np.std(e), np.std(e1), np.std(e2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(e, bins=15, label='$e$', alpha=0.5)
    ax.hist(e1, bins=15, label='$e_{2}$', alpha=0.5)
    ax.hist(e2, bins=15, label='$e_{1}$', alpha=0.5)
    ax.set_xlabel(r'$\delta e$ [no CTI - CDM03 corrected]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('ellipticityDelta.pdf')
    plt.close()

    r2 = (results['R2clean'] - results['R2reduced'])/results['R2clean']
    print 'delta R2 / R2: mean, std ', np.mean(r2), np.std(r2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(r2, bins=15, label='$R^{2}$')
    ax.set_xlabel(r'$\frac{\delta R^{2}}{R^{2}_{ref}}$ [no CTI - CDM03 corrected]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('sizeDelta.pdf')
    plt.close()


if __name__ == '__main__':
    log = lg.setUpLogger('testShapeMeasurement.log')

    results = testCTIcorrection(log, g.glob('reducedQ0_00_00stars*'), iterations=8, side=25)
    plotResults(results)