"""
A simple script to analyse the impact of PSF resolution.

:requires: NumPy
:requires: PyFITS:
:requires: matplotlib
:requires: VIS-PP

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
import pyfits as pf
import numpy as np
from analysis import shape
from support import logger as lg
from support import files as fileIO
import glob, cPickle, itertools


def calculateShapes(log, files, catalogue, cutout=85, iterations=50, sigma=0.75):
    """
    Calculate the shape and size of the PSFs in the files at the positions of the catalogue.

    :param log: logger instance
    :type log: instance
    :param files: names of the FITS files to process
    :type files: list
    :param catalogue: name of the input catalogue with object positions
    :type catalogue: str
    :param cutout: size of the cutout region [centre-cutout:centre+cutout+1]
    :type cutout: int
    :param iterations: number of iterations in the shape measurement
    :type iterations: int
    :param sigma: size of the gaussian weighting function
    :type sigma: float

    :return: [resolution, xcentre, ycentre, shape dictionary]
    :rtype: list
    """
    #shape measurement settings
    settings = dict(itereations=iterations, sigma=sigma)

    cat = np.loadtxt(catalogue)
    x = cat[:, 0]
    y = cat[:, 1]

    results = []

    for xc, yc in zip(x, y):
        for f in files:
            fh = pf.open(f, memmap=True)
            data = fh[1].data[yc-cutout:yc+cutout+1, xc-cutout:xc+cutout+1]
            reso = fh[1].header['PSFOVER']
            fh.close()

            sh = shape.shapeMeasurement(data, log, **settings)
            r = sh.measureRefinedEllipticity()

            results.append([reso, xc, yc, r])

    return results


def plotResults(results):
    """
    plot results

    :param results: results from calculateShapes
    :type results: list

    :return: None
    """
    xpos = []
    res = []
    e1 = []
    e2 = []
    e = []
    R2 = []
    for reso, xc, yc, d in results:
        xpos.append(xc)
        res.append(reso)
        e1.append(d['e1'])
        e2.append(d['e2'])
        e.append(d['ellipticity'])
        R2.append(d['R2'])

    xpos = np.asarray(xpos)
    res = np.asarray(res)
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    e = np.asarray(e)
    R2 = np.asarray(R2)

    #markers to loop over
    marker = itertools.cycle(('s', 'h', 'D', 'o', '*'))

    #e1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r', 'k', 'm', 'c', 'g'])

    for i in sorted(set(res)):
        msk = res == i
        ax.plot(xpos[msk]-np.round(xpos[msk], decimals=0), e1[msk],
                marker=marker.next(), linestyle='', label='Sampling=%i' % i)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel('X position')
    ax.set_ylabel(r'$e_{1}$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='best')
    plt.savefig('e1.pdf')
    plt.close()

    #e2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r', 'k', 'm', 'c', 'g'])

    for i in sorted(set(res)):
        msk = res == i
        ax.plot(xpos[msk]-np.round(xpos[msk], decimals=0), e2[msk],
                marker=marker.next(), linestyle='', label='Sampling=%i' % i)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel('X position')
    ax.set_ylabel(r'$e_{2}$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='best')
    plt.savefig('e2.pdf')
    plt.close()

    #e
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r', 'k', 'm', 'c', 'g'])

    for i in sorted(set(res)):
        msk = res == i
        ax.plot(xpos[msk]-np.round(xpos[msk], decimals=0), e[msk],
                marker=marker.next(), linestyle='', label='Sampling=%i' % i)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel('X position')
    ax.set_ylabel(r'$e$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='best')
    plt.savefig('e.pdf')
    plt.close()

    #eR2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r', 'k', 'm', 'c', 'g'])

    for i in sorted(set(res)):
        msk = res == i
        ax.plot(xpos[msk]-np.round(xpos[msk], decimals=0), R2[msk],
                marker=marker.next(), linestyle='', label='Sampling=%i' % i)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel('X position')
    ax.set_ylabel(r'$R^{2}$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='best')
    plt.savefig('R2.pdf')
    plt.close()


if __name__ == "__main__":
    log = lg.setUpLogger('resolutionTesting.log')

    #calculate, save and load results
    res = calculateShapes(log, glob.glob('Q0*stars*x.fits'), 'test.dat')
    fileIO.cPickleDumpDictionary(res, 'results.pk')
    res = cPickle.load(open('results.pk'))

    #plot results
    plotResults(res)