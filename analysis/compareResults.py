"""
Simple script that can be used to compare ellipticities.

:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import matplotlib
matplotlib.use('PDF')
import numpy as np
from support import sextutils
import matplotlib.pyplot as plt


def readData(SExtractorfile, myfile='ellipticities.txt'):
    """

    """
    sextractor = sextutils.se_catalog(SExtractorfile)
    mydata = sextutils.se_catalog(myfile)
    return sextractor, mydata


def plotResults(sexdata, mydata):
    """

    """
    xcent = sexdata.x_image - mydata.x
    ycent = sexdata.y_image - mydata.y
    msk = (xcent < 0.5) & (xcent > -0.5) & (ycent < 0.5) & (ycent > -0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xcent, ycent, 'bo', ms=2)
    ax.set_xlabel('Delta X (SExtractor - Gaussian Weighted)')
    ax.set_ylabel('Delta Y (SExtractor - Gaussian Weighted)')
    plt.savefig('CentroidingDifference.pdf')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig('CentroidingDifferenceZoomed.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(sexdata.ellipticity[msk], bins=15, normed=True, range=(0,1), alpha=0.7, hatch='/', label='SExtractor')
    ax.hist(mydata.ellipticity[msk], bins=15, normed=True, range=(0,1), alpha=0.5, hatch='x', label='Gaussian Weighted Moments')
    ax.set_xlabel('Ellipticity')
    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Ellipticities.pdf')
    plt.close()



if __name__ == '__main__':
    plotResults(*readData('test.cat'))
