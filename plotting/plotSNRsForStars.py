"""
This script can be used to generate simple plots.
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
import datetime
import numpy as np
from support import sextutils


def MagnitudeDistribution(catalog, mag=18., bins=16):
    """
    A simple plot to compare input and extracted magnitudes for a fixed magnitude stars.
    """
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.hist(catalog.mag_aper-mag, bins=bins, label='r=0.65 Aperture', alpha=0.2, normed=True, color='b')
    ax1.axvline(x=np.mean(catalog.mag_aper-mag), c='b' ,ls='-')
    ax1.hist(catalog.mag_auto-mag, bins=bins, label='Auto', alpha=0.3, normed=True, color='r')
    ax1.axvline(x=np.mean(catalog.mag_auto-mag), c='r', ls='-')

    #print np.std(catalog.mag_aper), np.std(catalog.mag_auto)

    ax1.set_xlabel('SExtractor Magnitude - Input Catalog')
    ax1.set_ylabel('PDF')

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')
    plt.savefig('MagDistributionSExtractor.pdf')
    plt.close()


def plotSourceFinderResults(file='objects.phot', mag=18., bins=12):
    """
    """
    data = sextutils.sextractor(file)

    offs = data.magnitude-mag
    xpos = np.mean(offs)
    std = np.std(data.magnitude)
    #print std, 1./std, xpos, np.median(offs)

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.hist(offs, bins=bins, label='r=0.65 Aperture', alpha=0.2, normed=True, color='b')
    ax1.axvline(x=xpos, c='b' ,ls='-')

    ax1.set_xlabel('Aperture Corrected Magnitude - Input Catalog')
    ax1.set_ylabel('PDF')

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')
    plt.savefig('MagDistributionSourceFinder.pdf')
    plt.close()

    counts = data.counts / 0.925969 - 1527.57315282 #for 24.5mag
    #counts = data.counts / 0.925969 - 608137.825681  #for 18mag
    xpos = np.mean(counts)
    std = np.std(counts)
    print np.mean(data.counts/0.925969)/std, np.mean(data.counts)/std, 1527.57315282/std

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.hist(counts, bins=bins, label='r=0.65 Aperture', alpha=0.2, normed=True, color='b')
    ax1.axvline(x=xpos, c='b' ,ls='-')

    ax1.set_xlabel('Aperture Correct Counts - Input Catalog')
    ax1.set_ylabel('PDF')

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')
    plt.savefig('CountDistributionSourceFinder.pdf')
    plt.close()


    avg = np.mean(data.ellipticity)
    std = np.std(data.ellipticity)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.hist(data.ellipticity, bins=bins, alpha=0.2, normed=True, color='b')
    ax1.axvline(x=avg, c='b' ,ls='-')

    ax1.text(ax1.get_xlim()[0]*1.02, ax1.get_ylim()[1]*0.95, r'$\bar{e} = %f$' % avg)
    ax1.text(ax1.get_xlim()[0]*1.02, ax1.get_ylim()[1]*0.9, r'$\sigma = %f$' % std)

    ax1.set_xlabel('Derived Ellipticity')
    ax1.set_ylabel('PDF')

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    plt.savefig('EllipticityDistributionSourceFinder.pdf')
    plt.close()


if __name__ == '__main__':
    cat = sextutils.sextractor('mergedNew.dat')
    #MagnitudeDistribution(cat)
    #plotSourceFinderResults()

    MagnitudeDistribution(cat, mag=24.5)
    plotSourceFinderResults(mag=24.5)
