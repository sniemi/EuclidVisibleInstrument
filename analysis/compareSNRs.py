"""
This script can be used to compare SNRs of SExtracted pointed sources to the radiometric calculations.
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
from support import sextutils
from analysis import ETC


def readCatalog(file='merged.dat'):
    """

    """
    catalog = sextutils.sextractor(file)
    return catalog


def compareMagnitudes(catalog):
    """

    """
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure(frameon=False)

    left, width = 0.1, 0.8
    rect1 = [left, 0.3, width, 0.65]
    rect2 = [left, 0.1, width, 0.2]

    ax1 = fig.add_axes(rect1, title='VIS Simulator Point Source Magnitudes')
    ax2 = fig.add_axes(rect2)  #left, bottom, width, height

    ax1.plot([14, 26], [14, 26], 'k-')
    ax1.errorbar(catalog.mag_input, catalog.mag_aper, yerr=catalog.magerr_aper,
                 c='b', ls='None', marker='o', ms=3, label='r=0.65 Aperture')
    ax1.errorbar(catalog.mag_input, catalog.mag_auto, yerr=catalog.magerr_auto,
                 c='r', ls='None', marker='s', ms=3, label='Auto')
    ax1.errorbar(catalog.mag_input, catalog.mag_best, yerr=catalog.magerr_best,
                 c='g', ls='None', marker='D', ms=3, label='Best')

    ax2.plot([14, 26], [0, 0], 'k-')
    ax2.errorbar(catalog.mag_input, catalog.mag_aper-catalog.mag_input, yerr=catalog.magerr_aper,
                 c='b', ls='None', marker='o', ms=3, label='r=0.65 Aperture')
    ax2.errorbar(catalog.mag_input, catalog.mag_auto-catalog.mag_input, yerr=catalog.magerr_auto,
                 c='r', ls='None', marker='s', ms=3, label='Auto')
    ax2.errorbar(catalog.mag_input, catalog.mag_best-catalog.mag_input, yerr=catalog.magerr_best,
                 c='g', ls='None', marker='D', ms=3, label='Best')

    ax2.set_xlabel('Input Magnitude')
    ax1.set_ylabel('Extracted Magnitude')
    ax2.set_ylabel('Ext - Inp')

    ax1.set_xticklabels([])
    ax1.set_yticks(ax1.get_yticks()[1:])
    ax2.set_yticks(ax2.get_yticks()[::2])

    ax1.set_xlim(14.5, 25.5)
    ax2.set_xlim(14.5, 25.5)

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')
    plt.savefig('magnitudes.pdf')
    plt.close()


def compareSNR(catalog):
    """

    """
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    #calculate input SNRs
    SNRs = ETC.SNR(ETC.VISinformation(), magnitude=catalog.mag_input, exposures=1, galaxy=False)
    Sextracted = catalog.flux_aper / catalog.fluxerr_aper

    fig = plt.figure(frameon=False)

    left, width = 0.1, 0.8
    rect1 = [left, 0.3, width, 0.65]
    rect2 = [left, 0.1, width, 0.2]

    ax1 = fig.add_axes(rect1, title='VIS Simulator SNR Comparison')
    ax2 = fig.add_axes(rect2)  #left, bottom, width, height

    ax1.plot([0, 3100], [0, 3100], 'k--')
    ax1.scatter(SNRs, Sextracted, c='b', marker='o', s=10, edgecolor='None', label='r=0.65 Aperture')

    ax2.plot([0, 3100], [1, 1], 'k--')
    ax2.scatter(SNRs, Sextracted / SNRs, c='b', marker='o', s=10, edgecolor='None')

    ax2.set_xlabel('Input SNR')
    ax1.set_ylabel('Extracted SNR')
    ax2.set_ylabel('Ext / Inp')

    ax1.set_xticklabels([])
    ax1.set_yticks(ax1.get_yticks()[1:])
    #ax2.set_yticks(ax2.get_yticks()[::2])

    ax1.set_xlim(-10, 3100)
    ax1.set_ylim(-50, 3100)
    ax2.set_xlim(-10, 3100)
    ax2.set_ylim(0.55, 1.45)

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')

    plt.savefig('SNRs.pdf')
    plt.close()




if __name__ == '__main__':
    cat = readCatalog()
    #compareMagnitudes(cat)
    compareSNR(cat)