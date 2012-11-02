"""
This script can be used to plot the distributions from which the cosmic rays are drawn from.

:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.2

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
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def readCosmicRayInformation(lengths, totals):
    """
    Reads in the cosmic ray track information from two input files.
    Stores the information to a dictionary and returns it.

    :param lengths: name of the file containing information about the lengths of cosmic rays
    :type lengths: str
    :param totals: name of the file containing information about the total energies of cosmic rays
    :type totals: str

    :return: cosmic ray track information
    :rtype: dict
    """
    crLengths = np.loadtxt(lengths)
    crDists = np.loadtxt(totals)

    return dict(cr_u=crLengths[:, 0], cr_cdf=crLengths[:, 1], cr_cdfn=np.shape(crLengths)[0],
                cr_v=crDists[:, 0], cr_cde=crDists[:, 1], cr_cden=np.shape(crDists)[0])


def plotCosmicRayInformation(data):
    """
    Generates plots to show the cosmic ray track length and energy distributions.

    :param data: cosmic ray information
    :return: None
    """
    fig = plt.figure()
    plt.title('Cosmic Ray Track Lengths')
    ax = fig.add_subplot(111)
    ax.semilogx(data['cr_u'], data['cr_cdf'])
    ax.set_xlabel('Length [pixels]')
    ax.set_ylabel('Cumulative Distribution Function')
    plt.savefig('LengthDistribution.pdf')
    plt.close()

    fig = plt.figure()
    plt.title('Cosmic Ray Track Energies')
    ax = fig.add_subplot(111)
    ax.semilogx(data['cr_v'], data['cr_cde'])
    ax.set_xlabel('Total Energy [counts]')
    ax.set_ylabel('Cumulative Distribution Function')
    plt.savefig('EnergyDistribution.pdf')
    plt.close()

    #for a single VIS quadrant
    cr_n = 2048 * 2066 * 0.014 / 43.263316 * 2.
    print int(np.floor(cr_n))

    #choose the length of the tracks
    #pseudo-random number taken from a uniform distribution between 0 and 1
    luck = np.random.rand(int(np.floor(cr_n)))

    #interpolate to right values
    ius = InterpolatedUnivariateSpline(data['cr_cdf'], data['cr_u'])
    data['cr_l'] = ius(luck)
    ius = InterpolatedUnivariateSpline(data['cr_cde'], data['cr_v'])
    data['cr_e'] = ius(luck)

    fig = plt.figure()
    plt.title('Cosmic Ray Track Energies (a single quadrant)')
    ax = fig.add_subplot(111)
    #ax.hist(np.log10(data['cr_e']), bins=35, normed=True)
    ax.hist(np.log10(data['cr_e']), bins=35)
    ax.set_xlabel(r'$\log_{10}($Total Energy [counts]$)$')
    #ax.set_ylabel('PDF')
    ax.set_ylabel(r'\#')
    plt.savefig('SingleQuadrantEnergies.pdf')
    plt.close()

    fig = plt.figure()
    plt.title('Cosmic Ray Track Lengths (a single quadrant)')
    ax = fig.add_subplot(111)
    #ax.hist(np.log10(data['cr_l']), bins=35, normed=True)
    ax.hist(np.log10(data['cr_l']), bins=35)
    ax.set_xlabel(r'$\log_{10}($Track Lengths [pixels]$)$')
    #ax.set_ylabel('PDF')
    ax.set_ylabel(r'\#')
    plt.savefig('SingleQuadrantLengths.pdf')
    plt.close()


if __name__ == '__main__':

    lengths = 'data/cdf_cr_length.dat'
    totals = 'data/cdf_cr_total.dat'

    cosmics = readCosmicRayInformation(lengths, totals)

    plotCosmicRayInformation(cosmics)