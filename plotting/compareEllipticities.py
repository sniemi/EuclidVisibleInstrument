"""
Simple script to compare ellipticies found with the analyse script
after creating some fake galaxies with generateGalaxies.py.

:reqiures: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
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
matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def loadData(filename):
    """
    Load data after galaxies.dat and ellipticities.txt has been joined.
    Assumed data columns are:

        1. derived x-coordinates
        2. derived y-coordinates
        3. derived ellipticity
        4. derived R2
        5. input x-coordinate
        6. input y-coordinate
        7. input magnitude
        8. input profile
        9. input radius
        10. input axial ratio
        11. input angle
        12. input theta
        13. distance separatation

    :return: data
    :rtype: ndarray
    """
    data = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6,8,9,10))
    return data


def deriveEllipticityFromAxialRatio(ar, theta):
    """
    Derives ellipticity from axial ration and position angle theta.

    :param ar: axial rations
    :type ar: float or ndarray
    :parma theta: position angle [deg]
    :type theta: float or ndarray

    :return: ellipticity
    :rtype: float or ndarray
    """
    e1 = (1. - ar**2) / (1. + ar**2) * np.cos(2.*np.deg2rad(theta))
    e2 = (1. - ar**2) / (1. + ar**2) * np.sin(2.*np.deg2rad(theta))
    return np.sqrt(e1*e1 + e2*e2)


def generatePlots(data):
    """
    Generate some plots from the input data. The assumed data columns are:

        1. derived x-coordinates
        2. derived y-coordinates
        3. derived ellipticity
        4. derived R2
        5. input x-coordinate
        6. input y-coordinate
        7. input magnitude
        8. input radius
        9. input axial ratio
        10. input angle

    :return: None
    """
    #derived values
    xder = data[:,0]
    yder = data[:,1]
    eder = data[:,2]
    #inputs
    xinp = data[:,4]
    yinp = data[:,5]
    maginp = data[:,6]
    radinp = data[:,7]
    ar = data[:,8]
    theta = data[:,9]
    einp = deriveEllipticityFromAxialRatio(ar, theta)

    #centroinding difference
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(xder-xinp, yder-yinp, c=maginp, s=20, marker='o',
                    cmap=cm.get_cmap('jet'), edgecolor='none', alpha=0.6)
    c1 = fig.colorbar(sc, shrink=0.7, fraction=0.05)
    c1.set_label('Magnitude')
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r'$\Delta x$ [pixels]')
    ax.set_ylabel(r'$\Delta y$ [pixels]')
    plt.savefig('Centroiding.pdf')
    plt.close()

    #ellipticity distributions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(eder, bins=20, alpha=0.1, normed=True, hatch='x', label='Derived')
    ax.hist(einp, bins=20, alpha=0.3, normed=True, hatch='/', label='Input')
    ax.set_xlabel('Ellipticity')
    ax.set_ylabel('Probability Density')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('Ellipticities.pdf')
    plt.close()

    #ellipticity difference as a  function of magnitude
    fig = plt.figure()
    fig.subplots_adjust(left=0.15, bottom=0.08,
                        right=0.93, top=0.95,
                        wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(111)
    sc = ax.scatter(maginp, einp/eder, c=radinp, s=20, marker='o',
                    cmap=cm.get_cmap('jet'), edgecolor='none', alpha=0.6)
    c1 = fig.colorbar(sc, shrink=0.7, fraction=0.05)
    c1.set_label('Radius')
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel(r'$\frac{e_\textrm{in}}{e_\textrm{der}}$')
    plt.savefig('EllipticityDifference.pdf')
    plt.close()




if __name__ == '__main__':

    data = loadData('joined.txt')
    generatePlots(data)