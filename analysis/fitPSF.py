"""
PSF Fitting
===========

This script can be used to fit a set of basis functions to a point spread function.

:requires: Scikit-learn
:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.1

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import matplotlib
#matplotlib.rc('text', usetex=True)  #pyMC does not like LaTeX output
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pyfits as pf
from scipy.optimize import curve_fit
import glob as g
from support import files as fileIO


def leastSQfit(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20):
    """
    A simple function returning a linear combination of 20 input parameters.

    :param x: data
    :type x: mdarray
    :param a0: first fitting coefficient
    :param a0: float
    :param a1: second fitting coefficient
    :param a1: float
    :param a2: third fitting coefficient
    :param a2: float
    :param a3: fourth fitting coefficient
    :param a3: float
    :param a4: fifth fitting coefficient
    :param a4: float
    :param a5: sixth fitting coefficient
    :param a5: float
    :param a6: seventh fitting coefficient
    :param a6: float
    :param a7: eight fitting coefficient
    :param a7: float
    :param a8: ninth fitting coefficient
    :param a8: float
    :param a9: tenth fitting coefficient
    :param a9: float
    :param a10: eleventh fitting coefficient
    :param a10: float
    :param a11: twelfth fitting coefficient
    :param a0: float
    :param a12: thirteenth fitting coefficient
    :param a0: float
    :param a13: fourteenth fitting coefficient
    :param a0: float
    :param a14: fifteenth fitting coefficient
    :param a0: float
    :param a15: sixteenth fitting coefficient
    :param a0: float
    :param a16: seventieth fitting coefficient
    :param a0: float
    :param a17: eighteenth fitting coefficient
    :param a0: float
    :param a18: nineteenth fitting coefficient
    :param a0: float
    :param a19: twentieth fitting coefficient
    :param a0: float
    :param a20: twentyfirst fitting coefficient
    :param a0: float

    :return: a linear combination of the data given the coefficients
    :rtype: ndarray
    """
    tmp = a0*x[0] + a1*x[1] + a2*x[2] + a3*x[3] + a4*x[4] + a5*x[5] + a6*x[6] + \
          a7*x[7] + a8*x[8] + a9*x[9] + a10*x[10] + a11*x[11] + a12*x[12] + a13*x[13] +\
          a14*x[14] + a15*x[15] + a16*x[16] + a17*x[17] + a18*x[18] + a19*x[19] + a20*x[20]
    return tmp


def visualise(popt, psf, modedata, output='PCA'):
    """
    Plot the fitting results, residuals and coefficients

    :param popt: fitting coefficients
    :type popt: array
    :param psf: data to which the modes were fitted to
    :type psf: ndarray
    :param modedata: each of the basis set modes that were fit to the data
    :type modedata: ndarray

    :return: None
    """
    #plot coefficients
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.axhline(y=0, c='k', ls=':')
    ax.semilogy(np.arange(len(popt))+1, np.abs(popt), 'bo-')
    ax.set_ylabel('abs(coefficient)')
    ax.set_xlabel('Basis set Coefficients')
    plt.savefig('%scoefficients.pdf' % output)

    #residuals
    fig3D = plt.figure(2, figsize=(7, 7))
    ax3D = Axes3D(fig3D)
    ims = []
    for x in range(len(popt)):
        vals = list(popt[:x+1]) + [0,]*(len(popt)-x-1)
        residual = (psf - leastSQfit(modedata, *vals)).reshape(201, 201)

        #2D figures
        fig = plt.figure()
        ax = fig.add_subplot(111)
        s = ax.imshow(residual, vmin=-0.001, vmax=0.001, origin='lower')
        c = fig.colorbar(s, ax=ax, shrink=0.7, fraction=0.05)
        c.set_label('Residual')
        plt.savefig('%sResidualCoeff%02d.pdf' % (output, (x+1)))

        #movie
        stopy, stopx = residual.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
        ims.append((ax3D.plot_surface(X, Y, residual, rstride=3, cstride=3,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False),))

    ax3D.set_xlabel('X [pixels]')
    ax3D.set_ylabel('Y [pixels]')
    ax3D.set_zlabel('Residual')

    anime = animation.ArtistAnimation(fig3D, ims, interval=2000, blit=True)
    anime.save('%sResidual.mp4' % output, fps=0.5)


def PCAleastSQ():
    """
    Perform least squares fitting using PCA basis set.

    :return: None
    """
    #to be fit
    psf = np.asarray(np.ravel(pf.getdata('PSF800.fits')[400:601, 400:601]), dtype=np.float64)
    #psf = np.asarray(np.ravel(pf.getdata('PSF800.fits')), dtype=np.float64)
    # average fit
    mean = np.asarray(np.ravel(pf.getdata('mean.fits')[400:601, 400:601]), dtype=np.float64)
    #mean = np.asarray(np.ravel(pf.getdata('mean.fits')), dtype=np.float64)

    #modes to be fitted
    modes = sorted(g.glob('modes/PCA*.fits'))
    modedata = [np.ravel(pf.getdata(file)[400:601, 400:601]) for file in modes]
    #modedata = [np.ravel(pf.getdata(file)) for file in modes]

    data = [mean,] + modedata
    data = np.asarray(data, dtype=np.float64)

    #least squares fitting
    popt, cov, info, mesg, ier = curve_fit(leastSQfit, data, psf, full_output=True)
    print info
    print mesg
    print ier
    print popt

    visualise(popt, psf, data)

    residual = (psf - leastSQfit(data, *popt)).reshape(201, 201)
    #residual = (psf - leastSQfit(data, *popt)).reshape(psf.shape)
    fileIO.writeFITS(residual, 'PCAresidual.fits',  int=False)


def ICAleastSq():
    """
    Perform least squares fitting using ICA basis set.

    :return: None
    """
    psf = np.asarray(np.ravel(pf.getdata('PSF800.fits')[400:601, 400:601]), dtype=np.float64)
    mean = np.asarray(np.ravel(pf.getdata('mean.fits')[400:601, 400:601]), dtype=np.float64)

    modes = sorted(g.glob('modes/ICA*.fits'))
    modedata = [np.ravel(pf.getdata(file)[400:601, 400:601]) for file in modes]

    data = [mean, ] + modedata
    data = np.asarray(data, dtype=np.float64)

    popt, cov, info, mesg, ier = curve_fit(leastSQfit, data, psf, full_output=True)
    print info
    print mesg
    print ier
    print popt

    visualise(popt, psf, data, output='ICA')

    residual = (psf - leastSQfit(data, *popt)).reshape(201, 201)
    fileIO.writeFITS(residual, 'ICAresidual.fits', int=False)


def BaysianPCAfitting():
    """
    Perform PCA basis set fitting using Bayesian Markov Chain Monte Carlo fitting technique.

    :requires: pyMC

    :return: None
    """
    import pymc
    import fitPSFmodel

    R = pymc.MCMC(fitPSFmodel, db='pickle', dbname='test.pickle', verbose=3)

    # populate and run it
    R.sample(iter=100000, burn=5000, thin=1)

    print 'Summary of Results:'
    print R.summary()
    R.write_csv('fitting.csv')

    #generate plots
    pymc.Matplot.plot(R)
    pymc.Matplot.summary_plot(R)

    #close MCMC to write database
    R.db.close()


if __name__ == '__main__':
    #PCAleastSQ()
    #ICAleastSq()
    BaysianPCAfitting()