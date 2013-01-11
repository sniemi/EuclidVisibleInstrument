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

:version: 0.2

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
import time


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
    :type popt: ndarray
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

    #load psf, mean, and PCA basis sets
    psf = np.asarray(np.ravel(pf.getdata('PSF800.fits')[400:601, 400:601]), dtype=np.float64)
    mean = np.asarray(np.ravel(pf.getdata('mean.fits')[400:601, 400:601]), dtype=np.float64)
    modes = sorted(g.glob('modes/PCA*.fits'))
    modedata = [np.ravel(pf.getdata(file)[400:601, 400:601]) for file in modes]
    data = [mean, ] + modedata
    data = np.asarray(data, dtype=np.float64)

    #from mean + basis modes
    #xdata = pymc.distributions.Uniform('x', -1.0, 1.0, observed=True, value=data, trace=False)
    xdata = data

    #set uniform priors for the basis set functions
    #these could actually be set based on the eigenvalues...
    a0 = pymc.distributions.Uniform('a0', 0.5, 1.5, value=1.0)
    a1 = pymc.distributions.Uniform('a1', -1.0, 1.0, value=-0.35)
    a2 = pymc.distributions.Uniform('a2', -1.0, 1.0, value=0.1)
    a3 = pymc.distributions.Uniform('a3', -1.0, 1.0, value=0.0)
    a4 = pymc.distributions.Uniform('a4', -1.0, 1.0, value=0.0)
    a5 = pymc.distributions.Uniform('a5', -1.0, 1.0, value=0.0)
    a6 = pymc.distributions.Uniform('a6', -1.0, 1.0, value=0.0)
    a7 = pymc.distributions.Uniform('a7', -1.0, 1.0, value=0.0)
    a8 = pymc.distributions.Uniform('a8', -1.0, 1.0, value=0.0)
    a9 = pymc.distributions.Uniform('a9', -1.0, 1.0, value=0.0)
    a10 = pymc.distributions.Uniform('a10', -1.0, 1.0, value=0.0)
    a11 = pymc.distributions.Uniform('a11', -1.0, 1.0, value=0.0)
    a12 = pymc.distributions.Uniform('a12', -1.0, 1.0, value=0.0)
    a13 = pymc.distributions.Uniform('a13', -1.0, 1.0, value=0.0)
    a14 = pymc.distributions.Uniform('a14', -1.0, 1.0, value=0.0)
    a15 = pymc.distributions.Uniform('a15', -1.0, 1.0, value=0.0)
    a16 = pymc.distributions.Uniform('a16', -1.0, 1.0, value=0.0)
    a17 = pymc.distributions.Uniform('a17', -1.0, 1.0, value=0.0)
    a18 = pymc.distributions.Uniform('a18', -1.0, 1.0, value=0.0)
    a19 = pymc.distributions.Uniform('a19', -1.0, 1.0, value=0.0)
    a20 = pymc.distributions.Uniform('a20', -1.0, 1.0, value=0.0)

    #model that is being fitted
    @pymc.deterministic(plot=False, trace=False)
    def model(x=xdata, a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8, a9=a9, a10=a10,
              a11=a11, a12=a12, a13=a13, a14=a14, a15=a15, a16=a16, a17=a17, a18=a18, a19=a19, a20=a20):
        tmp = a0 * x[0] + a1 * x[1] + a2 * x[2] + a3 * x[3] + a4 * x[4] + a5 * x[5] + a6 * x[6] +\
              a7 * x[7] + a8 * x[8] + a9 * x[9] + a10 * x[10] + a11 * x[11] + a12 * x[12] + a13 * x[13] +\
              a14 * x[14] + a15 * x[15] + a16 * x[16] + a17 * x[17] + a18 * x[18] + a19 * x[19] + a20 * x[20]
        return tmp

    #likelihood function, note that an inverse weighting has been applied... this could be something else too
    #y = pymc.distributions.Normal('y', mu=model, tau=1.e4/psf**2, value=psf, observed=True, trace=False)
    y = pymc.distributions.Normal('y', mu=model, tau=1.e4/np.abs(model)**2, value=psf, observed=True, trace=False)
    #for the real application, one should use pymc.distributions.mv_normal_cov_like
    #see also Lance's note Appendix A, equation 1.

    #store the model to a dictionary
    d = {'a0' : a0,
         'a1' : a1,
         'a2' : a2,
         'a3' : a3,
         'a4' : a4,
         'a5' : a5,
         'a6' : a6,
         'a7' : a7,
         'a8' : a8,
         'a9' : a9,
         'a10': a10,
         'a11': a11,
         'a12': a12,
         'a13': a13,
         'a14': a14,
         'a15': a15,
         'a16': a16,
         'a17': a17,
         'a18': a18,
         'a19': a19,
         'a20': a20,
         'f' : model,
         'y': y}

    print 'Will start running a chain...'
    R = pymc.MCMC(d, verbose=0)
    R.sample(iter=100000, burn=20000, thin=5)

    #generate plots
    pymc.Matplot.plot(R)
    pymc.Matplot.summary_plot(R)

    Rs = R.stats()
    print 'PyMC finished...'

    #calculate the residual residual given the found parameter values and write to a FITS file
    vals = [Rs['a0']['mean'],
            Rs['a1']['mean'],
            Rs['a2']['mean'],
            Rs['a3']['mean'],
            Rs['a4']['mean'],
            Rs['a5']['mean'],
            Rs['a6']['mean'],
            Rs['a7']['mean'],
            Rs['a8']['mean'],
            Rs['a9']['mean'],
            Rs['a10']['mean'],
            Rs['a11']['mean'],
            Rs['a12']['mean'],
            Rs['a13']['mean'],
            Rs['a14']['mean'],
            Rs['a15']['mean'],
            Rs['a16']['mean'],
            Rs['a17']['mean'],
            Rs['a18']['mean'],
            Rs['a19']['mean'],
            Rs['a20']['mean']]
    vals = np.asarray(vals)
    print vals
    print 'Calculating the residual...'
    residual = (psf - leastSQfit(data, *vals)).reshape(201, 201)
    print np.mean(residual), np.min(residual), np.max(residual), np.std(residual), np.median(residual)
    fileIO.writeFITS(residual, 'BayesianResidual.fits', int=False)

    #generate plots
    print 'Doing further visualisation...'
    visualise(vals, psf, data, output='Bayesian')


def test():
    """
    Simple test using both least squares and Bayesian MCMC fitting.
    Includes Poisson noise to the PSF prior fitting. The PSF is build randomly
    from the the mean PSF and the PCA components are being fitted.

    :return: None
    """
    import pymc

    #least squares fitting
    def lsq(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
              a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, background):
        tmp = a0 * x[0] + a1 * x[1] + a2 * x[2] + a3 * x[3] + a4 * x[4] + a5 * x[5] + a6 * x[6] +\
              a7 * x[7] + a8 * x[8] + a9 * x[9] + a10 * x[10] + a11 * x[11] + a12 * x[12] + a13 * x[13] +\
              a14 * x[14] + a15 * x[15] + a16 * x[16] + a17 * x[17] + a18 * x[18] + a19 * x[19] + a20 * x[20] +\
              background
        return tmp

    #generate random PSF from average
    mean = np.asarray(np.ravel(pf.getdata('mean.fits')[400:601, 400:601]), dtype=np.float64)
    #and the modes to be fitted
    modes = sorted(g.glob('modes/PCA*.fits'))
    modedata = [np.ravel(pf.getdata(file)[400:601, 400:601]) for file in modes]
    data = [mean,] + modedata
    modeldata = np.asarray(data, dtype=np.float64)

    #generate PSF from model data
    bgrd = 1e3
    coeffs = np.random.random_integers(-1e4, 1e4, size=20)
    coeffs = np.hstack((np.random.random_integers(1e4, 1e6), coeffs, bgrd))
    print coeffs
    psfNonoise = lsq(modeldata, *coeffs)
    #add Poisson Noise
    psf = np.random.poisson(psfNonoise.copy())
    fileIO.writeFITS(psfNonoise.reshape(201, 201), 'PSFNonoise.fits', int=False)
    fileIO.writeFITS(psf.reshape(201, 201), 'PSF.fits', int=False)

    if np.min(psf) < 0.:
        import sys
        sys.exit('increase background value and run again...')

    start = time.time()
    popt, cov, info, mesg, ier = curve_fit(lsq, modeldata, psf, full_output=True)
    print info
    print mesg
    print ier
    print popt
    print
    print coeffs / popt
    print 'Finished least squares in %e seconds' % (time.time() - start)
    #visualise(popt, psf, data)
    residual = (psfNonoise - lsq(modeldata.copy(), *popt)).reshape(201, 201) / np.max(psfNonoise)
    print np.mean(residual), np.min(residual), np.max(residual), np.std(residual), np.median(residual)
    fileIO.writeFITS(residual, 'LSQresidual.fits',  int=False)

    #Baysian
    #set uniform priors for the basis set functions
    #these could actually be set based on the eigenvalues...
    a0 = pymc.distributions.Uniform('a0', 1e4, 1e6, value=1e5)
    a1 = pymc.distributions.Uniform('a1', -1e4, 1e4, value=0.0)
    a2 = pymc.distributions.Uniform('a2', -1e4, 1e4, value=0.0)
    a3 = pymc.distributions.Uniform('a3', -1e4, 1e4, value=0.0)
    a4 = pymc.distributions.Uniform('a4', -1e4, 1e4, value=0.0)
    a5 = pymc.distributions.Uniform('a5', -1e4, 1e4, value=0.0)
    a6 = pymc.distributions.Uniform('a6', -1e4, 1e4, value=0.0)
    a7 = pymc.distributions.Uniform('a7', -1e4, 1e4, value=0.0)
    a8 = pymc.distributions.Uniform('a8', -1e4, 1e4, value=0.0)
    a9 = pymc.distributions.Uniform('a9', -1e4, 1e4, value=0.0)
    a10 = pymc.distributions.Uniform('a10', -1e4, 1e4, value=0.0)
    a11 = pymc.distributions.Uniform('a11', -1e4, 1e4, value=0.0)
    a12 = pymc.distributions.Uniform('a12', -1e4, 1e4, value=0.0)
    a13 = pymc.distributions.Uniform('a13', -1e4, 1e4, value=0.0)
    a14 = pymc.distributions.Uniform('a14', -1e4, 1e4, value=0.0)
    a15 = pymc.distributions.Uniform('a15', -1e4, 1e4, value=0.0)
    a16 = pymc.distributions.Uniform('a16', -1e4, 1e4, value=0.0)
    a17 = pymc.distributions.Uniform('a17', -1e4, 1e4, value=0.0)
    a18 = pymc.distributions.Uniform('a18', -1e4, 1e4, value=0.0)
    a19 = pymc.distributions.Uniform('a19', -1e4, 1e4, value=0.0)
    a20 = pymc.distributions.Uniform('a20', -1e4, 1e4, value=0.0)
    a21 = pymc.distributions.Uniform('a21', 0, 1e4, value=0.0)

    #model that is being fitted
    @pymc.deterministic(plot=False, trace=False)
    def model(x=modeldata, a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8, a9=a9, a10=a10,
              a11=a11, a12=a12, a13=a13, a14=a14, a15=a15, a16=a16, a17=a17, a18=a18, a19=a19, a20=a20, a21=a21):
        tmp = a0 * x[0] + a1 * x[1] + a2 * x[2] + a3 * x[3] + a4 * x[4] + a5 * x[5] + a6 * x[6] +\
              a7 * x[7] + a8 * x[8] + a9 * x[9] + a10 * x[10] + a11 * x[11] + a12 * x[12] + a13 * x[13] +\
              a14 * x[14] + a15 * x[15] + a16 * x[16] + a17 * x[17] + a18 * x[18] + a19 * x[19] + a20 * x[20] + a21
        return tmp

    #likelihood function, note that an inverse weighting has been applied... this could be something else too
    y = pymc.distributions.Poisson('y', mu=model, value=psf, observed=True, trace=False)

    #store the model to a dictionary
    d = {'a0' : a0,
         'a1' : a1,
         'a2' : a2,
         'a3' : a3,
         'a4' : a4,
         'a5' : a5,
         'a6' : a6,
         'a7' : a7,
         'a8' : a8,
         'a9' : a9,
         'a10': a10,
         'a11': a11,
         'a12': a12,
         'a13': a13,
         'a14': a14,
         'a15': a15,
         'a16': a16,
         'a17': a17,
         'a18': a18,
         'a19': a19,
         'a20': a20,
         'a21': a21,
         'f' : model,
         'y': y}

    print 'Will start running a chain...'
    start = time.time()
    R = pymc.MCMC(d, verbose=0)
    R.sample(iter=50000, burn=15000, thin=5)

    #generate plots
    pymc.Matplot.plot(R)
    pymc.Matplot.summary_plot(R)

    Rs = R.stats()
    print 'Finished MCMC in %e seconds' % (time.time() - start)

    #calculate the residual residual given the found parameter values and write to a FITS file
    vals = [Rs['a0']['mean'],
            Rs['a1']['mean'],
            Rs['a2']['mean'],
            Rs['a3']['mean'],
            Rs['a4']['mean'],
            Rs['a5']['mean'],
            Rs['a6']['mean'],
            Rs['a7']['mean'],
            Rs['a8']['mean'],
            Rs['a9']['mean'],
            Rs['a10']['mean'],
            Rs['a11']['mean'],
            Rs['a12']['mean'],
            Rs['a13']['mean'],
            Rs['a14']['mean'],
            Rs['a15']['mean'],
            Rs['a16']['mean'],
            Rs['a17']['mean'],
            Rs['a18']['mean'],
            Rs['a19']['mean'],
            Rs['a20']['mean'],
            Rs['a21']['mean']]
    vals = np.asarray(vals)
    print coeffs / vals
    print 'Calculating the residual...'
    residual = (psfNonoise - lsq(modeldata.copy(), *vals)).reshape(201, 201) / np.max(psfNonoise)
    print np.mean(residual), np.min(residual), np.max(residual), np.std(residual), np.median(residual)
    fileIO.writeFITS(residual, 'BayesianResidual.fits', int=False)


if __name__ == '__main__':
    #test()
    #PCAleastSQ()
    #ICAleastSq()
    BaysianPCAfitting()