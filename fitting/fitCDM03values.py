"""
Simple script to fit CDM03 CTI model parameters to measurements.

:requires: SciPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)
:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import matplotlib
matplotlib.use('PDF')
matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=18)
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'
matplotlib.rcParams['legend.fontsize'] = 11
import scipy.optimize
import cdm03
import numpy as np
from matplotlib import pyplot as plt
import simulator.logger as lg


def applyRadiationDamage(data, nt, sigma, taur, iquadrant=0, rdose=6e9):
    """
    Apply radian damage based on FORTRAN CDM03 model. The method assumes that
    input data covers only a single quadrant defined by the iquadrant integer.

    :param data: imaging data to which the CDM03 model will be applied to.
    :type data: ndarray



    :param iquandrant: number of the quadrant to process:
    :type iquandrant: int

    cdm03 - Function signature:
      sout = cdm03(sinp,iflip,jflip,dob,rdose,in_nt,in_sigma,in_tr,[xdim,ydim,zdim])
    Required arguments:
      sinp : input rank-2 array('f') with bounds (xdim,ydim)
      iflip : input int
      jflip : input int
      dob : input float
      rdose : input float
      in_nt : input rank-1 array('d') with bounds (zdim)
      in_sigma : input rank-1 array('d') with bounds (zdim)
      in_tr : input rank-1 array('d') with bounds (zdim)
    Optional arguments:
      xdim := shape(sinp,0) input int
      ydim := shape(sinp,1) input int
      zdim := len(in_nt) input int
    Return objects:
      sout : rank-2 array('f') with bounds (xdim,ydim)

    :Note: Because Python/NumPy arrays are different row/column based, one needs
           to be extra careful here. NumPy.asfortranarray will be called to get
           an array laid out in Fortran order in memory.


    :return: image that has been run through the CDM03 model
    :rtype: ndarray
    """
    #call Fortran routine
    CTIed = cdm03.cdm03(np.asfortranarray(data),
                        iquadrant % 2, iquadrant / 2,
                        0.0, rdose,
                        nt, sigma, taur,
                        [data.shape[0], data.shape[1], len(nt)])

    return CTIed


def fitfunc(p, x):
    """
    Functional form to be fitted.
    """
    #keep sigma and taur fixed
    nt = [5.0, 0.22, 0.2, 0.1, 0.043, 0.39, 1.0]
    sigma = [2.2e-13, 2.2e-13, 4.72e-15, 1.37e-16, 2.78e-17, 1.93e-17, 6.39e-18]
    taur = [0.00000082, 0.0003, 0.002, 0.025, 0.124, 16.7, 496.0]

    #params that are being fit
    #nt = p[:7]
    nt[1:4] = p#[:3]
    #taur = p[7:]
    #taur[:3] = p[3:]

    y = applyRadiationDamage(x.transpose(), nt, sigma, taur).transpose()[1063:1090, 0]

    #print y[2], x[1065, 0]
    return y


def plotPosition(values, profile, fits, xstart=1060, len=13, output='StartingPosition.pdf'):
    """
    Simple plotting script.
    """
    prof = np.average(profile, axis=1)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    plt.text(0.1, 0.95, r'$n_{t}=$'+str(np.around(fits['nt'], decimals=2)), va='top',  transform=ax.transAxes, fontsize=11)
    plt.text(0.1, 0.85, r'$\sigma=$'+str(fits['sigma']), va='top',  transform=ax.transAxes, fontsize=11)
    plt.text(0.1, 0.80, r'$\tau_{r}=$'+str(fits['taur']), va='top',  transform=ax.transAxes, fontsize=11)
    plt.semilogy(prof[xstart:xstart+len+3], 'rD-', label='Fitted')
    plt.semilogy(np.arange(len)+3, values[:len], 'bo', label='Data')
    plt.xlabel('Pixels')
    plt.ylabel('electrons')
    plt.legend(loc='best')
    plt.savefig(output, numpoints=1)
    plt.close()


if __name__ == '__main__':
    #set up logger
    log = lg.setUpLogger('fitting.log')

    #input measurement values (taken from an Excel spreadsheet)
    values = np.loadtxt('CTIdata.txt')
    vals = np.ones((2066, 1)) * 4.0
    ln = len(values)
    vals[1063:1063+ln, 0] = values
    vals[1075:, 0] = 3
    vals = vals[1063:1090,0]

    #data to be CTIed
    data = np.zeros((2066, 1))
    data[1053:1064, :] = 38000.

    #Values that were in the CDM03 model prior May 9th 2012
    nt = [5.0, 0.22, 0.2, 0.1, 0.043, 0.39, 1.0]
    sigma = [2.2e-13,2.2e-13,4.72e-15,1.37e-16,2.78e-17,1.93e-17,6.39e-18]
    taur = [0.00000082, 0.0003, 0.002, 0.025, 0.124, 16.7, 496.0]

    #get the starting profile and plot it
    profile1 = applyRadiationDamage(data.transpose(), nt, sigma, taur).transpose()
    plotPosition(vals, profile1, dict(nt=nt, sigma=sigma, taur=taur))

    #get the starting profile and plot it
    #profile2 = applyRadiationDamage(data.transpose()/10., nt, sigma, taur).transpose()
    #plotPosition(vals/10., profile2, dict(nt=nt, sigma=sigma, taur=taur), output='LowElectrons.pdf')

    #initial guesses for trap densities and release times
    #because of lack of data points to fit, I decided to keep sigma fixed
    #nt = [6.0, .44, 0.35, 0.1, 0.043, 0.39, 1.]
    #nt = [5.7, .6, 0.245, 0.1, 0.043, 0.39, 1.]
    #nt = [5.1, .18, 0.135, 0.1, 0.043, 0.39, 1.]
    nt = [5.1, .21, 0.135, 0.1, 0.043, 0.39, 1.] #best

    #write these to the log file
    log.info('Initial Guess Values:')
    log.info('nt='+str(nt))
    log.info('sigma='+str(sigma))
    log.info('taur='+str(taur))

    #combine to a single Python list
    #params = nt + taur
    params = nt[1:4] #+ taur[:3]

    #even/uneven weighting scheme
    weights = np.arange(27.)*0.01 + 0.095
    weights[7:] = 1.0
    #weights = np.ones(27.)

    #write out the weights
    log.info('Weights:')
    log.info(str(weights))

    #fitting with SciPy
    errfuncE = lambda p, x, y, errors: (fitfunc(p, x) - y)  / errors
    out = scipy.optimize.leastsq(errfuncE, params[:], args=(data, vals, weights), full_output=True,
                                 maxfev=100000, ftol=1e-14, xtol=1e-14)
    print out

    #new params
    #newnt = out[0][:7]
    newnt = list(nt)
    newnt[1:4] = out[0][:3]
    #newtaur = out[0][7:]
    newtaur = list(taur)
    #newtaur[:3] = out[0][3:]
    #newtaur = np.asarray(taur)
    #nt[:3] = out[0]
    print
    print newnt / np.asarray(nt)
    print
    print newtaur / np.asarray(taur)
    profile = applyRadiationDamage(data.transpose(), newnt, sigma, newtaur).transpose()
    plotPosition(vals, profile, dict(nt=newnt, sigma=sigma, taur=newtaur), output='EndPosition.pdf')

    #write the numbers to a log
    log.info('Final Values:')
    log.info('nt='+str(nt))
    log.info('sigma='+str(sigma))
    log.info('taur='+str(taur))
    log.info('Finished!')