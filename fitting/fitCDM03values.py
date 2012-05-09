"""
Simple script to fit CDM03 CTI model parameters to measurements.

:requires: SciPy
:requires: CDM03 (FORTRAN code, use f2py to compile)
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


def applyRadiationDamage(data, nt, sigma, taur, iquadrant=0, rdose=1e10):
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
    #params
    nt = p[:7]
    taur = p[7:]

    #keep sigma fixed
    sigma = np.array([2.2e-13,2.2e-13,4.72e-15,1.37e-16,2.78e-17,1.93e-17,6.39e-18])

    y = applyRadiationDamage(x.transpose(), nt, sigma, taur).transpose()[1063:1080, 0]
    return y



def plotPosition(values, profile, fits, xstart=1063, len=15, output='StartingPosition.pdf'):
    """
    Simple plotting script.
    """
    prof = np.average(profile, axis=1)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    plt.text(0.1, 0.95, r'$n_{t}=$'+str(np.around(fits['nt'], decimals=2)), va='top',  transform=ax.transAxes, fontsize=11)
    plt.text(0.1, 0.90, r'$\sigma=$'+str(fits['sigma']), va='top',  transform=ax.transAxes, fontsize=11)
    plt.text(0.1, 0.85, r'$\tau_{r}=$'+str(fits['taur']), va='top',  transform=ax.transAxes, fontsize=11)
    plt.semilogy(prof[xstart:xstart+len], 'rD-', label='Fitted')
    plt.semilogy(values[:len], 'bo', label='Data')
    plt.xlabel('Pixels')
    plt.ylabel('ADUs')
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
    vals = vals[1063:1080,0]

    #data to be CTIed
    data = np.zeros((2066, 1))
    data[1053:1064, :] = 38337.71

    #Values that were in the CDM03 model prior May 9th 2012
    nt = [5.0, 0.22, 0.2, 0.1, 0.043, 0.39, 1.0]
    sigma = [2.2e-13,2.2e-13,4.72e-15,1.37e-16,2.78e-17,1.93e-17,6.39e-18]
    taur = [0.00000082, 0.0003, 0.002, 0.025, 0.124, 16.7, 496.0]

    #get the starting profile and plot it
    profile1 = applyRadiationDamage(data.transpose(), nt, sigma, taur).transpose()
    plotPosition(vals, profile1, dict(nt=nt, sigma=sigma, taur=taur))

    #initial guesses for trap densities and release times
    #because of lack of data points to fit, I decided to keep sigma fixed
    nt = [ 4.7,  0.15,    0.17,  0.1,  0.043,       0.39,       1.        ]
    taur = [2.2e-06,   1.5e-04,   8.6e-06,   9.8e-06,  0.124,   16.7,   496.0]

    #write these to the log file
    log.info('Initial Guess Values:')
    log.info('nt='+str(nt))
    log.info('sigma='+str(sigma))
    log.info('taur='+str(taur))

    #combine to a single Python list
    params = nt + taur
    params = np.array(params)

    #even/uneven weighting scheme
    weights = np.arange(17.)*0.01 + 0.1
    weights[7:] = 1.0
    weights = np.ones(17.)

    #write out the weights
    log.info('Weights:')
    log.info(str(weights))

    #fitting with SciPy
    errfuncE = lambda p, x, y, errors: (fitfunc(p, x) - y)  / errors
    out = scipy.optimize.leastsq(errfuncE, params[:], args=(data, vals, weights), full_output=True,
                                 maxfev=10000000, ftol=1e-11, xtol=1e-11, factor=50)
    print out

    #new params
    taur = out[0][7:]
    nt = out[0][:7]
    print
    print nt
    print
    print taur
    profile = applyRadiationDamage(data.transpose(), nt, sigma, taur).transpose()
    plotPosition(vals, profile, dict(nt=nt, sigma=sigma, taur=taur), output='EndPosition.pdf')

    #write the numbers to a log
    log.info('Final Values:')
    log.info('nt='+str(nt))
    log.info('sigma='+str(sigma))
    log.info('taur='+str(taur))
    log.info('Finished!')