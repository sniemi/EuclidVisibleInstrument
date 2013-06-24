"""
Simple script to fit CDM03 CTI model parameters to measurements.

:requires: SciPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03bidir cdm03bidir.f90)
:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.2
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
import cdm03bidir
import numpy as np
from matplotlib import pyplot as plt
import support.logger as lg


def applyRadiationDamageBiDir(data, nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, iquadrant=0, rdose=1.6e10):
    """

    :return: image that has been run through the CDM03 model
    :rtype: ndarray
    """

    #read in trap information
    CTIed = cdm03bidir.cdm03(data,
                        iquadrant%2, iquadrant/2,
                        0.0, rdose,
                        nt_p, sigma_p, taur_p,
                        nt_s, sigma_s, taur_s,
                        [data.shape[0], data.shape[1], len(nt_p), len(nt_p)])
    return np.asanyarray(CTIed)


def fitfunc(p, x):
    """
    Functional form to be fitted.
    """
    #serial fixed
    nt_s = [20, 10, 2.]
    sigma_s = [6e-20, 1.13e-14, 5.2e016]
    taur_s = [2.38e-2, 1.7e-6, 2.2e-4]

    #keep sigma and taur fixed
    nt_p = [5.0, 0.22, 0.2, 0.1, 0.043, 0.39, 1.0]
    sigma_p = [2.2e-13, 2.2e-13, 4.72e-15, 1.37e-16, 2.78e-17, 1.93e-17, 6.39e-18]
    taur_p = [0.00000082, 0.0003, 0.002, 0.025, 0.124, 16.7, 496.0]

    #params that are being fit
    nt_p[1:4] = p#[:3]

    y = applyRadiationDamageBiDir(x.transpose(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s).transpose()[1063:1090, 0]

    return y


def plotPosition(values, profile, fits, xstart=1060, len=15, output='StartingPosition.pdf'):
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
    electrons = 43500.

    #set up logger
    log = lg.setUpLogger('fitting.log')

    #input measurement values
    datafolder = '/Users/smn2/EUCLID/CTItesting/data/'
    gain1 = 1.17

    tmp = np.loadtxt(datafolder + 'CCD204_05325-03-02_Hopkinson_EPER_data_200kHz_one-output-mode_1.6e10-50MeV.txt',
                     usecols=(0, 6)) #6 = 152.55K
    ind = tmp[:, 0]
    values = tmp[:, 1]
    values = values[ind > 0.]
    values *= gain1
    vals = np.ones((2066, 1)) * 4.0
    ln = len(values)
    vals[1063:1063+ln, 0] = values
    vals[1075:, 0] = 3
    vals = vals[1063:1090,0]

    #data to be CTIed
    #data = np.zeros((2066, 1))
    data = np.zeros((2066, 2048))
    data[1053:1064, :] = electrons

    #Initial CTI values
    f1 = 'cdm_euclid_parallel.dat'
    trapdata = np.loadtxt(f1)
    nt_p = trapdata[:, 0]
    sigma_p = trapdata[:, 1]
    taur_p = trapdata[:, 2]
    f2 = 'cdm_euclid_serial.dat'
    trapdata = np.loadtxt(f2)
    nt_s = trapdata[:, 0]
    sigma_s = trapdata[:, 1]
    taur_s = trapdata[:, 2]

    #get the starting profile and plot it
    profile1 = applyRadiationDamageBiDir(data.transpose(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s).transpose()
    plotPosition(vals, profile1, dict(nt=nt_p, sigma=sigma_p, taur=taur_p))

    #write these to the log file
    log.info('Initial Guess Values:')
    log.info('nt='+str(nt_p))
    log.info('sigma='+str(sigma_p))
    log.info('taur='+str(taur_p))

    #combine to a single Python list
    #params = nt + taur
    params = nt_p[1:4] #+ taur[:3]

    #even/uneven weighting scheme
    weights = np.arange(27.)*0.01 + 0.095
    weights[7:] = 1.0
    #weights = np.ones(27.)

    #write out the weights
    log.info('Weights:')
    log.info(str(weights))

    #fitting with SciPy
    errfuncE = lambda p, x, y, errors: (fitfunc(p, x) - y) / errors
    out = scipy.optimize.leastsq(errfuncE, params[:], args=(data, vals, weights), full_output=True,
                                 maxfev=10000000, ftol=1e-16, xtol=1e-16)
    print out

    #new params
    #newnt = out[0][:7]
    newnt_p = list(nt_p)
    newnt_p[1:4] = out[0]#[:3]
    #newtaur = out[0][7:]
    newtaur_p = list(taur_p)
    #newtaur[:3] = out[0][3:]
    #newtaur = np.asarray(taur)
    #nt[:3] = out[0]
    print
    print newnt_p / np.asarray(nt_p)
    print
    print newtaur_p / np.asarray(taur_p)
    profile = applyRadiationDamageBiDir(data.transpose(), newnt_p, sigma_p, newtaur_p, nt_s, sigma_s, taur_s).transpose()
    plotPosition(vals, profile, dict(nt=newnt_p, sigma=sigma_p, taur=newtaur_p), output='EndPosition.pdf')

    #write the numbers to a log
    log.info('Final Values:')
    log.info('nt='+str(nt_p))
    log.info('sigma='+str(sigma_p))
    log.info('taur='+str(taur_p))
    log.info('Finished!')
