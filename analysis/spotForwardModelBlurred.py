"""
CCD Spot Measurements
=====================

Analyse laboratory CCD PSF measurements by forward modelling to the data.

The methods used here seem to work reasonably well when the spot has been well centred. If however, the
spot is e.g. 0.3 pixels off then estimating the amplitude of the Airy disc becomes rather difficult.
Unfortunately this affects the following CCD PSF estimates as well and can lead to models that are
rather far from the truth. Also, if when the width of the CCD PSF kernel becomes narrow, say 0.2, pixels
it is very difficult to recover. This most likely results from an inadequate sampling. In this case it might
be more appropriate to use "cross"-type kernel.

Because the amplitude can be very tricky to estimate, the version 1.4 (and later) implement a meta-parameter
called peak, which is the peak counts in the image. This is then converted to the amplitude by using the centroid
estimate. Because the centroids are fitted for each image, the amplitude can also vary in the joint fits. This
seems to improve the joint fitting constrains. Note however that this does couple the radius of the Airy disc
as well, because the amplitude estimate uses the x, y, and radius information as well.

One question to address is how the smoothing of the Airy disc is done. So far I have assumed that the Gaussian that
represents defocus should be centred at the same location as the Airy disc. However, if the displacement if the
Airy disc is large, then the defocus term will move the Airy disc to the direction of the displacement and make
it more asymmetric. Another option is to use scipy.ndimage.filters.gaussian_filter which simply applies Gaussian
smoothing to the input image. Based on the testing carried out this does not seem to make a huge difference. The
latter (smoothing) will lead to more or less the same CCD PSF estimates, albeit with slightly higher residuals.
We therefore adopt a Gaussian kernel that is centred with the Airy disc.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: astropy
:requires: matplotlib
:requires: VISsim-Python
:requires: emcee
:requires: sklearn

:version: 0.1

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import matplotlib
#matplotlib.use('pdf')
#matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['image.interpolation'] = 'none'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyfits as pf
import numpy as np
import emcee
import scipy
import scipy.ndimage.measurements as m
from scipy import signal
from scipy import ndimage
from scipy.special import j1, jn_zeros
from support import files as fileIO
from astropy.modeling import models, fitting
import triangle
import glob as g
import os, datetime
from multiprocessing import Pool


__author__ = 'Sami-Matias Niemi'
__vesion__ = 1.1

#fixed parameters
cores = 8


def forwardModel(file, out='Data', wavelength=None, gain=3.1, size=10, burn=500, spotx=2888, spoty=3514, run=700,
                 simulation=False, truths=None, blurred=False):
    """
    Forward models the spot data found from the input file. Can be used with simulated and real data.

    Notes:
    - emcee is run three times as it is important to have a good starting point for the final run.
    """
    print '\n\n\n'
    print '_'*120
    print 'Processing:', file
    #get data and convert to electrons
    o = pf.getdata(file)*gain

    if simulation:
        data = o
    else:
        #roughly the correct location - to avoid identifying e.g. cosmic rays
        data = o[spoty-(size*3):spoty+(size*3)+1, spotx-(size*3):spotx+(size*3)+1].copy()

    #maximum position within the cutout
    y, x = m.maximum_position(data)

    #spot and the peak pixel within the spot, this is also the CCD kernel position
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()
    CCDy, CCDx = m.maximum_position(spot)
    print 'CCD Kernel Position (within the postage stamp):', CCDx, CCDy

    #bias estimate
    if simulation:
        bias = 9000.
        rn = 4.5
    else:
        bias = np.median(o[spoty-size: spoty+size, spotx-220:spotx-20]) #works for read o
        rn = np.std(o[spoty-size: spoty+size, spotx-220:spotx-20])

    print 'Readnoise (e):', rn
    if rn < 2. or rn > 6.:
        print 'NOTE: suspicious readout noise estimate...'
    print 'ADC offset (e):', bias

    #remove bias
    spot -= bias

    #save to file
    fileIO.writeFITS(spot, out+'small.fits', int=False)

    #make a copy ot generate error array
    data = spot.copy().flatten()
    #assume that uncertanties scale as sqrt of the values + readnoise
    #sigma = np.sqrt(data/gain + rn**2)
    tmp = data.copy()
    tmp[tmp + rn**2 < 0.] = 0.  #set highly negative values to zero
    var = tmp.copy() + rn**2
    #Gary B. said that actually this should be from the model or is biased,
    #so I only pass the readout noise part now

    #fit a simple model
    print 'Least Squares Fitting...'
    gaus = models.Gaussian2D(spot.max(), size, size, x_stddev=0.5, y_stddev=0.5)
    gaus.theta.fixed = True  #fix angle
    p_init = gaus
    fit_p = fitting.LevMarLSQFitter()
    stopy, stopx = spot.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
    p = fit_p(p_init, X, Y, spot)
    print p
    model = p(X, Y)
    fileIO.writeFITS(model, out+'BasicModel.fits', int=False)
    fileIO.writeFITS(model - spot, out+'BasicModelResidual.fits', int=False)

    #goodness of fit
    gof = (1./(np.size(data) - 5.)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof
    print 'Done\n\n'

    #maximum value
    max = np.max(spot)
    peakrange = (0.9*max, 1.7*max)
    sum = np.sum(spot)

    print 'Maximum Value:', max
    print 'Sum of the values:', sum
    print 'Peak Range:', peakrange

    #MCMC based fitting
    print 'Bayesian Model Fitting...'
    nwalkers = 1000

    # Initialize the sampler with the chosen specs.
    #Create the coordinates x and y
    x = np.arange(0, spot.shape[1])
    y = np.arange(0, spot.shape[0])
    #Put the coordinates in a mesh
    xx, yy = np.meshgrid(x, y)

    #Flatten the arrays
    xx = xx.flatten()
    yy = yy.flatten()

    print 'Fitting full model...'
    ndim = 7

    #Choose an initial set of positions for the walkers - fairly large area not to bias the results
    p0 = np.zeros((nwalkers, ndim))
    #peak, center_x, center_y, radius, focus, width_x, width_y = theta
    p0[:, 0] = np.random.normal(max, max/100., size=nwalkers)                 # peak value
    p0[:, 1] = np.random.normal(p.x_mean.value, 0.1, size=nwalkers)           # x
    p0[:, 2] = np.random.normal(p.y_mean.value, 0.1, size=nwalkers)           # y
    print 'Using initial guess [radius, focus, width_x, width_y]:', [0.5, 0.6, 0.05, 0.09]
    p0[:, 3] = np.random.normal(0.5, 0.01, size=nwalkers)                   # radius
    p0[:, 4] = np.random.normal(0.6, 0.01, size=nwalkers)                    # focus
    p0[:, 5] = np.random.normal(0.05, 0.001, size=nwalkers)                   # width_x
    p0[:, 6] = np.random.normal(0.09, 0.001, size=nwalkers)                   # width_y

    #initiate sampler
    pool = Pool(cores) #A hack Dan gave me to not have ghost processes running as with threads keyword
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, var, peakrange, spot.shape],
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=[xx, yy, data, rn**2, peakrange, spot.shape],
                                    pool=pool)

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)
    maxprob_index = np.argmax(prob)
    params_fit = pos[maxprob_index]
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    print 'Estimate:', params_fit
    sampler.reset()

    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
    _printResults(params_fit, errors_fit)

    #Best fit model
    peak, center_x, center_y, radius, focus, width_x, width_y = params_fit
    amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius, x_0=CCDx, y_0=CCDy)
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape(spot.shape)
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape(spot.shape)
    foc = signal.convolve2d(adata, focusdata, mode='same')
    CCDdata = np.array([[0.0, width_y, 0.0],
                        [width_x, (1.-width_y-width_y-width_x-width_x), width_x],
                        [0.0, width_y, 0.0]])
    fileIO.writeFITS(CCDdata, 'kernel.fits', int=False)
    model = signal.convolve2d(foc, CCDdata, mode='same')
    #save model
    fileIO.writeFITS(model, out+'model.fits', int=False)

    #residuals
    fileIO.writeFITS(model - spot, out+'residual.fits', int=False)
    fileIO.writeFITS(((model - spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)

    # a simple goodness of fit
    gof = (1./(np.size(data) - ndim)) * np.sum((model.flatten() - data)**2 / var)
    maxdiff = np.max(np.abs(model - spot))
    print 'GoF:', gof, ' Maximum difference:', maxdiff
    if maxdiff > 2e3 or gof > 4.:
        print '\nFIT UNLIKELY TO BE GOOD...\n'
    print 'Amplitude estimate:', amplitude

    #plot
    samples = sampler.chain.reshape((-1, ndim))
    extents = None
    if simulation:
        extents = [(0.91*truth, 1.09*truth) for truth in truths]
        extents[1] = (truths[1]*0.995, truths[1]*1.005)
        extents[2] = (truths[2]*0.995, truths[2]*1.005)
        extents[3] = (0.395, 0.425)
        extents[4] = (0.503, 0.517)
        truths[0] = _peakFromTruth(truths)
        print truths
    fig = triangle.corner(samples,
                          labels=['peak', 'x', 'y', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)#, extents=extents)
    fig.savefig(out+'Triangle.png')
    plt.close()
    pool.close()


def log_posterior(theta, x, y, z, var, peakrange, size):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_prior(theta, peakrange)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, z, var, size)


def log_prior(theta, peakrange):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    peak, center_x, center_y, radius, focus, width_x, width_y = theta
    if 7. < center_x < 14. and 7. < center_y < 14. and 1.e-4 < width_x < 0.25 and 1.e-4 < width_y < 0.3 and \
       peakrange[0] < peak < peakrange[1] and 0.4 < radius < 1. and 0.2 < focus < 1.5:
        return 0.
    else:
        return -np.inf


def log_likelihood(theta, x, y, data, var, size):
    """
    Logarithm of the likelihood function.
    """
    #unpack the parameters
    peak, center_x, center_y, radius, focus, width_x, width_y = theta

    #1)Generate a model Airy disc
    amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius, x_0=int(size[0]/2.-0.5), y_0=int(size[1]/2.-0.5))
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape(size)

    #2)Apply Focus
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape(size)
    model = signal.convolve2d(adata, focusdata, mode='same')

    #3)Apply CCD diffusion, approximated with a Gaussian
    CCDdata = np.array([[0.0, width_y, 0.0],
                        [width_x, (1.-width_y-width_y-width_x-width_x), width_x],
                        [0.0, width_y, 0.0]])
    model = signal.convolve2d(model, CCDdata, mode='same').flatten()

    #true for Gaussian errors
    #lnL = - 0.5 * np.sum((data - model)**2 / var)
    #Gary B. said that this should be from the model not data so recompute var (now contains rn**2)
    var += model.copy()
    lnL = - 0.5 * np.sum((data - model)**2 / var)

    return lnL


def _printResults(best_params, errors):
    """
    Print basic results.
    """
    print("=" * 60)
    print('Fitting with MCMC:')
    pars = ['peak', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y']
    print('*'*20 + ' Fitted parameters ' + '*'*20)
    for name, value, sig in zip(pars, best_params, errors):
        print("{:s} = {:e} +- {:e}" .format(name, value, sig))
    print("=" * 60)


def _printFWHM(sigma_x, sigma_y, sigma_xerr, sigma_yerr, req=10.8):
    """
    Print results and compare to the requirement at 800nm.
    """
    print("=" * 60)
    print 'FWHM (requirement %.1f microns):' % req
    print round(np.sqrt(_FWHMGauss(sigma_x)*_FWHMGauss(sigma_y)), 2), ' +/- ', \
          round(np.sqrt(_FWHMGauss(sigma_xerr)*_FWHMGauss(sigma_yerr)), 3) , ' microns'
    print 'x:', round(_FWHMGauss(sigma_x), 2), ' +/- ', round(_FWHMGauss(sigma_xerr), 3), ' microns'
    print 'y:', round(_FWHMGauss(sigma_y), 2), ' +/- ', round(_FWHMGauss(sigma_yerr), 3), ' microns'
    print("=" * 60)


def _FWHMGauss(sigma, pixel=12):
    """
    Returns the FWHM of a Gaussian with a given sigma.
    The returned values is in microns (pixel = 12microns).
    """
    return sigma*2*np.sqrt(2*np.log(2))*pixel


def _ellipticityFromGaussian(sigmax, sigmay):
    """
    Ellipticity
    """
    return np.abs((sigmax**2 - sigmay**2) / (sigmax**2 + sigmay**2))


def _ellipticityerr(sigmax, sigmay, sigmaxerr, sigmayerr):
    """
    Error on ellipticity.
    """
    e = _ellipticityFromGaussian(sigmax, sigmay)
    err = e * np.sqrt((sigmaxerr/e)**2 + (sigmayerr/e)**2)
    return err


def _R2FromGaussian(sigmax, sigmay, pixel=0.1):
    """
    R2.
    """
    return (sigmax*pixel)**2 + (sigmay*pixel)**2


def _R2err(sigmax, sigmay, sigmaxerr ,sigmayerr):
    """
    Error on R2.
    """
    err = np.sqrt((2*_R2FromGaussian(sigmax, sigmay))**2*sigmaxerr**2 +
                  (2*_R2FromGaussian(sigmax, sigmay))**2*sigmayerr**2)
    return err


def getFiles(mintime=(17, 20, 17), maxtime=(17, 33, 17), folder='data/30Jul/'):
    """
    Find all files between a minimum time and maximum time from a given folder.

    :param mintime: minimum time (h, min, s)
    :type mintime: tuple
    :param maxtime: maximum time (h, min, s)
    :type maxtime: tuple
    :param folder: folder from which FITS files are looked for
    :type folder: str

    :return: a list of file names that have been taken between the two times.
    :rtype: list
    """
    start = datetime.time(*mintime)
    stop = datetime.time(*maxtime)
    all = g.glob(folder + '*.fits')
    ret = []
    for f in all:
        path, file = os.path.split(f)
        numbs = [int(x) for x in file.replace('sEuclid.fits', '').split('_')]
        data = datetime.time(*numbs)
        if start <= data <= stop:
            ret.append(file)
    return [folder + f for f in ret]


def _plotDifferenceIndividualVsJoined(individuals, joined, title='800nm', sigma=3,
                                      requirementFWHM=10.8, requirementE=0.156, requirementR2=0.002,
                                      truthx=None, truthy=None, FWHMlims=(7.6, 10.3)):
    """
    Simple plot
    """
    ind = []
    for file in g.glob(individuals):
        print file
        ind.append(fileIO.cPicleRead(file))

    join = fileIO.cPicleRead(joined)
    xtmp = np.arange(len(ind)) + 1

    #plot FWHM
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.12, right=0.98)
    ax1.set_title(title)

    wxind = np.asarray([_FWHMGauss(data['wx']) for data in ind])
    wyind = np.asarray([_FWHMGauss(data['wy']) for data in ind])
    wxerr = np.asarray([sigma*_FWHMGauss(data['wxerr']) for data in ind])
    wyerr = np.asarray([sigma*_FWHMGauss(data['wyerr']) for data in ind])

    ax1.errorbar(xtmp, wxind, yerr=wxerr, fmt='o')
    ax1.errorbar(xtmp[-1]+1, _FWHMGauss(join['wx']), yerr=sigma*_FWHMGauss(join['wxerr']), fmt='s', c='r')
    ax2.errorbar(xtmp, wyind, yerr=wyerr, fmt='o')
    ax2.errorbar(xtmp[-1]+1, _FWHMGauss(join['wy']), yerr=sigma*_FWHMGauss(join['wyerr']), fmt='s', c='r')

    geommean = np.sqrt(wxind*wyind)
    err = np.sqrt(wxerr*wyerr)
    ax3.errorbar(xtmp, geommean, yerr=err, fmt='o')
    ax3.errorbar(xtmp[-1]+1, np.sqrt(_FWHMGauss(join['wx'])*_FWHMGauss(join['wy'])),
                 yerr=sigma*np.sqrt(_FWHMGauss(join['wxerr'])*_FWHMGauss(join['wyerr'])), fmt='s', c='r')

    #simulations
    if truthx is not None:
        ax1.axhline(y=_FWHMGauss(truthx), label='Input', c='g')
    if truthy is not None:
        ax2.axhline(y=_FWHMGauss(truthy), label='Input', c='g')
        ax3.axhline(y=np.sqrt(_FWHMGauss(truthx)*_FWHMGauss(truthy)), label='Input', c='g')

    #requirements
    if requirementFWHM is not None:
        ax1.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='--')
        ax2.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='--')
        ax3.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='-')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(visible=False)
    plt.sca(ax3)

    ltmp = np.hstack((xtmp, xtmp[-1]+1))
    plt.xticks(ltmp, ['Individual %i' % x for x in ltmp[:-1]] + ['Joint',], rotation=45)

    #ax1.set_ylim(7.1, 10.2)
    ax1.set_ylim(*FWHMlims)
    ax2.set_ylim(*FWHMlims)
    #ax2.set_ylim(8.6, 10.7)
    ax3.set_ylim(*FWHMlims)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax3.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel(r'FWHM$_{X} \, [\mu$m$]$')
    ax2.set_ylabel(r'FWHM$_{Y} \, [\mu$m$]$')
    #ax3.set_ylabel(r'FWHM$=\sqrt{FWHM_{X}FWHM_{Y}} \quad [\mu$m$]$')
    ax3.set_ylabel(r'FWHM$ \, [\mu$m$]$')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedFWHM%s.pdf' % title)
    plt.close()

    #plot R2 and ellipticity
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.12, right=0.98)
    ax1.set_title(title)

    R2x = [_R2FromGaussian(data['wx'], data['wy'])*1e3 for data in ind]
    errR2 = [sigma*1.e3*_R2err(data['wx'], data['wy'], data['wxerr'], data['wyerr']) for data in ind]
    ax1.errorbar(xtmp, R2x, yerr=errR2, fmt='o')
    ax1.errorbar(xtmp[-1]+1, _R2FromGaussian(join['wx'], join['wy'])*1e3,
                 yerr=sigma*1.e3*_R2err(join['wx'], join['wy'], join['wxerr'], join['wyerr']), fmt='s')

    ell = [_ellipticityFromGaussian(data['wx'], data['wy']) for data in ind]
    ellerr = [sigma*_ellipticityerr(data['wx'], data['wy'], data['wxerr'], data['wyerr']) for data in ind]
    ax2.errorbar(xtmp, ell, yerr=ellerr, fmt='o')
    ax2.errorbar(xtmp[-1]+1, _ellipticityFromGaussian(join['wx'], join['wy']),
                 yerr=sigma*_ellipticityerr(join['wx'], join['wy'], join['wxerr'], join['wyerr']), fmt='s')

    if requirementE is not None:
        ax2.axhline(y=requirementE, label='Requirement (800nm)', c='r')
    if requirementR2 is not None:
        ax1.axhline(y=requirementR2*1e3, label='Requirement (800nm)', c='r')

    #simulations
    if truthx and truthy is not None:
        ax2.axhline(y=_ellipticityFromGaussian(truthx, truthy), label='Input', c='g')
        ax1.axhline(y= _R2FromGaussian(truthx, truthy)*1e3, label='Input', c='g')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    ltmp = np.hstack((xtmp, xtmp[-1]+1))
    plt.xticks(ltmp, ['Individual%i' % x for x in ltmp[:-1]] + ['Joint',], rotation=45)

    ax1.set_ylim(0.0011*1e3, 0.003*1e3)
    ax2.set_ylim(0., 0.23)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel(r'$R^{2}$ [mas$^{2}$]')
    ax2.set_ylabel('ellipticity')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedR2e%s.pdf' % title)
    plt.close()


def _plotModelResiduals(id='simulated800nmJoint1', folder='results/', out='Residual.pdf', individual=False):
    """
    Generate a plot with data, model, and residuals.
    """
    #data
    if individual:
        data = pf.getdata(folder+id+'small.fits')
        data[data < 1] = 1.
        data = np.log10(data)
    else:
        data = pf.getdata(folder+id+'datafit.fits')
        data[data < 1] = 1.
        data = np.log10(data)
    #model
    model = pf.getdata(folder+id+'model.fits ')
    model[model < 1] = 1.
    model = np.log10(model)
    #residual
    residual = pf.getdata(folder+id+'residual.fits')
    #squared residual
    residualSQ = pf.getdata(folder+id+'residualSQ.fits')

    max = np.max((data.max(), model.max()))

    #figure
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax = [ax1, ax2, ax3, ax4]
    fig.subplots_adjust(hspace=0.05, wspace=0.3, top=0.95, bottom=0.02, left=0.02, right=0.9)
    ax1.set_title('Data')
    ax2.set_title('Model')
    ax3.set_title('Residual')
    ax4.set_title('$L^{2}$ Residual')

    im1 = ax1.imshow(data, interpolation='none', vmax=max, origin='lower', vmin=0.1)
    im2 = ax2.imshow(model, interpolation='none', vmax=max, origin='lower', vmin=0.1)
    im3 = ax3.imshow(residual, interpolation='none', origin='lower', vmin=-100, vmax=100)
    im4 = ax4.imshow(residualSQ, interpolation='none', origin='lower', vmin=0., vmax=10)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label(r'$\log_{10}(D_{i, j} \quad [e^{-}]$)')
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label(r'$\log_{10}(M_{i, j} \quad [e^{-}]$)')
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label(r'$M_{i, j} - D_{i, j}  \quad [e^{-}]$')
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label(r'$\frac{(M_{i, j} - D_{i, j})^{2}}{\sigma_{CCD}^{2}}$')

    for tmp in ax:
        plt.sca(tmp)
        plt.xticks(visible=False)
        plt.yticks(visible=False)

    plt.savefig(out)
    plt.close()


def plotAllResiduals():
    """
    Plot residuals of all model fits.
    """
    #Joint fits
    files = g.glob('results/J*.fits')
    individuals = [file for file in files if 'datafit' in file]
    for file in individuals:
        id = file.replace('results/', '').replace('datafit.fits', '')
        print 'processing:', id
        _plotModelResiduals(id=id, folder='results/', out='results/%sResidual.pdf' % id)

    #Individual fits
    files = g.glob('results/I*.fits')
    individuals = [file for file in files if 'model' in file]
    for file in individuals:
        id = file.replace('results/', '').replace('model.fits', '')
        print 'processing:', id
        _plotModelResiduals(id=id, folder='results/', out='results/%sResidual.pdf' % id, individual=True)


def _CCDkernel(CCDx=10, CCDy=10, width_x=0.35, width_y=0.4, size=21):
    """
    Generate a CCD PSF using a 2D Gaussian and save it to file.
    """
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    CCD = models.Gaussian2D(1., CCDx, CCDy, width_x, width_y, 0.)
    CCDdata = CCD.eval(xx, yy, 1., CCDx, CCDy, width_x, width_y, 0.).reshape((size, size))
    fileIO.writeFITS(CCDdata, 'CCDPSF.fits', int=False)
    return CCDdata


def _AiryDisc(amplitude=1.5e5, center_x=10.0, center_y=10.0, radius=0.5, focus=0.4, size=21,
              resample=False):
    """
    Generate a CCD PSF using a 2D Gaussian and save it to file.
    """
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape((size, size))
    fileIO.writeFITS(adata, 'AiryDisc.fits', int=False)
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
    model1 = signal.convolve2d(adata.copy(), focusdata, mode='same')
    model2 = scipy.ndimage.filters.gaussian_filter(adata, sigma=focus)
    fileIO.writeFITS(model1, 'AiryDiscDefocused1.fits', int=False)
    fileIO.writeFITS(model2, 'AiryDiscDefocused2.fits', int=False)

    if resample:
        import SamPy.image.manipulation as m
        tmp = m.frebin(model2, 21, nlout=21, total=False)
        tmp /= tmp.max()
        tmp *= 1.5e5
        fileIO.writeFITS(tmp, 'AiryDiscDefocused1Rebin.fits', int=False)

    return adata, model1


def plotModelExample(size=10):
    data = pf.getdata('testdata/stacked.fits') * 3.1
    y, x = m.maximum_position(data)
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()
    bias = np.median(data[y-size: y+size, x-220:x-20])
    print 'ADC offset (e):', bias
    spot -= bias
    spot[spot < 0.] = 1.

    #CCD PSF and Airy disc
    CCD = _CCDkernel()
    airy, defocused = _AiryDisc()

    #log all
    spot = np.log10(spot + 1.)
    CCD = np.log10(CCD + 1.)
    airy = np.log10(airy + 1.)
    defocused = np.log10(defocused + 1.)

    #figure
    matplotlib.rc('text', usetex=True)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax = [ax1, ax2, ax3, ax4]
    fig.subplots_adjust(hspace=0.05, wspace=0.3, top=0.95, bottom=0.02, left=0.02, right=0.9)

    ax1.set_title('Data')
    ax2.set_title('Airy Disc')
    ax3.set_title('CCD Kernel')
    ax4.set_title('Defocused Airy Disc')

    im1 = ax1.imshow(spot, interpolation='none', origin='lower', vmin=0., vmax=4.7)
    im2 = ax2.imshow(airy, interpolation='none', origin='lower', vmin=0., vmax=4.7)
    im3 = ax3.imshow(CCD, interpolation='none', origin='lower', vmin=0., vmax=0.05)
    im4 = ax4.imshow(defocused, interpolation='none', origin='lower', vmin=0., vmax=4.7)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label(r'$\log_{10}(D_{i, j} + 1 \quad [e^{-}]$)')
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label(r'$\log_{10}(A_{i, j} + 1 \quad [e^{-}]$)')
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label(r'$\log_{10}(G_{\textrm{CCD}} + 1 \quad [e^{-}])$')
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label(r'$\log_{10}(A_{i, j} \ast G_{\textrm{focus}} + 1 \quad [e^{-}])$')

    for tmp in ax:
        plt.sca(tmp)
        plt.xticks(visible=False)
        plt.yticks(visible=False)

    plt.savefig('ModelExample.pdf')
    plt.close()


def _printAnalysedData(folder='results/', id='J*.pkl'):
    print 'Wavelength   Median Intensity    Number of Files     SNR'
    structure = '%i     &   %.1f    &   %i    &     %.0f \\\\'
    for file in g.glob(folder+id):
        tmp = fileIO.cPicleRead(file)
        med = np.median(tmp['peakvalues'])
        wave = int(tmp['wavelength'].replace('nm', ''))
        out = structure %  (wave, med/1.e3, np.size(tmp['peakvalues']), med/(np.sqrt(med + 4.5**2)))
        print out


def _amplitudeFromPeak(peak, x, y, radius, x_0=10, y_0=10):
    """
    This function can be used to estimate an Airy disc amplitude from the peak pixel, centroid and radius.
    """
    rz = jn_zeros(1, 1)[0] / np.pi
    r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / rz)
    if r == 0.:
        return peak
    rt = np.pi * r
    z = (2.0 * j1(rt) / rt)**2
    amp = peak / z
    return amp


def _peakFromTruth(theta, size=21):
    """
    Derive the peak value from the parameters used for simulations.
    """
    amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    x = np.arange(0, size)
    y = np.arange(0, size)
    x, y = np.meshgrid(x, y)
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(x, y, amplitude, center_x, center_y, radius)
    return adata.max()


def _expectedValues():
    """
    These values are expected for well exposed spot data. The dictionary has a tuple for each wavelength.
    Note that for example focus is data set dependent and should be used only as an indicator of a possible value.

    keys: l600, l700, l800, l890

    tuple = [radius, focus, widthx, widthy]
    """
    out = dict(l600=(0.45, 0.40, 0.34, 0.32),
               l700=(0.47, 0.40, 0.32, 0.31),
               l800=(0.49, 0.41, 0.30, 0.30),
               l800l=(0.49, 0.41, 0.27, 0.27),
               l800m=(0.49, 0.41, 0.30, 0.30),
               l800h=(0.49, 0.41, 0.31, 0.31),
               l890=(0.54, 0.38, 0.29, 0.29))

    return out


def _simpleExample(CCDx=10, CCDy=10):
    spot = np.zeros((21, 21))
    #Create the coordinates x and y
    x = np.arange(0, spot.shape[1])
    y = np.arange(0, spot.shape[0])
    #Put the coordinates in a mesh
    xx, yy = np.meshgrid(x, y)

    peak, center_x, center_y, radius, focus, width_x, width_y = (200000, 10.1, 9.95, 0.5, 0.5, 0.03, 0.06)
    amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius, x_0=CCDx, y_0=CCDy)
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape(spot.shape)
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape(spot.shape)
    foc = signal.convolve2d(adata, focusdata, mode='same')
    fileIO.writeFITS(foc, 'TESTfocus.fits', int=False)
    CCDdata = np.array([[0.0, width_y, 0.0],
                        [width_x, (1.-width_y-width_y-width_x-width_x), width_x],
                        [0.0, width_y, 0.0]])
    model = signal.convolve2d(foc, CCDdata, mode='same')
    #save model
    fileIO.writeFITS(model, 'TESTkernel.fits', int=False)



def analyseOutofFocus():
    """

    """
    forwardModel('data/13_24_53sEuclid.fits', wavelength='l800', out='blurred800',
                 spotx=2983, spoty=3760, size=10, blurred=True, burn=10000, run=10000)


if __name__ == '__main__':
    analyseOutofFocus()
    #_simpleExample()