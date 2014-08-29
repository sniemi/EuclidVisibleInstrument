"""
CCD Spot Measurements
=====================

Analyse laboratory CCD PSF measurements by forward modelling.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: astropy
:requires: matplotlib
:requires: VISsim-Python
:requires: emcee

:version: 0.96

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
import pyfits as pf
import numpy as np
import emcee
import scipy.ndimage.measurements as m
from scipy import signal
from scipy import ndimage
from support import files as fileIO
from astropy.modeling import models, fitting
import triangle
import glob as g
import os, sys, datetime


def forwardModel(file, out='Data', gain=3.1, size=10, burn=10, spotx=2888, spoty=3514, run=100,
                 simulation=False, truths=None):
    """
    A single file to quickly test if the method works
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
    data[data + rn**2 < 0.] = 0.  #set highly negative values to zero
    #assume errors scale as sqrt of the values + readnoise
    #sigma = np.sqrt(data/gain + rn**2)
    sigma = np.sqrt(data + rn**2)
    #variance is the true noise model
    var = sigma**2

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

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
    print 'Done'

    #MCMC based fitting
    print 'Bayesian Fitting...'
    ndim = 7
    nwalkers = 1000

    #amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    # Choose an initial set of positions for the walkers using the Gaussian fit
    p0 = [np.asarray([1.5*max,#p.amplitude.value,
                      p.x_mean.value,
                      p.y_mean.value,
                      np.max([p.x_stddev.value, p.y_stddev.value]),
                      0.5,
                      0.3,
                      0.3]) + 1e-3*np.random.randn(ndim) for i in xrange(nwalkers)]

    # Initialize the sampler with the chosen specs.
    #Create the coordinates x and y
    x = np.arange(0, spot.shape[1])
    y = np.arange(0, spot.shape[0])
    #Put the coordinates in a mesh
    xx, yy = np.meshgrid(x, y)

    #Flatten the arrays
    xx = xx.flatten()
    yy = yy.flatten()

    #initiate sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, var], threads=7)

    # Run a burn-in
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)

    # Print out the mean acceptance fraction
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
    amplitudeE, center_xE, center_yE, radiusE, focusE, width_xE, width_yE = errors_fit
    _printResults(params_fit, errors_fit)

    #Best fit model
    amplitude, center_x, center_y, radius, focus, width_x, width_y = params_fit
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape(spot.shape)
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape(spot.shape)
    foc = signal.convolve2d(adata, focusdata, mode='same')
    CCD = models.Gaussian2D(1., CCDx, CCDy, width_x, width_y, 0.)
    CCDdata = CCD.eval(xx, yy, 1., CCDx, CCDy, width_x, width_y, 0.).reshape(spot.shape)
    model = signal.convolve2d(foc, CCDdata, mode='same')
    #save model
    fileIO.writeFITS(model, out+'model.fits', int=False)

    #residuals
    fileIO.writeFITS(model - spot, out+'residual.fits', int=False)
    fileIO.writeFITS(((model - spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)

    # a simple goodness of fit
    gof = (1./(np.size(data) - ndim)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof, ' Maximum difference:', np.max(np.abs(model - spot))

    #results and save results
    _printFWHM(width_x, width_y, errors_fit[5], errors_fit[6])
    res = dict(wx=width_x, wy=width_y, wxerr=width_xE, wyerr=width_yE, out=out,
               peakvalue=max, CCDmodel=CCD, CCDmodeldata=CCDdata, GoF=gof)
    fileIO.cPickleDumpDictionary(res, out+'.pkl')

    #plot
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    fig = triangle.corner(samples,
                          labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)
    fig.savefig(out+'Triangle.png')



def forwardModelJointFit(files, out, wavelength, gain=3.1, size=10, burn=10, run=100,
                         spotx=2888, spoty=3514, simulated=False, truths=None):
    """
    A single file to quickly test if the method works
    """
    print '\n\n\n'
    print '_'*120

    images = len(files)
    orig = []
    image = []
    noise = []
    peakvalues = []
    for file in files:
        print file
        #get data and convert to electrons
        o = pf.getdata(file)*gain

        if simulated:
            data = o
        else:
            #roughly the correct location - to avoid identifying e.g. cosmic rays
            data = o[spoty-(size*3):spoty+(size*3)+1, spotx-(size*3):spotx+(size*3)+1].copy()

        #maximum position within the cutout
        y, x = m.maximum_position(data)

        #spot and the peak pixel within the spot, this is also the CCD kernel position
        spot = data[y-size:y+size+1, x-size:x+size+1].copy()
        orig.append(spot.copy())
        CCDy, CCDx = m.maximum_position(spot)

        #bias estimate
        if simulated:
            bias = 9000.
            rn = 4.5
        else:
            bias = np.median(o[spoty-size: spoty+size, spotx-220:spotx-20])
            rn = np.std(o[spoty-size: spoty+size, spotx-220:spotx-20])

        print 'Readnoise (e):', rn
        if rn < 2. or rn > 6.:
            print 'NOTE: suspicious readout noise estimate...'
        print 'ADC offset (e):', bias

        #remove bias
        spot -= bias

        max = np.max(spot)
        print 'Maximum Value:', max
        peakvalues.append(max)

        #noise model
        variance = spot.copy() + rn**2

        #save to a list
        image.append(spot)
        noise.append(variance)

    #sensibility test, try to check if all the files in the fit are of the same dataset
    if np.std(peakvalues) > 3*np.sqrt(np.median(peakvalues)):
        #check for more than 3sigma outliers...
        print 'POTENTIAL OUTLIER, please check the input files...'
        print np.std(peakvalues), 3*np.sqrt(np.median(peakvalues))

    #MCMC based fitting
    ndim = 2*images + 5  #xpos, ypos for each image and single amplitude, radius, focus, and sigmaX and sigmaY
    nwalkers = 1000
    print 'Bayesian Fitting, model has %i dimensions' % ndim

    # Choose an initial set of positions for the walkers using the Gaussian fit
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    p0 = [np.asarray((([CCDx, CCDy]*images) +[max*1.05, 0.5, 0.5, 0.3, 0.3])) +
          1e-3*np.random.randn(ndim) for i in xrange(nwalkers)]

    # Initialize the sampler with the chosen specs.
    #Create the coordinates x and y
    x = np.arange(0, spot.shape[1])
    y = np.arange(0, spot.shape[0])
    #Put the coordinates in a mesh
    xx, yy = np.meshgrid(x, y)

    #Flatten the arrays
    xx = xx.flatten()
    yy = yy.flatten()

    #initiate sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorJoint, args=[xx, yy, image, noise], threads=7)

    # Run a burn-in
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)

    # Print out the mean acceptance fraction
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
    print params_fit

    #unpack the fixed parameters
    amplitude, radius, focus, width_x, width_y = params_fit[-5:]
    amplitudeE, radiusE, focusE, width_xE, width_yE = errors_fit[-5:]

    #print results
    _printFWHM(width_x, width_y, width_xE, width_yE)

    #save the best models per file
    size = size*2 + 1
    gofs = []
    for index, file in enumerate(files):
        #path, file = os.path.split(file)
        id = 'results/' + out + str(index)
        #X and Y are always in pairs
        center_x, center_y = params_fit[index:index+2]

        #1)Generate a model Airy disc
        airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
        adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape((size, size))

        #2)Apply Focus
        f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
        focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
        model = signal.convolve2d(adata, focusdata, mode='same')

        #3)Apply CCD diffusion, approximated with a Gaussian
        CCD = models.Gaussian2D(1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.)
        CCDdata = CCD.eval(xx, yy, 1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.).reshape((size, size))
        model = signal.convolve2d(model, CCDdata, mode='same')

        #save the data, model and residuals
        fileIO.writeFITS(orig[index], id+'data.fits', int=False)
        fileIO.writeFITS(image[index], id+'datafit.fits', int=False)
        fileIO.writeFITS(model, id+'model.fits', int=False)
        fileIO.writeFITS(model - image[index], id+'residual.fits', int=False)
        fileIO.writeFITS(((model - image[index])**2 / noise[index]), id+'residualSQ.fits', int=False)

        #a simple goodness of fit
        gof = (1./(np.size(image[index])*images - ndim)) * np.sum((model - image[index])**2 / noise[index])
        print 'GoF:', gof, ' Max difference', np.max(np.abs(model - image[index]))
        gofs.append(gof)

    #save results
    res = dict(wx=width_x, wy=width_y, wxerr=width_xE, wyerr=width_yE, files=files, out=out,
               wavelength=wavelength, peakvalues=np.asarray(peakvalues), CCDmodel=CCD, CCDmodeldata=CCDdata,
               GoFs=gofs)
    fileIO.cPickleDumpDictionary(res, 'results/' + out + '.pkl')

    #plot
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    fig = triangle.corner(samples, labels=['x', 'y']*images + ['amplitude', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)
    fig.savefig('results/' + out + 'Triangle.png')


def log_posterior(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, z, var)


def log_posteriorJoint(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_priorJoint(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihoodJoint(theta, x, y, z, var)


def log_prior(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    if 8. < center_x < 12. and 8. < center_y < 12. and 0. < width_x < 1. and 0. < width_y < 1. and \
       1.e2 < amplitude < 1.e6 and 0. < radius < 2. and 0. < focus < 1.:
        return 0.
    else:
        return -np.inf


def log_priorJoint(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    tmp = theta[-5:] #these are the last five i.e. amplitude, radius, focus, sigmaX, and sigmaY
    if all(3. < x < 16. for x in theta[:-5]) and 1.e2 < tmp[0] < 1.e6 and 0. < tmp[1] < 2. and 0. < tmp[2] < 1. and \
       0. < tmp[3] < 1. and 0. < tmp[4] < 1.:
        return 0.
    else:
        return -np.inf


def log_likelihood(theta, x, y, data, var, size=21):
    """
    Logarithm of the likelihood function.
    """
    #unpack the parameters
    amplitude, center_x, center_y, radius, focus, width_x, width_y = theta

    #1)Generate a model Airy disc
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape((size, size))

    #2)Apply Focus
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
    model = signal.convolve2d(adata, focusdata, mode='same')

    #3)Apply CCD diffusion, approximated with a Gaussian
    CCD = models.Gaussian2D(1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.)
    CCDdata = CCD.eval(x, y, 1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.).reshape((size, size))
    model = signal.convolve2d(model, CCDdata, mode='same').flatten()

    #true for Gaussian errors, but not really true here because of mixture of Poisson and Gaussian noise
    lnL = - 0.5 * np.sum((data - model)**2 / var)
    #others...
    #lnL = - 2. * np.sum((((data - model)**2) + np.abs(data - model))/var)
    #using L1 norm would be true for exponential distribution
    #lnL = - np.sum(np.abs(data - model) / var)

    return lnL


def log_likelihoodJoint(theta, x, y, data, var, size=21):
    """
    Logarithm of the likelihood function for joint fitting. Not really sure if this is right...
    """
    #unpack the parameters
    #[xpos, ypos]*images) +[amplitude, radius, focus])
    images = len(theta[:-5]) / 2
    amplitude, radius, focus, width_x, width_y = theta[-5:]

    lnL = 0.
    for tmp in range(images):
        #X and Y are always in pairs
        center_x, center_y = theta[tmp:tmp+2]

        #1)Generate a model Airy disc
        airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
        adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape((size, size))

        #2)Apply Focus
        f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
        focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
        model = signal.convolve2d(adata, focusdata, mode='same')

        #3)Apply CCD diffusion, approximated with a Gaussian
        CCD = models.Gaussian2D(1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.)
        CCDdata = CCD.eval(x, y, 1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.).reshape((size, size))
        model = signal.convolve2d(model, CCDdata, mode='same').flatten()

        lnL += - 0.5 * np.sum((data[tmp].flatten() - model)**2 / var[tmp].flatten())

    return lnL


def _printResults(best_params, errors):
    """
    Print basic results.
    """
    print("=" * 60)
    print('Fitting with MCMC:')
    pars = ['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y']
    print('*'*20 + ' Fitted parameters ' + '*'*20)
    for name, value, sig in zip(pars, best_params, errors):
        print("{:s} = {:e} +- {:e}" .format(name, value, sig))
    print("=" * 60)


def _simulate(theta=(1.e5, 10., 10.3, 0.6, 0.1, 10., 10., 0.33, 0.35), gain=3.1, bias=9000, rn=4.5, size=21,
              out='simulated.fits', Gaussian=True):
    """
    Generate simulated spot image with the assumed process.
    """
    #unpack the parameters
    amplitude, center_x, center_y, radius, focus, xCCD, yCCD, width_x, width_y = theta

    #Create the coordinates x and y
    x = np.arange(0, size)
    y = np.arange(0, size)
    #Put the coordinates in a mesh
    x, y = np.meshgrid(x, y)

    #1)Generate a model Airy disc
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape((size, size))

    #2)Apply Focus
    #data = ndimage.filters.gaussian_filter(adata, [width_y, width_x]) #no position
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
    data = signal.convolve2d(adata, focusdata, mode='same')

    #3)Apply CCD diffusion
    if Gaussian:
        #full Gaussian
        CCD = models.Gaussian2D(1., xCCD, yCCD, width_x, width_y, 0.)
        d = CCD.eval(x, y, 1.,xCCD, yCCD, width_x, width_y, 0.).reshape((size, size))
        CCDdata = signal.convolve2d(data, d, mode='same')
    else:
        #CCD kernel -- high flux
        kernel = np.array([[0.01/4., 0.05, 0.01/4.],
                          [0.075, 0.74, 0.075],
                          [0.01/4., 0.05, 0.01/4.]])
        fileIO.writeFITS(kernel, 'kernel.fits', int=False)
        CCDdata = ndimage.convolve(data, kernel)

    #4)Add Poisson noise
    rounded = np.rint(CCDdata)
    residual = CCDdata.copy() - rounded #ugly workaround for multiple rounding operations...
    rounded[rounded < 0.0] = 0.0
    CCDdata = np.random.poisson(rounded).astype(np.float64)
    CCDdata += residual

    #5)Add ADC offset level
    CCDdata += bias

    #6)Add readnoise
    CCDdata += np.random.normal(loc=0.0, scale=rn, size=CCDdata.shape)

    #7)Convert to DNs
    CCDdata = np.round(CCDdata/gain)

    #save to a file
    fileIO.writeFITS(CCDdata, out)


def _printFWHM(sigma_x, sigma_y, sigma_xerr, sigma_yerr, req=10.8):
    """
    Print results and compare to the requirement
    """
    print("=" * 60)
    print 'FWHM (requirement %.1f microns):' % req
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

    """
    return np.abs((sigmax**2 - sigmay**2) / (sigmax**2 + sigmay**2))


def _R2FromGaussian(sigmax, sigmay, pixel=0.1):
    """

    """
    return (sigmax*pixel)**2 + (sigmay*pixel)**2


def RunTestSimulations(both=False):
    """
    A set of simulated spots and analysis.
    """
    print("|" * 120)
    print 'Joint Fitting Simulation'
    #a joint fit test - vary only the x and y positions
    theta1 = (2.e5, 9.9, 10.03, 0.45, 0.5, 10., 10., 0.296, 0.335)
    theta2 = (2.e5, 10.05, 9.95, 0.45, 0.5, 10., 10., 0.296, 0.335)
    theta3 = (2.e5, 9.98, 10.1, 0.45, 0.5, 10., 10., 0.296, 0.335)
    theta4 = (2.e5, 10.0, 10.1, 0.45, 0.5, 10., 10., 0.296, 0.335)
    theta5 = (2.e5, 10.1, 9.99, 0.45, 0.5, 10., 10., 0.296, 0.335)

    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        _simulate(theta=theta, out='simulated/simulatedJoint%i.fits' %i)
        forwardModel(file='simulated/simulatedJoint%i.fits' %i, out='simulatedResults/RunI%i' %i, simulation=True,
                     truths=[theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]])
        print 'Simulation Parameters'
        print 'amplitude, center_x, center_y, radius, focus, width_x, width_y'
        print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
        print("=" * 60)

    truths = [theta1[1], theta1[2], theta2[1], theta2[2], theta3[1], theta3[2], theta4[1], theta4[2],
              theta5[1], theta5[2], theta1[0], theta4[3], theta1[4], theta1[7], theta1[8]]
    forwardModelJointFit(g.glob('simulated/simulatedJoint?.fits'),
                         out='simulated800nmJoint', wavelength='800nm', simulated=True,
                         truths=truths)

    print 'True width_x and widht_y:', theta1[7], theta1[8]

    #test plots
    if both:
        _plotDifferenceIndividualVsJoined(individuals='simulatedResults/RunI*.pkl',
                                          joined='results/simulated800nmJoint.pkl',
                                          title='Simulated', truthx=theta1[7], truthy=theta1[8])

        #different simulation sets
        print("|" * 120)
        print 'Single Fitting Simulations'
        theta1 = (2.e5, 9.95, 10.3, 0.45, 0.5, 10., 10., 0.33, 0.35)
        theta2 = (1.e5, 10.1, 10.1, 0.55, 0.45, 10., 10., 0.38, 0.36)
        theta3 = (8.e4, 10., 10.2, 0.4, 0.55, 10., 10., 0.25, 0.35)
        theta4 = (5.e4, 10.1, 10.3, 0.42, 0.48, 10., 10., 0.30, 0.28)
        theta5 = (1.e5, 10., 10.2, 0.5, 0.45, 10., 10., 0.35, 0.31)
        thetas = [theta1, theta2, theta3, theta4, theta5]

        for i, theta in enumerate(thetas):
            _simulate(theta=theta, out='simulated/simulatedSmall%i.fits' %i)
            forwardModel(file='simulated/simulatedSmall%i.fits' %i, out='simulatedResults/Run%i' %i, simulation=True,
                         truths=[theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]])
            print("=" * 60)
            print 'Simulation Parameters'
            print 'amplitude, center_x, center_y, radius, focus, width_x, width_y'
            print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
            print("=" * 60)


def RunData(files, out='testdata'):
    """
    A set of test data to analyse.
    """
    for i, file in enumerate(files):
        forwardModel(file=file, out='results/%s%i' % (out, i))


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


def individualRuns():
    """
    Execute all spot data analysis runs individually.
    """
    #800 nm
    RunData(getFiles(mintime=(15, 12, 20), maxtime=(15, 24, 16), folder='data/31Jul/'), out='I800nm5k')
    RunData(getFiles(mintime=(15, 28, 40), maxtime=(15, 39, 21), folder='data/31Jul/'), out='I800nm10k')
    RunData(getFiles(mintime=(15, 43, 24), maxtime=(15, 51, 47), folder='data/31Jul/'), out='I800nm20k')
    RunData(getFiles(mintime=(15, 56, 11), maxtime=(16, 02, 58), folder='data/31Jul/'), out='I800nm30k')
    RunData(getFiles(mintime=(16, 12, 39), maxtime=(16, 18, 25), folder='data/31Jul/'), out='I800nm38k')
    RunData(getFiles(mintime=(16, 21, 52), maxtime=(16, 26, 16), folder='data/31Jul/'), out='I800nm50k')
    RunData(getFiles(mintime=(16, 32, 02), maxtime=(16, 35, 23), folder='data/31Jul/'), out='I800nm54k')
    #700 nm
    RunData(getFiles(mintime=(17, 20, 17), maxtime=(17, 33, 17), folder='data/30Jul/'), out='I700nm5k')
    RunData(getFiles(mintime=(17, 37, 35), maxtime=(17, 46, 51), folder='data/30Jul/'), out='I700nm9k')
    RunData(getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/'), out='I700nm52k')
    RunData(getFiles(mintime=(17, 58, 18), maxtime=(17, 59, 31), folder='data/30Jul/'), out='I700nm32k')
    #600 nm
    RunData(getFiles(mintime=(15, 22, 00), maxtime=(15, 36, 32), folder='data/30Jul/'), out='I600nm5k')
    RunData(getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/'), out='I600nm54k')
    RunData(getFiles(mintime=(15, 52, 07), maxtime=(16, 06, 32), folder='data/30Jul/'), out='I600nm10k')
    #890 nm
    RunData(getFiles(mintime=(13, 37, 37), maxtime=(13, 50, 58), folder='data/01Aug/'), out='I890nm5k')
    RunData(getFiles(mintime=(14, 00, 58), maxtime=(14, 11, 54), folder='data/01Aug/'), out='I890nm10k')
    RunData(getFiles(mintime=(14, 17, 57), maxtime=(14, 25, 49), folder='data/01Aug/'), out='I890nm30k')
    RunData(getFiles(mintime=(14, 30, 03), maxtime=(14, 34, 37), folder='data/01Aug/'), out='I890nm50k')


def jointRuns():
    """
    Execute all spot data analysis runs fitting jointly.
    """
    #800 nm
    forwardModelJointFit(getFiles(mintime=(15, 12, 20), maxtime=(15, 24, 16), folder='data/31Jul/'),
                         out='J800nm5k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(15, 28, 40), maxtime=(15, 39, 21), folder='data/31Jul/'),
                         out='J800nm10k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(15, 43, 24), maxtime=(15, 51, 47), folder='data/31Jul/'),
                         out='J800nm20k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(15, 56, 11), maxtime=(16, 02, 58), folder='data/31Jul/'),
                         out='J800nm30k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 12, 39), maxtime=(16, 18, 25), folder='data/31Jul/'),
                         out='J800nm38k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 21, 52), maxtime=(16, 26, 16), folder='data/31Jul/'),
                         out='J800nm50k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 32, 02), maxtime=(16, 35, 23), folder='data/31Jul/'),
                         out='J800nm54k', wavelength='800nm')
    #700 nm
    forwardModelJointFit(getFiles(mintime=(17, 20, 17), maxtime=(17, 33, 17), folder='data/30Jul/'),
                         out='J700nm5k', wavelength='700nm')
    forwardModelJointFit(getFiles(mintime=(17, 37, 35), maxtime=(17, 46, 51), folder='data/30Jul/'),
                         out='J700nm9k', wavelength='700nm')
    forwardModelJointFit(getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/'),
                         out='J700nm52k', wavelength='700nm')
    forwardModelJointFit(getFiles(mintime=(17, 58, 18), maxtime=(17, 59, 31), folder='data/30Jul/'),
                         out='J700nm32k', wavelength='700nm')
    #600 nm
    forwardModelJointFit(getFiles(mintime=(15, 22, 00), maxtime=(15, 36, 32), folder='data/30Jul/'),
                         out='J600nm5k', wavelength='600nm')
    forwardModelJointFit(getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/'),
                         out='J600nm54k', wavelength='600nm')
    forwardModelJointFit(getFiles(mintime=(15, 52, 07), maxtime=(16, 06, 32), folder='data/30Jul/'),
                         out='J600nm10k', wavelength='600nm')
    #890 nm
    forwardModelJointFit(getFiles(mintime=(13, 37, 37), maxtime=(13, 50, 58), folder='data/01Aug/'),
                         out='J890nm5k', wavelength='890nm')
    forwardModelJointFit(getFiles(mintime=(14, 00, 58), maxtime=(14, 11, 54), folder='data/01Aug/'),
                         out='J890nm10k', wavelength='890nm')
    forwardModelJointFit(getFiles(mintime=(14, 17, 57), maxtime=(14, 25, 49), folder='data/01Aug/'),
                         out='J890nm30k', wavelength='890nm')
    forwardModelJointFit(getFiles(mintime=(14, 30, 03), maxtime=(14, 34, 37), folder='data/01Aug/'),
                         out='J890nm50k', wavelength='890nm')


def _plotDifferenceIndividualVsJoined(individuals, joined, title='800nm', truthx=None, truthy=None):
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
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.95, bottom=0.15, left=0.12, right=0.93)
    plt.title(title)

    ax1.errorbar(xtmp, [_FWHMGauss(data['wx']) for data in ind], yerr=[_FWHMGauss(data['wxerr']) for data in ind], fmt='o')
    ax1.errorbar(xtmp[-1]+1, _FWHMGauss(join['wx']), yerr=_FWHMGauss(join['wxerr']), fmt='s')
    ax2.errorbar(xtmp, [_FWHMGauss(data['wy']) for data in ind], yerr=[_FWHMGauss(data['wyerr']) for data in ind], fmt='o')
    ax2.errorbar(xtmp[-1]+1, _FWHMGauss(join['wy']), yerr=_FWHMGauss(join['wyerr']), fmt='s')

    #simulations
    if truthx is not None:
        ax1.axhline(y=_FWHMGauss(truthx), label='Truth', c='g')
    if truthy is not None:
        ax2.axhline(y=_FWHMGauss(truthy), label='Truth', c='g')

    #requirements
    ax1.axhline(y=10.8, label='Requirement', c='r')
    ax2.axhline(y=10.8, label='Requirement', c='r')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(xtmp, ['Individual%i' % x for x in xtmp], rotation=45)

    ax1.set_ylim(5.5, 12.5)
    ax2.set_ylim(5.5, 12.5)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel('X FWHM [microns]')
    ax2.set_ylabel('Y FWHM [microns]')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedFWHM%s.pdf' % title)
    plt.close()

    #plot R2 and ellipticity
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.95, bottom=0.15, left=0.12, right=0.93)
    plt.title(title)

    ax1.errorbar(xtmp, [_R2FromGaussian(data['wx'], data['wy'])*1e3 for data in ind],
                 yerr=[_R2FromGaussian(data['wxerr'], data['wyerr'])*1e3 for data in ind], fmt='o')
    ax1.errorbar(xtmp[-1]+1, _R2FromGaussian(join['wx'], join['wy'])*1e3,
                 yerr=_R2FromGaussian(join['wxerr'], join['wyerr'])*1e3, fmt='s')

    ax2.errorbar(xtmp, [_ellipticityFromGaussian(data['wy'], data['wx']) for data in ind],
                 yerr=[_ellipticityFromGaussian(data['wyerr'], data['wxerr']) for data in ind], fmt='o')
    ax2.errorbar(xtmp[-1]+1, _ellipticityFromGaussian(join['wy'], join['wx']),
                 yerr=_ellipticityFromGaussian(join['wyerr'], join['wxerr']), fmt='s')

    ax2.axhline(y=0.156, label='Requirement', c='r')
    ax1.axhline(y=0.002*1e3, label='Requirement', c='r')

    #simulations
    if truthx and truthy is not None:
        ax2.axhline(y=_ellipticityFromGaussian(truthx, truthy), label='Truth', c='g')
        ax1.axhline(y= _R2FromGaussian(truthx, truthy)*1e3, label='Truth', c='g')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(xtmp, ['Individual%i' % x for x in xtmp], rotation=45)

    ax1.set_ylim(0.001*1e3, 0.004*1e3)
    ax2.set_ylim(0., 0.33)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel(r'$R^{2}$ [mas$^{2}$]')
    ax2.set_ylabel('ellipticity')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedR2e%s.pdf' % title)
    plt.close()


def RunTest():
    """
    Test runs with test data sets
    """
    #Individual Fits
    RunData(g.glob('testdata/15*.fits'), out='test800nm')
    RunData(g.glob('testdata/17*.fits'), out='test700nm')

    #Joint Fit (same signal level and wavelength, but separate files) - test data set
    forwardModelJointFit(g.glob('testdata/15*.fits'), out='test800nmJoint', wavelength='800nm')
    forwardModelJointFit(g.glob('testdata/17*.fits'), out='test700nmJoint', wavelength='700nm')

    #test plots
    _plotDifferenceIndividualVsJoined(individuals='testresults/test700nm?.pkl', joined='testresults/test700nmJoint.pkl',
                                      title='700nm')
    _plotDifferenceIndividualVsJoined(individuals='testresults/test800nm?.pkl', joined='testresults/test800nmJoint.pkl',
                                      title='800nm')


if __name__ == '__main__':
    #Simulated spots and analysis
    RunTestSimulations()

    #Real Runs
    jointRuns()
    individualRuns()

    #Testing set
    #RunTest()
