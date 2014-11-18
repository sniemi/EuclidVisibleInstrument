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

:version: 1.9

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
from scipy import optimize
from scipy.special import j1, jn_zeros
from sklearn.gaussian_process import GaussianProcess
from support import files as fileIO
from astropy.modeling import models, fitting
import triangle
import glob as g
import os, datetime
from multiprocessing import Pool


__author__ = 'Sami-Matias Niemi'
__vesion__ = 1.8

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

    if wavelength is None:
        p0[:, 3] = np.random.uniform(.45, 0.55, size=nwalkers)                     # radius
        p0[:, 4] = np.random.uniform(.40, 0.45, size=nwalkers)                     # focus
        p0[:, 5] = np.random.uniform(.35, 0.45, size=nwalkers)                     # width_x
        p0[:, 6] = np.random.uniform(.35, 0.45, size=nwalkers)                     # width_y
    else:
        tmp = _expectedValues()[wavelength]
        if blurred:
            print 'Using initial guess [radius, focus, width_x, width_y]:', [tmp[0], 1.2, tmp[2], tmp[3]]
            p0[:, 3] = np.random.normal(tmp[0], 0.01, size=nwalkers)                   # radius
            p0[:, 4] = np.random.normal(1.2, 0.01, size=nwalkers)                       # focus
            p0[:, 5] = np.random.normal(tmp[2], 0.01, size=nwalkers)                   # width_x
            p0[:, 6] = np.random.normal(tmp[3], 0.01, size=nwalkers)                   # width_y
        else:
            print 'Using initial guess [radius, focus, width_x, width_y]:', tmp
            p0[:, 3] = np.random.normal(tmp[0], 0.01, size=nwalkers)                   # radius
            p0[:, 4] = np.random.normal(tmp[1], 0.01, size=nwalkers)                   # focus
            p0[:, 5] = np.random.normal(tmp[2], 0.01, size=nwalkers)                   # width_x
            p0[:, 6] = np.random.normal(tmp[3], 0.01, size=nwalkers)                   # width_y

    #initiate sampler
    pool = Pool(cores) #A hack Dan gave me to not have ghost processes running as with threads keyword
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, var, peakrange, spot.shape],
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=[xx, yy, data, rn**2, peakrange, spot.shape, blurred],
                                    pool=pool)

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    print best_pos
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    sampler.reset()

    print "Running an improved estimate..."
    pos, prob, state = sampler.run_mcmc(pos, burn)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    sampler.reset()
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
    amplitudeE, center_xE, center_yE, radiusE, focusE, width_xE, width_yE = errors_fit
    _printResults(params_fit, errors_fit)

    #Best fit model
    peak, center_x, center_y, radius, focus, width_x, width_y = params_fit
    amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius, x_0=CCDx, y_0=CCDy)
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
    maxdiff = np.max(np.abs(model - spot))
    print 'GoF:', gof, ' Maximum difference:', maxdiff
    if maxdiff > 2e3 or gof > 4.:
        print '\nFIT UNLIKELY TO BE GOOD...\n'
    print 'Amplitude estimate:', amplitude

    #results and save results
    _printFWHM(width_x, width_y, errors_fit[5], errors_fit[6])
    res = dict(wx=width_x, wy=width_y, wxerr=width_xE, wyerr=width_yE, out=out,
               peakvalue=max, CCDmodel=CCD, CCDmodeldata=CCDdata, GoF=gof,
               maximumdiff=maxdiff, fit=params_fit)
    fileIO.cPickleDumpDictionary(res, out+'.pkl')

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


def forwardModelJointFit(files, out, wavelength, gain=3.1, size=10, burn=500, run=800,
                         spotx=2888, spoty=3514, simulated=False, truths=None):
    """
    Forward models the spot data found from the input files. Models all data simultaneously so that the Airy
    disc centroid and shift from file to file. Assumes that the spot intensity, focus, and the CCD PSF kernel
    are the same for each file. Can be used with simulated and real data.

    Note, however, that because the amplitude of the Airy disc is kept fixed, but the centroids can shift the
    former most likely forces the latter to be very similar. This is fine if we assume that the spots did not
    move during the time data were accumulated. So, this approach is unlikely to work if the projected spot
    drifted during the data acquisition.
    """
    print '\n\n\n'
    print '_'*120

    images = len(files)
    orig = []
    image = []
    noise = []
    rns = []
    peakvalues = []
    xestimate = []
    yestimate = []
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

        #set highly negative values to zero
        spot[spot + rn**2 < 0.] = 0.

        print 'Least Squares Fitting...'
        gaus = models.Gaussian2D(spot.max(), size, size, x_stddev=0.5, y_stddev=0.5)
        gaus.theta.fixed = True  #fix angle
        p_init = gaus
        fit_p = fitting.LevMarLSQFitter()
        stopy, stopx = spot.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
        p = fit_p(p_init, X, Y, spot)
        print p

        max = np.max(spot)
        s = spot.sum()
        print 'Maximum Value:', max
        print 'Sum:', s
        print ''

        peakvalues.append(max)

        #noise model
        variance = spot.copy() + rn**2

        #save to a list
        image.append(spot)
        noise.append(variance)
        xestimate.append(p.x_mean.value)
        yestimate.append(p.y_mean.value)
        rns.append(rn**2)

    #sensibility test, try to check if all the files in the fit are of the same dataset
    if np.std(peakvalues) > 5*np.sqrt(np.median(peakvalues)):
        #check for more than 5sigma outliers, however, this is very sensitive to the centroiding of the spot...
        print '\n\n\nPOTENTIAL OUTLIER, please check the input files...'
        print np.std(peakvalues), 5*np.sqrt(np.median(peakvalues))

    peakvalues = np.asarray(peakvalues)
    peak = np.median(peakvalues)
    peakrange = (0.95*np.min(peakvalues), 1.7*np.max(peakvalues))

    print '\nPeak Estimate:', peak
    print 'Peak Range:', peakrange

    #MCMC based fitting
    ndim = 2*images + 5  #xpos, ypos for each image and single amplitude, radius, focus, and sigmaX and sigmaY
    nwalkers = 1000
    print '\n\nBayesian Fitting, model has %i dimensions' % ndim

    # Choose an initial set of positions for the walkers using the Gaussian fit
    tmp = _expectedValues()['l' + wavelength.replace('nm', '')]
    print 'Using initial guess [radius, focus, width_x, width_y]:', tmp
    p0 = np.zeros((nwalkers, ndim))
    for x in xrange(images):
        p0[:, 2*x] = np.random.normal(xestimate[x], 0.1, size=nwalkers)         # x
        p0[:, 2*x+1] = np.random.normal(yestimate[x], 0.1, size=nwalkers)       # y
    p0[:, -5] = np.random.normal(peak, peak/100., size=nwalkers)                # amplitude
    p0[:, -4] = np.random.normal(tmp[0], 0.01, size=nwalkers)                   # radius
    p0[:, -3] = np.random.normal(tmp[1], 0.01, size=nwalkers)                   # focus
    p0[:, -2] = np.random.normal(tmp[2], 0.01, size=nwalkers)                   # width_x
    p0[:, -1] = np.random.normal(tmp[3], 0.01, size=nwalkers)                   # width_y

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
    pool = Pool(cores) #A hack Dan gave me to not have ghost processes running as with threads keyword
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorJoint,
                                    args=[xx, yy, image, rns, peakrange, spot.shape], pool=pool)
                                   # args=[xx, yy, image, noise, peakrange, spot.shape], pool=pool)

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    print best_pos
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    #run another burn-in
    print "Running an improved estimate..."
    pos, prob, state = sampler.run_mcmc(pos, burn)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    sampler.reset()

    # Starting from the final position in the improved chain
    print "Running final MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
    print params_fit

    #unpack the fixed parameters
    peak, radius, focus, width_x, width_y = params_fit[-5:]
    peakE, radiusE, focusE, width_xE, width_yE = errors_fit[-5:]

    #print results
    _printFWHM(width_x, width_y, width_xE, width_yE)

    #save the best models per file
    size = size*2 + 1
    gofs = []
    mdiff = []
    for index, file in enumerate(files):
        #path, file = os.path.split(file)
        id = 'results/' + out + str(index)
        #X and Y are always in pairs
        center_x = params_fit[2*index]
        center_y = params_fit[2*index+1]

        #1)Generate a model Airy disc
        amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius,
                                       x_0=int(size/2.-0.5), y_0=int(size/2.-0.5))
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
        maxdiff = np.max(np.abs(model - image[index]))
        print 'GoF:', gof, ' Max difference', maxdiff
        gofs.append(gof)
        mdiff.append(maxdiff)
        print 'Amplitude Estimate:', amplitude

    if np.asarray(mdiff).max() > 3e3 or np.asarray(gofs).max() > 4.:
        print '\nFIT UNLIKELY TO BE GOOD...\n'

    #save results
    res = dict(wx=width_x, wy=width_y, wxerr=width_xE, wyerr=width_yE, files=files, out=out,
               wavelength=wavelength, peakvalues=np.asarray(peakvalues), CCDmodel=CCD, CCDmodeldata=CCDdata,
               GoFs=gofs, fit=params_fit, maxdiff=mdiff)
    fileIO.cPickleDumpDictionary(res, 'results/' + out + '.pkl')

    #plot
    samples = sampler.chain.reshape((-1, ndim))
    #extents = None
    #if simulated:
    #    extents = [(0.9*truth, 1.1*truth) for truth in truths]
    #    print extents
    if simulated:
        tr = truths[:-5]
        peaks = []
        for x in xrange(images):
            xcen = tr[2*x]
            ycen = tr[2*x+1]
            theta = [truths[-5], xcen, ycen, truths[-4], truths[-3], truths[-2], truths[-1]]
            peaks.append(_peakFromTruth(theta))
        print peaks
        truths[-5] = np.median(np.asarray(peaks))
    fig = triangle.corner(samples, labels=['x', 'y']*images + ['peak', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)#, extents=extents)
    fig.savefig('results/' + out + 'Triangle.png')
    plt.close()
    pool.close()


def log_posterior(theta, x, y, z, var, peakrange, size, blurred):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_prior(theta, peakrange, blurred)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, z, var, size)


def log_posteriorJoint(theta, x, y, z, var, peakrange, size):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_priorJoint(theta, peakrange)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihoodJoint(theta, x, y, z, var, size)


def log_prior(theta, peakrange, blurred):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    peak, center_x, center_y, radius, focus, width_x, width_y = theta
    if blurred:
        if 14. < center_x < 28. and 14. < center_y < 28. and 0.1 < width_x < 0.6 and 0.1 < width_y < 0.6 and \
           peakrange[0] < peak < peakrange[1] and 0.4 < radius < 3.5 and 0.1 < focus < 3.5:
            return 0.
        else:
            return -np.inf
    else:
        if 7. < center_x < 14. and 7. < center_y < 14. and 0.1 < width_x < 0.6 and 0.1 < width_y < 0.6 and \
           peakrange[0] < peak < peakrange[1] and 0.4 < radius < 0.7 and 0.1 < focus < 0.7:
            return 0.
        else:
            return -np.inf


def log_priorJoint(theta, peakrange):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    if all(7. < x < 14. for x in theta[:-5]) and peakrange[0] < theta[-5] < peakrange[1] and 0.4 < theta[-4] < 0.7 and \
       0.1 < theta[-3] < 0.7 and 0.1 < theta[-2] < 0.6 and 0.1 < theta[-1] < 0.6:
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
    CCD = models.Gaussian2D(1., size[0]/2.-0.5, size[1]/2.-0.5, width_x, width_y, 0.)
    CCDdata = CCD.eval(x, y, 1., size[0]/2.-0.5, size[1]/2.-0.5, width_x, width_y, 0.).reshape(size)
    model = signal.convolve2d(model, CCDdata, mode='same').flatten()

    #true for Gaussian errors
    #lnL = - 0.5 * np.sum((data - model)**2 / var)
    #Gary B. said that this should be from the model not data so recompute var (now contains rn**2)
    var += model.copy()
    lnL = - 0.5 * np.sum((data - model)**2 / var)

    return lnL


def log_likelihoodJoint(theta, x, y, data, var, size):
    """
    Logarithm of the likelihood function for joint fitting. Not really sure if this is right...
    """
    #unpack the parameters
    #[xpos, ypos]*images) +[amplitude, radius, focus])
    images = len(theta[:-5]) / 2
    peak, radius, focus, width_x, width_y = theta[-5:]

    lnL = 0.
    for tmp in xrange(images):
        #X and Y are always in pairs
        center_x = theta[2*tmp]
        center_y = theta[2*tmp+1]

        #1)Generate a model Airy disc
        amplitude = _amplitudeFromPeak(peak, center_x, center_y, radius,
                                       x_0=int(size[0]/2.-0.5), y_0=int(size[1]/2.-0.5))
        airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
        adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape(size)

        #2)Apply Focus, no normalisation as smoothing
        f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
        focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape(size)
        model = signal.convolve2d(adata, focusdata, mode='same')

        #3)Apply CCD diffusion, approximated with a Gaussian -- max = 1 as centred
        CCD = models.Gaussian2D(1., size[0]/2.-0.5, size[1]/2.-0.5, width_x, width_y, 0.)
        CCDdata = CCD.eval(x, y, 1., size[0]/2.-0.5, size[1]/2.-0.5, width_x, width_y, 0.).reshape(size)
        model = signal.convolve2d(model, CCDdata, mode='same').flatten()

        #lnL += - 0.5 * np.sum((data[tmp].flatten() - model)**2 / var[tmp].flatten())
        #Gary B. said that this should be from the model not data so recompute var (now contains rn**2)
        var = var[tmp] + model.copy()
        lnL += - 0.5 * np.sum((data[tmp].flatten() - model)**2 / var)

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
        d = CCD.eval(x, y, 1.,xCCD, yCCD, width_x, width_y, 0.).reshape((size, size)) #max = 1 as centred
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


def RunTestSimulations(newfiles=True):
    """
    A set of simulated spots and analysis.

    Note that this is not the best test case for the joint fitting, because the amplitudes are fixed.
    This means that the peak pixels can have quite different values. With the ones selected here,
    the fourth is actually somewhat different from the others. It is actually quite nice that we
    can still recover the CCD PSF.
    """
    print("|" * 120)
    print 'SIMULATED DATA'
    #a joint fit test - vary only the x and y positions
    #It is misleading to keep the amplitude fixed as it is the counts in the peak pixel that matters.
    #If the Airy were centred perfectly then we could keep the amplitude fixed. In this case the individual
    #fits will work and can recover the 200k amplitude, but it is more problematic for the joint fit.
    theta1 = (2.e5, 9.9, 10.03, 0.47, 0.41, 10., 10., 0.291, 0.335)
    theta2 = (2.e5, 10.1, 9.97, 0.47, 0.41, 10., 10., 0.291, 0.335)
    theta3 = (2.e5, 9.97, 10.1, 0.47, 0.41, 10., 10., 0.291, 0.335)
    theta4 = (2.e5, 10.02, 9.9, 0.47, 0.41, 10., 10., 0.291, 0.335)
    theta5 = (2.e5, 10.1, 10., 0.47, 0.41, 10., 10., 0.291, 0.335)

    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        if newfiles:
            print 'Generating a new file with the following parameters:'
            _simulate(theta=theta, out='simulated/simulatedJoint%i.fits' %i)

            print 'amplitude, x, y, radius, focus, width_x, width_y'
            print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
            print("=" * 60)

        forwardModel(file='simulated/simulatedJoint%i.fits' %i, out='simulatedResults/RunI%i' %i, simulation=True,
                     truths=[theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]])

    #plot residuals
    _plotModelResiduals(id='RunI0', folder='simulatedResults/', out='Residual0.pdf', individual=True)
    _plotModelResiduals(id='RunI1', folder='simulatedResults/', out='Residual1.pdf', individual=True)
    _plotModelResiduals(id='RunI2', folder='simulatedResults/', out='Residual2.pdf', individual=True)
    _plotModelResiduals(id='RunI3', folder='simulatedResults/', out='Residual3.pdf', individual=True)
    _plotModelResiduals(id='RunI4', folder='simulatedResults/', out='Residual4.pdf', individual=True)

    #joint fit
    truths = [theta1[1], theta1[2], theta2[1], theta2[2], theta3[1], theta3[2], theta4[1], theta4[2],
              theta5[1], theta5[2], theta1[0], theta4[3], theta1[4], theta1[7], theta1[8]]
    forwardModelJointFit(g.glob('simulated/simulatedJoint?.fits'),
                         out='simulated800nmJoint', wavelength='800nm', simulated=True,
                         truths=truths)

    print 'True width_x and widht_y:', theta1[7], theta1[8]

    #plot residuals
    _plotModelResiduals(id='simulated800nmJoint0', folder='results/', out='ResidualJ0.pdf')
    _plotModelResiduals(id='simulated800nmJoint1', folder='results/', out='ResidualJ1.pdf')
    _plotModelResiduals(id='simulated800nmJoint2', folder='results/', out='ResidualJ2.pdf')
    _plotModelResiduals(id='simulated800nmJoint3', folder='results/', out='ResidualJ3.pdf')
    _plotModelResiduals(id='simulated800nmJoint4', folder='results/', out='ResidualJ4.pdf')

    #test plots
    _plotDifferenceIndividualVsJoined(individuals='simulatedResults/RunI*.pkl',
                                      joined='results/simulated800nmJoint.pkl',
                                      title='Simulated Data', truthx=theta1[7], truthy=theta1[8],
                                      requirementE=None, requirementFWHM=None, requirementR2=None)


def RunTestSimulations2(newfiles=True):
    #different simulation sets
    print("|" * 120)
    print 'Single Fitting Simulations'
    theta1 = (1.e6, 9.65, 10.3, 0.6, 0.45, 10., 10., 0.28, 0.33)    # ok recovery, sigmax has long tail towards 0.
    theta2 = (5.e5, 10.3, 10.2, 0.55, 0.45, 10., 10., 0.38, 0.36)   # well recovered
    theta3 = (8.e4, 10.0, 10.1, 0.4, 0.55, 10., 10., 0.25, 0.35)    # ok, recovery, but sigmax has long tail towards 0.
    theta4 = (5.e5, 10.1, 10.3, 0.42, 0.48, 10., 10., 0.30, 0.28)   # sigmax and sigmay not perfectly recovered
    theta5 = (2.e5, 9.95, 10.3, 0.45, 0.5, 10., 10., 0.33, 0.35)    # good recovery
    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        if newfiles:
            _simulate(theta=theta, out='simulated/simulatedSmall%i.fits' %i)
        forwardModel(file='simulated/simulatedSmall%i.fits' %i, out='simulatedResults/Run%i' %i, simulation=True,
                     truths=[theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]])
        print("=" * 60)
        print 'Simulation Parameters'
        print 'amplitude, center_x, center_y, radius, focus, width_x, width_y'
        print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
        print("=" * 60)


def RunData(files, wavelength=None, out='testdata'):
    """
    A set of test data to analyse.
    """
    for i, file in enumerate(files):
        forwardModel(file=file, out='results/%s%i' % (out, i), wavelength=wavelength)


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


def AllindividualRuns():
    """
    Execute all spot data analysis runs individually.
    """
    #800 nm
    RunData(getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/'), out='I800nm')
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


def analyseData800nm():
    """
    Execute spot data analysis.
    """
    #800 nm
    RunData(getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/'), out='I800nm') #0.31, 0.3
    forwardModelJointFit(getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/'),
                         out='J800nm', wavelength='800nm') #0.31, 0.3


def analyseData700nm():
    #700 nm
    forwardModelJointFit(getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/'),
                     out='J700nm52k', wavelength='700nm') #around 0.2, 0.307 (x not good)
    forwardModelJointFit(getFiles(mintime=(17, 58, 18), maxtime=(17, 59, 31), folder='data/30Jul/'),
                         out='J700nm32k', wavelength='700nm') #


def analyseData600nm():
    #600 nm
    forwardModelJointFit(getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/'),
                     out='J600nm54k', wavelength='600nm') #around 0.299, 0.333


def analyseData890nm():
    #890 nm
    forwardModelJointFit(getFiles(mintime=(14, 17, 57), maxtime=(14, 25, 49), folder='data/01Aug/'),
                         out='J890nm30k', wavelength='890nm') #0.33, 0.35, these are surprising...
    forwardModelJointFit(getFiles(mintime=(14, 30, 03), maxtime=(14, 34, 37), folder='data/01Aug/'),
                     out='J890nm50k', wavelength='890nm') #around 0.28, 0.29, these are more realistic


def analyseData800nmBrighterFatter():
    #For Brighter-Fatter
    forwardModelJointFit(getFiles(mintime=(15, 12, 20), maxtime=(15, 24, 16), folder='data/31Jul/'),
                         out='J800nm5k', wavelength='800nm') #0.21, 0.35, not probably reliable
    forwardModelJointFit(getFiles(mintime=(15, 28, 40), maxtime=(15, 39, 21), folder='data/31Jul/'),
                         out='J800nm10k', wavelength='800nm') #0.2, 0.325, not probably reliable
    forwardModelJointFit(getFiles(mintime=(15, 43, 24), maxtime=(15, 51, 47), folder='data/31Jul/'),
                         out='J800nm20k', wavelength='800nm') #
    forwardModelJointFit(getFiles(mintime=(15, 56, 11), maxtime=(16, 02, 58), folder='data/31Jul/'),
                         out='J800nm30k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 12, 39), maxtime=(16, 18, 25), folder='data/31Jul/'),
                         out='J800nm38k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 21, 52), maxtime=(16, 26, 16), folder='data/31Jul/'),
                         out='J800nm50k', wavelength='800nm')
    forwardModelJointFit(getFiles(mintime=(16, 32, 02), maxtime=(16, 35, 23), folder='data/31Jul/'),
                         out='J800nm54k', wavelength='800nm')

def AlljointRuns():
    """
    Execute all spot data analysis runs fitting jointly.
    """
    #800 nm
    forwardModelJointFit(getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/'),
                         out='J800nm', wavelength='800nm') #0.31, 0.3
    forwardModelJointFit(getFiles(mintime=(15, 12, 20), maxtime=(15, 24, 16), folder='data/31Jul/'),
                         out='J800nm5k', wavelength='800nm') #0.28 0.31
    forwardModelJointFit(getFiles(mintime=(15, 28, 40), maxtime=(15, 39, 21), folder='data/31Jul/'),
                         out='J800nm10k', wavelength='800nm') #0.27 0.29
    forwardModelJointFit(getFiles(mintime=(15, 43, 24), maxtime=(15, 51, 47), folder='data/31Jul/'),
                         out='J800nm20k', wavelength='800nm') #0.27 0.28
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
                         out='J700nm5k', wavelength='700nm') # 0.28 0.32
    forwardModelJointFit(getFiles(mintime=(17, 37, 35), maxtime=(17, 46, 51), folder='data/30Jul/'),
                         out='J700nm9k', wavelength='700nm') # 0.27 0.32
    forwardModelJointFit(getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/'),
                         out='J700nm52k', wavelength='700nm') # 0.26 0.31
    forwardModelJointFit(getFiles(mintime=(17, 58, 18), maxtime=(17, 59, 31), folder='data/30Jul/'),
                         out='J700nm32k', wavelength='700nm')
    #600 nm
    forwardModelJointFit(getFiles(mintime=(15, 22, 00), maxtime=(15, 36, 32), folder='data/30Jul/'),
                         out='J600nm5k', wavelength='600nm') #0.27 0.31
    forwardModelJointFit(getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/'),
                         out='J600nm54k', wavelength='600nm') #0.299, 0.333
    forwardModelJointFit(getFiles(mintime=(15, 52, 07), maxtime=(16, 06, 32), folder='data/30Jul/'),
                         out='J600nm10k', wavelength='600nm') #0.28 0.32
    #890 nm
    forwardModelJointFit(getFiles(mintime=(13, 37, 37), maxtime=(13, 50, 58), folder='data/01Aug/'),
                         out='J890nm5k', wavelength='890nm') #0.28 0.35
    forwardModelJointFit(getFiles(mintime=(14, 00, 58), maxtime=(14, 11, 54), folder='data/01Aug/'),
                         out='J890nm10k', wavelength='890nm') #0.28 0.33
    forwardModelJointFit(getFiles(mintime=(14, 17, 57), maxtime=(14, 25, 49), folder='data/01Aug/'),
                         out='J890nm30k', wavelength='890nm') #0.3 0.33
    forwardModelJointFit(getFiles(mintime=(14, 30, 03), maxtime=(14, 34, 37), folder='data/01Aug/'),
                         out='J890nm50k', wavelength='890nm') #0.3 0.3


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


def RunTest():
    """
    Test runs with test data sets
    """
    #800nm
    RunData(g.glob('testdata/15*.fits'), out='test800nm')
    forwardModelJointFit(g.glob('testdata/15*.fits'), out='test800nmJoint', wavelength='800nm')
    _plotDifferenceIndividualVsJoined(individuals='results/test800nm?.pkl', joined='results/test800nmJoint.pkl',
                                      title='800nm')
    #700nm
    RunData(g.glob('testdata/17*.fits'), out='test700nm')
    forwardModelJointFit(g.glob('testdata/17*.fits'), out='test700nmJoint', wavelength='700nm')
    _plotDifferenceIndividualVsJoined(individuals='results/test700nm?.pkl', joined='results/test700nmJoint.pkl',
                                      title='700nm')


def generateTestPlots(folder='results/'):
    #800nm
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm?.pkl', joined=folder+'J800nm.pkl', title='800nm')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm5k?.pkl', joined=folder+'J800nm5k.pkl', title='800nm5k')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm10k?.pkl', joined=folder+'J800nm10k.pkl', title='800nm10k')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm30k?.pkl', joined=folder+'J800nm30k.pkl', title='800nm30k')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm38k?.pkl', joined=folder+'J800nm38k.pkl', title='800nm38k')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm50k?.pkl', joined=folder+'J800nm50k.pkl', title='800nm50k')
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm54k?.pkl', joined=folder+'J800nm54k.pkl', title='800nm54k')
    #700nm
    _plotDifferenceIndividualVsJoined(individuals=folder+'I700nm5k?.pkl', joined=folder+'J700nm5k.pkl', title='700nm5k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I700nm9k?.pkl', joined=folder+'J700nm9k.pkl', title='700nm9k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I700nm32k?.pkl', joined=folder+'J700nm32k.pkl', title='700nm32k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I700nm52k?.pkl', joined=folder+'J700nm52k.pkl', title='700nm52k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    #600nm
    _plotDifferenceIndividualVsJoined(individuals=folder+'I600nm5k?.pkl', joined=folder+'J600nm5k.pkl', title='600nm5k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I600nm10k?.pkl', joined=folder+'J600nm10k.pkl', title='600nm10k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I600nm54k?.pkl', joined=folder+'J600nm54k.pkl', title='600nm54k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    #890nm
    _plotDifferenceIndividualVsJoined(individuals=folder+'I890nm5k?.pkl', joined=folder+'J890nm5k.pkl', title='890nm5k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I890nm10k?.pkl', joined=folder+'J890nm10k.pkl', title='890nm10k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I890nm30k?.pkl', joined=folder+'J890nm30k.pkl', title='890nm30k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)
    _plotDifferenceIndividualVsJoined(individuals=folder+'I890nm50k?.pkl', joined=folder+'J890nm50k.pkl', title='890nm50k',
                                      requirementE=None, requirementR2=None, requirementFWHM=None)


def plotBrighterFatter(out='BrighterFatter.pdf', requirementFWHM=10.8, sigma=3):
    """
    Plot the CCD PSF size intensity relation.
    """
    matplotlib.rc('text', usetex=True)

    #600nm
    low = [fileIO.cPicleRead(file) for file in g.glob('results/I600nm5k*.pkl')]
    med = [fileIO.cPicleRead(file) for file in g.glob('results/I600nm10k*.pkl')]
    high = [fileIO.cPicleRead(file) for file in g.glob('results/I600nm54k*.pkl')]
    data = (low, med, high)
    fluxes600 = []
    fluxes600err = []
    wx600 = []
    wxerrs = []
    wy600 = []
    wyerrs = []
    for x in data:
        wx = np.median([d['wx'] for d in x])
        wxerr = np.median([d['wxerr'] for d in x])
        wy = np.median([d['wy'] for d in x])
        wyerr = np.median([d['wyerr'] for d in x])
        wx600.append(wx)
        wxerrs.append(wxerr)
        wy600.append(wy)
        wyerrs.append(wyerr)
        fluxes600.append(np.median([d['peakvalue'] for d in x]))
        fluxes600err.append(np.std([d['peakvalue'] for d in x]))

    #wx600 = _FWHMGauss(np.asarray(wx600))
    #wy600 = _FWHMGauss(np.asarray(wy600))
    fluxes600 = np.asarray(fluxes600)
    fluxes600err = np.asarray(fluxes600err)
    #averaged over many runs
    wx600 = _FWHMGauss(np.asarray([0.26, 0.28, 0.316]))
    wx600err = _FWHMGauss(np.asarray(wxerr))
    wy600 = _FWHMGauss(np.asarray([0.30, 0.31, 0.3315560]))
    wy600err = _FWHMGauss(np.asarray(wyerr))
    w600 = np.sqrt(wx600*wy600)
    w600err = np.sqrt(wx600err*wy600err)
    print wx600
    print wy600
    print fluxes600

    #800nm
    nom = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm?.pkl')]
    nom5k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm5k*.pkl')]
    nom10k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm10k*.pkl')]
    nom20k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm20k*.pkl')]
    nom30k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm30k*.pkl')]
    nom38k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm38k*.pkl')]
    nom50k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm50k*.pkl')]
    nom54k = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm54k*.pkl')]

    data = (nom5k, nom10k, nom20k, nom30k, nom38k, nom50k, nom54k, nom)
    fluxes800 = []
    fluxes800err = []
    wx800 = []
    wxerrs = []
    wy800 = []
    wyerrs = []
    for x in data:
        wx = np.max([d['wx'] for d in x])
        wxerr = np.max([d['wxerr'] for d in x])
        wy = np.max([d['wy'] for d in x])
        wyerr = np.max([d['wyerr'] for d in x])
        wx800.append(wx)
        wxerrs.append(wxerr)
        wy800.append(wy)
        wyerrs.append(wyerr)
        fluxes800.append(np.median([d['peakvalue'] for d in x]))
        fluxes800err.append(np.std([d['peakvalue'] for d in x]))

    #wx800 = _FWHMGauss(np.asarray(wx800))
    #wy800 = _FWHMGauss(np.asarray(wy800))
    fluxes800 = np.asarray(fluxes800)
    fluxes800err = np.asarray(fluxes800err)
    #averaged over many runs
    wx800pix = np.asarray([0.241, 0.248, 0.245, 0.26, 0.28, 0.29, 0.304, 0.30848382])
    wx800 = _FWHMGauss(wx800pix)
    wx800err = _FWHMGauss(np.asarray(wxerr))
    wy800pix = np.asarray([0.2401, 0.251, 0.246, 0.27, 0.285, 0.294, 0.298, 0.2972725])
    wy800 = _FWHMGauss(wy800pix)
    wy800err = _FWHMGauss(np.asarray(wyerr))
    w800 = np.sqrt(wx800*wy800)
    w800err = np.sqrt(wx800err*wy800err)
    print wx800
    print wy800
    print fluxes800

    #plot FWHM
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.93)
    ax1.set_title('CCD273 PSF Intensity Dependency')

    ax1.errorbar(fluxes600, wx600, yerr=sigma*wx600err, xerr=sigma*fluxes600err, fmt='o', label='600nm', c='g')
    ax2.errorbar(fluxes600, wy600, yerr=sigma*wy600err, xerr=sigma*fluxes600err, fmt='o', label='600nm', c='g')
    ax3.errorbar(fluxes600, w600, yerr=sigma*w600err, xerr=sigma*fluxes600err, fmt='o', label='600nm', c='g')

    ax1.errorbar(fluxes800, wx800, yerr=sigma*wx800err, xerr=sigma*fluxes800err, fmt='s', label='800nm', c='m')
    ax2.errorbar(fluxes800, wy800, yerr=sigma*wy800err, xerr=sigma*fluxes800err, fmt='s', label='800nm', c='m')
    ax3.errorbar(fluxes800, w800, yerr=sigma*w800err, xerr=sigma*fluxes800err, fmt='s', label='800nm', c='m')

    #linear fits
    z1 = np.polyfit(fluxes600, wx600, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(fluxes600, wy600, 1)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(fluxes600, w600, 1)
    p3 = np.poly1d(z3)
    x = np.linspace(0., fluxes600.max()*1.05, 100)
    #ax1.plot(x, p1(x), 'g-')
    #ax2.plot(x, p2(x), 'g-')
    #ax3.plot(x, p3(x), 'g-')
    print '600nm fits:'
    print p1
    print p2
    print p3

    # Bayesian
    p600x, params600x, errors600x, outliers600x = linearFitWithOutliers(fluxes600, wx600, wx600err,
                                                                        outtriangle='BF600FWHMx.png')
    p600y, params600y, errors600y, outliers600y = linearFitWithOutliers(fluxes600, wy600, wy600err,
                                                                        outtriangle='BF600FWHMy.png')
    p600, params600, errors600, outliers600 = linearFitWithOutliers(fluxes600, w600, w600err,
                                                                    outtriangle='BF600FWHM.png')
    print params600x[::-1], errors600x[::-1]
    print params600y[::-1], errors600y[::-1]
    print params600[::-1], errors600[::-1]

    ax1.plot(x, params600x[0] + params600x[1]*x, 'g-')
    #ax1.plot(fluxes600[outliers600x], wx600[outliers600x], 'ro', ms=20, mfc='none', mec='red')
    ax2.plot(x, params600y[0] + params600y[1]*x, 'g-')
    #ax2.plot(fluxes600[outliers600y], wy600[outliers600y], 'ro', ms=20, mfc='none', mec='red')
    ax3.plot(x, params600[0] + params600[1]*x, 'g-')
    #ax3.plot(fluxes600[outliers600], w600[outliers600], 'ro', ms=20, mfc='none', mec='red')

    z1 = np.polyfit(fluxes800, wx800, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(fluxes800, wy800, 1)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(fluxes800, w800, 1)
    p3 = np.poly1d(z3)
    x = np.linspace(0., fluxes800.max()*1.05, 100)
    #ax1.plot(x, p1(x), 'm-')
    #ax2.plot(x, p2(x), 'm-')
    #ax3.plot(x, p3(x), 'm-')
    print '800nm fits:'
    print p1
    print p2
    print p3

    # Bayesian
    p800x, params800x, errors800x, outliers800x = linearFitWithOutliers(fluxes800, wx800, wx800err,
                                                                        outtriangle='BF800FWHMx.png')
    p800y, params800y, errors800y, outliers800y = linearFitWithOutliers(fluxes800, wy800, wy800err,
                                                                        outtriangle='BF800FWHMy.png')
    p800, params800, errors800, outliers800 = linearFitWithOutliers(fluxes800, w800, w800err,
                                                                    outtriangle='BF800FWHM.png')
    print params800x[::-1], errors800x[::-1]
    print params800y[::-1], errors800y[::-1]
    print params800[::-1], errors800[::-1]

    ax1.plot(x, params800x[0] + params800x[1]*x, 'm-')
    #ax1.plot(fluxes800[outliers800x], wx800[outliers800x], 'ro', ms=20, mfc='none', mec='red')
    ax2.plot(x, params800y[0] + params800y[1]*x, 'm-')
    #ax2.plot(fluxes800[outliers800y], wy800[outliers800y], 'ro', ms=20, mfc='none', mec='red')
    ax3.plot(x, params800[0] + params800[1]*x, 'm-')
    #ax3.plot(fluxes800[outliers800], w800[outliers800], 'ro', ms=20, mfc='none', mec='red')

    #requirements
    #ax1.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='--')
    #ax2.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='--')
    #ax3.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r', ls='-')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(visible=False)
    plt.sca(ax3)

    ax1.set_ylim(4.3, 10.8)
    ax2.set_ylim(4.3, 10.8)
    ax3.set_ylim(4.3, 10.8)
    ax1.set_xlim(0., fluxes800.max()*1.05)
    ax2.set_xlim(0., fluxes800.max()*1.05)
    ax3.set_xlim(0., fluxes800.max()*1.05)

    ax3.set_ylabel(r'FWHM $\, [\mu$m$]$')
    ax1.set_ylabel(r'FWHM$_{X} \, [\mu$m$]$')
    ax2.set_ylabel(r'FWHM$_{Y} \, [\mu$m$]$')
    ax3.set_xlabel(r'Intensity $\quad [e^{-}]$')
    ax1.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')
    ax2.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')
    ax3.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')
    plt.savefig(out)
    plt.close()

    print 'R2:'
    R2 = _R2FromGaussian(wx800pix, wy800pix)*1e3
    errR2 = _R2err(wx800pix, wy800pix, wxerr, wyerr)*1e3
    print R2
    p, params, errors, outliers = linearFitWithOutliers(fluxes800, R2, errR2, outtriangle='BF800R2.png')
    print params[::-1], errors[::-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.93)
    ax1.set_title('CCD273 PSF Intensity Dependency')
    ax1.errorbar(fluxes800, R2, yerr=sigma*errR2, fmt='o', label='800nm', c='b')
    ax1.plot(x, params[0] + params[1]*x, 'm-')
    ax1.plot(fluxes800[outliers], R2[outliers], 'ro', ms=20, mfc='none', mec='red')
    ax1.set_ylabel(r'$R^{2} \, [$mas$^{2}]$')

    print 'Ellipticity'
    ell = _ellipticityFromGaussian(wx800pix, wy800pix)
    print ell
    ellerr = _ellipticityerr(wx800pix, wy800pix, wxerr, wyerr)
    p, params, errors, outliers = linearFitWithOutliers(fluxes800, ell, ellerr, outtriangle='BF800ell.png')
    print params[::-1], errors[::-1]

    ax2.errorbar(fluxes800, ell, yerr=sigma*ellerr, fmt='s', label='800nm', c='m')
    ax2.plot(x, params[0] + params[1]*x, 'm-')
    ax2.plot(fluxes800[outliers], ell[outliers], 'ro', ms=20, mfc='none', mec='red')
    ax2.axhline(y=0.156, label='Requirement', c='r')
    ax1.axhline(y=2., label='Requirement', c='r')
    ax1.set_xlim(0., fluxes800.max()*1.05)
    ax2.set_xlim(0., fluxes800.max()*1.05)
    ax1.set_ylim(0.8, 2.1)
    ax2.set_ylim(-0.01, 0.165)
    ax2.set_ylabel('Ellipticity')
    ax2.set_xlabel(r'Intensity $\quad [e^{-}]$')
    ax1.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')
    ax2.legend(shadow=True, fancybox=True, numpoints=1)

    plt.savefig('R2ellIntensity.pdf')
    plt.close()


def linearFitWithOutliers(x, y, e, outtriangle='linear.png'):
    """
    Linear fitting with outliers
    """
    # theta will be an array of length 2 + N, where N is the number of points
    # theta[0] is the intercept, theta[1] is the slope,
    # and theta[2 + i] is the weight g_i
    def log_prior(theta):
        #g_i needs to be between 0 and 1
        if (all(x > 0. for x in theta[2:]) and all(x < 1. for x in theta[2:])) and \
            0. < theta[0] < 10. and 0. < theta[1] < 0.1:
            return 0
        else:
            return -np.inf  # recall log(0) = -inf

    def log_likelihood(theta, x, y, e, sigma_B):
        dy = y - theta[0] - theta[1] * x
        g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
        logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
        logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy / sigma_B) ** 2
        return np.sum(np.logaddexp(logL1, logL2))

    def log_posterior(theta, x, y, e, sigma_B):
        return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)


    #find starting point
    def squared_loss(theta, x=x, y=y, e=e):
        dy = y - theta[0] - theta[1] * x
        return np.sum(0.5 * (dy / e) ** 2)
    theta1 = optimize.fmin(squared_loss, [0, 0], disp=False)

    ndim = 2 + len(x)   # number of parameters in the model
    nwalkers = 200      # number of MCMC walkers
    nburn = 5000        # "burn-in" period to let chains stabilize
    nsteps = 50000      # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    starting_guesses = np.zeros((nwalkers, ndim))
    starting_guesses[:, :2] = np.random.normal(theta1, 1, (nwalkers, 2))
    starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))

    #initiate sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, e, 20])

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(starting_guesses, nburn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    sampler.reset()

    print "Running an improved estimate..."
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    sampler.reset()
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #sample shape = (nwalkers, nsteps, ndim)
    sample = sampler.chain.reshape(-1, ndim)

    params = np.mean(sample[:, :2], 0)
    g = np.mean(sample[:, 2:], 0)
    outliers = (g < 0.5)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index][:2]
    errors = [sampler.flatchain[:, i].std() for i in xrange(ndim)][:2]

    fig = triangle.corner(sample, labels=['intercept' , 'slope'] + len(x)*['Gi',])
    fig.savefig(outtriangle)
    plt.close()


    return params, params_fit, errors, outliers


def powerlawFitWithOutliers(x, y, e, outtriangle='power.png'):
    """
    Linear fitting with outliers
    """
    x = np.asarray(x)
    y = np.asarray(y)
    e = np.asarray(e)
    # theta will be an array of length 2 + N, where N is the number of points
    # theta[0] is the amplitude, theta[1] is the power,
    # and theta[2 + i] is the weight g_i
    def log_prior(theta):
        #g_i needs to be between 0 and 1 and limits for the amplitude and power
        if (all(tmp > 0. for tmp in theta[2:]) and all(tmp < 1. for tmp in theta[2:])) and \
            -2. < theta[1] < -0.05 and 0. < theta[0] < 3.e2:
            return 0
        else:
            return -np.inf  # recall log(0) = -inf

    def log_likelihood(theta, x, y, e, sigma_B):
        dy = y - theta[0] * x**theta[1]
        g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
        logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
        logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy / sigma_B) ** 2
        return np.sum(np.logaddexp(logL1, logL2))

    def log_posterior(theta, x, y, e, sigma_B):
        return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)

    #find starting point
    def squared_loss(theta, x=x, y=y, e=e):
        dy = y - theta[0] * x**theta[1]
        return np.sum(0.5 * (dy / e) ** 2)

    theta1 = optimize.fmin(squared_loss, [10, -0.3], disp=False)

    ndim = 2 + len(x)   # number of parameters in the model
    nwalkers = 400      # number of MCMC walkers
    nburn = 1000        # "burn-in" period to let chains stabilize
    nsteps = 10000      # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    starting_guesses = np.zeros((nwalkers, ndim))
    starting_guesses[:, :2] = np.random.normal(theta1, 1, (nwalkers, 2))
    starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))

    #initiate sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, e, 10])

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(starting_guesses, nburn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    sampler.reset()

    print "Running an improved estimate..."
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    sampler.reset()
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #sample shape = (nwalkers, nsteps, ndim)
    sample = sampler.chain.reshape(-1, ndim)

    params = np.mean(sample[:, :2], 0)
    g = np.mean(sample[:, 2:], 0)
    outliers = (g < 0.5)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors and print best fits
    params_fit = pos[maxprob_index][:2]
    errors = [sampler.flatchain[:, i].std() for i in xrange(ndim)][:2]

    fig = triangle.corner(sample, labels=['amplitude', 'power'] + len(x)*['Gi', ])
    fig.savefig(outtriangle)
    plt.close()

    return params, params_fit, errors, outliers


def plotLambdaDependency(folder='results/', analysis='good', sigma=3):
    """
    Plot CCD PSF size as a function of wavelength and compare it to Euclid VIS requirement.
    """
    matplotlib.rc('text', usetex=True)
    if 'ind' in analysis:
        print 'Individual Results'
        data800 = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm*.pkl')]
        data600 = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm54*.pkl')]
        data700 = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm52*.pkl')]
        data890 = [fileIO.cPicleRead(file) for file in g.glob('results/I800nm50*.pkl')]
        data = (data600, data700, data800, data890)
        datacontainer = []
        for x in data:
            wx = np.median([d['wx'] for d in x])
            wxerr = np.median([d['wxerr'] for d in x])
            wy = np.median([d['wy'] for d in x])
            wyerr = np.median([d['wyerr'] for d in x])
            dat = dict(wx=wx, wy=wy, wxerr=wxerr, wyerr=wyerr)
            datacontainer.append(dat)
        data = datacontainer
        waves = [600, 700, 800, 890]
    elif 'join' in analysis:
        print 'Joint Results'
        data800nm = fileIO.cPicleRead(folder+'J800nm.pkl')
        data600nm = fileIO.cPicleRead(folder+'J600nm54k.pkl')
        data700nm = fileIO.cPicleRead(folder+'J700nm52k.pkl')
        data890nm = fileIO.cPicleRead(folder+'J890nm50k.pkl')
        data = (data600nm, data700nm, data800nm, data890nm)
        waves = [int(d['wavelength'].replace('nm', '')) for d in data]
    else:
        print 'Using subset of data'
        #data600nm = fileIO.cPicleRead(folder+'G600nm0.pkl')
        data600nm = fileIO.cPicleRead(folder+'J600nm54k.pkl')
        #data700nm = fileIO.cPicleRead(folder+'G700nm0.pkl')
        data700nm = fileIO.cPicleRead(folder+'J700nm52k.pkl')
        #data800nm = fileIO.cPicleRead(folder+'G800nm0.pkl')
        data800nm = fileIO.cPicleRead(folder+'J800nm.pkl')
        #data890nm = fileIO.cPicleRead(folder+'G890nm0.pkl')
        data890nm = fileIO.cPicleRead(folder+'J890nm50k.pkl')
        data = (data600nm, data700nm, data800nm, data890nm)
        waves = [600, 700, 800, 890]

    wx = np.asarray([_FWHMGauss(d['wx']) for d in data])
    wxerr = np.asarray([_FWHMGauss(d['wxerr']) for d in data])
    wypix = np.asarray([d['wy'] for d in data])
    wy = _FWHMGauss(wypix)
    wyerrpix = np.asarray([d['wyerr'] for d in data])
    wyerr = _FWHMGauss(wyerrpix)
    #hand derived -- picked the averages of the fits from many many runs
    wxpix = np.asarray([0.315, 0.31, 0.305, 0.29])
    wx = _FWHMGauss(wxpix)
    wxerrpix = np.asarray([0.01, 0.011, 0.012, 0.015])
    wxerr = _FWHMGauss(wxerrpix)
    # wy = np.asarray([_FWHMGauss(d) for d in [0.33, 0.31, 0.295, 0.29]])
    # wyerr = np.asarray([_FWHMGauss(d) for d in [0.01, 0.011, 0.013, 0.015]])
    waves = np.asarray(waves)

    w = np.sqrt(wx*wy)
    werr = np.sqrt(wxerr*wyerr)

    print zip(waves, w)

    #plot FWHM
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.95)
    ax1.set_title('CCD273 PSF Wavelength Dependency')

    ax1.errorbar(waves, wx, yerr=sigma*wxerr/3., fmt='o', label='Data')
    ax2.errorbar(waves, wy, yerr=sigma**wyerr/3., fmt='o', label='Data')
    ax3.errorbar(waves, w, yerr=sigma*werr, fmt='o', label='Data')

    #fit a power law
    fitfunc = lambda p, x: p[0] * x ** p[1]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    fit1, success = optimize.leastsq(errfunc, [1, -0.2],  args=(waves, wx))
    fit2, success = optimize.leastsq(errfunc, [1, -0.2],  args=(waves, wy))
    fit3, success = optimize.leastsq(errfunc, [1, -0.2],  args=(waves, w))

    #requirement
    alpha=0.2
    x = np.arange(500, 950, 1)
    y = 37*x**-alpha
    # compute the best fit function from the best fit parameters
    corrfit1 = fitfunc(fit1, x)
    corrfit2 = fitfunc(fit2, x)
    corrfit3 = fitfunc(fit3, x)
    print 'Slope:', fit1[1]
    print 'Slope:', fit2[1]
    print 'Slope [requirement < -0.2]:', fit3[1]

    #ax1.plot(x, corrfit1, 'k-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (fit1[1]))
    #ax2.plot(x, corrfit2, 'k-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (fit2[1]))
    ax3.plot(x, y, 'r-', label=r'Requirement: $\alpha \leq - %.1f$' % alpha)
    #ax3.plot(x, corrfit3, 'k-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (fit3[1]))

    # Bayesian
    shift = 0.
    waves -= shift
    px, paramsx, errorsx, outliersx = powerlawFitWithOutliers(waves, wx, wxerr, outtriangle='WFWHMx.png')
    py, paramsy, errorsy, outliersy = powerlawFitWithOutliers(waves, wy, wyerr, outtriangle='WFWHMy.png')
    p, params, errors, outliers = powerlawFitWithOutliers(waves, w, werr, outtriangle='WFWHM.png')
    print paramsx[::-1], errorsx[::-1]
    print paramsy[::-1], errorsy[::-1]
    print params[::-1], errors[::-1]

    ax1.plot(x, paramsx[0]*(x-shift)**paramsx[1], 'g-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (paramsx[1]))
    ax2.plot(x, paramsy[0]*(x-shift)**paramsy[1], 'g-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (paramsy[1]))
    ax3.plot(x, params[0]*(x-shift)**params[1], 'g-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (params[1]))

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(visible=False)
    plt.sca(ax3)

    ax1.set_ylim(6.6, 13.5)
    ax2.set_ylim(6.6, 13.5)
    ax3.set_ylim(6.6, 13.5)
    ax1.set_xlim(550, 900)
    ax2.set_xlim(550, 900)
    ax3.set_xlim(550, 900)

    ax1.set_ylabel(r'FWHM$_{X} \, [\mu$m$]$')
    ax2.set_ylabel(r'FWHM$_{Y} \, [\mu$m$]$')
    ax3.set_ylabel(r'FWHM$\, [\mu$m$]$')
    ax3.set_xlabel('Wavelength [nm]')
    ax1.legend(shadow=True, fancybox=True, loc='best', numpoints=1)
    ax2.legend(shadow=True, fancybox=True, loc='best', numpoints=1)
    ax3.legend(shadow=True, fancybox=True, loc='best', numpoints=1)
    plt.savefig('LambdaDependency.pdf')
    plt.close()

    print 'R2:'
    R2 = _R2FromGaussian(wxpix, wypix)*1e3
    print zip(waves, R2)
    errR2 = _R2err(wxpix, wypix, wxerrpix, wyerrpix)*1e3
    p, params, errors, outliers = powerlawFitWithOutliers(waves, R2, errR2, outtriangle='WR2.png')
    print params[::-1], errors[::-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.93)
    ax1.set_title('CCD273 PSF Wavelength Dependency')
    ax1.errorbar(waves, R2, yerr=sigma*errR2, fmt='o', label='Data')
    ax1.plot(x, params[0] * (x - shift)**params[1], 'm-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (params[1]))
    #ax1.plot(waves[outliers], R2[outliers], 'ro', ms=20, mfc='none', mec='red')
    ax1.set_ylabel(r'R^{2} \, [$mas$^{2}]$')
    ax1.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')

    print 'Ellipticity:'
    ell = _ellipticityFromGaussian(wxpix, wypix) + 1
    print zip(waves, ell)
    ellerr = _ellipticityerr(wxpix, wypix, wxerrpix, wyerrpix)
    p, params, errors, outliers = powerlawFitWithOutliers(waves, ell, ellerr, outtriangle='Well.png')
    print params[::-1], errors[::-1]

    fitfunc = lambda p, x: p[0] * x ** p[1]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    fit1, success = optimize.leastsq(errfunc, [2., -0.1],  args=(waves, ell), maxfev=100000)
    print fit1[::-1]

    ax2.errorbar(waves, ell, yerr=sigma*ellerr, fmt='o', label='Data')
    ax2.plot(x, params[0] * (x - shift)**params[1], 'm-', label=r'Power Law Fit: $\alpha \sim %.2f $' % (params[1]))
    #ax2.plot(waves[outliers], ell[outliers], 'ro', ms=20, mfc='none', mec='red')
    ax1.legend(shadow=True, fancybox=True, numpoints=1, loc='lower right')
    ax2.legend(shadow=True, fancybox=True, numpoints=1)
    ax2.set_ylabel('Ellipticity')

    ax1.set_ylim(0.65, 2.5)
    ax2.set_ylim(1+-0.01, 1+0.16)

    plt.sca(ax1)
    plt.xticks(visible=False)

    plt.savefig('LambdaR2ell.pdf')
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


def plotPaperFigures(folder='results/'):
    """
    Generate Figures of the Experimental Astronomy paper.
    """
    #model Example
    _AiryDisc(amplitude=1e6, center_x=10.0, center_y=10.0, radius=0.5, focus=0.4, size=21)
    _CCDkernel(CCDx=10, CCDy=10, width_x=0.35, width_y=0.4, size=21)
    plotModelExample()

    #simulation figures
    theta1 = (2.e5, 9.9, 10.03, 0.41, 0.51, 10., 10., 0.291, 0.335)
    try:
        _plotDifferenceIndividualVsJoined(individuals='simulatedResults/RunI*.pkl',
                                          joined='results/simulated800nmJoint.pkl',
                                          title='Simulated Data: CCD PSF Recovery', truthx=theta1[7], truthy=theta1[8],
                                          requirementE=None, requirementFWHM=None, requirementR2=None)
        _plotModelResiduals(id='RunI1', folder='simulatedResults/', out='Residual1.pdf', individual=True)
    except:
        print 'No simulated data to plot...'

    #real data
    _plotDifferenceIndividualVsJoined(individuals=folder+'I800nm?.pkl', joined=folder+'J800nm.pkl', title='800nm',
                                      FWHMlims=(7.3, 11.8))

    _plotModelResiduals(id='I800nm2', folder=folder, out='ResidualData.pdf', individual=True)
    _plotModelResiduals(id='RunI2', folder='simulatedResults/', out='Residual2.pdf', individual=True)

    #_plotModelResiduals(id='G600nm0', folder=folder, out='ResidualG600.pdf', individual=True)
    #_plotModelResiduals(id='G700nm0', folder=folder, out='ResidualG700.pdf', individual=True)
    _plotModelResiduals(id='G800nm0', folder=folder, out='ResidualG800.pdf', individual=True)
    #_plotModelResiduals(id='G890nm0', folder=folder, out='ResidualG890.pdf', individual=True)

    #wavelength dependency
    plotLambdaDependency()

    #brighter fatter
    plotBrighterFatter()


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


def _testDifficultCases():
    """
    These are a few files that seem to be very difficult to fit. The main reason is that the spot
    was rather off-centred making it difficult to estimate the amplitude of the Airy disc.
    These files seem to give way too sharp kernels. This might be related to the fact that t
    """
    files = getFiles(mintime=(16, 21, 52), maxtime=(16, 26, 16), folder='data/31Jul/')
    RunData([files[0], ], out='I800nm50k', wavelength='l800')
    # ============================================================
    # Fitting with MCMC:
    # ******************** Fitted parameters ********************
    # peak = 1.876476e+05 +- 5.178195e+03
    # center_x = 1.013078e+01 +- 5.653792e-02
    # center_y = 9.747014e+00 +- 7.721453e-02
    # radius = 4.660135e-01 +- 7.392010e-03
    # focus = 4.625621e-01 +- 2.905099e-02
    # width_x = 1.339326e-01 +- 3.069773e-02
    # width_y = 1.685128e-01 +- 2.864525e-02
    # ============================================================
    # GoF: 63.013339033  Maximum difference: 27583.5231737
    #
    # FIT UNLIKELY TO BE GOOD...
    #
    # Amplitude estimate: 908507.619723
    # ============================================================
    # FWHM (requirement 10.8 microns):
    # 4.25  +/-  0.838  microns
    # x: 3.78  +/-  0.867  microns
    # y: 4.76  +/-  0.809  microns
    # ============================================================


    files = getFiles(mintime=(15, 43, 24), maxtime=(15, 51, 47), folder='data/31Jul/')
    RunData([files[0], ], out='I800nm20k', wavelength='l800')
    #Kernel probably too narrow...
    # ============================================================
    # Fitting with MCMC:
    # ******************** Fitted parameters ********************
    # peak = 6.347887e+04 +- 7.413094e+02
    # center_x = 1.010584e+01 +- 5.058798e-02
    # center_y = 9.761372e+00 +- 5.927853e-02
    # radius = 4.584184e-01 +- 3.703650e-03
    # focus = 4.391148e-01 +- 7.021983e-03
    # width_x = 1.519947e-01 +- 3.559739e-02
    # width_y = 1.838032e-01 +- 4.041398e-02
    # ============================================================
    # GoF: 3.18858069585  Maximum difference: 1676.70415821
    # Amplitude estimate: 242256.307521
    # ============================================================
    # FWHM (requirement 10.8 microns):
    # 4.72  +/-  1.072  microns
    # x: 4.3  +/-  1.006  microns
    # y: 5.19  +/-  1.142  microns
    # ============================================================


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


def runGood():
    """
    Analyse data that are well centred. These fits are more reliable.
    """
    forwardModelJointFit(getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/'),
                     out='J600nm54k', wavelength='600nm') #kernel around 0.3, 0.33
    forwardModelJointFit(getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/'),
                     out='J700nm52k', wavelength='700nm') #around 0.3, 0.31
    RunData([getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/')[2],], out='G800nm',
             wavelength='l800') #around 0.305/315 and 0.295/0.3
    forwardModelJointFit(getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/'),
                         out='J800nm', wavelength='800nm') #around 0.3, 0.3
    forwardModelJointFit(getFiles(mintime=(14, 30, 03), maxtime=(14, 34, 37), folder='data/01Aug/'),
                     out='J890nm50k', wavelength='890nm') #around 0.285, 0.29


def runBrighterFatter():
    """
    Special runs for brighter-fatter effect.
    Most of the 800nm data were very poorly centred.
    """
    RunData([getFiles(mintime=(15, 12, 20), maxtime=(15, 24, 16), folder='data/31Jul/')[0],], out='I800nmlow',
            wavelength='l800l')
    RunData([getFiles(mintime=(15, 28, 40), maxtime=(15, 39, 21), folder='data/31Jul/')[2],], out='I800nmmed',
            wavelength='l800m')
    RunData([getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/')[4],], out='I800nmhigh',
            wavelength='l800h')


def runWavelengthDependency():
    """
    Run the best for the wavelength dependency.
    """
    RunData([getFiles(mintime=(15, 39, 58), maxtime=(15, 47, 58), folder='data/30Jul/')[0],], out='I600nmwave',
            wavelength='l600')
    RunData([getFiles(mintime=(17, 48, 35), maxtime=(17, 56, 03), folder='data/30Jul/')[0],], out='I700nmwave',
            wavelength='l700')
    RunData([getFiles(mintime=(15, 40, 07), maxtime=(15, 45, 14), folder='data/29Jul/')[0],], out='I800nmwave',
            wavelength='l800')
    RunData([getFiles(mintime=(14, 17, 57), maxtime=(14, 25, 49), folder='data/01Aug/')[4],], out='I890nmwave',
            wavelength='l890')


def doAll():
    try:
        RunTestSimulations()
    except:
        print 'Cannot run Simulations 1'
    try:
        RunTestSimulations2()
    except:
        print 'Cannot run Simulations 2'

    #Data Analysis -- real spots
    try:
        runGood()
    except:
        print 'Cannot run goods'
    try:
        analyseData600nm()
    except:
        print 'Cannot run 600nm'
    try:
        analyseData700nm()
    except:
        print 'Cannot run 700nm'
    try:
        analyseData800nm()
    except:
        print 'Cannot run 800nm'
    try:
        analyseData890nm()
    except:
        print 'Cannot run 890nm'
    try:
        analyseData800nmBrighterFatter()
    except:
        print 'Cannot run Brighter-Fatter'

    #plots
    try:
        plotPaperFigures()
    except:
        print 'Cannot generate paper figures'
    try:
        generateTestPlots()
    except:
        print 'Cannot generate test plots'

    #All Data
    try:
        AlljointRuns()
    except:
        print 'Cannot run all joint runs'
    try:
        AllindividualRuns()
    except:
        print 'Cannot run all individual runs'
    try:
        plotAllResiduals()
    except:
        print 'Cannot plot all residuals'

    #Special Runs
    try:
        forwardModelJointFit(getFiles(mintime=(15, 03, 29), maxtime=(15, 41, 01), folder='data/22Jul/'),
                             out='J800nmDrift', wavelength='800nm', spotx=2985, spoty=3774) #0.3, 0.3
    except:
        print 'Cannot run 800nm drift data'
    try:
        forwardModelJointFit(getFiles(mintime=(14, 56, 18), maxtime=(15, 19, 42), folder='data/30Jul/'),
                             out='J600nm20', wavelength='600nm') #0.295, 0.33
    except:
        print 'Cannot run 600nm special set'

    #test some of the cases, which seem to be more difficult to fit
    try:
        _testDifficultCases()
    except:
        print 'Cannot run difficult cases'


def analyseOutofFocus():
    """

    """
    forwardModel('data/13_59_05sEuclid.fits', wavelength='l700', out='blurred700',
                 spotx=2985, spoty=3774, size=20, blurred=True)
    forwardModel('data/13_24_53sEuclid.fits', wavelength='l800', out='blurred800',
                 spotx=2983, spoty=3760, size=10, blurred=True)


if __name__ == '__main__':
    #doAll()
    #_printAnalysedData('results/')

    #Simulated spots and analysis
    #RunTestSimulations()
    #RunTestSimulations2()

    #Data Analysis -- real spots
    #runBrighterFatter()
    #runWavelengthDependency()
    #runGood()
    #analyseData600nm()
    #analyseData700nm()
    #analyseData800nm()
    #analyseData890nm()
    #analyseData800nmBrighterFatter()

    #Special Runs
    #forwardModelJointFit(getFiles(mintime=(15, 03, 29), maxtime=(15, 41, 01), folder='data/22Jul/'),
    #                     out='J800nmDrift', wavelength='800nm', spotx=2985, spoty=3774)
    #forwardModelJointFit(getFiles(mintime=(14, 56, 18), maxtime=(15, 19, 42), folder='data/30Jul/'),
    #                     out='J600nm20', wavelength='600nm')

    #plots
    #plotPaperFigures()
    #generateTestPlots()
    #plotBrighterFatter()
    #plotLambdaDependency()

    #All Data
    #AlljointRuns()
    #AllindividualRuns()
    #plotAllResiduals()

    #Testing set
    #RunTest()

    #test some of the cases, which seem to be more difficult to fit
    #_testDifficultCases()

    analyseOutofFocus()