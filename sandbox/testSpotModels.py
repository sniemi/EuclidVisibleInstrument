"""
Simply for testing optimising purposes, do not use with real data!
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
import scipy.ndimage.measurements as m
from scipy import signal
from support import files as fileIO
from astropy.modeling import models, fitting
import triangle
from multiprocessing import Pool
import glob as g


def forwardModel(file, out='Data', gain=3.1, size=10, burn=20, spotx=2888, spoty=3514, run=50,
                 simulation=False, truths=None):
    """
    Forward models the spot data found from the input file. Can be used with simulated and real data.

    Notes:
    - The emcee is run three times as it is important to have a good starting point for the final run.
    - It is very important to have the amplitude well estimated, otherwise it is difficult to get good parameter estimates.
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
    data[data + rn**2 < 0.] = 0.  #set highly negative values to zero
    #assume errors scale as sqrt of the values + readnoise
    #sigma = np.sqrt(data/gain + rn**2)
    var = data.copy() + rn**2

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

    #MCMC based fitting
    print 'Bayesian Fitting...'
    ndim = 7
    nwalkers = 1000

    #Choose an initial set of positions for the walkers - fairly large area not to bias the results
    #amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = np.random.uniform(max, 2.*max, size=nwalkers)     # amplitude
    p0[:, 1] = np.random.uniform(7., 14., size=nwalkers)         # x
    p0[:, 2] = np.random.uniform(7., 14., size=nwalkers)         # y
    p0[:, 3] = np.random.uniform(.1, 1., size=nwalkers)          # radius
    p0[:, 4] = np.random.uniform(.1, 1., size=nwalkers)          # focus
    p0[:, 5] = np.random.uniform(.1, 0.5, size=nwalkers)         # width_x
    p0[:, 6] = np.random.uniform(.1, 0.5, size=nwalkers)         # width_y

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
    pool = Pool(7) #A hack Dan gave me to not have ghost processes running as with threads keyword
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, var], pool=pool)

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, burn)
    sampler.reset()
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
    samples = sampler.chain.reshape((-1, ndim))
    extents = None
    if simulation:
        extents = [(0.91*truth, 1.09*truth) for truth in truths]
        extents[1] = (truths[1]*0.995, truths[1]*1.005)
        extents[2] = (truths[2]*0.995, truths[2]*1.005)
        extents[3] = (0.395, 0.425)
        extents[4] = (0.503, 0.517)
    fig = triangle.corner(samples,
                          labels=['amplitude', 'x', 'y', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)#, extents=extents)
    fig.savefig(out+'Triangle.png')

    pool.close()


def forwardModelJointFit(files, out, wavelength, gain=3.1, size=10, burn=50, run=100,
                         spotx=2888, spoty=3514, simulated=False, truths=None):
    """
    Forward models the spot data found from the input files. Models all data simultaneously so that the Airy
    disc centroid and shift from file to file. Assumes that the spot intensity, focus, and the CCD PSF kernel
    are the same for each file. Can be used with simulated and real data.
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

        max = np.max(spot)
        print 'Maximum Value:', max
        peakvalues.append(max)

        #noise model
        variance = spot.copy() + rn**2

        #save to a list
        image.append(spot)
        noise.append(variance)

    #sensibility test, try to check if all the files in the fit are of the same dataset
    if np.std(peakvalues) > 5*np.sqrt(np.median(peakvalues)):
        #check for more than 5sigma outliers, however, this is very sensitive to the centroiding of the spot...
        print 'POTENTIAL OUTLIER, please check the input files...'
        print np.std(peakvalues), 5*np.sqrt(np.median(peakvalues))

    #MCMC based fitting
    ndim = 2*images + 5  #xpos, ypos for each image and single amplitude, radius, focus, and sigmaX and sigmaY
    nwalkers = 1000
    print 'Bayesian Fitting, model has %i dimensions' % ndim

    # Choose an initial set of positions for the walkers using the Gaussian fit
    p0 = np.zeros((nwalkers, ndim))
    for x in xrange(images):
        p0[:, 2*x] = np.random.uniform(7., 14., size=nwalkers)      # x
        p0[:, 2*x+1] = np.random.uniform(7., 14., size=nwalkers)    # y
    p0[:, -5] = np.random.uniform(max, 2.*max, size=nwalkers)       # amplitude
    p0[:, -4] = np.random.uniform(.1, 1., size=nwalkers)            # radius
    p0[:, -3] = np.random.uniform(.1, 1., size=nwalkers)            # focus
    p0[:, -2] = np.random.uniform(.1, 0.5, size=nwalkers)           # width_x
    p0[:, -1] = np.random.uniform(.1, 0.5, size=nwalkers)           # width_y

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
    pool = Pool(7) #A hack Dan gave me to not have ghost processes running as with threads keyword
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorJoint, args=[xx, yy, image, noise], pool=pool)

    # Run a burn-in and set new starting position
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)
    best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
    pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, burn)
    sampler.reset()
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
        id = 'test/' + out + str(index)
        #X and Y are always in pairs
        center_x = params_fit[2*index]
        center_y = params_fit[2*index+1]

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
    fileIO.cPickleDumpDictionary(res, 'test/' + out + '.pkl')

    #plot
    samples = sampler.chain.reshape((-1, ndim))
    #extents = None
    #if simulated:
    #    extents = [(0.9*truth, 1.1*truth) for truth in truths]
    #    print extents
    fig = triangle.corner(samples, labels=['x', 'y']*images + ['amplitude', 'radius', 'focus', 'width_x', 'width_y'],
                          truths=truths)#, extents=extents)
    fig.savefig('test/' + out + 'Triangle.png')

    pool.close()


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
    if 7. < center_x < 14. and 7. < center_y < 14. and 0.1 < width_x < 1. and 0.1 < width_y < 1. and \
       1.e3 < amplitude < 1.e6 and 0. < radius < 2. and 0. < focus < 1.:
        return 0.
    else:
        return -np.inf


def log_priorJoint(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    tmp = theta[-5:] #these are the last five i.e. amplitude, radius, focus, sigmaX, and sigmaY
    if all(7. < x < 14. for x in theta[:-5]) and 1.e3 < tmp[0] < 1.e6 and 0. < tmp[1] < 2. and 0. < tmp[2] < 1. and \
       0.1 < tmp[3] < 1. and 0.1 < tmp[4] < 1.:
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
    #lnL = - 0.5 * np.sum((data - model)**2 / var)
    #others...
    lnL = - 2. * np.sum((((data - model)**2) + np.abs(data - model))/var) #does not get the amplitude easily right
    #using L1 norm would be true for exponential distribution
    #lnL = - np.sum(np.abs(data - model) / var)

    # data[data < 1.] = 0.
    # model[model < 1.] = 1.
    # lnL1 = - np.sum(model - data*np.log(model))
    # lnL2 = - 0.5 * np.sum((data - model)**2 / var)
    # #lnL = np.logaddexp(lnL1, lnL2)
    # lnL = lnL1 + lnL2

    return lnL


def log_likelihoodJoint(theta, x, y, data, var, size=21):
    """
    Logarithm of the likelihood function for joint fitting. Not really sure if this is right...
    """
    #unpack the parameters
    #[xpos, ypos]*images) +[amplitude, radius, focus])
    images = len(theta[:-5]) / 2
    amplitude, radius, focus, width_x, width_y = theta[-5:]

    data[data < 1.] = 0.

    lnL = 0.
    for tmp in xrange(images):
        #X and Y are always in pairs
        center_x = theta[2*tmp]
        center_y = theta[2*tmp+1]

        #1)Generate a model Airy disc
        airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
        adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape((size, size))

        #2)Apply Focus, no normalisation as smoothing
        f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
        focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
        model = signal.convolve2d(adata, focusdata, mode='same')

        #3)Apply CCD diffusion, approximated with a Gaussian -- max = 1 as centred
        CCD = models.Gaussian2D(1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.)
        CCDdata = CCD.eval(x, y, 1., size/2.-0.5, size/2.-0.5, width_x, width_y, 0.).reshape((size, size))
        model = signal.convolve2d(model, CCDdata, mode='same').flatten()

        lnL += - 0.5 * np.sum((data[tmp].flatten() - model)**2 / var[tmp].flatten())
        # model[model < 1.] = 1.
        # lnL1 = - np.sum(model - data*np.log(model))
        # lnL2 = - 0.5 * np.sum((data - model)**2 / var)
        # #lnL += np.logaddexp(lnL1, lnL2)
        # lnL += lnL1 + lnL2

    return lnL


def log_posteriorG(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_priorG(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihoodG(theta, x, y, z, var)



def log_priorG(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    if 7. < center_x < 14. and 7. < center_y < 14. and 0.1 < width_x < 1. and 0.1 < width_y < 1. and \
       1.e3 < amplitude < 1.e6 and 0. < radius < 2. and 0. < focus < 1.:
        return 0.
    else:
        return -np.inf


def log_likelihoodG(theta, x, y, data, var, size=21):
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

    return lnL


def log_posteriorC(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_priorC(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihoodC(theta, x, y, z, var)


def log_priorC(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    amplitude, center_x, center_y, radius, focus, width_x, width_y , width_d = theta
    if 8. < center_x < 12. and 8. < center_y < 12. and 0. <= width_x < 0.25 and 0. <= width_y < 0.25 and \
       0. <= width_d < 0.1 and 1.e2 < amplitude < 1.e6 and 0. < radius < 1. and 0. < focus < 1.:
        return 0.
    else:
        return -np.inf


def log_likelihoodC(theta, x, y, data, var, size=21):
    """
    Logarithm of the likelihood function.
    """
    #unpack the parameters
    amplitude, center_x, center_y, radius, focus, width_x, width_y, width_d = theta

    #1)Generate a model Airy disc
    airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    adata = airy.eval(x, y, amplitude, center_x, center_y, radius).reshape((size, size))

    #2)Apply Focus
    f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    focusdata = f.eval(x, y, 1., center_x, center_y, focus, focus, 0.).reshape((size, size))
    focusmodel = signal.convolve2d(adata, focusdata, mode='same')

    #3)Apply CCD diffusion kernel
    kernel = np.array([[width_d, width_y, width_d],
                       [width_x, 1., width_x],
                       [width_d, width_y, width_d]])
    kernel /= kernel.sum()
    #model = ndimage.convolve(focusmodel, kernel)
    model = signal.convolve2d(focusmodel, kernel, mode='same').flatten()

    #true for Gaussian errors, but not really true here because of mixture of Poisson and Gaussian noise
    lnL = - 0.5 * np.sum((data - model)**2 / var)

    return lnL


def forwardModelTest(file, CCDPSFmodel='Gaus', out='Data', gain=3.1, size=10, spotx=2888, spoty=3514,
                     burn=100, run=200, nwalkers=1000):
    """
    A single file to quickly test if the method works
    """
    #get data and convert to electrons
    print '\n\n\n'
    print '_'*120
    print 'Processing:', file

    o = pf.getdata(file)*gain

    #roughly the correct location - to avoid identifying e.g. cosmic rays
    data = o[spoty-(size*3):spoty+(size*3)+1, spotx-(size*3):spotx+(size*3)+1].copy()

    #maximum position within the cutout
    y, x = m.maximum_position(data)

    #spot and the peak pixel within the spot, this is also the CCD kernel position
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()
    CCDy, CCDx = m.maximum_position(spot)

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

    fileIO.writeFITS(model, out+'BasicModelG.fits', int=False)
    fileIO.writeFITS(model - spot, out+'BasicModelResidualG.fits', int=False)

    airy = models.AiryDisk2D(spot.max(), size, size, 0.6)
    p_init = airy
    fit_p = fitting.LevMarLSQFitter()
    a = fit_p(p_init, X, Y, spot)
    print a
    model = p(X, Y)

    fileIO.writeFITS(model, out+'BasicModelA.fits', int=False)
    fileIO.writeFITS(model - spot, out+'BasicModelResidualA.fits', int=False)

    #goodness of fit
    gof = (1./(len(data)-5.)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof
    print 'Done'

    #MCMC based fitting
    if 'Gaus' in CCDPSFmodel:
        ndim = 7
        print 'Model with a Gaussian CCD PSF, %i dimensions' % ndim

        #Choose an initial set of positions for the walkers - fairly large area not to bias the results
        #amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
        p0 = np.zeros((nwalkers, ndim))
        p0[:, 0] = np.random.uniform(max, 2.*max, size=nwalkers)     # amplitude
        p0[:, 1] = np.random.uniform(7., 14., size=nwalkers)         # x
        p0[:, 2] = np.random.uniform(7., 14., size=nwalkers)         # y
        p0[:, 3] = np.random.uniform(.1, 1., size=nwalkers)          # radius
        p0[:, 4] = np.random.uniform(.1, 1., size=nwalkers)          # focus
        p0[:, 5] = np.random.uniform(.1, 0.5, size=nwalkers)         # width_x
        p0[:, 6] = np.random.uniform(.1, 0.5, size=nwalkers)         # width_y

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
        pool = Pool(7) #A hack Dan gave me to not have ghost processes running as with threads keyword
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorG, args=[xx, yy, data, var], pool=pool)

        # Run a burn-in and set new starting position
        print "Burning-in..."
        pos, prob, state = sampler.run_mcmc(p0, burn)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
        pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        # Starting from the final position in the burn-in chain
        print "Running MCMC..."
        pos, prob, state = sampler.run_mcmc(pos, burn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)

        # Print out the mean acceptance fraction
        print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

        #Get the index with the highest probability
        maxprob_index = np.argmax(prob)

        #Get the best parameters and their respective errors and print best fits
        params_fit = pos[maxprob_index]
        errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
        _printResults2(params_fit, errors_fit, model=CCDPSFmodel)

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
        fileIO.writeFITS(((model-spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)

        #results
        _printFWHM(width_x, width_y, errors_fit[5], errors_fit[6])

        #plot
        samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
        fig = triangle.corner(samples,
                              labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y'])
        fig.savefig(out+'Triangle.png')

    elif 'Cross' in CCDPSFmodel:
        ndim = 8
        print 'Model with a Cross CCD PSF, %i dimensions' % ndim

        #amplitude, center_x, center_y, radius, focus, width_x, width_y, width_d = theta
        # Choose an initial set of positions for the walkers using the Gaussian fit
        p0 = [np.asarray([1.3*max,#p.amplitude.value,
                          p.x_mean.value,
                          p.y_mean.value,
                          np.max([p.x_stddev.value, p.y_stddev.value]),
                          0.5,
                          0.08,
                          0.1,
                          0.01]) + 1e-3*np.random.randn(ndim) for i in xrange(nwalkers)]

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
        pool = Pool(7) #A hack Dan gave me to not have ghost processes running as with threads keyword
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorC, args=[xx, yy, data, var], pool=pool)


        # Run a burn-in and set new starting position
        print "Burning-in..."
        pos, prob, state = sampler.run_mcmc(p0, burn)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
        pos = emcee.utils.sample_ball(best_pos, best_pos/100., size=nwalkers)
        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        # Starting from the final position in the burn-in chain
        print "Running MCMC..."
        pos, prob, state = sampler.run_mcmc(pos, burn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, run, rstate0=state)

        # Print out the mean acceptance fraction
        print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

        #Get the index with the highest probability
        maxprob_index = np.argmax(prob)

        #Get the best parameters and their respective errors and print best fits
        params_fit = pos[maxprob_index]
        errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]
        _printResults2(params_fit, errors_fit, model=CCDPSFmodel)

        #Best fit model
        amplitude, center_x, center_y, radius, focus, width_x, width_y, width_d = params_fit
        airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
        adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape(spot.shape)
        f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
        focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape(spot.shape)
        foc = signal.convolve2d(adata, focusdata, mode='same')

        #3)Apply CCD diffusion kernel
        kernel = np.array([[width_d, width_y, width_d],
                           [width_x, 1., width_x],
                           [width_d, width_y, width_d]])
        kernel /= kernel.sum()
        model = signal.convolve2d(foc, kernel, mode='same')

        #save model
        fileIO.writeFITS(model, out+'model.fits', int=False)

        #residuals
        fileIO.writeFITS(model - spot, out+'residual.fits', int=False)
        fileIO.writeFITS(((model-spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)

        #results
        print kernel
        gaus = models.Gaussian2D(kernel.max(), 1.5, 1.5, x_stddev=0.3, y_stddev=0.3)
        gaus.theta.fixed = True
        p_init = gaus
        fit_p = fitting.LevMarLSQFitter()
        stopy, stopx = kernel.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
        p = fit_p(p_init, X, Y, kernel)
        #print p
        _printFWHM(p.x_stddev.value, p.y_stddev.value, errors_fit[5], errors_fit[6])

        #plot
        samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
        fig = triangle.corner(samples,
                              labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y', 'width_d'])
        fig.savefig(out+'Triangle.png')

    # a simple goodness of fit
    gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof, ' Maximum difference:', np.max(np.abs(model - spot))


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


def _printResults2(best_params, errors, model):
    """
    Print basic results.
    """
    print("=" * 60)
    print('Fitting with MCMC:')
    if 'Gaus' in model:
        pars = ['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y']
        print('*'*20 + ' Fitted parameters ' + '*'*20)
        for name, value, sig in zip(pars, best_params, errors):
            print("{:s} = {:e} +- {:e}" .format(name, value, sig))
        print("=" * 60)
    elif 'Cross' in model:
        pars = ['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y', 'width_d']
        print('*'*20 + ' Fitted parameters ' + '*'*20)
        for name, value, sig in zip(pars, best_params, errors):
            print("{:s} = {:e} +- {:e}" .format(name, value, sig))
        print("=" * 60)


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


def _plotModelResiduals(id='simulated800nmJoint1', folder='test/', out='Residual.pdf', individual=False):
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
    ax4.set_title('Normalised Residual')

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
    cbar3.set_label(r'$D_{i, j} - M_{i, j} \quad [e^{-}]$')
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label(r'$\frac{(D_{i, j} - M_{i, j})^{2}}{\sigma_{CCD}^{2}}$')

    for tmp in ax:
        plt.sca(tmp)
        plt.xticks(visible=False)
        plt.yticks(visible=False)

    plt.savefig(out)
    plt.close()


def _plotDifferenceIndividualVsJoined(individuals, joined, title='800nm', sigma=3,
                                      requirementFWHM=10.8, requirementE=0.156, requirementR2=0.002,
                                      truthx=None, truthy=None, FWHMlims=(7.5, 10.3)):
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
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.98)
    ax1.set_title(title)

    ax1.errorbar(xtmp, [_FWHMGauss(data['wx']) for data in ind], yerr=[sigma*_FWHMGauss(data['wxerr']) for data in ind],
                 fmt='o')
    ax1.errorbar(xtmp[-1]+1, _FWHMGauss(join['wx']), yerr=sigma*_FWHMGauss(join['wxerr']), fmt='s')
    ax2.errorbar(xtmp, [_FWHMGauss(data['wy']) for data in ind], yerr=[sigma*_FWHMGauss(data['wyerr']) for data in ind],
                 fmt='o')
    ax2.errorbar(xtmp[-1]+1, _FWHMGauss(join['wy']), yerr=sigma*_FWHMGauss(join['wyerr']), fmt='s')

    #simulations
    if truthx is not None:
        ax1.axhline(y=_FWHMGauss(truthx), label='Truth', c='g')
    if truthy is not None:
        ax2.axhline(y=_FWHMGauss(truthy), label='Truth', c='g')

    #requirements
    if requirementFWHM is not None:
        ax1.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r')
        ax2.axhline(y=requirementFWHM, label='Requirement (800nm)', c='r')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)

    ltmp = np.hstack((xtmp, xtmp[-1]+1))
    plt.xticks(ltmp, ['Individual %i' % x for x in ltmp[:-1]] + ['Joint',], rotation=45)

    ax1.set_ylim(*FWHMlims)
    ax2.set_ylim(*FWHMlims)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel(r'FWHM$_{X} \quad [\mu$m$]$')
    ax2.set_ylabel(r'FWHM$_{Y} \quad [\mu$m$]$')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedFWHM%s.pdf' % title)
    plt.close()

    #plot R2 and ellipticity
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.93, bottom=0.17, left=0.11, right=0.98)
    ax1.set_title(title)

    ax1.errorbar(xtmp, [_R2FromGaussian(data['wx'], data['wy'])*1e3 for data in ind],
                 yerr=[sigma*1.e3*_R2FromGaussian(data['wxerr'], data['wyerr']) for data in ind], fmt='o')
    ax1.errorbar(xtmp[-1]+1, _R2FromGaussian(join['wx'], join['wy'])*1e3,
                 yerr=sigma*1.e3*_R2FromGaussian(join['wxerr'], join['wyerr']), fmt='s')

    ax2.errorbar(xtmp, [_ellipticityFromGaussian(data['wy'], data['wx']) for data in ind],
                 yerr=[_ellipticityFromGaussian(data['wyerr'], data['wxerr'])/30. for data in ind], fmt='o')
    ax2.errorbar(xtmp[-1]+1, _ellipticityFromGaussian(join['wy'], join['wx']),
                 yerr=_ellipticityFromGaussian(join['wyerr'], join['wxerr'])/30., fmt='s')

    if requirementE is not None:
        ax2.axhline(y=requirementE, label='Requirement (800nm)', c='r')
    if requirementR2 is not None:
        ax1.axhline(y=requirementR2*1e3, label='Requirement (800nm)', c='r')

    #simulations
    if truthx and truthy is not None:
        ax2.axhline(y=_ellipticityFromGaussian(truthx, truthy), label='Truth', c='g')
        ax1.axhline(y= _R2FromGaussian(truthx, truthy)*1e3, label='Truth', c='g')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    ltmp = np.hstack((xtmp, xtmp[-1]+1))
    plt.xticks(ltmp, ['Individual%i' % x for x in ltmp[:-1]] + ['Joint',], rotation=45)

    ax1.set_ylim(0.001*1e3, 0.004*1e3)
    ax2.set_ylim(0., 0.33)
    ax1.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)
    ax2.set_xlim(xtmp.min()*0.9, (xtmp.max() + 1)*1.05)

    ax1.set_ylabel(r'$R^{2}$ [mas$^{2}$]')
    ax2.set_ylabel('ellipticity')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig('IndividualVsJoinedR2e%s.pdf' % title)
    plt.close()


def TestUsingSimulations():
    """
    A set of simulated spots and analysis.
    """
    print("|" * 120)
    print 'SIMULATED DATA'
    #a joint fit test - vary only the x and y positions
    theta1 = (2.e5, 9.9, 10.03, 0.41, 0.51, 10., 10., 0.296, 0.335)
    theta2 = (2.e5, 10.1, 9.97, 0.41, 0.51, 10., 10., 0.296, 0.335)
    theta3 = (2.e5, 9.97, 10.1, 0.41, 0.51, 10., 10., 0.296, 0.335)
    theta4 = (2.e5, 10.08, 10.04, 0.41, 0.51, 10., 10., 0.296, 0.335)
    theta5 = (2.e5, 10.1, 9.97, 0.41, 0.51, 10., 10., 0.296, 0.335)

    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        forwardModel(file='simulated/simulatedJoint%i.fits' %i, out='test/RunI%i' %i, simulation=True,
                     truths=[theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]])

        print 'amplitude, x, y, radius, focus, width_x, width_y'
        print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
        print("=" * 60)

    #plot residuals
    _plotModelResiduals(id='RunI0', folder='test/', out='test/Residual0.pdf', individual=True)
    _plotModelResiduals(id='RunI1', folder='test/', out='test/Residual1.pdf', individual=True)
    _plotModelResiduals(id='RunI2', folder='test/', out='test/Residual2.pdf', individual=True)
    _plotModelResiduals(id='RunI3', folder='test/', out='test/Residual3.pdf', individual=True)
    _plotModelResiduals(id='RunI4', folder='test/', out='test/Residual4.pdf', individual=True)

    #joint fit
    truths = [theta1[1], theta1[2], theta2[1], theta2[2], theta3[1], theta3[2], theta4[1], theta4[2],
              theta5[1], theta5[2], theta1[0], theta4[3], theta1[4], theta1[7], theta1[8]]
    forwardModelJointFit(g.glob('simulated/simulatedJoint?.fits'),
                         out='simulated800nmJoint', wavelength='800nm', simulated=True,
                         truths=truths)

    print 'True width_x and widht_y:', theta1[7], theta1[8]

    #plot residuals
    _plotModelResiduals(id='simulated800nmJoint0', folder='test/', out='ResidualJ0.pdf')
    _plotModelResiduals(id='simulated800nmJoint1', folder='test/', out='ResidualJ1.pdf')
    _plotModelResiduals(id='simulated800nmJoint2', folder='test/', out='ResidualJ2.pdf')
    _plotModelResiduals(id='simulated800nmJoint3', folder='test/', out='ResidualJ3.pdf')
    _plotModelResiduals(id='simulated800nmJoint4', folder='test/', out='ResidualJ4.pdf')

    #test plots
    _plotDifferenceIndividualVsJoined(individuals='test/RunI*.pkl',
                                      joined='test/simulated800nmJoint.pkl',
                                      title='Test', truthx=theta1[7], truthy=theta1[8],
                                      requirementE=None, requirementFWHM=None, requirementR2=None)


if __name__ == '__main__':
    #file = 'testdata/17_52_04sEuclid.fits' #Gaussian works better
    file = 'testdata/15_41_20sEuclid.fits' #Gaussian works better
    #file = 'testdata/15_34_34sEuclid.fits' #Gaussian works better

    # forwardModelTest(file=file, out='modeltest/' + file.replace('.fits', '').replace('testdata/', '') + 'G')
    # forwardModelTest(file=file, CCDPSFmodel='Cross',
    #                  out='modeltest/' + file.replace('.fits', '').replace('testdata/', '') + 'C')

    TestUsingSimulations()