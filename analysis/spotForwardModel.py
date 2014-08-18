"""
CCD Spot Measurements
=====================

Analyse laboratory CCD PSF measurements by forward modelling.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python
:requires: emcee

:version: 0.3

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import pyfits as pf
import numpy as np
import emcee
import scipy.ndimage.measurements as m
import triangle
import pymc
import glob as g
from scipy import signal
from scipy import ndimage
from support import files as fileIO
from astropy.modeling import models, fitting, SerialCompositeModel


def log_prior(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    width_x, width_y, amp, x0, y0, rad = theta
    if 0. < width_x < 0.8 and 0. < width_y < 0.8 and 1.e4 < amp < 1.e6 and 8. < x0 < 12. and 8. < y0 < 12. \
            and 0. < rad < 1.:
        return 0.
    else:
        return -np.inf


def log_likelihood(theta, x, y, data, var):
    """
    This is not quite right...
    """
    width_x, width_y, amp, x0, y0, rad = theta

    #Airy disc
    airy = models.AiryDisk2D(amp, x0, y0, rad)
    adata = airy.eval(x, y, amp, x0, y0, rad).reshape((21, 21))

    #Convolve the Airy disc with the Gaussian broadening kernel to derive the model
    model = ndimage.filters.gaussian_filter(adata, [width_y, width_x]).flatten()

    #true for Gaussian errors, but not really true here because of mixture of Poisson and Gaussian noise
    #lnL = - 0.5 * np.sum((data - model)**2 / var)
    #others...
    #lnL = - 2. * np.sum(data*np.log(((data - model)**2) / var) + np.abs(data - model)) #only peak matters
    #lnL = - 2. * np.sum(np.log(((data - model)**2) / var) + np.abs(data - model))  #works pretty well
    #lnL = - 2. * np.sum(((data - model)**2) / var + np.abs(data - model))  #similar as above, maybe slightly better
    lnL = - 2. * np.sum((((data - model)**2) + np.abs(data - model))/var)  #similar as above, maybe slightly better
    #using L1 norm would be true for exponential distribution
    #lnL = - np.sum(np.abs(data - model) / var)

    return lnL


# def log_likelihoodForwards(theta, x, y, data, var):
#     """
#     This could be used to mimic forward modelling
#     """
#     width_x, width_y, amp, x0, y0, rad = theta
#     size = 21
#
#     lnL = []
#     for dummy in range(10):
#         #1)Generate a model Airy disc
#         airy = models.AiryDisk2D(amp, x0, y0, rad)
#         amodel = airy.eval(x, y, amp, x0, y0, rad).reshape((size, size))
#
#         #2)Convolve with a Gaussian
#         model = ndimage.filters.gaussian_filter(amodel, [width_y, width_y])
#
#         #3)Add Poisson noise
#         rounded = np.rint(model)
#         residual = model.copy() - rounded #ugly workaround for multiple rounding operations...
#         rounded[rounded < 0.0] = 0.0
#         model = np.random.poisson(rounded).astype(np.float64)
#         model += residual
#
#         #4)Add readnoise
#         model += np.random.normal(loc=0.0, scale=4.5, size=model.shape)
#         model = model.flatten()
#
#         lnL.append(- 2. * np.sum((((data - model)**2) + np.abs(data - model))/var))
#
#     #return the median of the log likelihoods
#     return np.median(np.asarray(lnL))


def log_posterior(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, z, var)


# def log_prior(theta, fixHeight=False):
#     """
#     Priors, limit the values to a range but otherwise flat.
#     """
#     if fixHeight:
#         centre_x, centre_y, width_x, width_y, amp, x0, y0, rad = theta
#         if 7. < centre_x < 14. and 7. < centre_y < 14. and 0. < width_x < 3. and 0. < width_y < 3. \
#            and 4.e4 < amp < 1.e6 and 7. < x0 < 14. and 7. < y0 < 14. and 0. < rad < 3.:
#             return 0.
#         else:
#             return -np.inf
#
#     else:
#         height, centre_x, centre_y, width_x, width_y, amp, x0, y0, rad = theta
#         if 0.5 < height < 5. and 8. < centre_x < 12. and 8. < centre_y < 12. and \
#            0.1 < width_x < 1. and 0.1 < width_y < 1. and \
#            1.e4 < amp < 1.e6 and 8. < x0 < 12. and 8. < y0 < 12. and 0.1 < rad < 1.5:
#             return 0.
#         else:
#             return -np.inf
#
#
# def log_likelihood(theta, x, y, data, var, fixHeight=False):
#     """
#     This is not quite right...
#     """
#     if fixHeight:
#         center_x, center_y, width_x, width_y, amp, x0, y0, rad = theta
#         gaussian = models.Gaussian2D(1., center_x, center_y, width_x, width_y, 0.)
#         gdata = gaussian.eval(x, y, 1., center_x, center_y, width_x, width_y, 0.).reshape((21, 21))
#
#     else:
#         height, center_x, center_y, width_x, width_y, amp, x0, y0, rad = theta
#         gaussian = models.Gaussian2D(height, center_x, center_y, width_x, width_y, 0.)
#         gdata = gaussian.eval(x, y, height, center_x, center_y, width_x, width_y, 0.).reshape((21, 21))
#
#     #Airy disc
#     airy = models.AiryDisk2D(amp, x0, y0, rad)
#     adata = airy.eval(x, y, amp, x0, y0, rad).reshape((21, 21))
#
#     #Convolve the Airy disc with the Gaussian broadening kernel to derive the model
#     model = signal.convolve2d(adata, gdata, mode='same').flatten()
#
#     #true for Gaussian errors, but not really true here because of mixture of Poisson and Gaussian noise
#     #lnL = - 0.5 * np.sum((data - model)**2 / var)
#     #others...
#     #lnL = - 2. * np.sum(data*np.log(((data - model)**2) / var) + np.abs(data - model)) #only peak matters
#     #lnL = - 2. * np.sum(np.log(((data - model)**2) / var) + np.abs(data - model))  #works pretty well
#     #lnL = - 2. * np.sum(((data - model)**2) / var + np.abs(data - model))  #similar as above, maybe slightly better
#     lnL = - 2. * np.sum((((data - model)**2) + np.abs(data - model))/var)  #similar as above, maybe slightly better
#     #using L1 norm would be true for exponential distribution
#     #lnL = - np.sum(np.abs(data - model) / var)
#
#     return lnL


def printResults(best_params, errors):
    """
    Print output
    """
    print("=" * 60)
    print('Fitting with MCMC:')
    #pars = ['height', 'xcentre', 'ycentre', 'sigmax', 'sigmay', 'amplitude', 'x0', 'y0', 'radius']
    pars = ['sigmax', 'sigmay', 'amplitude', 'x0', 'y0', 'radius']
    print('*'*20 + ' Fitted parameters ' + '*'*20)
    for name, value, sig in zip(pars, best_params, errors):
        print("{:s} = {:e} +- {:e}" .format(name, value, sig))
    print("=" * 60)


def testSimpleFit(file='16_12_39sEuclid.fits', gain=3.1, size=10, A=True):
    #get data and convert to electrons
    data = pf.getdata(file)*gain

    #maximum position
    y, x = m.maximum_position(data)

    #spot
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()

    #bias estimate
    bias = np.median(data[y-size: y+size, x-100:x-20])

    #remove bias
    spot -= bias

    #save to file
    fileIO.writeFITS(spot, 'small.fits', int=False)

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

    #fit a simple model
    print 'Least Squares Fitting...'
    gaus = models.Gaussian2D(spot.max(), size, size, x_stddev=0.5, y_stddev=0.5)
    gaus.theta.fixed = True  #fix angle
    airy = models.AiryDisk2D(spot.max(), size, size, 1.)
    if A:
        p_init = airy
    else:
        p_init = gaus
    fit_p = fitting.NonLinearLSQFitter()
    stopy, stopx = spot.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
    p = fit_p(p_init, X, Y, spot)
    print p
    model = p(X, Y)

    fileIO.writeFITS(model, 'SimpleModel.fits', int=False)
    fileIO.writeFITS(model - spot, 'SimpleModelResidual.fits', int=False)

    #make a copy ot generate error array
    data = spot.copy().flatten()
    #remove negatives
    data[data < 0.] = 0.
    #assume errors scale as sqrt of the values
    sigma = np.sqrt(data)
    sigma[sigma < 1.] = 1.
    #sigma = np.ones_like(data) # no scaling with errors...

    #goodness of fit
    gof = (1./(len(data)-5.)) * np.sum((model.flatten() - data)**2 / sigma)
    print 'GoF:', gof
    print 'Done'


def test(file='15_42_46sEuclid.fits', out='Data', gain=3.1, size=10, burn=1000):
    """
    A single file to quickly test if the method works
    """
    #get data and convert to electrons
    data = pf.getdata(file)*gain

    #maximum position
    y, x = m.maximum_position(data)

    #spot
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()

    #bias estimate
    bias = np.median(data[y-size: y+size, x-100:x-20])
    print 'ADC offset:', bias

    #remove bias
    spot -= bias

    #save to file
    fileIO.writeFITS(spot, out+'small.fits', int=False)

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

    #fit a simple model
    print 'Least Squares Fitting...'
    gaus = models.Gaussian2D(spot.max(), size, size, x_stddev=0.5, y_stddev=0.5)
    gaus.theta.fixed = True  #fix angle
    p_init = gaus
    fit_p = fitting.NonLinearLSQFitter()
    stopy, stopx = spot.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
    p = fit_p(p_init, X, Y, spot)
    print p
    model = p(X, Y)

    fileIO.writeFITS(model, out+'BasicModel.fits', int=False)
    fileIO.writeFITS(model - spot, out+'BasicModelResidual.fits', int=False)

    #make a copy ot generate error array
    data = spot.copy().flatten()
    #assume errors scale as sqrt of the values + readnoise
    sigma = np.sqrt(data + 4.5**2) #readnoise = 4.5

    #goodness of fit
    gof = (1./(len(data)-5.)) * np.sum((model.flatten() - data)**2 / sigma)
    print 'GoF:', gof
    print 'Done'

    #MCMC based fitting
    print 'Bayesian Fitting...'

    # Airy + Gaussian model has 9 free parameters
    ndim = 6
    nwalkers = 900

    # Choose an initial set of positions for the walkers
    p0 = [np.asarray([0.3, 0.3, max*1.5, size, size, 0.6]) + 1e-3*np.random.randn(ndim) for i in xrange(nwalkers)]

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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, sigma], threads=7)

    # Run a burn-in
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, 3000, rstate0=state)

    # Print out the mean acceptance fraction
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]

    #model
    width_x, width_y, amp, x0, y0, rad = params_fit
    airy = models.AiryDisk2D(amp, x0, y0, rad)
    adata = airy.eval(xx, yy, amp, x0, y0, rad).reshape(spot.shape)
    model = ndimage.filters.gaussian_filter(adata, [width_y, width_y])

    #save model
    fileIO.writeFITS(model, out+'model.fits', int=False)

    #residuals
    fileIO.writeFITS(model - spot, out+'residual.fits', int=False)
    fileIO.writeFITS(((model-spot)**2 / sigma.reshape(spot.shape)), out+'residualSQ.fits', int=False)

    #goodness of fit
    gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / sigma)
    print 'GoF:', gof

    #Print the output
    printResults(params_fit, errors_fit)

    print 'Gaussian FWHM (x, y) = ', round(_FWHMGauss(width_x)), round(_FWHMGauss(width_y), 3), ' microns'
    print("=" * 60)

    #plot
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    fig = triangle.corner(samples, labels=['widthX', 'widthY', 'amp', 'x0', 'y0', 'rad'])
    fig.savefig(out+'Triangle.png')


def testFullGaussian(file='15_42_46sEuclid.fits', gain=3.1, size=10, burn=100, single=True):
    """
    A single file to quickly test if the method works
    """
    #get data and convert to electrons
    data = pf.getdata(file)*gain

    #maximum position
    y, x = m.maximum_position(data)

    #spot
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()

    #bias estimate
    bias = np.median(data[y-size: y+size, x-100:x-20])
    print 'ADC offset:', bias

    #remove bias
    spot -= bias

    #save to file
    fileIO.writeFITS(spot, 'small.fits', int=False)

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

    #fit a simple model
    print 'Least Squares Fitting...'
    gaus = models.Gaussian2D(spot.max(), size, size, x_stddev=0.5, y_stddev=0.5)
    gaus.theta.fixed = True  #fix angle
    airy = models.AiryDisk2D(spot.max(), size, size, 1.)
    if single:
        p_init = gaus
    else:
        #p_init = airy.add_model(gaus, mode='s') #combine the models
        p_init = SerialCompositeModel([airy, gaus])
    fit_p = fitting.NonLinearLSQFitter()
    stopy, stopx = spot.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
    p = fit_p(p_init, X, Y, spot)
    print p
    model = p(X, Y)

    fileIO.writeFITS(model, 'BasicModel.fits', int=False)
    fileIO.writeFITS(model - spot, 'BasicModelResidual.fits', int=False)

    #make a copy ot generate error array
    data = spot.copy().flatten()
    #assume errors scale as sqrt of the values + readnoise
    sigma = np.sqrt(data + 4.5**2) #readnoise = 4.5

    #goodness of fit
    gof = (1./(len(data)-5.)) * np.sum((model.flatten() - data)**2 / sigma)
    print 'GoF:', gof
    print 'Done'

    #MCMC based fitting
    print 'Bayesian Fitting...'

    # Airy + Gaussian model has 9 free parameters
    ndim = 9
    #ndim = 8; height = 1.
    nwalkers = 800

    # Choose an initial set of positions for the walkers
    p0 = [np.asarray([1., size, size, 0.3, 0.3, max*1.5, size, size, 0.6]) + 1e-3*np.random.randn(ndim)
          for i in xrange(nwalkers)]
    #p0 = [np.asarray([5, 5, 0.3, 0.3, 5.6e4, 5, 5, 0.6]) + 1e-3*np.random.randn(ndim) for i in xrange(nwalkers)]

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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, data, sigma], threads=7)

    # Run a burn-in
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, burn)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, 1600, rstate0=state)

    # Print out the mean acceptance fraction
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]

    #model
    height, center_x, center_y, width_x, width_y, amp, x0, y0, rad = params_fit
    #center_x, center_y, width_x, width_y, amp, x0, y0, rad = params_fit
    gaussian = models.Gaussian2D(height, center_x, center_y, width_x, width_y, 0.)
    gdata = gaussian.eval(xx, yy, height, center_x, center_y, width_x, width_y, 0.).reshape(spot.shape)
    airy = models.AiryDisk2D(amp, x0, y0, rad)
    adata = airy.eval(xx, yy, amp, x0, y0, rad).reshape(spot.shape)
    model = signal.convolve2d(adata, gdata, mode='same')

    #save model
    fileIO.writeFITS(model, 'model.fits', int=False)

    #residuals
    fileIO.writeFITS(model - spot, 'residual.fits', int=False)
    fileIO.writeFITS(((model-spot)**2 / sigma.reshape(spot.shape)), 'residualSQ.fits', int=False)

    #goodness of fit
    gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / sigma)
    print 'GoF:', gof

    #Print the output
    printResults(params_fit, errors_fit)

    print 'Gaussian FWHM (x, y) = ', round(width_x*2*np.sqrt(2*np.log(2))*12, 3), \
           round(width_y*2*np.sqrt(2*np.log(2))*12, 3), ' microns'
    print("=" * 60)

    #plot
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    fig = triangle.corner(samples, labels=['height', 'X', 'Y', 'widthX', 'widthY', 'amp', 'x0', 'y0', 'rad'])
    fig.savefig('triangle.png')


def testPyMC(file='15_42_46sEuclid.fits', gain=3.1, size=10):
    #get data and convert to electrons
    data = pf.getdata(file)*gain

    #maximum position
    y, x = m.maximum_position(data)

    #spot
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()

    #bias estimate
    bias = np.median(data[y-size: y+size, x-100:x-20])
    print 'ADC offset estimate:', bias

    #remove bias
    spot = spot.copy() - bias
    spot[spot < 0] = 0.
    #assume errors scale as sqrt of the values
    sigma = np.sqrt(data + 4.5**2) #readnoise = 4.5

    #save to file
    fileIO.writeFITS(spot, 'smallPyMC.fits', int=False)

    #maximum value
    max = np.max(spot)
    print 'Maximum Value:', max

    #Create the coordinates x and y
    x = np.arange(0, spot.shape[1])
    y = np.arange(0, spot.shape[0])
    #Put the coordinates in a mesh and flatten
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    #Priors
    height = pymc.Uniform('GaussianHeight', 0.99, 1.01, value=1.)
    centre_x = pymc.Uniform('GaussianCentreX', 0, 2*size, value=size)
    centre_y = pymc.Uniform('GaussianCentreY', 0, 2*size, value=size)
    width_x = pymc.Uniform('GaussianWidthX', 0., 2., value=0.2)
    width_y = pymc.Uniform('GaussianWidthY', 0., 2., value=0.2)
    amp = pymc.Uniform('AiryAmplitude', max*0.8, max*3, value=max)
    x0 = pymc.Uniform('AiryCentreX', 0, 2*size, value=size)
    y0 = pymc.Uniform('AiryCentreY', 0, 2*size, value=size)
    rad = pymc.Uniform('AiryRadius', 0.1, 1., value=0.4)
    #bias = pymc.Uniform('bias', 2000, 4000, value=bias) #this cannot be used with Poisson likelihood function

    #model
    @pymc.deterministic(plot=False, trace=False)
    def model(height=height, centre_x=centre_x, centre_y=centre_y, width_x=width_x, width_y=width_y,
              amp=amp, x0=x0, y0=y0, rad=rad):
        #Gaussian
        gaussian = models.Gaussian2D(height.item(), centre_x.item(), centre_y.item(), width_x.item(), width_y.item(), 0.)
        gdata = gaussian.eval(xx, yy, height.item(), centre_x.item(), centre_y.item(), width_x.item(), width_y.item(), 0.).reshape(spot.shape)

        #Airy disc
        airy = models.AiryDisk2D(amp.item(), x0.item(), y0.item(), rad.item())
        adata = airy.eval(xx, yy, amp.item(), x0.item(), y0.item(), rad.item()).reshape(spot.shape)

        #Convolve the Airy disc with the Gaussian broadening kernel
        model = signal.convolve2d(adata, gdata, mode='same').flatten()

        #add ADC offset level
        #model += bias.item()

        return model

    #likelihood function
    y = pymc.Poisson('y', mu=model, value=spot.flatten(), observed=True, trace=False)
    #y = pymc.Normal('y', mu=model, tau=1./sigma, value=spot.flatten(), observed=True, trace=False)

    #store the model to a dictionary
    d = {'height': height,
         'centre_x': centre_x,
         'centre_y': centre_y,
         'width_x': width_x,
         'width_y': width_y,
         'amp': amp,
         'x0': x0,
         'y0': y0,
         'rad': rad,
         #'bias': bias,
         'f': model,
         'y': y}

    R = pymc.MCMC(d)

    #good starting position
    print 'Finding the maximum a-posterior...'
    map_ = pymc.MAP(d)
    map_.fit()#method='fmin_powell'
    print height.value, centre_x.value, centre_y.value, width_x.value, width_y.value, amp.value, x0.value, y0.value, rad.value#, bias.value

    print 'Will start running a chain...'
    R.sample(iter=100000, burn=5000, thin=2)

    #print out summaries
    R.height.summary()
    R.centre_x.summary()
    R.centre_y.summary()
    R.width_x.summary()
    R.width_y.summary()
    R.amp.summary()
    R.x0.summary()
    R.y0.summary()
    R.rad.summary()
    #R.bias.summary()

    #generate plots
    pymc.Matplot.plot(R, common_scale=False)
    pymc.Matplot.summary_plot(R)

    #rename stats
    Rs = R.stats()
    height = Rs['GaussianHeight']['mean']
    center_x = Rs['GaussianCentreX']['mean']
    center_y = Rs['GaussianCentreY']['mean']
    width_x = Rs['GaussianWidthX']['mean']
    width_y = Rs['GaussianWidthY']['mean']
    amp = Rs['AiryAmplitude']['mean']
    x0 = Rs['AiryCentreX']['mean']
    y0 = Rs['AiryCentreY']['mean']
    rad = Rs['AiryRadius']['mean']
    #bias = Rs['bias']['mean']

    #model
    gaussian = models.Gaussian2D(height, center_x, center_y, width_x, width_y, 0.)
    gdata = gaussian.eval(xx, yy, height, center_x, center_y, width_x, width_y, 0.).reshape(spot.shape)
    airy = models.AiryDisk2D(amp, x0, y0, rad)
    adata = airy.eval(xx, yy, amp, x0, y0, rad).reshape(spot.shape)
    model = signal.convolve2d(adata, gdata, mode='same') #+ bias

    #save model
    fileIO.writeFITS(model, 'modelPyMC.fits', int=False)

    #residual
    fileIO.writeFITS(model - spot, 'residualPyMC.fits', int=False)

    #goodness of fit
    gof = (1./(len(spot.flatten())-9.)) * np.sum((model - spot)**2 / sigma)
    print 'GoF:', gof

    print 'Gaussian FWHM (x, y) = ', round(width_x*2*np.sqrt(2*np.log(2))*12, 3), \
           round(width_y*2*np.sqrt(2*np.log(2))*12, 3), ' microns'


def _simulate(theta=(3., 3., 2.3e5, 128., 128., 5.), gain=3.1, bias=9000, rn=4.5, size=256,
              out='simulated.fits', Gaussian=True):
    """
    Generate simulated spot image with the assumed process.
    """
    #unpack the parameters
    #height, center_x, center_y, width_x, width_y, amp, x0, y0, rad = theta
    width_x, width_y, amp, x0, y0, rad = theta

    #Create the coordinates x and y
    x = np.arange(0, size)
    y = np.arange(0, size)
    #Put the coordinates in a mesh
    x, y = np.meshgrid(x, y)

    #1)Generate a model Airy disc
    airy = models.AiryDisk2D(amp, x0, y0, rad)
    adata = airy.eval(x, y, amp, x0, y0, rad).reshape((size, size))

    #2)Convolve
    if Gaussian:
        data = ndimage.filters.gaussian_filter(adata, [width_y, width_x])
        #full Gaussian is not really needed if we are interested in the convolution kernel width
        #gaussian = models.Gaussian2D(height, center_x, center_y, width_x, width_y, 0.)
        #gdata = gaussian.eval(x, y, height, center_x, center_y, width_x, width_y, 0.).reshape((size, size))
        #data = signal.convolve2d(adata, gdata, mode='same')
    else:
        #CCD kernel -- high flux
        kernel = np.array([[0.01/4., 0.05, 0.01/4.],
                          [0.075, 0.74, 0.075],
                          [0.01/4., 0.05, 0.01/4.]])
        fileIO.writeFITS(kernel, 'kernel.fits', int=False)
        data = ndimage.convolve(adata, kernel)

    #2)Add Poisson noise
    rounded = np.rint(data)
    residual = data.copy() - rounded #ugly workaround for multiple rounding operations...
    rounded[rounded < 0.0] = 0.0
    data = np.random.poisson(rounded).astype(np.float64)
    data += residual

    #3)Add ADC offset level
    data += bias

    #4)Add readnoise
    data += np.random.normal(loc=0.0, scale=rn, size=data.shape)

    #5)Convert to DNs
    data = np.round(data/gain)

    #save to a file
    fileIO.writeFITS(data, out, int=False)


def _FWHMGauss(sigma, pixel=12):
    return sigma*2*np.sqrt(2*np.log(2))*pixel


def _averageGaussianAiry():
    A = pf.getdata('SimpleModel.fits')
    G = pf.getdata('BasicModel.fits')
    mean = (A+G)/2.
    data = pf.getdata('small.fits')

    fileIO.writeFITS(mean, 'AveragedModel.fits', int=False)
    fileIO.writeFITS(mean-data, 'AveragedModelResidual.fits', int=False)


def _stackFiles(id='15_4*.fits'):
    data = np.asarray([pf.getdata(file) for file in g.glob(id)])
    med = np.median(data, axis=0)
    fileIO.writeFITS(med, 'stacked.fits', int=False)


def RunTestSimulations():
    """
    A set of simulated spots and analysis.
    """
    theta1 = (0.35, 0.3, 2.3e5, 10., 10.2, 0.6)
    theta2 = (0.456, 0.448, 2.3e5, 9.92, 10.1, 0.527) #close to 15_42_46sEuclid.fits
    theta3 = (0.55, 0.45, 2.3e5, 10., 10., 0.55)
    theta4 = (0.30, 0.35, 2.3e5, 10.3, 10., 0.55)
    theta5 = (0.35, 0.30, 2.3e5, 10., 10.3, 0.55)
    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        _simulate(theta=theta, size=21, out='simulatedSmall%i.fits' %i)
        test(file='simulatedSmall%i.fits' %i, out='Run%i' %i)
        print("=" * 60)
        print 'Truth:', theta
        print 'FWHM (x, y) = ', round(_FWHMGauss(theta[0]), 3), round(_FWHMGauss(theta[1]), 3), ' microns'
        print("=" * 60)

    # theta = (0.456, 0.448, 2.55e5, 9.92, 10.1, 0.6) #close to 15_42_46sEuclid.fits
    # _simulate(theta=theta, size=21, out='simulatedSmallKernel.fits', Gaussian=False)
    # test(file='simulatedSmallKernel.fits')


if __name__ == '__main__':
    #Simulated spots and analysis
    RunTestSimulations()

    #MCMC - for real data
    #test()
    #test(file='15_22_00sEuclid.fits', out='Data600nmLow') #low counts, 600nm
    #test(file='stacked.fits', out='DataStack')
    #testPyMC(file='stacked.fits')
