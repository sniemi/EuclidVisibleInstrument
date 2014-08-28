import pyfits as pf
import numpy as np
import emcee
import scipy.ndimage.measurements as m
from scipy import signal
from support import files as fileIO
from astropy.modeling import models, fitting
import triangle


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
    if 8. < center_x < 12. and 8. < center_y < 12. and 0.25 < width_x < 0.5 and 0.25 < width_y < 0.5 and \
       1.e2 < amplitude < 1.e6 and 0. < radius < 1. and 0. < focus < 1.:
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


def forwardModel(file, CCDPSFmodel='Gaus', out='Data', gain=3.1, size=10, spotx=2888, spoty=3514,
                 burn=10, run=100, nwalkers=1000):
    """
    A single file to quickly test if the method works
    """
    #get data and convert to electrons
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

    fileIO.writeFITS(model, out+'BasicModel.fits', int=False)
    fileIO.writeFITS(model - spot, out+'BasicModelResidual.fits', int=False)

    #goodness of fit
    gof = (1./(len(data)-5.)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof
    print 'Done'

    #MCMC based fitting
    if 'Gaus' in CCDPSFmodel:
        ndim = 7
        print 'Model with a Gaussian CCD PSF, %i dimensions' % ndim

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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorG, args=[xx, yy, data, var], threads=7)

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
        _printResults(params_fit, errors_fit, model=CCDPSFmodel)

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
        fig = triangle.corner(samples, labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y'])
        fig.savefig(out+'Triangle.png')

    elif 'Cross' in CCDPSFmodel:
        ndim = 8
        print 'Model with a Cross CCD PSF, %i dimensions' % ndim

        #amplitude, center_x, center_y, radius, focus, width_x, width_y, width_d = theta
        # Choose an initial set of positions for the walkers using the Gaussian fit
        p0 = [np.asarray([1.5*max,#p.amplitude.value,
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorC, args=[xx, yy, data, var], threads=7)

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
        _printResults(params_fit, errors_fit, model=CCDPSFmodel)

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
        fig = triangle.corner(samples, labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y', 'width_d'])
        fig.savefig(out+'Triangle.png')

    # a simple goodness of fit
    gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof


def _printResults(best_params, errors, model):
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


if __name__ == '__main__':
    file = 'testdata/17_52_04sEuclid.fits' #Gaussian works better
    #file = 'testdata/15_41_20sEuclid.fits' #Gaussian works better
    #file = 'testdata/15_34_34sEuclid.fits' #Gaussian works better

    forwardModel(file=file, out='modeltest/' + file.replace('.fits', '').replace('testdata/', '') + 'G')
    forwardModel(file=file, CCDPSFmodel='Cross', out='modeltest/' + file.replace('.fits', '').replace('testdata/', '') + 'C')
