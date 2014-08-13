"""
Simple example how to use emcee to fit 2D gaussian to noisy data...
"""
from support import files as fileIO
import numpy as np
import emcee
from astropy.modeling import models


def log_prior(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    #limit the values to 0 < theta < 256
    if (all(theta > 0) and all(theta < 256)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf


def log_likelihood(theta, x, y, z, var):
    """
    This is probably not quite right...
    """
    height, center_x, center_y, width_x, width_y = theta
    model = models.Gaussian2D(height, center_x, center_y, width_x, width_y, 0.)
    zz = model.eval(x, y, height, center_x, center_y, width_x, width_y, 0)
    #true for Gaussian case
    chi2 = - 0.5 * np.sum((z - zz)**2 / var)
    return chi2


def log_posterior(theta, x, y, z, var):
    """
    Posterior probability: combines the prior and likelihood.
    """
    return log_prior(theta) + log_likelihood(theta, x, y, z, var)


def fakeData():
    """
    Generate some fake data i.e. a 2D Gaussian with noise.
    """
    #Create the coordinates x and y
    x = np.arange(0, 256)
    y = np.arange(0, 256)
    #Put the coordinates in a mesh
    xx, yy = np.meshgrid(x, y)

    #get Gaussian with fixed params
    model = models.Gaussian2D(50, 123, 135, 20, 35.5)
    zz = model.eval(xx, yy, 50, 123, 135, 20, 35.5, 0)

    #Flatten the arrays
    xx = xx.flatten()
    yy = yy.flatten()
    #add some noise to zz
    zz = zz.flatten() + np.random.normal(0.0, 2., len(xx))
    sigma = np.ones(len(xx))
    return xx, yy, zz, sigma


def printResults(best_params, errors):
    """
    Print output
    """
    print("=" * 60)
    print('Gaussian fitting with MCMC:')
    print("=" * 60)
    pars = ['height', 'xcentre', 'ycentre', 'sigmax', 'sigmay']
    print('*'*15 + ' Fitted parameters ' + '*'*15)
    for name, value, sig in zip(pars, best_params, errors):
        print("{:s} = {:e} +- {:e}" .format(name, value, sig))


if __name__ == '__main__':
    xx, yy, zz, sigma = fakeData()
    #save file
    fileIO.writeFITS(zz.reshape(256, 256), 'Gaussian.fits', int=False)

    # Gaussian with 5 parameters
    ndim = 5

    # We'll sample with 400 walkers
    nwalkers = 400

    # Choose an initial set of positions for the walkers.
    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

    # Initialize the sampler with the chosen specs.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xx, yy, zz, sigma], threads=6)

    # Run a burn-in.
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, 500)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1500
    # steps. (rstate0 is the state of the internal random number generator)
    print "Running MCMC..."
    pos, prob, state = sampler.run_mcmc(pos, 1500, rstate0=state)

    # Print out the mean acceptance fraction.
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    #Get the index with the highest probability
    maxprob_index = np.argmax(prob)

    #Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:,i].std() for i in xrange(ndim)]

    #model
    m = models.Gaussian2D(params_fit[0],params_fit[1],params_fit[2],params_fit[3],params_fit[4], 0.)
    model = m.eval(xx, yy, params_fit[0],params_fit[1],params_fit[2],params_fit[3],params_fit[4], 0).reshape(256, 256)
    fileIO.writeFITS(model, 'model.fits', int=False)

    #residual
    fileIO.writeFITS(zz.reshape(256, 256) - model, 'residual.fits', int=False)

    #goodness of fit
    gof = (1./(len(zz)-ndim)) * np.sum((model.ravel() - zz)**2 / sigma**2)
    print gof

    #Print the output
    printResults(params_fit, errors_fit)
