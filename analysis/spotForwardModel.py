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

:version: 0.7

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


def forwardModel(file, out='Data', gain=3.1, size=10, burn=100, run=2000, simulation=False):
    """
    A single file to quickly test if the method works
    """
    #get data and convert to electrons
    data = pf.getdata(file)*gain

    #maximum position within the full frame
    y, x = m.maximum_position(data)

    #spot and the peak pixel within the spot, this is also the CCD kernel position
    spot = data[y-size:y+size+1, x-size:x+size+1].copy()
    CCDy, CCDx = m.maximum_position(spot)

    #bias estimate
    if simulation:
        bias = 9000.
        rn = 4.5
    else:
        bias = np.median(data[y-size: y+size, x-100:x-20]) #works for read data
        rn = np.std(data[y-size: y+size, x-100:x-20])

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
    fit_p = fitting.NonLinearLSQFitter()
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
    print 'Bayesian Fitting...'
    ndim = 7
    nwalkers = 1000

    #amplitude, center_x, center_y, radius, focus, width_x, width_y = theta
    # Choose an initial set of positions for the walkers using the Gaussian fit
    p0 = [np.asarray([1.5*max,#p.amplitude.value,
                      p.x_mean.value,
                      p.y_mean.value,
                      np.max([p.x_stddev.value, p.y_stddev.value]),
                      0.3,
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
    fileIO.writeFITS(((model-spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)

    # a simple goodness of fit
    gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / var)
    print 'GoF:', gof

    #results
    _printFWHM(width_x, width_y, errors_fit[5], errors_fit[6])

    #plot
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    fig = triangle.corner(samples, labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y'])
    fig.savefig(out+'Triangle.png')

    return width_x, width_y, errors_fit[5], errors_fit[6]


def forwardModelJointFit(files, out='Data', gain=3.1, size=10, burn=50, run=500):
    """
    A single file to quickly test if the method works
    """
    images = len(files)
    image = []
    noise = []
    for file in files:
        #get data and convert to electrons
        data = pf.getdata(file)*gain

        #maximum position within the full frame
        y, x = m.maximum_position(data)

        #spot and the peak pixel within the spot, this is also the CCD kernel position
        spot = data[y-size:y+size+1, x-size:x+size+1].copy()
        CCDy, CCDx = m.maximum_position(spot)

        bias = np.median(data[y-size: y+size, x-100:x-20]) #works for read data
        rn = np.std(data[y-size: y+size, x-100:x-20])

        print 'Readnoise (e):', rn
        if rn < 2. or rn > 6.:
            print 'NOTE: suspicious readout noise estimate...'
        print 'ADC offset (e):', bias

        #remove bias
        spot -= bias

        max = np.max(spot)
        print 'Maximum Value:', max

        #noise model
        variance = spot.copy() + rn**2

        #save to a list
        image.append(spot)
        noise.append(variance)


    #MCMC based fitting
    ndim = 2*images + 5  #xpos, ypos for each image and single amplitude, radius, focus, and sigmaX and sigmaY
    nwalkers = 1000
    print 'Bayesian Fitting, model has %i dimensions' % ndim

    # Choose an initial set of positions for the walkers using the Gaussian fit
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    p0 = [np.asarray((([CCDx, CCDy]*images) +[max, 0.5, 0.5, 0.3, 0.3])) +
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
    #print params_fit
    #_printResults(params_fit, errors_fit)


    # #Best fit model
    # amplitude, center_x, center_y, radius, focus, width_x, width_y = params_fit
    # airy = models.AiryDisk2D(amplitude, center_x, center_y, radius)
    # adata = airy.eval(xx, yy, amplitude, center_x, center_y, radius).reshape(spot.shape)
    # f = models.Gaussian2D(1., center_x, center_y, focus, focus, 0.)
    # focusdata = f.eval(xx, yy, 1., center_x, center_y, focus, focus, 0.).reshape(spot.shape)
    # foc = signal.convolve2d(adata, focusdata, mode='same')
    # CCD = models.Gaussian2D(1., CCDx, CCDy, width_x, width_y, 0.)
    # CCDdata = CCD.eval(xx, yy, 1., CCDx, CCDy, width_x, width_y, 0.).reshape(spot.shape)
    # model = signal.convolve2d(foc, CCDdata, mode='same')
    # #save model
    # fileIO.writeFITS(model, out+'model.fits', int=False)
    #
    # #residuals
    # fileIO.writeFITS(model - spot, out+'residual.fits', int=False)
    # fileIO.writeFITS(((model-spot)**2 / var.reshape(spot.shape)), out+'residualSQ.fits', int=False)
    #
    # # a simple goodness of fit
    # gof = (1./(len(data)-ndim)) * np.sum((model.flatten() - data)**2 / var)
    # print 'GoF:', gof
    #
    # #results
    # _printFWHM(width_x, width_y, errors_fit[5], errors_fit[6])
    #
    # #plot
    # samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # fig = triangle.corner(samples, labels=['amplitude', 'center_x', 'center_y', 'radius', 'focus', 'width_x', 'width_y'])
    # fig.savefig(out+'Triangle.png')
    #
    # return width_x, width_y, errors_fit[5], errors_fit[6]



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
    if 8. < center_x < 12. and 8. < center_y < 12. and 0.1 < width_x < 0.5 and 0.1 < width_y < 0.5 and \
       1.e2 < amplitude < 1.e6 and 0. < radius < 1. and 0. < focus < 1.:
        return 0.
    else:
        return -np.inf


def log_priorJoint(theta):
    """
    Priors, limit the values to a range but otherwise flat.
    """
    #[xpos, ypos]*images) +[amplitude, radius, focus, sigmaX, sigmaY])
    #tmp = theta[:-5]
    # if 8. < all(tmp[::2]) < 12. and 8. < all(tmp[1::2]) < 12. and \
    #    0.1 < theta[-2] < 0.5 and 0.1 < theta[-1] < 0.5 and \
    #    1.e2 < theta[-5] < 1.e6 and 0. < theta[-4] < 1. and 0. < theta[-3] < 1.:
    #     return 0.
    if 3. < theta[0] < 16. and 3. < theta[1] < 16. and \
       3. < theta[2] < 16. and 3. < theta[3] < 16. and \
        3. < theta[4] < 16. and 3. < theta[5] < 16. and \
        3. < theta[6] < 16. and 3. < theta[7] < 16. and \
        3. < theta[8] < 16. and 3. < theta[9] < 16. and \
       0. < theta[13] < 2. and 0. < theta[14] < 2. and \
       1.e2 < theta[10] < 1.e6 and 0. < theta[11] < 2. and 0. < theta[12] < 2.:
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
    Logarithm of the likelihood function.
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

        lnL += - 0.5 * np.sum((data[tmp] - model)**2 / var[tmp])

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


def RunTestSimulations():
    """
    A set of simulated spots and analysis.
    """
    #different simulation sets
    theta1 = (2.e5, 10., 10.3, 0.45, 0.5, 10., 10., 0.33, 0.35)
    theta2 = (1.e5, 10., 10.1, 0.55, 0.45, 10., 10., 0.38, 0.36)
    theta3 = (8.e4, 10., 10.2, 0.4, 0.55, 10., 10., 0.25, 0.35)
    theta4 = (5.e4, 10., 10.3, 0.42, 0.48, 10., 10., 0.30, 0.28)
    theta5 = (1.e5, 10., 10.2, 0.5, 0.45, 10., 10., 0.35, 0.31)
    thetas = [theta1, theta2, theta3, theta4, theta5]

    for i, theta in enumerate(thetas):
        _simulate(theta=theta, out='simulated/simulatedSmall%i.fits' %i)
        forwardModel(file='simulated/simulatedSmall%i.fits' %i, out='results/Run%i' %i, simulation=True)
        print("=" * 60)
        print 'Simulation Parameters'
        print 'amplitude, center_x, center_y, radius, focus, width_x, width_y'
        print theta[0], theta[1], theta[2], theta[3], theta[4], theta[7], theta[8]
        print("=" * 60)


def RunTestData(files, out='testdata'):
    """
    A set of test data to analyse.
    """
    widthx = []
    widthxerr = []
    widthy = []
    widthyerr = []
    for i, file in enumerate(files):
        print 'Processing:', file
        wx, wy, wxe, wye = forwardModel(file=file, out='results/%s%i' % (out, i))
        widthx.append(wx)
        widthxerr.append(wxe)
        widthy.append(wy)
        widthyerr.append(wye)

    wx = _FWHMGauss(np.asarray(widthx))
    wxe = _FWHMGauss(np.asarray(widthxerr))
    wy = _FWHMGauss(np.asarray(widthy))
    wye = _FWHMGauss(np.asarray(widthyerr))
    res = dict(wx=wx, wy=wy, wxerr=wxe, wyerr=wye, files=files)
    fileIO.cPickleDumpDictionary(res, out+'.pkl')
    testDataPlot(out+'.pkl', out=out)


def testDataPlot(file='testData.pkl', out='test'):
    """
    A simple plot to show test results.
    """
    data = fileIO.cPicleRead(file)
    xtmp = np.arange(len(data['files'])) + 1

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0, top=0.95, bottom=0.15, left=0.12, right=0.93)

    ax1.errorbar(xtmp, data['wx'], yerr=data['wxerr'], fmt='o')
    ax2.errorbar(xtmp, data['wy'], yerr=data['wyerr'], fmt='o')
    ax1.axhline(y=10.8, label='Requirement', c='r')
    ax2.axhline(y=10.8, label='Requirement', c='r')

    plt.sca(ax1)
    plt.xticks(visible=False)
    plt.sca(ax2)
    plt.xticks(xtmp, [x.replace('.fits', '').replace('Euclid', '').replace('testdata/', '') for x in data['files']],
               rotation=45)

    ax1.set_ylim(0.2, 15)
    ax2.set_ylim(0.2, 15)
    ax1.set_xlim(xtmp.min()*0.9, xtmp.max()*1.05)
    ax2.set_xlim(xtmp.min()*0.9, xtmp.max()*1.05)

    ax1.set_ylabel('X FWHM [microns]')
    ax2.set_ylabel('Y FWHM [microns]')
    ax1.legend(shadow=True, fancybox=True)
    plt.savefig(out+'.pdf')
    plt.close()


if __name__ == '__main__':
    #MCMC - test data set
    RunTestData(g.glob('testdata/17*.fits'), out='test700nm')
    #RunTestData(g.glob('testdata/15*.fits'), out='test800nm')

    #Joint Fit
    #forwardModelJointFit(g.glob('testdata/15*.fits'))

    #Simulated spots and analysis
    #RunTestSimulations()
