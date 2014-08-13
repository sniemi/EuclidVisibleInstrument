import pymc
import emcee
import numpy as np
import matplotlib.pyplot as plt
from pymc.Matplot import plot
from scipy import optimize
import matplotlib.pyplot as plt


def pymc_linear_fit_withoutliers(data1, data2, data1err=None, data2err=None,
                                 print_results=False, intercept=True, nsample=50000, burn=5000,
                                 thin=5, return_MC=False, guess=None, verbose=0):
    """
    Use pymc to fit a line to data with outliers, assuming outliers
    come from a broad, uniform distribution that cover all the data.

    :param data1: xdata
    :param data2: ydata
    :param data1err: x errors
    :param data2err: y errors
    :param print_results: whether or not to print out the results
    :param intercept: whether or not to fit for intercept
    :param nsample: number of samples
    :param burn: number of burn-in samples
    :param thin: thinnening value
    :param return_MC: whether or not to return the pymc MCMC instance
    :param guess: initial guessues for slope and intercept
    :param verbose: verbosity level of MCMC sampler
    """
    if guess is None:
        guess = (0, 0)

    xmu = pymc.distributions.Uninformative(name='x_observed', value=0)

    if data1err is None:
        xdata = pymc.distributions.Normal('x', mu=xmu, observed=True, value=data1, tau=1, trace=False)
    else:
        xtau = pymc.distributions.Uninformative(name='x_tau', value=1.0 / data1err ** 2, observed=True, trace=False)
        xdata = pymc.distributions.Normal('x', mu=xmu, observed=True, value=data1, tau=xtau, trace=False)

    d = {'slope': pymc.distributions.Uninformative(name='slope', value=guess[0]),
         'badvals': pymc.distributions.DiscreteUniform('bad', 0, 1, value=[False] * len(data2)),
         'bady': pymc.distributions.Uniform('bady', min(data2 - data2err), max(data2 + data2err), value=data2)}

    if intercept:
        d['intercept'] = pymc.distributions.Uninformative(name='intercept', value=guess[1])

        @pymc.deterministic(trace=False)
        def model(x=xdata, slope=d['slope'], intercept=d['intercept'], badvals=d['badvals'], bady=d['bady']):
            return (x * slope + intercept) * (True - badvals) + badvals * bady
    else:
        @pymc.deterministic(trace=False)
        def model(x=xdata, slope=d['slope'], badvals=d['badvals'], bady=d['bady']):
            return x * slope * (True - badvals) + badvals * bady

    d['f'] = model

    if data2err is None:
        ydata = pymc.distributions.Normal('y', mu=model, observed=True, value=data2, tau=1, trace=False)
    else:
        ytau = pymc.distributions.Uninformative(name='y_tau', value=1.0 / data2err ** 2, observed=True, trace=False)
        ydata = pymc.distributions.Normal('y', mu=model, observed=True, value=data2, tau=ytau, trace=False)
    d['y'] = ydata

    MC = pymc.MCMC(d)
    MC.sample(nsample, burn=burn, thin=thin, verbose=verbose)

    MCs = MC.stats()
    m, em = MCs['slope']['mean'], MCs['slope']['standard deviation']

    if intercept:
        b, eb = MCs['intercept']['mean'], MCs['intercept']['standard deviation']

    if print_results:
        print "MCMC Best fit y = %g x" % (m),

        if intercept:
            print " + %g" % (b)
        else:
            print ""
        print "m = %g +/- %g" % (m, em)

        if intercept:
            print "b = %g +/- %g" % (b, eb)
        print "Chi^2 = %g, N = %i" % (((data2 - (data1 * m)) ** 2).sum(), data1.shape[0] - 1)

    if return_MC:
        return MC

    if intercept:
        return m, b
    else:
        return m


def run1():
    #fake data [x, y, yerr, xyerr]
    data = np.array([[201, 592, 61, 9],
                     [244, 401, 25, 4],
                     [47, 583, 58, 11],
                     [287, 402, 15, 7],
                     [203, 495, 21, 5],
                     [58, 173, 15, 9],
                     [210, 479, 27, 4],
                     [202, 504, 14, 4],
                     [198, 510, 30, 11],
                     [158, 416, 16, 7],
                     [165, 393, 14, 5],
                     [201, 442, 25, 5],
                     [157, 317, 52, 5],
                     [131, 311, 16, 6],
                     [166, 400, 34, 6],
                     [160, 337, 31, 5],
                     [186, 423, 42, 9],
                     [125, 334, 26, 8],
                     [218, 533, 16, 6],
                     [146, 344, 22, 5],
                     [150, 300, 23, 10],
                     [270, 620, 40, 15]])

    #rename columns
    xdata, ydata = data[:, 0], data[:, 1]
    xerr, yerr = data[:, 3], data[:, 2]

    #perform MCMC
    MC = pymc_linear_fit_withoutliers(xdata, ydata, data1err=xerr, data2err=yerr, return_MC=True)
    MC.sample(100000, burn=1000, verbose=0)

    #show the results
    fig = plt.figure()

    #plot the confidence levels
    low25 = np.linspace(20,300)*MC.stats()['slope']['quantiles'][2.5] + MC.stats()['intercept']['quantiles'][2.5]
    top97 = np.linspace(20,300)*MC.stats()['slope']['quantiles'][97.5] + MC.stats()['intercept']['quantiles'][97.5]
    plt.fill_between(np.linspace(20,300), low25, top97, color='k', alpha=0.1, label='2.5/97.5 quartile')

    #plot the average results
    plt.plot(np.linspace(20,300), np.linspace(20,300)*MC.stats()['slope']['mean'] + MC.stats()['intercept']['mean'],
             color='k', linewidth=1, label='Average fit')

    #plot data
    plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, color='b', label='data', fmt='o')

    #show likely outliers
    plt.plot(xdata[MC.badvals.value.astype('bool')], ydata[MC.badvals.value.astype('bool')], 'rs',
             label='likely outliers')

    plt.xlim(20, 300)
    plt.legend(shadow=True, fancybox=True, scatterpoints=1, numpoints=1, loc='upper left')
    plt.savefig('test.pdf')
    plt.close()

    #MCMC plot
    plot(MC)


def run2():
    #fake data
    x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
                  40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
    y = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
                  53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
    e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
                   2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])

    def squared_loss(theta, x=x, y=y, e=e):
        dy = y - theta[0] - theta[1] * x
        return np.sum(0.5 * (dy / e) ** 2)

    theta1 = optimize.fmin(squared_loss, [0, 0], disp=False)
    xfit = np.linspace(0, 100)

    def huber_loss(t, c=3):
        return ((abs(t) < c) * 0.5 * t ** 2
                + (abs(t) >= c) * -c * (0.5 * c - abs(t)))

    def total_huber_loss(theta, x=x, y=y, e=e, c=3):
        return huber_loss((y - theta[0] - theta[1] * x) / e, c).sum()

    theta2 = optimize.fmin(total_huber_loss, [0, 0], disp=False)


    # theta will be an array of length 2 + N, where N is the number of points
    # theta[0] is the intercept, theta[1] is the slope,
    # and theta[2 + i] is the weight g_i
    def log_prior(theta):
        #g_i needs to be between 0 and 1
        if (all(theta[2:] > 0) and all(theta[2:] < 1)):
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


    ndim = 2 + len(x)  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 10000  # "burn-in" period to let chains stabilize
    nsteps = 15000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    np.random.seed(0)
    starting_guesses = np.zeros((nwalkers, ndim))
    starting_guesses[:, :2] = np.random.normal(theta1, 1, (nwalkers, 2))
    starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, e, 50])
    sampler.run_mcmc(starting_guesses, nsteps)

    sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
    sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)

    #traces
    #plt.plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1)
    #plt.xlabel('intercept')
    #plt.ylabel('slope')


    theta3 = np.mean(sample[:, :2], 0)
    g = np.mean(sample[:, 2:], 0)
    outliers = (g < 0.5)

    plt.errorbar(x, y, e, fmt='.k', ecolor='gray')
    plt.plot(xfit, theta1[0] + theta1[1] * xfit, color='lightgray')
    plt.plot(xfit, theta2[0] + theta2[1] * xfit, color='lightgray')
    plt.plot(xfit, theta3[0] + theta3[1] * xfit, color='black')
    plt.plot(x[outliers], y[outliers], 'ro', ms=20, mfc='none', mec='red')
    plt.title('Maximum Likelihood fit: Bayesian Marginalization')
    plt.savefig('test2.pdf')
    plt.close()


if __name__ == "__main__":
    #run1()
    run2()