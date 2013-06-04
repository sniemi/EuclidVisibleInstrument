import pymc
from pymc.Matplot import plot


def pymc_linear_fit_withoutliers(data1, data2, data1err=None, data2err=None,
                                 print_results=True, intercept=True, nsample=50000, burn=5000,
                                 thin=2, return_MC=False, guess=None, verbose=0):
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

    d = {'slope': pymc.distributions.Uninformative(name='slope', value=guess[0], doc='Slope of the straight line'),
         'badvals': pymc.distributions.DiscreteUniform('bad', lower=0, upper=1, value=[False] * len(data2), doc='Outliers'),
         'bady': pymc.distributions.Uniform('bady', np.min(data2 - data2err), np.max(data2 + data2err), value=data2)}

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

    #generate plots
    plot(MC)

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


def outlier_detection():
    """

    :return:
    """
    import numpy as np
    import pylab as pl
    import matplotlib.font_manager
    from scipy import stats

    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope

    # Example settings
    n_samples = 200
    outliers_fraction = 0.25
    clusters_separation = [0, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1),
        "robust covariance estimator": EllipticEnvelope(contamination=.1)}

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = 0

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)
        # Data generation
        X1 = 0.3 * np.random.randn(0.5 * n_inliers, 2) - offset
        X2 = 0.3 * np.random.randn(0.5 * n_inliers, 2) + offset
        X = np.r_[X1, X2]
        # Add outliers
        X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

        # Fit the model with the One-Class SVM
        pl.figure(figsize=(10, 5))
        for i, (clf_name, clf) in enumerate(classifiers.iteritems()):
            # fit the data and tag outliers
            clf.fit(X)
            y_pred = clf.decision_function(X).ravel()
            threshold = stats.scoreatpercentile(y_pred,
                                                100 * outliers_fraction)
            y_pred = y_pred > threshold
            n_errors = (y_pred != ground_truth).sum()
            # plot the levels lines and the points
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            subplot = pl.subplot(1, 2, i + 1)
            subplot.set_title("Outlier detection")
            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                             cmap=pl.cm.Blues_r)
            a = subplot.contour(xx, yy, Z, levels=[threshold],
                                linewidths=2, colors='red')
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')
            b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
            c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
            subplot.axis('tight')
            subplot.legend(
                [a.collections[0], b, c],
                ['learned decision function', 'true inliers', 'true outliers'],
                prop=matplotlib.font_manager.FontProperties(size=11))
            subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
            subplot.set_xlim((-7, 7))
            subplot.set_ylim((-7, 7))
        pl.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)

    pl.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from pymc.Matplot import plot

    #fake data [x, y, yerr, xyerr]
    data = np.array([[201, 592, 61, 9],
                     [244, 401, 25, 10],
                     [47, 583, 258, 11],
                     [287, 402, 85, 27],
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
    MC = pymc_linear_fit_withoutliers(xdata, ydata, data1err=xerr, data2err=yerr, return_MC=True,
                                      nsample=1000000, burn=50000, verbose=1)

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