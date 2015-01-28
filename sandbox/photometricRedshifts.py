"""
Photometric Redshifts
=====================

This scripts shows simple methods to derive photometric redshifts using machine learning.

:requires: pandas
:requires: numpy
:requires: scikit-learn
:requires: matplotlib

:author: Sami-Matias Niemi (s.niemi@ucl.ac.uk)
:version: 0.6
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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import copy


def loadKaggledata(folder='/Users/sammy/Google Drive/MachineLearning/photo-z/kaggleData/',
                   useErrors=True):
    """
    Load Kaggle photometric redshift competition data. These data are from 2012 and at low-z.

    train: ID, u, g, r, i, z, uErr, gErr, rErr, iErr, zErr, redshift
    query: ID, u, g, r, i, z, uErr, gErr, rErr, iErr, zErr
    solution: ID, redshift, estimatedRedshiftError
    """
    filename = folder + 'train.csv'
    data = pd.read_csv(filename, index_col=0, usecols=['ID', 'u', 'g', 'r', 'i', 'z',
                                                       'modelmagerr_u', 'modelmagerr_g',
                                                       'modelmagerr_r', 'modelmagerr_i',
                                                       'modelmagerr_z', 'redshift'])
    if useErrors:
        data_features = data[['u', 'g', 'r', 'i', 'z',
                              'modelmagerr_u', 'modelmagerr_g',
                              'modelmagerr_r', 'modelmagerr_i',
                              'modelmagerr_z']]
    else:
        data_features = data[['u', 'g', 'r', 'i', 'z']]
    data_redshifts = data[['redshift']]

    X_train, X_test, y_train, y_test = train_test_split(data_features.values, data_redshifts.values,
                                                        random_state=42)
    y_test = y_test.ravel()
    y_train = y_train.ravel()

    print "feature vector shape=", data_features.values.shape
    print 'Training sample shape=', X_train.shape
    print 'Testing sample shape=', X_test.shape
    print 'Target training redshift sample shape=', y_train.shape
    print 'Testing redshift sample shape=',  y_test.shape

    return X_train, X_test, y_train, y_test



def loadSDSSdata(folder='/Users/sammy/Google Drive/MachineLearning/AstroSDSS/', filename="qso10000.csv",
                 plot=False):
    """
    Load SDSS QSO data. The redshift range is rather broard from about 0.3 to 6.
    """
    filename = folder + filename
    qsos = pd.read_csv(filename,index_col=0, usecols=["objid","dered_r","spec_z","u_g_color",
                                                      "g_r_color","r_i_color","i_z_color","diff_u",
                                                      "diff_g1","diff_i","diff_z"])

    qsos = qsos[(qsos["dered_r"] > -9999) & (qsos["g_r_color"] > -10) & (qsos["g_r_color"] < 10)]
    qso_features = copy.copy(qsos)
    qso_redshifts = qsos["spec_z"]
    del qso_features["spec_z"]

    if plot:
        ## truncate the color at z=2.5 just to keep some contrast.
        norm = mpl.colors.Normalize(vmin=min(qso_redshifts.values), vmax=2.5)
        cmap = cm.jet_r
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        pd.scatter_matrix(qso_features[0:2000], alpha=0.2, figsize=[15, 15],
                          color=m.to_rgba(qso_redshifts.values))
        plt.savefig('Sample.pdf')
        plt.close()

    X_train, X_test, y_train, y_test = train_test_split(qso_features.values, qso_redshifts.values,
                                                        random_state=42)

    print "feature vector shape=", qso_features.values.shape
    print 'Training sample shape=', X_train.shape
    print 'Testing sample shape=', X_test.shape

    return X_train, X_test, y_train, y_test


def randomForest(X_train, X_test, y_train, y_test, search=True):
    """
    A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying decision
    trees on various sub-samples of the dataset and use averaging to improve the
    predictive accuracy and control over-fitting.

    Runs a grid search to look for the best parameters. For the test case, these
    were found to be the best:

    """
    if search:
        # parameter values over which we will search
        parameters = {'min_samples_split': [2, 8, 15],
                      'min_samples_leaf': [1, 3, 10],
                     'max_features': [None, 'auto', 'sqrt'],
                     'max_depth': [None, 5, 10]}
        rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, verbose=1)
        #note: one can run out of memory if using n_jobs=-1..
        rf_tuned = grid_search.GridSearchCV(rf, parameters, scoring='r2', n_jobs=2, verbose=1, cv=3)
    else:
        rf_tuned = RandomForestRegressor(n_estimators=5000, max_depth=None,
                                         max_features='sqrt',
                                         min_samples_split=5, min_samples_leaf=3,
                                         n_jobs=-1, verbose=1)
       #n_estimators=5000 will take about 36GB of RAM

    print '\nTraining...'
    rf_optimised = rf_tuned.fit(X_train, y=y_train)
    print 'Done'

    if search:
        print 'The best score and estimator:'
        print(rf_optimised.best_score_)
        print(rf_optimised.best_estimator_)

    print '\nPredicting...'
    predicted = rf_optimised.predict(X_test)
    expected = y_test.copy()
    print 'Done'

    return predicted, expected
    
    
def SupportVectorRegression(X_train, X_test, y_train, y_test, search):
    """
    Support Vector Regression.
    """
    if search:
        # parameter values over which we will search
        parameters = {'C': [0.1, 0.5, 1., 1.5, 2.],
                     'kernel': ['rbf', 'sigmoid', 'poly'],
                     'degree': [3, 5]}
        s = SVR()
        clf = grid_search.GridSearchCV(s, parameters, scoring='r2',
                                       n_jobs=-1, verbose=1, cv=3)
    else:
        clf = SVR(verbose=1)
    
    print '\nTraining...'
    clf.fit(X_train, y_train)
    print 'Done'

    if search:
        print 'The best score and estimator:'
        print(clf.best_score_)
        print(clf.best_estimator_)
        print 'Best hyperparameters:'
        print clf.best_params_
    
    print '\nPredicting...'
    predicted = clf.predict(X_test)
    expected = y_test.copy()    
    print 'Done'

    return predicted, expected    
 
    
def BayesianRidge(X_train, X_test, y_train, y_test, search=True):
    """
    """
    print '\nTraining...'
    clf = linear_model.BayesianRidge(n_iter=1000, tol=1e-3, alpha_1=1., 
                                     fit_intercept=True, normalize=False, verbose=1)
    clf.fit(X_train, y_train)
    print 'Done'
    
    print '\nPredicting...'
    predicted = clf.predict(X_test)
    expected = y_test.copy()    
    print 'Done'

    return predicted, expected    


def GradientBoostingRegressor(X_train, X_test, y_train, y_test, search):
    """
    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.
    
     Among the most important hyperparameters for GBRT are:

        #. number of regression trees (n_estimators)
        #. depth of each individual tree (max_depth)
        #. loss function (loss)
        #. learning rate (learning_rate)
    """
    if search:
        # parameter values over which we will search
        parameters = {'loss': ['ls', 'huber'],
                     'learning_rate': [0.01, 0.2, 0.4, 0.6, 1.0],
                     'max_depth': [3, 5, 7, 9],
                     'max_features': ['auto', None]}
        s = GBR(n_estimators=1000, verbose=1)
        #note: one can run out of memory if using n_jobs=-1..
        clf = grid_search.GridSearchCV(s, parameters, scoring='r2',
                                       n_jobs=-1, verbose=1, cv=3)
    else:
        clf = GBR(verbose=1, n_estimators=5000, min_samples_leaf=3,
                  learning_rate=0.05, max_features='auto', loss='huber',
                  max_depth=7)
        
    print '\nTraining...'    
    clf.fit(X_train, y_train)
    print 'Done'

    if search:
        print 'The best score and estimator:'
        print(clf.best_score_)
        print(clf.best_estimator_)
        print 'Best hyperparameters:'
        print clf.best_params_
    
    print '\nPredicting...'
    predicted = clf.predict(X_test)
    expected = y_test.copy()    
    print 'Done'

    return predicted, expected    


def GradientBoostingRegressorTestPlots(X_train, X_test, y_train, y_test, n_estimators=100):
    """
    An important diagnostic when using GBRT in practise is the so-called deviance
    plot that shows the training/testing error (or deviance) as a function of the
    number of trees.
    """
    def fmt_params(params):
        return ", ".join("{0}={1}".format(key, val) for key, val in params.iteritems())
        
    def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', 
                      test_color='#d7191c', alpha=1.0):
        """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
        test_dev = np.empty(n_estimators)
    
        for i, pred in enumerate(est.staged_predict(X_test)):
           test_dev[i] = est.loss_(y_test, pred)
    
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()
            
        ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label, 
                 linewidth=2, alpha=alpha)
        ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, 
                 label='Train %s' % label, linewidth=2, alpha=alpha)
        ax.set_ylabel('Error')
        ax.set_xlabel('n_estimators')
        return test_dev, ax

    est = GBR(n_estimators=n_estimators, verbose=1)
    est.fit(X_train, y_train)
    
    test_dev, ax = deviance_plot(est, X_test, y_test)
    ax.legend(loc='upper right')
    ax.annotate('Lowest test error', xy=(test_dev.argmin() + 1, test_dev.min() + 0.02), xycoords='data',
                xytext=(150, 1.0), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
    plt.savefig('GBRdeviance.pdf')
    plt.close()
        
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for params, (test_color, train_color) in [({'min_samples_leaf': 1},
                                                ('#d7191c', '#2c7bb6')),
                                              ({'min_samples_leaf': 4},
                                               ('#fdae61', '#abd9e9'))]:
        est = GBR(n_estimators=n_estimators, verbose=1)
        est.set_params(**params)
        est.fit(X_train, y_train)
        
        test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                                     train_color=train_color, test_color=test_color)
    plt.legend(loc='upper right')
    plt.savefig('GBRTree.pdf')
    plt.close()
    
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for params, (test_color, train_color) in [({'learning_rate': 0.5},
                                                ('#d7191c', '#2c7bb6')),
                                              ({'learning_rate': 1.},
                                               ('#fdae61', '#abd9e9'))]:
        est = GBR(n_estimators=n_estimators, verbose=1)
        est.set_params(**params)
        est.fit(X_train, y_train)
        
        test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                                     train_color=train_color, test_color=test_color)
    plt.legend(loc='upper right')
    plt.savefig('GBRShrinkage.pdf')
    plt.close()


def plotResults(predicted, expected, output):
    """
    Generate a simple plot demonstrating the results.
    """
    var = metrics.explained_variance_score(expected, predicted)
    mae = metrics.mean_absolute_error(expected, predicted)
    mse = metrics.mean_squared_error(expected, predicted)
    r2 = metrics.r2_score(expected, predicted)
    rms = np.sqrt(np.mean((expected - predicted) ** 2))

    print output
    print 'Explained variance (best possible score is 1.0, lower values are worse):', var
    print 'Mean Absolute Error (best is 0.0):', mae
    print 'Mean Squred Error (best is 0.0):', mse
    print 'R2 score (best is 1.0):', r2
    print 'RMS:', rms
    print '\n\n\n'

    title = 'RMS=%.4f, MSE=%.4f, R2=%.3f' % (rms, mse, r2)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title(title)
    ax1.scatter(expected, predicted, alpha=0.2, s=5)
    ax1.set_xlabel("Spectroscopic Redshift")
    ax1.set_ylabel("Photo-z")
    ax1.plot([0, 8], [0, 8], '-r')
    ax1.set_xlim(0, 1.1*expected.max())
    ax1.set_ylim(0, 1.1*expected.max())
    plt.savefig(output+'Results.pdf')
    plt.close()


def runRandomForestSDSSQSO(search=False):
    """
    Simple Random Forest on SDSSQSOs
    """
    X_train, X_test, y_train, y_test = loadSDSSdata()
    predictedRF, expectedRF = randomForest(X_train, X_test, y_train, y_test, search=search)
    plotResults(predictedRF, expectedRF, output='RandomForestSDSSQSOs')


def runRandomForestKaggle(useErrors=True, search=False):
    """
    Simple Random Forest on Kaggle training data.
    """
    X_train, X_test, y_train, y_test = loadKaggledata(useErrors=useErrors)
    predictedRF, expectedRF = randomForest(X_train, X_test, y_train, y_test, search=search)
    plotResults(predictedRF, expectedRF, output='RandomForestKaggleErrors')


def runBayesianRidgeKaggle(useErrors=True):
    X_train, X_test, y_train, y_test = loadKaggledata(useErrors=useErrors)
    predicted, expected = BayesianRidge(X_train, X_test, y_train, y_test)
    plotResults(predicted, expected, output='BayesianRidgeKaggleErrors')
    
    
def runSupportVectorRegression(useErrors=False, search=False):
    """
    Really slow...
    """
    X_train, X_test, y_train, y_test = loadKaggledata(useErrors=useErrors)
    predicted, expected = SupportVectorRegression(X_train, X_test, y_train, y_test, search)
    plotResults(predicted, expected, output='SVRKaggleErrors')    


def runGradientBoostingRegressor(useErrors=True, search=False):
    """
    Run Gradient Boosting on Kaggle training data.
    """
    X_train, X_test, y_train, y_test = loadKaggledata(useErrors=useErrors)
    GradientBoostingRegressorTestPlots(X_train, X_test, y_train, y_test)
    predicted, expected = GradientBoostingRegressor(X_train, X_test, y_train, y_test, search)
    plotResults(predicted, expected, output='GBRKaggleErrors')    

    
if __name__ == '__main__':
    #runRandomForestKaggle(search=True)
    #runRandomForestSDSSQSO()
    runRandomForestKaggle()
    #runBayesianRidgeKaggle()
    #runSupportVectorRegression()
    #runGradientBoostingRegressor()