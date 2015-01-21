"""
Photometric Redshifts
=====================

This scripts shows simple methods to derive photometric redshifts using machine learning.

:requires: pandas
:requires: scikit-learn
:requires: matplotlib

:author: Sami-Matias Niemi (s.niemi@ucl.ac.uk)
:version: 0.1
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
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

    X_train, X_test, y_train, y_test = train_test_split(data_features.values, data_redshifts.values)
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

    X_train, X_test, y_train, y_test = train_test_split(qso_features.values, qso_redshifts.values)

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
        parameters = {'n_estimators': [10, 50, 100, 200, 500, 1000],
                     'max_features': [None, 'auto', 'sqrt'],
                     'max_depth': [None, 5, 10]}
        rf = RandomForestRegressor()
        #note: one can run out of memory if using n_jobs=-1..
        rf_tuned = grid_search.GridSearchCV(rf, parameters, scoring='r2', n_jobs=2, verbose=1, cv=3)
    else:
        #ok for QSO sample:
        rf_tuned = RandomForestRegressor(n_estimators=1000, max_depth=None, max_features='sqrt',
                                         n_jobs=-1, verbose=1)
        #ok for kaggle sample:

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


def plotResults(predicted, expected, output):
    """
    Generate a simple plot demonstrating the results.
    """
    var = metrics.explained_variance_score(expected, predicted)
    mae = metrics.mean_absolute_error(expected, predicted)
    mse = metrics.mean_squared_error(expected, predicted)
    r2 = metrics.r2_score(expected, predicted)

    print output
    print 'Explained variance (best possible score is 1.0, lower values are worse):', var
    print 'Mean Absolute Error (best is 0.0):', mae
    print 'Mean Squred Error (best is 0.0):', mse
    print 'R2 score (best is 1.0):', r2
    print '\n\n\n'

    title = 'Var=%.3f, MAE=%.3f, MSE=%.3f, R2=%.3f' % (var, mae, mse, r2)

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


if __name__ == '__main__':
    #runRandomForestKaggle(search=True)
    #runRandomForestSDSSQSO()
    runRandomForestKaggle()
