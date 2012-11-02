"""

"""
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
import pyfits as pf
import numpy as np
from analysis import shape
from support import logger as lg


def testShapeMeasurementAlgorithm(log, file='data/psf1x.fits', psfs=5000,
                                  sigma=0.75, iterations=4):
    """

    :param log:
    :param file:
    :param psfs:
    :param sigma:
    :param iterations:
    :return:
    """
    #read in PSF and rescale to avoid rounding or truncation errors
    data = pf.getdata(file)
    data /= np.max(data)

    scales = np.random.random_integers(2e2, 2e5, psfs)

    settings = dict(sigma=sigma, iterations=iterations)

    e = []
    R2 = []
    for scale in scales:
        sh = shape.shapeMeasurement(data.copy()*scale, log, **settings)
        results = sh.measureRefinedEllipticity()

        e.append(results['ellipticity'])
        R2.append(results['R2'])

    return np.asarray(e), np.asarray(R2)


def plotDistribution(data, xlabel, output):
    """

    :param data:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=15, color='g', normed=True, log=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PDF')
    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    #start the script
    log = lg.setUpLogger('testShapeMeasurement.log')

    e, R2 = testShapeMeasurementAlgorithm(log)
    plotDistribution(e, 'Ellipticity', 'ellipticity.pdf')
    plotDistribution(R2, r'Size $(R^{2})$', 'size.pdf')
