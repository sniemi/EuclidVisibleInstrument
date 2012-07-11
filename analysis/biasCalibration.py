"""
This simple script can be used to study the number of bias frames required for a given
PSF ellipticity knowledge level.

"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
import pyfits as pf
import numpy as np
import math, pprint, datetime
from analysis import shape
from support import logger as lg



def bias(number, shape, level=1000, readnoise=4.5):
    """
    Generates a VIS super bias by median comibining the number of bias frames genereated.

    :param number: number of bias readouts to combine
    :type number: int
    :param shape: shape of the image array in (y, x)
    :type shape: tuple or list
    :param level: bias level to add (default = 1000)
    :type level: int
    :param readnoise: readout electronics noise in electrons
    :type readnoise: float

    :return: median combined bias frame
    :rtype: ndarray
    """
    biases = np.random.normal(loc=0.0, scale=math.sqrt(readnoise), size=(shape[0], shape[1], number)) # + level
    bias = np.median(biases.astype(np.int), axis=2, overwrite_input=True)
    print np.mean(bias)
    return bias.astype(np.int)


def generateplots(ref, values, limit=3):
    """
    Create a simple plot to show the results.
    """
    x = np.arange(len(values)) + 1
    y = np.abs((np.asarray(values) - ref) / ref) * 1e5
    compliant = x[ y < limit]
    pprint.pprint(compliant)

    fig = plt.figure()
    plt.title('VIS Bias Calibration (%s)' % datetime.datetime.isoformat(datetime.datetime.now()))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'bo-')
    ax.plot(x, [limit,]*len(x), 'g:')
    ax.set_ylim(-1e-7, 1e3)

    #ax2 = zoomed_inset_axes(ax, 2.0, loc=1)
    ax2 = inset_axes(ax , width='50%', height=1.1, loc=1)
    ax2.plot(x, y, 'bo-')
    ax2.plot(x, [limit,]*len(x), 'g:')
    ax2.set_xlim(x[-10], x[-1])
    ax2.set_ylim(-1e-7, 50.0)
    #mark_inset(ax, ax2, loc1=1, loc2=2, fc='none', ec='0.5')
    #plt.xticks(visible=False)
    #plt.yticks(visible=False)

    try:
        plt.text(0.5, 0.4,
                 r'At least %i bias frames required for $\Delta e < 3 \times 10^{-5}$' % np.min(compliant),
                 ha='center',
                 va='center',
                 transform=ax.transAxes)
    except:
        pass

    #ax2.set_ylabel(r'$\Delta e \ [10^{-5}]$')
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\Delta e \ [10^{-5}]$')
    plt.savefig('BiasCalibration.pdf')


if __name__ == '__main__':
    number = 100    #number of frames to combine
    sigma = 0.75
    times = 100     #number of samples for a given number of frames to averega
    #sigma = 1.0

    log = lg.setUpLogger('biasCalibration.log')
    log.info('Testing bias level calibration...')
    log.info('Testing with %i bias frames...' % number)

    file = 'psf1x.fits'
    log.info('Processing file %s' % file)

    #download data without noise or bias and scale it to 30k
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 4.0e4

    #derive the reference value
    settings = dict(sigma=sigma)
    sh = shape.shapeMeasurement(data, log, **settings)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    reference = results['ellipticity']

    #loop over the number of bias frames and derive ellipticity
    es = []
    for number in xrange(number):
        tmp = []
        for x in xrange(times):
            biased = data.copy() + bias(number+1, data.shape)
            sh = shape.shapeMeasurement(biased, log, **settings)
            results = sh.measureRefinedEllipticity()
            tmp.append(results['ellipticity'])
        es.append(np.mean(np.asarray(tmp)))
    generateplots(reference, es)


