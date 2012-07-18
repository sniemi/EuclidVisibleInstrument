"""
This simple script can be used to study the number of bias frames required for a given
PSF ellipticity knowledge level.

:requires: PyFITS
:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
#import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
import pyfits as pf
import numpy as np
import math, pprint, datetime, cPickle, itertools
from analysis import shape
from support import logger as lg


def bias(number, shape, readnoise=4.5, level=0):
    """
    Generates a VIS super bias by median comibining the number of bias frames genereated.

    :param number: number of bias readouts to combine
    :type number: int
    :param shape: shape of the image array in (y, x)
    :type shape: tuple or list
    :param readnoise: readout electronics noise in electrons
    :type readnoise: float
    :param level: bias level to add (default = 0)
    :type level: int

    :return: median combined bias frame
    :rtype: ndarray
    """
    biases = np.random.normal(loc=0.0, scale=math.sqrt(readnoise), size=(shape[0], shape[1], number)) + level
    bias = np.median(biases.astype(np.int), axis=2, overwrite_input=True)
    return bias.astype(np.int)


def generateplots(ref, values, limit=3):
    """
    Create a simple plot to show the results.
    """
    x = np.arange(len(values[0])) + 1
    y1 = np.abs((np.asarray(values[0]) - ref[0])) * 1e5
    y2 = np.abs((np.asarray(values[1]) - ref[1])) * 1e5
    compliant = x[(y1 < limit) & (y2 < limit)]
    pprint.pprint(compliant)

    fig = plt.figure()
    plt.title('VIS Bias Calibration (%s)' % datetime.datetime.isoformat(datetime.datetime.now()))

    #part one
    ax = fig.add_subplot(111)
    l1, = ax.plot(x, y1, 'bo-')
    l2, = ax.plot(x, y2, 'rs-')
    ax.fill_between(x, np.ones(len(x))*limit, 100, facecolor='red', alpha=0.08)
    r, = ax.plot(x, [limit,]*len(x), 'g--')
    ax.set_ylim(-1e-7, 50.0)
    ax.set_xlim(1, np.max(x))

    plt.legend((l1, l2, r), (r'$e_{1}$', r'$e_{2}$', 'requirement'),
               shadow=True, fancybox=True, loc='upper left')

    #inset
    #ax2 = zoomed_inset_axes(ax, 2.0, loc=1)
    ax2 = inset_axes(ax , width='50%', height=1.1, loc=1)
    ax2.plot(x, y1, 'bo-')
    ax2.plot(x, y2, 'rs-')
    ax2.fill_between(x, np.ones(len(x))*limit, 100, facecolor='red', alpha=0.08)
    ax2.plot(x, [limit,]*len(x), 'g--')
    ax2.set_xlim(x[-10], x[-1])
    ax2.set_ylim(-0.1, 10.0)
    #mark_inset(ax, ax2, loc1=1, loc2=2, fc='none', ec='0.5')
    #plt.xticks(visible=False)
    #plt.yticks(visible=False)

    try:
        plt.text(0.5, 0.4,
                 r'At least %i bias frames required for $\Delta e_{1,2}  < 3 \times 10^{-5}$' % np.min(compliant),
                 ha='center',
                 va='center',
                 transform=ax.transAxes)
    except:
        pass

    #ax2.set_ylabel(r'$\Delta e \ [10^{-5}]$')
    ax.set_xlabel('Number of Bias Frames Median Combined')
    ax.set_ylabel(r'$\Delta e_{i}\ , \ \ \ i \in [1,2] \ \ \ \ [10^{-5}]$')

    plt.savefig('BiasCalibration.pdf')


def cPickleDumpDictionary(dictionary, output):
    """
    Dumps a dictionary of data to a cPickled file.

    :param dictionary: a Python data container does not have to be a dictionary
    :param output: name of the output file

    :return: None
    """
    out = open(output, 'wb')
    cPickle.dump(dictionary, out)
    out.close()


def singlePSFtest(log):
    """
    Runs a test with a single PSF.
    """
    #variables to be changed
    numbers = 50            #number of frames to combine
    sigma = 0.75            #size of the Gaussian weighting function [default = 0.75]
    times = 2              #number of samples for a given number of frames to average
    file = 'psf1x.fits'     #input file to use for the PSF

    log.info('Testing with %i bias frames...' % numbers)
    log.info('Processing file %s' % file)

    #read in data without noise or bias level and scale it to 10k ADUs (35k electrons)
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 3.5e4

    #derive the reference value from the scaled data
    settings = dict(sigma=sigma)
    sh = shape.shapeMeasurement(data, log, **settings)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    reference1 = results['e1']
    reference2 = results['e2']

    #loop over the number of bias frames and derive ellipticity
    e1s = []
    e2s = []
    for number in xrange(numbers):
        print '%i / %i' % (number+1, numbers)
        tmp1 = []
        tmp2 = []
        for x in xrange(times):
            biased = data.copy() + bias(number+1, data.shape)
            sh = shape.shapeMeasurement(biased, log, **settings)
            results = sh.measureRefinedEllipticity()
            tmp1.append(results['e1'])
            tmp2.append(results['e2'])
        e1s.append(np.mean(np.asarray(tmp1)))
        e2s.append(np.mean(np.asarray(tmp2)))

    #save output
    out = dict(reference=[reference1, reference2], ellipticities=[e1s, e2s])
    cPickleDumpDictionary(out, 'data.pk')

    #generate a plot
    generateplots([reference1, reference2], [e1s, e2s])


def generateSurface(x, y):
    tmpx = x / 1000.
    tmpy = y / 1000.
    #tmpz = tmpx + 1.05*tmpx**2 + 0.65*tmpy**2 + 0.7*tmpx**3 + 0.5*tmpy**3 + 0.4*tmpx**2*tmpy + 0.75*tmpx*tmpy + 990
    tmpz = tmpx + 1.05*tmpx**2 + 0.65*tmpy**2 + 0.7*tmpx**3 + 0.5*tmpy**3    + 990
    return tmpz.astype(np.int)


def addReadoutNoise(data, readnoise=4.5, number=1):
    """
    Add readout noise to the input data. The readout noise is the median of the number of frames.

    :param data: input data to which the readout noise will be added to
    :type data: ndarray
    :param readnoise: standard deviation of the read out noise [electrons]
    :type readnoise: float
    :param number: number of read outs to median combine before adding to the data
    :type number: int

    :return: data + read out noise
    :rtype: ndarray [same as input data]
    """
    shape = data.shape
    biases = np.random.normal(loc=0.0, scale=math.sqrt(readnoise), size=(shape[0], shape[1], number))
    bias = np.median(biases.astype(np.int), axis=2, overwrite_input=True)
    return data + bias


def generate3Dplot(X, Y, Z, output, zoom=None):
    """
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    if zoom is not None:
        tmp = np.log10(Z[zoom[2]:zoom[3], zoom[0]:zoom[1]])
        ax.plot_surface(X[zoom[2]:zoom[3], zoom[0]:zoom[1]],
                        Y[zoom[2]:zoom[3], zoom[0]:zoom[1]],
                        tmp, rstride=1, cstride=1)
        ax.set_zlim(tmp.min(), tmp.max())
        ax.set_zlabel(r'$\log_{10}(ADUs)$')
    else:
        ax.plot_wireframe(X, Y, (X*0)+1000.0, rstride=100, cstride=100, color='r')
        ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.5)
        ax.set_zlabel('ADUs')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.savefig(output)
    plt.close()


def surfaceTest(log, xsize=2048, ysize=2066, readout=True):
    """
    Test with a bias surface on a quadrant.
    """
    file = 'psf1x.fits'     #input file to use for the PSF
    sigma = 0.75            #size of the Gaussian weighting function [default = 0.75]

    ysize = xsize

    log.info('Processing file %s' % file)
    #read in data without noise or bias level and scale it to 10k ADUs (35k electrons)
    data = pf.getdata(file)
    data /= np.max(data)
    data *= 3.5e4
    data += 1000

    #derive the reference value from the scaled data
    settings = dict(sigma=sigma)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    results = sh.measureRefinedEllipticity()
    sh.writeFITS(results['GaussianWeighted'], file.replace('.fits', 'Gweighted.fits'))
    reference1 = results['e1']
    reference2 = results['e2']

    #generate a quadrant surface
    xs = np.linspace(0, xsize-1, xsize) + 1
    ys = np.linspace(0, ysize-1, ysize) + 1
    X, Y = np.meshgrid(xs, ys)
    Z = generateSurface(X, Y)
    zm = generateSurface(xs, ys)
    generate3Dplot(X, Y, Z, 'Surface.pdf')

    #add readout noise
    biased = addReadoutNoise(Z.copy())
    generate3Dplot(X, Y, biased, 'SurfaceNoise.png')

    #fit a surface to the noisy data
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(xs, ys, zm)
    ZZ = polyval2d(X, Y, m)
    generate3Dplot(X, Y, ZZ, 'SurfaceFitted.pdf')
    generate3Dplot(X, Y, ZZ/Z, 'SurfaceResidual.pdf')

    #add PSF
    #data -= 1000
    #Z[1000:1000+data.shape[0], 1000:1000+data.shape[1]] += data
    #generate3Dplot(X, Y, Z, 'SurfacePSF.pdf', zoom=[1075, 1095, 1075, 1095])



def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z



if __name__ == '__main__':
    #start the script
    log = lg.setUpLogger('biasCalibration.log')
    log.info('Testing bias level calibration...')

    surfaceTest(log)

    #singlePSFtest(log)