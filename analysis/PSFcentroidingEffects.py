"""
Simple script to test the impact of centroiding on PSF ellipticity and size.

:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.1

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
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
from scipy import ndimage
import cPickle, math
from support import files as fileIO


def testCentroidingImpact(log, psf='/Users/sammy/EUCLID/vissim-python/data/psf12x.fits',
                          xrange=12, yrange=12, zoom=12, iterations=50):
    """

    :return:
    """
    settings = dict(sampling=zoom / 12.0, itereations=iterations)

    data = pf.getdata(psf)

    res = []
    for x in range(xrange):
        for y in range(yrange):
            yind, xind = np.indices(data.shape)
            xind += x
            yind += y
            tmp = ndimage.map_coordinates(data.copy(), [yind, xind], order=1, mode='nearest')
            psf = ndimage.zoom(tmp, 1.0/zoom, order=0)

            sh = shape.shapeMeasurement(psf, log, **settings)
            results = sh.measureRefinedEllipticity()
            res.append([x, y, results])

    return res


def frebin(image, nsout, nlout=1, total=False):
    """
    Shrink or expand the size of an array an arbitary amount using interpolation.
    Conserves flux by ensuring that each input pixel is equally represented
    in the output array.

    .. todo:: one could do the binning faster if old and new outputs are modulo 0

    .. Note:: modelled after the IDL code frebin.pro, so this may not be the fastest solution

    :param image: input image, 1-d or 2-d ndarray
    :param nsout: number of samples in the output image, numeric scalar
    :param nlout: number of lines (ydir) in the output image, numeric scalar
    :param total: Use of the total conserves surface flux. If True, the output pixels
                  will be the sum of pixels within the appropriate box of the input image.
                  Otherwise they will be the average.

    :return: binned array
    :rtype: ndarray
    """
    shape = image.shape
    if nlout != 1:
        nl = shape[0]
        ns = shape[1]
    else:
        nl = nlout
        ns = shape[0]

    sbox = ns / float(nsout)
    lbox = nl / float(nlout)

    ns1 = ns - 1
    nl1 = nl - 1

    if nl == 1:
        #1D case
        result = np.zeros(nsout)
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            #add pixel valeus from istart to istop and subtract fraction pixel
            #from istart to rstart and fraction pixel from rstop to istop
            result[i] = np.sum(image[istart:istop + 1]) - frac1 * image[istart] - frac2 * image[istop]

        if total:
            return result
        else:
            return result / (float(sbox) * lbox)
    else:
        #2D case, first bin in second dimension
        temp = np.zeros((nlout, ns))
        result = np.zeros((nsout, nlout))

        #first lines
        for i in range(nlout):
            rstart = i * lbox
            istart = int(rstart)
            rstop = rstart + lbox

            if int(rstop) < nl1:
                istop = int(rstop)
            else:
                istop = nl1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                temp[i, :] = (1.0 - frac1 - frac2) * image[istart, :]
            else:
                temp[i, :] = np.sum(image[istart:istop + 1, :], axis=0) -\
                             frac1 * image[istart, :] - frac2 * image[istop, :]

        temp = np.transpose(temp)

        #then samples
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
            else:
                result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0) -\
                               frac1 * temp[istart, :] - frac2 * temp[istop, :]

        if total:
            return np.transpose(result)
        else:
            return np.transpose(result) / (sbox * lbox)


def Gaussian2D(x, y, sigmax, sigmay, xsize=512, ysize=512):
    """
    Create a two-dimensional Gaussian centered on x, y.

    :param x: x coordinate of the centre
    :type x: float
    :param y: y coordinate of the centre
    :type y: float
    :param sigmax: standard deviation of the Gaussian in x-direction
    :type sigmax: float
    :param sigmay: standard deviation of the Gaussian in y-direction
    :type sigmay: float
    :param xsize: number of pixels in xdirection
    :type xsize: int
    :param ysize: number of pixels in ydirection
    :type ysize: int

    :return: circular Gaussian 2D profile and x and y mesh grid
    :rtype: dict
    """
    #x and y coordinate vectors
    Gyvect = np.arange(1, ysize + 1)
    Gxvect = np.arange(1, xsize + 1)

    #meshgrid
    Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)

    #normalizers
    sigx = 1. / (2. * sigmax**2)
    sigy = 1. / (2. * sigmay**2)

    #gaussian
    exponent = (sigx * (Gxmesh - x)**2 + sigy * (Gymesh - y)**2)
    Gaussian = np.exp(-exponent) / (2. * math.pi * sigmax*sigmay)

    output = dict(GaussianXmesh=Gxmesh, GaussianYmesh=Gymesh, Gaussian=Gaussian)

    return output


def plotCentroids(results):
    """

    :param results:
    :return:
    """
    # e
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    for x, y, res in results:
        dist = np.sqrt(x**2 + y**2)
        ax.scatter(dist, res['e1'], c='m', marker='*', s=35)
        ax.scatter(dist, res['e2'], c='y', marker='s', s=35)
        ax.scatter(dist, res['ellipticity'], c='r', marker='o', s=35)

    #for the legend
    ax.scatter(dist, res['e1'], c='m', marker='*', label=r'$e_{1}$')
    ax.scatter(dist, res['e2'], c='y', marker='s', label=r'$e_{2}$')
    ax.scatter(dist, res['ellipticity'], c='r', marker='o', label=r'$e$')

    ax.set_xlabel('Centroid Offset $\sqrt{x^{2} + y^{2}} \quad$ [1/12 pixels]')
    ax.set_ylabel(r'$e_{i}\ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig('e.pdf')
    plt.close()

    #delta e
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    e = []
    e1 = []
    e2 = []
    d = []
    for x, y, res in results:
        d.append(np.sqrt(x**2 + y**2))
        e1.append(res['e1'])
        e2.append(res['e2'])
        e.append(res['ellipticity'])
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    m1 = np.mean(e1)
    m2 = np.mean(e2)
    d1 = e1 - m1
    d2 = e2 - m2

    ax.axhline(y=0.0)
    ax.scatter(d, d1, c='m', marker='*', s=50, label=r'$\Delta e_{1}$')
    ax.scatter(d, d2, c='y', marker='s', label=r'$\Delta e_{2}$')
    ax.scatter(d, np.sqrt(d1*d1 + d2*d2), c='r', marker='o', label=r'$\Delta e$')

    ax.set_xlim(-0.7, 17)
    ax.set_ylim(-0.01, 0.01)

    ax.set_xlabel('Centroid Offset $\sqrt{x^{2} + y^{2}} \quad$ [1/12 pixels]')
    ax.set_ylabel(r'$e_{i} - \bar{e}_{i}\ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8)
    plt.savefig('deltae.pdf')
    plt.close()

    #size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    for x, y, res in results:
        dist = np.sqrt(x**2 + y**2)
        ax.scatter(dist, res['R2'], c='m', marker='*', s=35)

    #for the legend
    ax.scatter(dist, res['R2'], c='m', marker='*', label=r'$R^{2}$')

    ax.set_xlabel('Centroid Offset $\sqrt{x^{2} + y^{2}} \quad$ [1/12 pixels]')
    ax.set_ylabel(r'$R^{2}$ [pixels$^{2}$]')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8, loc='upper left')
    plt.savefig('size.pdf')
    plt.close()

    #delta R2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #loop over the number of bias frames combined
    r = []
    d = []
    for x, y, res in results:
        d.append(np.sqrt(x**2 + y**2))
        r.append(res['R2'])
    r = np.asarray(r)

    ax.axhline(y=0.0)
    ax.scatter(d, r-np.mean(r), c='m', marker='*', label=r'$R^{2}$')

    ax.set_xlim(-0.7, 17)
    #ax.set_ylim(-0.01, 0.01)

    ax.set_xlabel('Centroid Offset $\sqrt{x^{2} + y^{2}} \quad$ [1/12 pixels]')
    ax.set_ylabel(r'$R^{2} - \bar{R}^{2}$ [pixels$^{2}$]')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8)
    plt.savefig('deltasize.pdf')
    plt.close()


def testCentroidingImpactSingleDirection(log, psf='/Users/sammy/EUCLID/vissim-python/data/psf12x.fits',
                                         canvas=16, ran=13, zoom=12, iterations=50, sigma=0.75,
                                         gaussian=False, interpolation=False, save=True):
    """

    :return:
    """
    settings = dict(sampling=zoom / 12.0, itereations=iterations, sigma=sigma)

    if gaussian:
        data = Gaussian2D(256., 256., 25, 25)['Gaussian']
        data *= 1e5
    else:
        data = pf.getdata(psf)

    xres = []
    print 'X shifts'
    for x in range(ran):
        tmp = data.copy()[canvas:-canvas, canvas-x:-canvas-x]

        if interpolation:
            if gaussian:
                psf = frebin(tmp, 40, nlout=40)
            else:
                size = tmp.shape[0] / 12
                psf = frebin(tmp, size, nlout=size, total=True)
        else:
            psf = ndimage.zoom(tmp, 1.0/zoom, order=0)

        if save:
            out = 'PSFx%i' % x
            if gaussian:
                out += 'Gaussian'
            if interpolation:
                out += 'Interpolated'
            out += '.fits'
            fileIO.writeFITS(psf, out, int=False)

        sh = shape.shapeMeasurement(psf, log, **settings)
        results = sh.measureRefinedEllipticity()
        xres.append([x, results])
        print x, psf.shape, np.sum(psf), np.max(psf), results['e1'], results['e2'], results['ellipticity']

    yres = []
    print 'Y shifts'
    for y in range(ran):
        tmp = data.copy()[canvas-y:-canvas-y, canvas:-canvas]

        if interpolation:
            if gaussian:
                psf = frebin(tmp, 40, nlout=40)
            else:
                size = tmp.shape[0] / 12
                psf = frebin(tmp, size, nlout=size, total=True)
        else:
            #yind, xind = np.indices(tmp.shape)
            #yind += y
            #tmp = ndimage.map_coordinates(tmp.copy(), [yind, xind], order=1)

            psf = ndimage.zoom(tmp, 1.0/zoom, order=0)

        sh = shape.shapeMeasurement(psf, log, **settings)
        results = sh.measureRefinedEllipticity()
        yres.append([y, results])
        print y, psf.shape, np.sum(psf), np.max(psf), results['e1'], results['e2'], results['ellipticity']

    return xres, yres


def plotCentroidsSingle(results, output='X', title='Nominal'):
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.09)
    ax.set_title(title + ' PSF')
    #loop over the number of bias frames combined
    e = []
    e1 = []
    e2 = []
    d = []
    for x, res in results:
        d.append(x)
        e1.append(res['e1'])
        e2.append(res['e2'])
        e.append(res['ellipticity'])
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    m1 = np.mean(e1) #e1[0]
    m2 = np.mean(e2) #e2[0]
    d1 = e1 - m1
    d2 = e2 - m2

    ax.axhline(y=0.0)
    ax.scatter(d, d1, c='m', marker='*', s=50, label=r'$\Delta e_{1}$')
    ax.scatter(d, d2, c='y', marker='s', label=r'$\Delta e_{2}$')
    ax.scatter(d, np.sqrt(d1*d1 + d2*d2), c='r', marker='o',
               label=r'$\Delta e = \sqrt{(\Delta e_{1})^{2} + (\Delta e_{2})^{2}}$')

    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(-0.002, 0.002)

    ax.set_xlabel('Centroid Offset [1/12 pixels]')
    ax.set_ylabel(r'$e_{i} - \bar{e}_{i}\ , \ \ \ i \in [1,2]$')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8,
               loc='lower right')
    plt.savefig('deltae%s.pdf' % output)
    plt.close()

    #delta R2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.09)
    ax.set_title(title + ' PSF')
    #loop over the number of bias frames combined
    r = []
    d = []
    for x, res in results:
        d.append(x)
        r.append(res['R2'])
    r = np.asarray(r)
    dr = r-np.mean(r)#r[0]

    ax.axhline(y=0.0)
    ax.scatter(d, dr, c='m', marker='*', label=r'$R^{2}$')

    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(-0.1, 0.1)

    ax.set_xlabel('Centroid Offset [1/12 pixels]')
    ax.set_ylabel(r'$R^{2} - \bar{R}^{2} \quad$ [pixels$^{2}$]')

    plt.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.8)
    plt.savefig('deltasize%s.pdf' % output)
    plt.close()


if __name__ == "__main__":
    log = lg.setUpLogger('centroidTesting.log')

    #res = testCentroidingImpact(log)
    #fileIO.cPickleDumpDictionary(res, 'centroidTesting.pk')
    #res = cPickle.load(open('centroidTesting.pk'))
    #plotCentroids(res)

    #PSF
    print 'Real PSF'
    xres, yres = testCentroidingImpactSingleDirection(log)
    fileIO.cPickleDumpDictionary(xres, 'centroidTestingX.pk')
    fileIO.cPickleDumpDictionary(yres, 'centroidTestingY.pk')
    xres = cPickle.load(open('centroidTestingX.pk'))
    yres = cPickle.load(open('centroidTestingY.pk'))
    plotCentroidsSingle(xres)
    plotCentroidsSingle(yres, output='Y')

    # #PSF interpolated
    # print 'Real PSF, interpolated'
    # xres, yres = testCentroidingImpactSingleDirection(log, interpolation=True)
    # fileIO.cPickleDumpDictionary(xres, 'centroidTestingXinterpolated.pk')
    # fileIO.cPickleDumpDictionary(yres, 'centroidTestingYinterpolated.pk')
    # xres = cPickle.load(open('centroidTestingXinterpolated.pk'))
    # yres = cPickle.load(open('centroidTestingYinterpolated.pk'))
    # plotCentroidsSingle(xres, output='Xinterpolated')
    # plotCentroidsSingle(yres, output='Yinterpolated')

    #Gaussian test
    print 'Gaussian'
    xres, yres = testCentroidingImpactSingleDirection(log, gaussian=True)
    fileIO.cPickleDumpDictionary(xres, 'centroidTestingXGaussian.pk')
    fileIO.cPickleDumpDictionary(yres, 'centroidTestingYGaussian.pk')
    xres = cPickle.load(open('centroidTestingXGaussian.pk'))
    yres = cPickle.load(open('centroidTestingYGaussian.pk'))
    plotCentroidsSingle(xres, output='XGaussian', title='Gaussian')
    plotCentroidsSingle(yres, output='YGaussian', title='Gaussian')