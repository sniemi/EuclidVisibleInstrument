"""
Helper functions related to surface fitting and a few examples to demonstrate.

:requires: NumPy
:requires: SciPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import math, itertools
import scipy, scipy.signal


def polyfit2d(x, y, z, order=3):
    """
    Fits a given order polynomial to 2D data.

    .. Note:: x and y arrays have to be of the same length. This function does not support non-square
              grids.

    :param x: x-data [in 1D formal]
    :type x: numpy array
    :param y: y-data [in 1D formal]
    :type y: numpy array
    :param z: the dependent data [in 1D formal]
    :type z: numpy array
    :param order: order of the polynomial to be fit
    :type order: int

    :return: coefficients defining a polynomial surface
    :rtype: ndarray
    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    coeffs, _, _, _ = np.linalg.lstsq(G, z)
    return coeffs


def polyval2d(x, y, coeffs):
    """
    Evaluates polynomial surface of a given coefficients on a given grid.

    .. Note:: x and y arrays have to be of the same length. This function does not support non-square
              grids.


    :param x: x-coordinates at which to evaluate [meshgrid]
    :type x: ndarray
    :param y: y-coordinates at which to evaluate [meshgrid]
    :type y: ndarray
    :param coeffs: coefficients of the polynomial surface, results of e.g. polyfit2d function
    :type coeffs: ndarray

    :return: evaluated values
    :rtype: ndarray
    """
    order = int(np.sqrt(len(coeffs))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(coeffs, ij):
        z += a * x**i * y**j
    return z


def polyfit2DSMN(x, y, z, order=3):
    """
    .. Warning:: DO NOT USE!
    """
    #Exponents of the polynomial
    exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]

    #Build matrix of system of equation
    A = np.empty((x.size, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (x**exp[0]) * (y**exp[1])

    #Compute the (Moore-Penrose) pseudo-inverse of a matrix
    Ainv = np.linalg.pinv(A)
    out = Ainv[1].reshape((2066, 2066)) * z
    return out


def sgolay2d(z, window_size, order, derivative=None):
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
       raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate(exps):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def exampleUsingFiltering():
    #generate bias surface and noise it
    X, Y, Z = generateBias()
    biased = addReadoutNoise(Z.copy())

    # filter it
    Zf = sgolay2d(biased, window_size=51, order=3)

    m = plt.matshow(Z)
    c1 = plt.colorbar(m, shrink=0.7, fraction=0.05)
    c1.set_label('BIAS level [ADUs]')
    plt.savefig('Bias.png')
    plt.close()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    plt.savefig('Bias3D.png')
    plt.close()

    m = plt.matshow(biased)
    c1 = plt.colorbar(m, shrink=0.7, fraction=0.05)
    c1.set_label('BIAS level [ADUs]')
    plt.savefig('BiasNoise.png')
    plt.close()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, biased, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    plt.savefig('BiasNoise3D.png')
    plt.close()

    m = plt.matshow(Zf)
    c1 = plt.colorbar(m, shrink=0.7, fraction=0.05)
    c1.set_label('BIAS level [ADUs]')
    plt.savefig('BiasSmoothed.png')
    plt.close()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Zf, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    plt.savefig('BiasSmoothed3D.png')
    plt.close()

    m = plt.matshow(Zf/Z)
    c1 = plt.colorbar(m, shrink=0.7, fraction=0.05)
    c1.set_label('BIAS level [ADUs]')
    plt.savefig('BiasResidual.png')
    plt.close()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Zf-Z, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_zlim(-2.0, 2.0)
    ax.set_zlabel(r'$\Delta$BIAS [ADUs]')
    plt.savefig('BiasResidual3D.png')
    plt.close()


def exampleNoNoiseNoInt(numdata=2066, floor=995, xsize=2048, ysize=2066):
    # generate random data
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    z = y - x + 0.78*x**2 + 15.0*y**2 - 1.75*x*y + 10.0*x**3 + 0.3*y**3 + floor
    print z.max(), z.min(), z.mean()

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x, y, z)
    print 'Example Coefficients'
    print m
    print

    # Evaluate it on a rectangular grid
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize),
                         np.linspace(y.min(), y.max(), ysize))
    zz = polyval2d(xx, yy, m)

    # generate 2D plot
    im = plt.imshow(zz, extent=(0, ysize, xsize, 0))
    c1 = plt.colorbar(im)
    c1.set_label('BIAS level [ADUs]')
    plt.scatter(x*numdata, y*numdata, c=z)
    plt.xlim(0, ysize)
    plt.ylim(0, xsize)
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('example2Dfit.png')
    plt.close()
    #and 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx*xsize, yy*ysize, zz, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('BIAS [ADUs]')
    plt.savefig('exampleSurfaceFit.png')
    plt.close()


def example(numdata=2066, floor=995, xsize=2048, ysize=2066):
    # generate random data
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize),
                         np.linspace(y.min(), y.max(), ysize))
    zclean = (yy - xx + 0.78*xx**2 + 15.0*yy**2 - 1.75*xx*yy + 10.0*xx**3 + 0.3*yy**3 + floor)#.astype(np.int)

    z = addReadoutNoise(zclean)
    print z.max(), z.min(), z.mean()

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(xx.ravel(), yy.ravel(), z.ravel())
    print 'Example Coefficients'
    print m
    print

    # Evaluate it on a rectangular grid
    zz = polyval2d(xx, yy, m)

    # generate 2D plot
    im = plt.imshow(z, extent=(0, ysize, xsize, 0))
    c1 = plt.colorbar(im)
    c1.set_label('BIAS [ADUs]')
    plt.xlim(0, ysize)
    plt.ylim(0, xsize)
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('exampleNoise2D.png')
    plt.close()
    #and 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx*xsize, yy*ysize, z, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('BIAS [ADUs]')
    plt.savefig('exampleNoise.png')
    plt.close()

    # generate 2D plot
    im = plt.imshow(zz, extent=(0, ysize, xsize, 0))
    c1 = plt.colorbar(im)
    c1.set_label('BIAS [ADUs]')
    plt.xlim(0, ysize)
    plt.ylim(0, xsize)
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('exampleNoise2Dfit.png')
    plt.close()
    #and 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx*xsize, yy*ysize, zz, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('BIAS [ADUs]')
    plt.savefig('exampleNoiseSurfaceFit.png')
    plt.close()


    # generate 2D plot
    im = plt.imshow(zclean-zz, extent=(0, ysize, xsize, 0))
    c1 = plt.colorbar(im)
    c1.set_label(r'$\Delta$BIAS [ADUs]')
    plt.xlim(0, ysize)
    plt.ylim(0, xsize)
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('exampleNoise2DResidual.png')
    plt.close()
    #and 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx*xsize, yy*ysize, zclean-zz, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel(r'$\Delta$BIAS [ADUs]')
    plt.savefig('exampleNoiseSurfaceResidual.png')
    plt.close()


def exampleAnimation(numdata=2066, floor=995, xsize=2048, ysize=2066, biases=25):
    # generate random data
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize),
                         np.linspace(y.min(), y.max(), ysize))
    zclean = yy - xx + 0.78*xx**2 + 15.0*yy**2 - 1.75*xx*yy + 10.0*xx**3 + 0.3*yy**3 + floor

    fig = plt.figure()
    ax = Axes3D(fig)

    ims = []
    for num in xrange(biases):
        z = addReadoutNoise(zclean.copy(), number=num+1)
        m = polyfit2d(xx.ravel(), yy.ravel(), z.ravel())
        zz = polyval2d(xx, yy, m)

        #append surface for animation
        ims.append((ax.plot_surface(xx*xsize, yy*ysize, zclean-zz, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet),))

    ax.set_title('Bias Surface Fitting')

    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel(r'$\Delta$BIAS [ADUs]')

    im_ani = animation.ArtistAnimation(fig, ims, interval=3000, repeat_delay=3000, blit=True)
    im_ani.save('Bias.mp4')



def generateBias(xsize=2066, ysize=2066, spread=30.0, min=990):
    """
    Generate a mock bias surface.
    """
    #coordinate axis and mesh grid
    xs = np.arange(xsize) + 1
    ys = np.arange(ysize) + 1
    x, y = np.meshgrid(xs, ys)

    #generate surface
    Z = y - x + 0.78*x**2 + 0.1*y**2 - x*y + 3.0*x**3 + 0.3*y**3

    #normalize the surface
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * spread + min

    print np.min(Z), np.max(Z), np.mean(Z)

    return x, y, Z.astype(np.int)


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


def generate3Dplot(X, Y, Z, output):
    """
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_wireframe(X, Y, (X*0)+1000.0, rstride=100, cstride=100, color='r')
    ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.5)
    ax.set_zlabel('ADUs')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    #exampleNoNoiseNoInt()
    #example()
    #exampleUsingFiltering()
    exampleAnimation()
