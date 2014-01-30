"""
Simple script to generate a power spectrum from shear field
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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculates the azimuthally averaged radial profile.

    :param image: image
    :type image: ndarray
    :param center: The [x,y] pixel coordinates used as the centre. The default is
                   None, which then uses the center of the image (including fractional pixels).
    :type center: list or None
    :param stddev: if specified, return the azimuthal standard deviation instead of the average
    :param returnradii: if specified, return (radii_array, radial_profile)
    :param return_nr: if specified, return number of pixels per radius *and* radius
    :param binsize: size of the averaging bin.  Can lead to strange results if
                    non-binsize factors are used to specify the center and the binsize is too large.
    :param weights: can do a weighted average instead of a simple average if this keyword parameter
                    is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
                    set weights and stddev.
    :param steps: if specified, will return a double-length bin array and radial profile
    :param interpnan: Interpolate over NAN values, i.e. bins where there is no data
    :param left: passed to interpnan to set the extrapolated values
    :param right: passed to interpnan to set the extrapolated values

    """
    #indeces
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2., (y.max() - y.min()) / 2.])

    #radial distances from centre
    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape, dtype=np.float64)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat, bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    #radial profiles
    if stddev:
        radial_prof = np.array([image.flat[whichbin == b].std() for b in xrange(1, nbins + 1)])
    else:
        #radial_prof = np.array([(image * weights).flat[whichbin == b].sum() / float(weights.flat[whichbin == b].sum())
        #                        / float(binsize) for b in xrange(1, nbins + 1)])
        radial_prof = np.array([(image * weights).flat[whichbin == b].sum() / float(weights.flat[whichbin == b].sum())
                                for b in xrange(1, nbins + 1)])
    if interpnan:
        radial_prof = np.interp(bin_centers, bin_centers[radial_prof == radial_prof],
                            radial_prof[radial_prof == radial_prof], left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof


def randomEllipticityField(n):
    """
    Generate a random shear field of n x n.
    """
    phi = np.random.rand(n, n) * 2. * np.pi #%generate an array of phi values on an nxn grid between 0 and 2pi
    r = 1./np.random.rand(n, n) #%generate an array of r values on an nxn grid a/b where b/a < 1

    e = np.exp(2j * phi) * (1. - r**2)/(1. + r**2) #complex ellipticity; this is an nxn complex matrix

    return e


def linearPowerSpectrum(n, loc=0., scale=1.5, plot=True):
    """
    Generate a Gaussian Random Field with a power law power (in l) spectrum.
    """
    #gaussian random field around zero with sigma=1 and FFT it
    gaussian = np.random.normal(loc=loc, scale=scale, size=(n, n))
    fG = np.fft.fft2(gaussian)

    #generate a cone surface
    #x and y coordinate vectors
    Gyvect = np.arange(1, n + 1)
    Gxvect = np.arange(1, n + 1)
    #meshgrid
    Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)
    P = n - np.sqrt((Gxmesh - n/2.)**2 + (Gymesh - n/2.)**2)

    #normalise to unity
    P -= P.min()
    P /= P.max()

    if plot:
        intercept = n * np.pi**2 / np.sqrt(2048. / n) * scale / np.sqrt(2)
        inpslope = (np.max(P) - np.min(P)) / np.sqrt(2048. / n) * scale * 2. * np.pi**2
        print 'input slope and intercept:', inpslope, intercept

        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot_surface(Gxmesh, Gymesh, P, rstride=250, cstride=250)
        ax2.imshow(P, origin='lower', interpolation='none')
        ax3.plot(P[np.unravel_index(P.argmax(), P.shape)[0], :], 'r-')
        ax3.plot(P[:, np.unravel_index(P.argmax(), P.shape)[1]], 'b--')

        ax3.set_xlim(0, n)

        plt.savefig('linear.png')

    #multiply the Fourier of the gaussian random field with the shape and FFT back
    multi = fG * P
    out = np.fft.ifft2(multi)

    if plot:
        field = np.abs(out.copy())

        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot_surface(Gxmesh, Gymesh, field, rstride=250, cstride=250)
        ax2.imshow(field, origin='lower', interpolation='none')
        ax3.plot(field[np.unravel_index(field.argmax(), field.shape)[0], :], 'r-')
        ax3.plot(field[:, np.unravel_index(field.argmax(), field.shape)[1]], 'b--')

        ax3.set_xlim(0, n)

        plt.savefig('linearfield.png')

    return out


def flatPowerSpectrum(n, loc=0., scale=1.5, plot=True):
    """
    Generate a Gaussian Random Field with a power law power (in l) spectrum.
    """
    #gaussian random field around zero with sigma=1 and FFT it
    gaussian = np.random.normal(loc=loc, scale=scale, size=(n, n))
    fG = np.fft.fft2(gaussian)

    #flat surface
    P = np.ones((n, n))

    if plot:
        intercept = n * np.pi**2 / np.sqrt(1024. / n) * scale * 2
        inpslope = (np.max(P) - np.min(P)) / np.sqrt(2048. / n) * scale * 2. * np.pi**2
        print 'input slope and intercept:', inpslope, intercept

        #meshgrid, needed for the plot
        Gxvect = np.arange(1, n + 1)
        Gxmesh, Gymesh = np.meshgrid(Gxvect, Gxvect)

        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot_surface(Gxmesh, Gymesh, P, rstride=250, cstride=250)
        ax2.imshow(P, origin='lower', interpolation='none')
        ax3.plot(P[np.unravel_index(P.argmax(), P.shape)[0], :], 'r-')
        ax3.plot(P[:, np.unravel_index(P.argmax(), P.shape)[1]], 'b--')

        ax3.set_xlim(0, n)

        plt.savefig('flat.png')

    #multiply the Fourier of the gaussian random field with the shape and FFT back
    multi = fG * P
    out = np.fft.ifft2(multi)

    if plot:
        field = np.abs(out.copy())

        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot_surface(Gxmesh, Gymesh, field, rstride=250, cstride=250)
        ax2.imshow(field, origin='lower', interpolation='none')
        ax3.plot(field[np.unravel_index(field.argmax(), field.shape)[0], :], 'r-')
        ax3.plot(field[:, np.unravel_index(field.argmax(), field.shape)[1]], 'b--')

        ax3.set_xlim(0, n)

        plt.savefig('flatfield.png')

    return out


def powerSpectrum(shearField, n):
    """
    From shear field to 2D and binned 1D power spectra.
    See e.g. Kitching et al. (2011) GREAT 10 handbook Appendix B.

    :param shearField: 2D shear field
    :type shearField: numpy 2D array
    """
    #2D FFT of the shear field
    e_FFT = np.fft.fft2(shearField)
    e_FFT = np.fft.fftshift(e_FFT)

    #make a 2D array of the position angles
    xposition = np.tile(np.arange(n)+1, (n, 1))
    yposition = xposition.copy()

    #make a complex l-vector l_x + il_y (not the angle as in arbitrary units at the moment)
    l = 2 * np.pi / (xposition + 1j * yposition)

    #rotate the FFT of the shear field
    rotate_comp = np.conj(l)**2 / np.abs(l)**2
    rotate = rotate_comp * e_FFT

    #inverse back to real space -> E + iB
    e_IFFT = np.fft.ifft(rotate)
    e_real = np.real(e_IFFT)
    #b = np.imag(e_IFFT)  #could also check the b-mode

    # FFT the E-mode
    e_real_fft = np.fft.fft2(e_real)

    #calculate the modulus of the E-mode
    e_mod = np.abs(e_real_fft)

    #rotate is now the 2D power spectrum in l_x,l_y space
    #now need to bin the rotate in |l| in azimuthal (angular) bins about the central value
    rd, profile = azimuthalAverage(e_mod, binsize=5, returnradii=True)

    return e_FFT, rd, profile


def plot(e, e_FFT, rd, profile, output='test.pdf'):
    """
    Plot results
    """
    #linear fit to the C(l), exclude beginning and end
    fit = np.polyfit(rd[10:-10], profile[10:-10], 1, full=True)
    p = np.poly1d(fit[0])
    print 'derived slope and intercept:', fit[0][0], fit[0][1]

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_title(r'$|e|$')
    ax2.set_title('2D Power Spectrum')
    ax3.set_title('1D Binned Power Spectrum')

    e = np.abs(e)
    e_FFT = np.abs(e_FFT)

    ax1.imshow(e, origin='lower', interpolation='none', vmin=np.min(e)*1.1, vmax=np.max(e)*0.9)
    ax2.imshow(e_FFT, origin='lower', interpolation='none', vmin=np.min(e_FFT)*1.1, vmax=np.max(e_FFT)*0.9)
    ax3.plot(rd, p(rd), 'r-', label='fit')
    ax3.plot(rd, profile, label='power')

    ax3.set_xlabel('l')
    ax3.set_ylabel('C(l)')

    ax3.set_xlim(0, np.floor(np.sqrt((e_FFT.shape[0]/2.)**2 + (e_FFT.shape[1]/2.)**2)-1))

    plt.savefig(output)


def doAllflat(n):
    e = flatPowerSpectrum(n)
    power, rd, profile = powerSpectrum(e, n)
    plot(e, power, rd, profile, output='flat.pdf')


def doAlllinear(n):
    e = linearPowerSpectrum(n)
    power, rd, profile = powerSpectrum(e, n)
    plot(e, power, rd, profile, output='linear.pdf')


def doAllrandom(n):
    e = randomEllipticityField(n)
    power, rd, profile = powerSpectrum(e, n)
    plot(e, power, rd, profile, output='random.pdf')


if __name__ == '__main__':
    n = 2048 #size of the array (n, n)

    print '\n\nflat:'
    doAllflat(n)

    print '\n\nlinear:'
    doAlllinear(n)