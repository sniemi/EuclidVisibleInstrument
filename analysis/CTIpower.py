"""
A simple script to study the effects of CTI trailing to weak lensing and power spectrum.

:requires: NumPy
:requires: SciPy
:requires: matplotlib

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
import numpy as np
from scipy import fftpack


def plotCTIeffect(data):
    """
    Data is assumed to be in the following format:
    x [deg]     y [deg] g1_parallel   g1_serial    g1_total    g2_total

    :param data:
    :return:
    """
    x = data[:, 0]
    y = data[:, 1]
    g1_parallel = data[:, 2]
    g1_serial = data[:, 3]
    g1_total = data[:, 4]
    g2_total = data[:, 5]

    M = np.sqrt(g1_parallel*g1_parallel + g1_serial*g1_serial)
    fig = plt.figure()
    plt.suptitle('CTI trails: G1 Parellel vs G1 Serial')
    ax1 = fig.add_subplot(111)
    Q = ax1.quiver(x, y, g1_parallel, g1_serial, M)#, headwidth=1, headlength=2)
    qk = ax1.quiverkey(Q, 0.9, 1.05, 1, r'$\sqrt{g_{par}^{2} + g_{ser}^{2}}$', labelpos='E', fontproperties={'weight': 'bold'})
    ax1.set_xlabel('X [deg]')
    ax1.set_ylabel('Y [deg]')
    ax1.set_xlim(x.min()*.99, x.max()*1.01)
    ax1.set_ylim(y.min()*.99, y.max()*1.01)
    plt.savefig('CTIG1.pdf')
    plt.close()

    M = np.sqrt(g1_total * g1_total + g2_total * g2_total)
    fig = plt.figure()
    plt.suptitle('CTI trails: G1 vs G2')
    ax1 = fig.add_subplot(111)
    Q = ax1.quiver(x, y, g1_total, g2_total, M)#, headwidth=1, headlength=2)
    qk = ax1.quiverkey(Q, 0.9, 1.05, 1, r'$\sqrt{g_{1}^{2} + g_{2}^{2}}$', labelpos='E', fontproperties={'weight': 'bold'})
    ax1.set_xlabel('X [deg]')
    ax1.set_ylabel('Y [deg]')
    ax1.set_xlim(x.min()*.99, x.max()*1.01)
    ax1.set_ylim(y.min()*.99, y.max()*1.01)
    plt.savefig('CTITotal.pdf')
    plt.close()

def azimuthalAverageSimple(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is preferred...

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
        radial_prof = np.array([(image * weights).flat[whichbin == b].sum() / float(weights.flat[whichbin == b].sum())
                                / float(binsize) for b in xrange(1, nbins + 1)])

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


def plot(data, fourier, radius, profile, xi, yi, output, title, log=False):
    """

    :param data:
    :param fourier:
    :param profile:
    :param output:
    :param title:
    :param log:
    :return:
    """
    fig = plt.figure(figsize=(14, 6.8))
    plt.suptitle(title)
    plt.suptitle(r'Residual CTI Shear', x=0.25, y=0.262)
    if log:
        plt.suptitle(r'$\log_{10}$(2D Power Spectrum + 1)', x=0.515, y=0.262)
    else:
        plt.suptitle(r'2D Power Spectrum', x=0.515, y=0.262)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    fig.subplots_adjust(wspace=0.3)

    i1 = ax1.imshow(data, origin='lower', interpolation='none')
    if log:
        i2 = ax2.imshow(np.log10(fourier[:len(yi) / 2, :len(xi) / 2]+1.), origin='lower', interpolation='none',
                        vmin=0., vmax=2.6)
    else:
        i2 = ax2.imshow(fourier[:len(yi) / 2, :len(xi) / 2], origin='lower', interpolation='none')
    plt.colorbar(i1, ax=ax1, orientation='horizontal')#, ticks=[-0.018, -0.013, -0.008, -0.002])
    plt.colorbar(i2, ax=ax2, orientation='horizontal')

    flat = np.ones(len(profile))*1e-6
    ax3.loglog(radius, radius**2*flat, 'r-', label=r'flat: $10^{-6}$')
    ax3.loglog(radius, radius**2*(flat + profile), 'b-', label='flat + CTI') #flat + CTI

    ax1.set_xlabel(r'X [pixel]')
    ax1.set_ylabel(r'Y [pixel]')
    ax2.set_xlabel(r'$l_{x}$')
    ax2.set_ylabel(r'$l_{y}$')
    ax3.set_xlabel(r'$l(\sqrt{x^{2} + y^{2}})$')
    ax3.set_ylabel(r'$l^{2}C(l)$')

    ax3.set_xlim(10, 500)
    ax3.set_ylim(1e-4, 1e3)

    ax3.legend(shadow=True, fancybox=True, loc='lower right')

    plt.savefig(output)
    plt.close()


def plotPower(data):
    """

    :param data:
    :return:
    """
    x = data[:, 0]
    y = data[:, 1]
    g1_total = data[:, 4]
    g2_total = data[:, 5]

    xi = np.unique(x)
    yi = np.unique(y)

    #G1
    data = g1_total.reshape((len(yi), len(xi)))
    F = fftpack.fft2(data)
    psd2D = np.abs(fftpack.fftshift(F.copy()))
    fourierSpectrum = np.abs(F.copy())
    rd, profile = azimuthalAverage(psd2D, binsize=1., returnradii=True)
    profile /= 4.  #quadrants all were averaged...

    plot(data, fourierSpectrum, rd, profile, xi, yi, 'FourierRealSpaceG1.pdf', 'Fourier Analysis of CTI Residual Shear (G1)')
    plot(data, fourierSpectrum, rd, profile, xi, yi, 'FourierRealSpaceG1log.pdf',
         'Fourier Analysis of CTI Residual Shear (G1)', log=True)

    #G2
    data = g2_total.reshape((len(yi), len(xi)))
    F = fftpack.fft2(data)
    psd2D = np.abs(fftpack.fftshift(F.copy()))
    fourierSpectrum = np.abs(F.copy())
    rd, profile = azimuthalAverage(psd2D, binsize=1., returnradii=True)
    profile /= 4.

    plot(data, fourierSpectrum, rd, profile, xi, yi, 'FourierRealSpaceG2.pdf', 'Fourier Analysis of CTI Residual Shear (G2)')
    plot(data, fourierSpectrum, rd, profile, xi, yi, 'FourierRealSpaceG2log.pdf',
         'Fourier Analysis of CTI Residual Shear (G2)', log=True)


if __name__ == '__main__':
    data = np.loadtxt('spurious_shears_cti.txt')

    #plotCTIeffect(data)
    plotPower(data)