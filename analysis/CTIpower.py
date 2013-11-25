"""
Impact of CTI Trailing on Shear Power Spectrum
==============================================

A simple script to study the effects of CTI trailing to weak lensing shear power spectrum.

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


class ps:
        """
        A class to calculate the power spectrum.
        """
        def __init__(self, g, step=48.0, size=10.0, nbin=20):
                """
                Attributes :
                - "g" is a complex square array representing a measurement at each point of a square grid.
                - "step" is the pixel step size in both x and y = postage stamp size = 48 for great10
                - "size" is is the total grid width (= height), in degrees, 10 for great10
                - "nbin" : number of ell bins for the power spectrum
                """
                self.g = g
                self.step = step
                self.size = size
                self.nbin2 = nbin


        def setup(self):
                """
                Set up stuff like l ranges
                """
                self.n = self.g.shape[0] # The width = height of the array
                if self.g.shape[1] != self.n:
                        sys.exit("Only square arrays !")

                self.radstep =  (self.size / self.n) * (np.pi / 180.0) # Angular step size in radians

                bigl = self.size * np.pi / 180.0

                self.max_l_mode = 2.0 * np.pi / self.radstep
                self.min_l_mode = 2.0 * np.pi / (self.size * np.pi/180.0)
                self.nyquist_deg = self.size / self.n

                print "Range of l modes : %f to %f" % (self.min_l_mode, self.max_l_mode)

                #print "Builing logarithmics l bins ..."
                self.dlogl = (self.max_l_mode - self.min_l_mode)/(self.nbin2 - 1.0)
                lbin = self.min_l_mode + (self.dlogl * (np.arange(self.nbin2))) -1.0 + 0.00001

                nbin = 2 * self.n

                # Creating a complex wavevector

                self.el1 = 2.0 * np.pi * (np.arange(self.n)  - ((self.n-1)/2.0) + 0.001) / bigl

                self.lvec = np.zeros((self.n,self.n), dtype = np.complex)
                icoord = np.zeros((self.n,self.n))
                jcoord = np.zeros((self.n,self.n))

                for i1 in range(self.n): # warning different python/matlab convention, i1 starts at 0
                        l1 = self.el1[i1]
                        for j1 in range(self.n):
                                l2 = self.el1[j1]
                                self.lvec[i1,j1] = np.complex(l1, l2)
                                icoord[i1,j1] = i1+1
                                jcoord[i1,j1] = j1+1



        def create(self):
                """
                Calculate the actual power spectrum
                """
                #% Estimate E and B modes assuming linear-KS.
                gfieldft = np.fft.fftshift(np.fft.fft2(self.g))
                gkapi = np.conjugate(self.lvec) * np.conjugate(self.lvec) * gfieldft / (self.lvec * np.conjugate(self.lvec))
                gkapi = np.fft.ifft2(np.fft.ifftshift(gkapi))

                gkapft = np.fft.fftshift(np.fft.fft2(np.real(gkapi)))
                gbetft = np.fft.fftshift(np.fft.fft2(np.imag(gkapi)))

                self.gCEE_2 = np.real(gkapft)**2.0 + np.imag(gkapft)**2.0 # E mode power
                self.gCBB_2 = np.real(gbetft)**2.0 + np.imag(gbetft)**2.0 # B mode power
                self.gCEB_2 = np.dot(np.real(gkapft), np.real(gbetft)) - np.dot(np.imag(gkapft), np.imag(gbetft)) # EB cross power



        def angavg(self):
                """
                Angular average of the spectrum
                """

                self.gPowEE = np.zeros(self.nbin2)
                self.gPowBB = np.zeros(self.nbin2)
                self.gPowEB = np.zeros(self.nbin2)
                self.ll = np.zeros(self.nbin2)
                dll = np.zeros(self.nbin2)

                for i1 in range(self.n): # start at 0
                        l1 = self.el1[i1]
                        for j1 in range(self.n):
                                l2 = self.el1[j1]
                                l = np.sqrt(l1*l1 + l2*l2)
                                #print l

                                if ( l <= self.max_l_mode and l >= self.min_l_mode) :
                                        ibin = int(np.round((l + 1 - self.min_l_mode) / self.dlogl))
                                        self.gPowEE[ibin] += self.gCEE_2[i1,j1] * l
                                        self.gPowBB[ibin] += self.gCBB_2[i1,j1] * l
                                        self.gPowEB[ibin] += self.gCEB_2[i1,j1] * l
                                else:
                                        print "Hmm, l out of min-max range, this part should be improved ..."

                                self.ll[ibin] = l # the array of l values
                                if ibin > 1:
                                        dll[ibin] = self.ll[ibin+1] - self.ll[ibin] # ibin starts from 0

                self.gPowEE /= (self.n**4 * self.dlogl)
                self.gPowBB /= (self.n**4 * self.dlogl)
                self.gPowEB /= (self.n**4 * self.dlogl)


        def plot(self, title="Power Spectrum"):
                """
                Plot it
                """
                plt.loglog(self.ll, self.gPowEE, "r.-", label="E mode")
                plt.loglog(self.ll, self.gPowBB, "b.-", label="B mode")
                plt.xlabel("Wavenumber l")
                plt.ylabel("Power [l^2 C_l / (2 pi)]")
                plt.title(title)
                plt.legend()
                plt.show()

def plotCTIeffect(data):
    """
    This function plots the CTI impact on shear. The Data is assumed to be in the following format::

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
    A simple algorithm to calculate the azimuthally averaged radial profile.

    :param image: image
    :type image: ndarray
    :param center: The [x,y] pixel coordinates used as the centre. The default is
                   None, which then uses the center of the image (including fractional pixels).
    :type center: list or None

    :return: radial profile
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
    #g1 = data[:, 2]
    #g2 = data[:, 3]
    g1_total = data[:, 4]
    g2_total = data[:, 5]

    g1 = g1_total
    g2 = g2_total

    xi = np.unique(x)
    yi = np.unique(y)

    g1 = g1.reshape((len(yi), len(xi)))
    g2 = g2.reshape((len(yi), len(xi)))

    ein = np.vectorize(complex)(g1, g2)
    myps = ps(ein, step=5.0, size=30.0, nbin=500)
    myps.setup()
    myps.create()
    myps.angavg()
    myps.plot(title="CTI Residual")


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