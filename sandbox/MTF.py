"""
MTF and PSF
===========

These functions can be used to address the CCD requirements, which are written for an MTF
while PERD requirements are for a PSF.

.. Note:: The frequency nu_0 is the Nyquist limit for the CCD, which is defined as:
          nu_0 = 1 / (2p) ,
          where p is the pixel pitch in mm. Hence, for VIS the nu_0 is about 41.666.

Some links:
http://www.dspguide.com/CH25.PDF
http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html#sec-5
http://mathworld.wolfram.com/FourierTransformGaussian.html
https://github.com/GalSim-developers/GalSim/wiki/Optics-Module-usage
http://www.e2v.com/e2v/assets/File/documents/imaging-space-and-scientific-sensors/Papers/ccdtn105.pdf
http://aberrator.astronomy.net/html/mtf.html
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
from matplotlib import animation
import numpy as np
import math
from scipy import optimize
from scipy.interpolate import interp1d


def FWHM(sigma):
    """
    Calculates FWHM from sigma assuming a Gaussian profile.

    :param sigma: standard deviation

    :return: FWHM
    """
    return 2.*np.sqrt(2.*np.log(2.))*sigma


def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D numpy array, conveniently.

    @param image            the numpy array to be circular shifted.
    @param (iroll, jroll)   the roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)


def kxky(array_shape=(256, 256)):
    """Return the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.

    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.

    Adopts Numpy array index ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    @param array_shape   the Numpy array shape desired for `kx, ky`.
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)


def circular2DGaussian(array_size, sigma):
    """
    Create a circular symmetric Gaussian centered on x, y.

    :param sigma: standard deviation of the Gaussian, note that sigma_x = sigma_y = sigma
    :type sigma: float

    :return: circular Gaussian 2D
    :rtype: ndarray
    """
    x = array_size[1] / 2.
    y = array_size[0] / 2.

    #x and y coordinate vectors
    Gyvect = np.arange(1, array_size[0] + 1)
    Gxvect = np.arange(1, array_size[1] + 1)

    #meshgrid
    Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)

    #normalizers
    sigmax = 1. / (2. * sigma**2)
    sigmay = sigmax #same sigma in both directions, thus same normalizer

    #gaussian
    exponent = (sigmax * (Gxmesh - x)**2 + sigmay * (Gymesh - y)**2)
    #Gaussian = np.exp(-exponent) / (2. * math.pi * sigma*sigma)
    Gaussian = np.exp(-exponent) / np.sqrt(2. * math.pi * sigma*sigma)

    return Gaussian


def pupilImage(array_shape=(512, 512), size=1., dx=1.):
    """
    Generates a pupil image.

    :param array_shape:
    :param size:
    :param dx:
    :return:
    """
    lam_over_diam = 2.
    kmax_internal = dx * 2. * np.pi / lam_over_diam
    kx, ky = kxky(array_shape)
    rho = np.sqrt((kx ** 2 + ky ** 2) / (.5 * kmax_internal) ** 2)
    in_pupil = (rho < size)
    wf = np.zeros(array_shape, dtype=complex)
    wf[in_pupil] = 1.
    return wf


def PSF(wf, array_shape=(512, 512), flux=1., dx=1.):
    """
    Derives a PSF from pupil image.

    :param wf:
    :param array_shape:
    :param flux:
    :param dx:
    :return:
    """
    ftwf = np.fft.fft2(wf.copy())
    im = roll2d((ftwf.copy() * ftwf.copy().conj()).real, (array_shape[0] / 2, array_shape[1] / 2))
    psf = im * (flux / (im.sum() * dx ** 2))
    return psf


def MTF(wf):
    """
    Derives an MTF from pupil image.

    :param wf:
    :return: MTF
    """
    ftwf = np.fft.fft2(wf.copy())
    stf = np.fft.ifft2((ftwf * ftwf.conj()).real)
    otf = stf / stf[0, 0].real
    MTF = np.abs(otf)
    return MTF


def Example(array_shape=(512, 512), size=1., dx=1.):
    wf = pupilImage(array_shape=array_shape, size=size, dx=dx)
    psf = PSF(wf, array_shape=array_shape, dx=dx)
    mtf = MTF(wf)
    plotPSFMTF(psf, mtf, array_shape=array_shape)


def GaussianExample(array_shape=(512, 512), sigma=3.):
    psf = circular2DGaussian(array_size=array_shape, sigma=sigma)
    mtf = np.abs(np.fft.fft2(psf))
    plotPSFMTF(psf, mtf, array_shape=array_shape)


def plotPSFMTF(psf, mtf, array_shape):
    profilePSF = np.sum(psf.copy(), axis=0)
    profilePSF /= np.max(profilePSF)
    profileMTF = np.sum(mtf.copy(), axis=0)
    profileMTF /= np.max(profileMTF)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(np.log10(psf))
    ax2.imshow(np.log10(mtf))
    ax3.plot(profilePSF)
    ax4.plot(profileMTF)
    ax4.set_xlim(0, array_shape[1] / 2.)
    ax4.axvline(x=array_shape[1] / 4., c='r', ls='--', label='Nyquist?')
    plt.legend()
    plt.show()


def GaussianAnimation(array_shape=(512, 512), frames=15):
    """
    Animation showing how MTF changes as the size of the Gaussian PSF grows.

    :param array_shape: size of the simulation array
    :param frames: number of frames in the animation

    :return: None
    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.imshow(np.zeros((1,1)), origin='lower')
    ax2.imshow(np.zeros((1,1)), origin='lower')
    ax3.plot([])
    ax4.plot([])

    sigma_text = ax1.text(0.5, 1.07, '', transform=ax1.transAxes, horizontalalignment='center')
    nyquist_text = ax2.text(0.5, 1.07, '', transform=ax2.transAxes, horizontalalignment='center')

    ax3.set_xlim(array_shape[1] / 2. - 10, array_shape[1] / 2. + 10)
    ax4.set_xlim(0, array_shape[1] / 2.)
    ax4.axvline(x=array_shape[1] / 4., c='r', ls='--', label='Nyquist?')

    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    ax2.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels')

    ax3.set_ylabel('Normalized Flux')
    ax3.set_xlabel('Pixels')
    ax4.set_ylabel('MTF')

    plt.legend(shadow=True, fancybox=True)

    def init():
        # initialization function: plot the background of each frame
        ax1.imshow(np.zeros((1,1)), origin='lower')
        ax2.imshow(np.zeros((1,1)), origin='lower')
        ax3.plot([], [])
        ax4.plot([], [])
        sigma_text.set_text(' ')
        nyquist_text.set_text(' ')
        return ax1, ax2, ax3, ax4, sigma_text, nyquist_text


    def animate(i):
        s = np.logspace(-0.3, 0.55, frames)[i]
        psf = circular2DGaussian(array_size=array_shape, sigma=s)
        mtf = np.abs(np.fft.fft2(psf))
        mtf /= np.max(mtf)

        profilePSF = np.sum(psf.copy(), axis=0)
        profilePSF /= np.max(profilePSF)
        profileMTF = np.sum(mtf.copy(), axis=0)
        profileMTF /= np.max(profileMTF)

        ax1.imshow(np.log10(psf), origin='lower')
        ax2.imshow(np.log10(mtf), origin='lower')
        ax3.plot(profilePSF)
        ax4.plot(profileMTF)

        sigma_text.set_text('$\sigma = %.1f$, FWHM $= %.3f$' % (s, FWHM(s)))

        i = interp1d(np.arange(len(profileMTF)), profileMTF, kind='cubic')
        x = i(array_shape[1] / 4.)
        nyquist_text.set_text('MTF $\sim %0.3f$ at Nyquist?' % (np.abs(x)))

        return ax1, ax2, ax3, ax4, sigma_text, nyquist_text

    #note that the frames defines the number of times animate functions is being called
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)
    anim.save('Gaussian.mp4', fps=1)
    plt.close()


def requirement(alpha=0.2, w=2.):
    """
    Plots the requirements, both for PSF and MTF and compares them.

    :param alpha: power law slope
    :param w: wavenumber
    :return: None
    """
    #from MTF
    wave = [550, 750, 900]
    mtf = np.asarray([0.3, 0.35, 0.4])
    forGaussian = np.sqrt(- np.log(mtf) * 4 * np.log(2) / np.pi**2 / w**2)

    # fit
    fitfunc = lambda p, x: p[0] * x ** p[1]
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    fit, success = optimize.leastsq(errfunc, [1, -0.2],  args=(wave, forGaussian))

    #requirement
    x = np.arange(500, 950, 1)
    y = x**-alpha

    # compute the best fit function from the best fit parameters
    corrfit = fitfunc(fit, x)

    plt.plot(x, y, label=r'PERD: $\alpha = - %.1f$' % alpha)
    plt.plot(wave, forGaussian, 'rs', label='MTF Requirement')
    plt.plot(x, corrfit, 'r--', label=r'Fit: $\alpha \sim %.3f $' % (fit[1]))

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('FWHM [Arbitrary, u=%i]' % w)
    plt.legend(shadow=True, fancybox=True, numpoints=1)
    plt.savefig('requirementAlpha.pdf')
    plt.close()

    logx = np.log10(x)
    logy = np.log10(y)

    print 'Slope in log-log:'
    print (logy[1] - logy[0]) / (logx[1] - logx[0]), (logy[11] - logy[10]) / (logx[11] - logx[10])

    logyy = np.log(forGaussian)
    logxx = np.log(wave)
    print '\nSlope from MTF:'
    print (logyy[1] - logyy[0]) / (logxx[1] - logxx[0])
    print (logyy[2] - logyy[0]) / (logxx[2] - logxx[0])
    print (logyy[2] - logyy[1]) / (logxx[2] - logxx[1])


def compareAnalytical(array_shape=(256, 256), nyq=16.):
    """
    Compares an analytical derivation of FWHM - MTF relation to numerical solution.
    This is only valid for Gaussian PSFs.

    :param array_shape:
    :param nyq: cutout frequency (16)

    :return: None
    """
    #inverse of the nyquist as defined 1/w**2
    w = 1. / nyq

    res = []
    sigma = np.logspace(-0.55, 0.75, 50)
    for s in sigma:
        psf = circular2DGaussian(array_size=array_shape, sigma=s)
        mtf = np.abs(np.fft.fft2(psf))
        profileMTF = np.sum(mtf, axis=0)
        profileMTF /= np.max(profileMTF)

        i = interp1d(np.arange(len(profileMTF)), profileMTF, kind='cubic')
        x = i(array_shape[1] / nyq)
        res.append(x)

    res = np.asarray(res)

    analytical = np.sqrt(- np.log(res.copy()) * 4 * np.log(2) / np.pi**2 / w**2)

    plt.title('Gaussian PSF')
    plt.plot(res, FWHM(sigma), 'r-', label=r'numerical, $\nu \sim %i$' % (array_shape[1]/nyq))
    plt.plot(res, analytical, 'b--', label=r'analytical, $\omega \sim %i$' % nyq)

    plt.ylabel('FWHM [arbitrary]')
    plt.xlabel('MTF [normalized]')

    plt.legend(shadow=True, fancybox=True)
    plt.savefig('MTFFWHM.pdf')
    plt.close()


if __name__ == '__main__':
    requirement()
    #compareAnalytical()
    #GaussianAnimation()
    #Example()
