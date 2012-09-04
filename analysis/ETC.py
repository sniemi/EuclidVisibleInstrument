"""
Calculating Exposure Times and Limiting Magnitude
=================================================

This file provides a simple functions to calculate exposure times or limiting magnitudes.
The file also provides a function that returns VIS related information such as pixel
size, dark current, gain, and zeropoint.

:requires: NumPy
:requires: SciPy
:requires: matplotlib

:version: 0.1

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
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
from scipy import interpolate
import math, datetime


def VISinformation():
    """
    Returns a dictionary describing VIS.
    """
    out = dict(readnoise=4.5, pixel_size=0.1, dark=0.001, sky_background=22.34, zodiacal=22.942,
               diameter=1.3, galaxy_fraction=0.836, star_fraction=0.928243, peak_fraction=0.261179,
               zeropoint=25.58, gain=3.5, sky_high=21.74, sky_low=22.94)

    apsize = calculateAperture(out)
    out.update(dict(aperture_size = apsize))

    return out


def calculateAperture(info):
    """
    pi * (diameter / 2. / pixel_size)**2
    """
    out = math.pi * (info['diameter']/2./info['pixel_size'])**2
    return out


def exposureTime(info, magnitude, snr=10.0, exposures=3, fudge=0.7, galaxy=True):
    """
    Returns the exposure time for a given magnitude.

    :param info: information describing the instrument
    :type info: dict
    :param magnitude: the magnitude of the objct
    :type magnitude: float or ndarray
    :param snr: signal-to-noise ratio required [default=10].
    :type snr: float
    :param exposures: number of exposures that the object is present in
    :type exposures: int
    :param fudge: the fudge parameter to which to use to scale the snr to SExtractor required [default=0.7]
    :type fudge: float
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean

    :return: exposure time [seconds]
    :rtype: float or ndarray
    """
    snr /= fudge

    if galaxy:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['galaxy_fraction']
    else:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction']

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    tmp = flux_in_aperture + (sky + instrument + info['dark'])*info['aperture_size']

    first = snr**2 * tmp * exposures
    #root = np.sqrt(first**2 + (2*flux_in_aperture*exposures*snr*info['readnoise'])**2 * exposures*info['aperture_size'])
    root = np.sqrt(first**2 + (2*flux_in_aperture*exposures*snr*(info['readnoise']**2 + (info['gain']/2.)**2 * info['aperture_size'])))
    div = 2*flux_in_aperture**2*exposures**2

    return (first + root)/div


def limitingMagnitude(info, exp=565, snr=10.0, exposures=3, fudge=0.7, galaxy=True):
    """
    Calculates the limiting magnitude for a given exposure time, number of exposures and minimum signal-to-noise
    level to be reached.

    :param info: instrumental information such as zeropoint and background
    :type info: dict
    :param exp: exposure time [seconds]
    :type exp: float or ndarray
    :param snr: the minimum signal-to-noise ratio to be reached.
                .. Note:: This is couple to the fudge parameter: snr_use = snr / fudge
    :type snr: float or ndarray
    :param exposures: number of exposures [default = 3]
    :type exposures: int
    :param fudge: a fudge factor to divide the given signal-to-noise ratio with to reach to the required snr.
                  This is mostly due to the fact that SExtractor may require a higher snr than what calculated
                  otherwise.
    :type fudge: float
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean

    :return: limiting magnitude given the input information
    :rtype: float or ndarray
    """
    snr /= fudge
    totalexp = exposures*exp

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    #tmp = 4*((sky+instrument+info['dark']) * info['aperture_size'] * totalexp + exposures * info['readnoise']**2 * info['aperture_size'])
    tmp = 4*((sky+instrument+info['dark']) * info['aperture_size'] * totalexp + exposures * (info['readnoise']**2 + (info['gain']/2.)**2 * info['aperture_size']))
    root = np.sqrt(snr**2 + tmp)

    if galaxy:
        lg = 2.5*np.log10(((0.5*(snr**2 + snr*root))/totalexp)/info['galaxy_fraction'])
    else:
        lg = 2.5*np.log10(((0.5*(snr**2 + snr*root))/totalexp)/info['star_fraction'])

    out = info['zeropoint'] - lg

    return out


def SNR(info, magnitude=24.5, exptime=565.0, exposures=3, galaxy=True, background=True):
    """
    Calculates the signal-to-noise ratio for an object of a given magnitude in a given exposure time and a
    number of exposures.

    :param info: instrumental information such as zeropoint and background
    :type info: dict
    :param magnitude: input magnitude of an object(s)
    :type magnitude: float or ndarray
    :param exptime: exposure time [seconds]
    :type exptime: float
    :param exposures: number of exposures [default = 3]
    :type exposures: int
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean
    :param background: whether to include background from sky, instrument, and dark current [default=True]
    :type background: boolean

    :return: signal-to-noise ratio
    :rtype: float or ndarray
    """
    if galaxy:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['galaxy_fraction']
    else:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction']

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background
    bgr = (sky + instrument + info['dark']) * info['aperture_size'] * exptime

    if not background:
        bgr = 0.0

    nom = flux_in_aperture * exptime
    #denom = np.sqrt(nom + bgr + info['readnoise']**2 * info['aperture_size'])
    denom = np.sqrt(nom + bgr + (info['readnoise']**2 + (info['gain']/2.)**2 * info['aperture_size']))

    return nom / denom * np.sqrt(exposures)


def SNRproptoPeak(info, exptime=565.0, exposures=1):
    """
    Calculates the relation between the signal-to-noise ratio and the electrons in the peak pixel.

    :param info: instrumental information such as zeropoint and background
    :type info: dict

    :return: signal-to-noise ratio
    :rtype: float or ndarray
    """
    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background
    bgr = (sky + instrument + info['dark']) * info['aperture_size'] * exptime

    snr = []
    peak = []
    for magnitude in np.arange(10, 27, 1)[::-1]:
        nom = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction'] * exptime
        peak_pixel = nom * info['peak_fraction']

        #denom = np.sqrt(nom + bgr + (info['readnoise']**2 * info['aperture_size']))
        denom = np.sqrt(nom + bgr + (info['readnoise']**2 + (info['gain']/2.)**2 * info['aperture_size']))

        result = nom / denom * np.sqrt(exposures)
        snr.append(result)
        peak.append(peak_pixel)

    peak = np.asarray(peak)
    snr = np.asanyarray(snr)

    f = interpolate.interp1d(snr, peak, kind='cubic')
    sn = np.asarray([5, 10, 50, 100, 200, 500, 1000])
    vals = f(sn)

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('VIS Nominal System PSF: Peak Pixel')
    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    ax.loglog(peak, snr, 'bo-', label=r'SNR $\propto$ counts')

    ax.axvline(x=220000, c='r', ls='--', label=r'Saturation $> 220000 e^{-}$')
    ax.fill_between(np.linspace(220000, np.max(peak), 3), 1e5, 1.0, facecolor='red', alpha=0.08, label=r'Saturation $> 220000 e^{-}$')

    i = 0
    for val, s in zip(vals, sn):
        plt.text(0.05, 0.9-i, r'snr = %i, peak = %.1f $e^{-}$' % (s, val), fontsize=10, transform=ax.transAxes)
        i += 0.03

    ax.set_ylabel('Signal-to-Noise Ratio [nominal 565s]')
    ax.set_xlabel(r'Counts in the Peak Pixel $[e^{-}]$')

    ax.set_xlim(xmax=np.max(peak))

    plt.legend(shadow=True, fancybox=True, loc='lower right')
    plt.savefig('Peak.pdf')
    plt.close()


if __name__ == '__main__':
    magnitude = 20.8989
    exptime = 565.0

    info = VISinformation()

    exp = exposureTime(info, magnitude)
    limit = limitingMagnitude(info, exp=exptime)
    snr = SNR(info, magnitude=magnitude, exptime=exptime, exposures=1, galaxy=False)

    print 'Exposure time required to reach SNR=10 (or 14.29) for a %.2f magnitude galaxy is %.1f' % (magnitude, exp)
    print 'SNR=%f for %.2fmag object if exposure time is %.2f' % (snr, magnitude, exptime)
    print 'Limiting magnitude of a galaxy for %.2f second exposure is %.2f' % (exptime, limit)

    #SNRproptoPeak(info)
