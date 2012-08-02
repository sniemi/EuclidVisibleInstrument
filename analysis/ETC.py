"""
Exposure Times
==============

This file provides a simple functions to calculate exposure times or limiting magnitudes.
The file also provides a function that returns VIS related information such as pixel
size, dark current, gain, and zeropoint.

:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import numpy as np
import math


def VISinformation():
    """
    Returns a dictionary describing VIS.
    """
    out = dict(readnoise=4.5, pixel_size=0.1, dark=0.001, sky_background=22.34, diameter=1.3, fraction=0.836,
               zeropoint=25.58, zodiacal=22.942, gain=3.5)

    apsize = calculateAperture(out)
    out.update(dict(aperture_size = apsize))

    return out


def calculateAperture(info):
    """
    pi * (diameter / pixel_size)**2 / 4
    """
    out = math.pi * (info['diameter']/2./info['pixel_size'])**2
    return out


def exposureTime(info, magnitude, snr=10.0, exposures=3, fudge=0.7):
    """
    Returns the exposure time for a given magnitude.

    :param info: information describing the instrument
    :type info: dict
    :param magnitude: the magnitude of the objct
    :type magnitude: float
    :param snr: signal-to-noise ratio required [default=10].
    :type snr: float
    :param exposures: number of exposures that the object is present in
    :type exposures: int
    :param fudge: the fudge parameter to which to use to scale the snr to SExtractor required [default=0.7]
    :type fudge: float

    :return: exposure time (of an individual exposure) [seconds]
    :rtype: float
    """
    snr /= fudge

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['fraction']
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    tmp = flux_in_aperture + (sky + instrument + info['dark'])*info['aperture_size']

    first = snr**2 * tmp * exposures
    root = np.sqrt(first**2 + (2*flux_in_aperture*exposures*snr*info['readnoise'])**2 * exposures*info['aperture_size'])
    div = 2*flux_in_aperture**2*exposures**2

    return (first + root)/div


def limitingMagnitude(info, exp=565, snr=10.0, exposures=3, fudge=0.7):
    """
    Calculates the limiting magnitude for a given exposure time
    """
    snr /= fudge
    totalexp = exposures*exp

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    tmp = 4*((sky+instrument+info['dark']) * info['aperture_size'] * totalexp + exposures * info['readnoise']**2 * info['aperture_size'])
    root = np.sqrt(snr**2 + tmp)
    lg = 2.5*np.log10(((0.5*(snr**2 + snr*root))/totalexp)/info['fraction'])
    out = info['zeropoint'] - lg

    return out


def SNR(info, magnitude=24.5, exptime=565.0, exposures=3):
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

    :return: signal-to-noise ratio
    :rtype: float or ndarray
    """
    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['fraction']
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    nom = flux_in_aperture * exptime
    denom = np.sqrt(nom + (sky + instrument + info['dark']) * exptime * info['aperture_size'] +
                    info['readnoise']**2 * info['aperture_size'])

    return nom / denom * np.sqrt(exposures)


if __name__ == '__main__':
    magnitude = 24.5
    exptime = 565.0

    info = VISinformation()

    exp = exposureTime(info, magnitude)
    limit = limitingMagnitude(info, exp=exptime)
    snr = SNR(info, magnitude=magnitude, exptime=exp)


    print 'Exposure time required to reach SNR=10 (or 14.29) for a %.2f magnitude galaxy is %.1f' % (magnitude, exp)
    print 'SNR=%f for %.2fmag object if exposure time is %.2f' % (snr, magnitude, exp)
    print 'Limiting magnitude for %.2f second exposure is %.2f' % (exptime, limit)
