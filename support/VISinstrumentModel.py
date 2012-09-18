"""
VIS Instrument Model
====================

The file provides a function that returns VIS related information such as pixel
size, dark current, gain, zeropoint, and sky background.

:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.4
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
import datetime, math
import numpy as np


def VISinformation():
    """
    Returns a dictionary describing VIS. The following information is provided::

         'beta': 0.6
         'bias': 1000.0
         'cosmic_bkgd': 0.172
         'dark': 0.001
         'diameter': 1.3
         'dob': 0
         'e_adu': 3.5
         'fullwellcapacity': 200000
         'fwc': 175000
         'gain': 3.5
         'galaxy_fraction': 0.836
         'magzero': 17059000000.0
         'ovrscanx': 20
         'peak_fraction': 0.261179
         'pixel_size': 0.1
         'prescanx': 50
         'rdose': 30000000000.0
         'readnoise': 4.5
         'readout': 4.5
         'readtime': 88.0
         'sfwc': 730000.0
         'sky_background': 22.34
         'sky_high': 21.74
         'sky_low': 22.94
         'st': 5e-06
         'star_fraction': 0.928243
         'svg': 1e-10
         't': 0.01024
         'trapfile': 'cdm_euclid.dat'
         'vg': 6e-11
         'vth': 11680000.0
         'xsize': 2048
         'ysize': 2066
         'zeropoint': 25.58
         'zodiacal': 22.942

    :return: instrument model parameters
    :rtype: dict
    """
    out = dict(readnoise=4.5, pixel_size=0.1, dark=0.001, sky_background=22.34, zodiacal=22.942,
               diameter=1.3, galaxy_fraction=0.836, star_fraction=0.928243, peak_fraction=0.261179,
               zeropoint=25.58, gain=3.5, sky_high=21.74, sky_low=22.94, magzero=1.7059e10,
               fullwellcapacity=200000, readout=4.5, bias=1000.0, cosmic_bkgd=0.172, e_adu=3.5,
               xsize=2048, ysize=2066, prescanx=50, ovrscanx=20, readtime=88., apCorrection=0.925969)

    out.update({'dob' : 0, 'rdose' : 3e10, 'trapfile' : 'cdm_euclid.dat',
                'beta' : 0.6, 'fwc' : 175000, 'vth' : 1.168e7, 't' : 1.024e-2, 'vg' : 6.e-11,
                'st' : 5.e-6, 'sfwc' : 730000., 'svg' : 1.0e-10})

    apsize = math.pi * (out['diameter']/2./out['pixel_size'])**2
    out.update(dict(aperture_size = apsize))

    return out


def CCDnonLinearityModel(data):
    """
    This function provides a non-linearity model for a VIS CCD273.

    :param data: data to which the non-linearity model is being applied to
    :type data: float, int or ndarray

    :return: input data after conversion with the non-linearity model
    :rtype: float or ndarray
    """
    data[data < 1e4] -= np.sqrt(data[data < 1e4])
    data[0] = 1
    data[data > 1.5e5] -= np.sqrt(data[data > 1.5e5])
    return data


def testNonLinearity():
    """
    A simple test to plot the current non-linearity model.
    """
    vis = VISinformation()
    data = np.linspace(1, vis['fullwellcapacity'], 10000)
    nonlin = CCDnonLinearityModel(data.copy())

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())

    fig = plt.figure(frameon=False)

    left, width = 0.1, 0.8
    rect1 = [left, 0.3, width, 0.65]
    rect2 = [left, 0.1, width, 0.2]

    ax1 = fig.add_axes(rect1, title='VIS Non-linearity Model')
    ax2 = fig.add_axes(rect2)  #left, bottom, width, height

    ax1.plot(data, data, 'k-')
    ax1.plot(data, nonlin, 'r-', label='Model')

    ax2.plot(data, data/nonlin, 'g-')

    ax1.set_xticklabels([])
    ax2.set_xlabel('Real Charge [electrons]')
    ax1.set_ylabel('Output Charge [electrons]')
    ax2.set_ylabel('Real / Out')

    ax1.set_xlim(0, vis['fullwellcapacity'])
    ax2.set_xlim(0, vis['fullwellcapacity'])
    ax1.set_ylim(0, vis['fullwellcapacity'])

    ax1.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax1.transAxes, alpha=0.2)
    ax1.legend(shadow=True, fancybox=True, numpoints=1, scatterpoints=1, markerscale=1.0, loc='upper left')
    plt.savefig('NonlinearityModel.pdf')
    plt.close()


if __name__ == '__main__':
    testNonLinearity()