"""
Background Subtraction
======================

This scripts can be used to study the impact of background subtraction errors on the shape measurements.

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
import numpy as np


def simpleAnalytical(value=130, size=(50, 50), readnoise=4.5, gain=3.1, req=1.):
    """
    A simple function to test the area of pixels needed (defined by size) to derive the
    pixel value to the level of required number of electrons given the readout noise and
    the gain of the system.

    :param value: the floor level in electrons to be found [default = 130]
    :type value: int
    :param size: area describing the number of pixels available [default = (50, 50)]
    :type size: tuple
    :param readnoise: readout noise of the full detection chain [default = 4.5]
    :type readnoise: float
    :param gain: gain of the detection system [default = 3.1]
    :type gain: float
    :param req: required level to reach in electrons [default = 1]
    :type req: float

    :return: none
    """
    stars = 2000
    mc = 500
    fail = 0
    for a in range(mc):
        for x in range(stars):
            data = np.round(((np.random.poisson(value, size=size) +
                              np.random.normal(loc=0, scale=readnoise, size=size)) / gain)).astype(np.int)
            derived = data * gain - value

            if np.mean(derived) > req:
                print 'Not enough pixels to derive the floor level to %.1f electron level' % req
                print np.mean(derived), np.median(derived), np.std(derived)
                fail += 1

    print 'Failed %i times' % fail
    print np.mean(derived), np.median(derived), np.std(derived)


if __name__ == '__main__':
    simpleAnalytical()
