"""


:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import math
import numpy as np


def modifyG10catalog(file, xshift=560, yshift=560):
    """

    """
    data = np.loadtxt(file, skiprows=1)

    pixshift = data[:, 0] / 2.

    x = data[:, 26] #+ pixshift #- xshift
    y = data[:, 27] #+ pixshift #- yshift

    padisk = data[:, 24]

    nphoton = data[:, 11]
    magnitude =  - (np.log10(nphoton) - 12.881) / 0.4

    q_dev = data[:, 16]
    pa_dev = data[:, 17]
    e1 = (1. - q_dev)/(1. + q_dev) * np.cos(2.*pa_dev*math.pi/180.)
    e2 = (1. - q_dev)/(1. + q_dev) * np.sin(2.*pa_dev*math.pi/180.)
    ellipticity = np.sqrt(e1*e1 + e2*e2)

    #msk = (x > 0) & (y > 0) & (x <= 4096) & (y <= 4132)
    msk = padisk > 0.995

    x = x[msk]
    y = y[msk]
    magnitude = magnitude[msk]
    ellipticity = ellipticity[msk]

    fh = open('reducedCatalogue.txt', 'w')
    for a, b, c, d in zip(x, y, magnitude, ellipticity):
        fh.write('%f %f %f %f\n' % (a,b,c,d))
    fh.close()


if __name__ == '__main__':
        modifyG10catalog('euclid_chip_realisation_0001.dat')