"""

:requires: NumPy
:requires: PyFITS
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pyfits as pf
import math, os, datetime


def Gaussian2D(x, y, sizex, sizey, sigmax, sigmay):
    """
    Create a circular symmetric Gaussian centered on x, y.

    :param x: x coordinate of the centre
    :type x: float
    :param y: y coordinate of the centre
    :type y: float
    :param sigmax: standard deviation of the Gaussian in x-direction
    :type sigmax: float
    :param sigmay: standard deviation of the Gaussian in y-direction
    :type sigmay: float

    :return: circular Gaussian 2D profile and x and y mesh grid
    :rtype: dict
    """
    #x and y coordinate vectors
    Gyvect = np.arange(1, sizey + 1)
    Gxvect = np.arange(1, sizex + 1)

    #meshgrid
    Gxmesh, Gymesh = np.meshgrid(Gxvect, Gyvect)

    #normalizers
    sigx = 1. / (2. * sigmax**2)
    sigy = 1. / (2. * sigmay**2)

    #gaussian
    exponent = (sigx * (Gxmesh - x)**2 + sigy * (Gymesh - y)**2)
    Gaussian = np.exp(-exponent) / (2. * math.pi * sigmax*sigmay)

    output = dict(GaussianXmesh=Gxmesh, GaussianYmesh=Gymesh, Gaussian=Gaussian)

    return output


def plot3D(data):
    fig = plt.figure(figsize=(12,12))
    rect = fig.add_subplot(111, visible=False).get_position()
    ax = Axes3D(fig, rect)
    surf = ax.plot_surface(data['GaussianXmesh'],
                           data['GaussianYmesh'],
                           data['Gaussian'],
                           rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.savefig('gaussian.pdf')


def writeFITSfile(data, output):
    """
    Write out FITS files using PyFITS.

    :param data: data to write to a FITS file
    :type data: ndarray
    :param output: name of the output file
    :type output: string

    :return: None
    """
    if os.path.isfile(output):
        os.remove(output)

    #create a new FITS file, using HDUList instance
    ofd = pf.HDUList(pf.PrimaryHDU())

    #new image HDU
    hdu = pf.ImageHDU(data=data)

    #update and verify the header
    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
    hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
    hdu.verify('fix')

    ofd.append(hdu)

    #write the actual file
    ofd.writeto(output)


if __name__ == '__main__':
    gaussian2d = Gaussian2D(100, 100, 200, 200, 35.0, 15.0)
    plot3D(gaussian2d)
    writeFITSfile(gaussian2d['Gaussian'], 'gaussian.fits')