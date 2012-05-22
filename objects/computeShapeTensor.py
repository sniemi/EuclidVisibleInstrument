"""
Quick and dirty script to compute shape tensor of a FITS image.

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import numpy as np
import pyfits as pf
import glob as g


def computeShapeTensor(data):
    """
    Computes a shape tensor from 2D imaging array.

    :Warning: This function has been adapted from Fortran
              and thus is very slow because of the nested
              loops. Need to be rewritten.

    :param data: imaging data as a numpy array
    :type data: ndarray

    :return: half of the size of the object in x and y direction
    :rtype: dict
    """
    data = data.transpose()
    xdim, ydim = data.shape

    Qxx = 0.
    Qxy = 0.
    Qyy = 0.
    for i in range(xdim):
        for j in range(ydim):
            Qxx += data[j, i] * (i - 0.5 * (xdim - 1)) * (i - 0.5 * (xdim - 1))
            Qxy += data[j, i] * (i - 0.5 * (xdim - 1)) * (j - 0.5 * (ydim - 1))
            Qyy += data[j, i] * (j - 0.5 * (ydim - 1)) * (j - 0.5 * (ydim - 1))

    shx = (Qxx + Qyy + np.sqrt((Qxx - Qyy)**2 + 4. * Qxy * Qxy)) / 2.
    shy = (Qxx + Qyy - np.sqrt((Qxx - Qyy)**2 + 4. * Qxy * Qxy)) / 2.

    shapex = np.sqrt(shx / np.sum(data))
    shapey = np.sqrt(shy / np.sum(data))

    return dict(shapex=shapex, shapey=shapey)


if __name__ == '__main__':
    files = g.glob('*.fits')

    for file in files:
        print 'File = %s' % file
        data = pf.getdata(file)
        shape = computeShapeTensor(data)
        print shape