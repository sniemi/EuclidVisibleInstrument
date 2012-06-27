"""
A simple script to test how to simulate CCD bleeding effects.

:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import pyfits as pf
import numpy as np
import os, datetime


def CCDBleeding(image, wfc=200000):
    #loop over each column, as bleeding is modelled column-wise
    for i, column in enumerate(image.T):
        sum = 0.
        for j, value in enumerate(column):
            #first round - from bottom to top (need to half the bleeding)
            overload = value - wfc
            if overload > 0.:
                overload /= 2.
                image[j, i] -= overload
                sum += overload
            elif sum > 0.:
                if -overload > sum:
                    overload = -sum
                image[j, i] -= overload
                sum += overload

    for i, column in enumerate(image.T):
        sum = 0.
        for j, value in enumerate(column[::-1]):
            #second round - from top to bottom (bleeding was half'd already, so now full)
            overload = value - wfc
            if overload > 0.:
                image[-j-1, i] -= overload
                sum += overload
            elif sum > 0.:
                if -overload > sum:
                    overload = -sum
                image[-j-1, i] -= overload
                sum += overload

    return image


def writeFITSfile(data, filename, unsigned16bit=False):
    """
    Writes out a simple FITS file.

    :param data: data to be written
    :type data: ndarray
    :param filename: name of the output file
    :type filename: str
    :param unsigned16bit: whether to scale the data using bzero=32768
    :type unsigned16bit: bool

    :return: None
    """
    if os.path.isfile(filename):
        os.remove(filename)

    #create a new FITS file, using HDUList instance
    ofd = pf.HDUList(pf.PrimaryHDU())

    #new image HDU
    hdu = pf.ImageHDU(data=data)

    #convert to unsigned 16bit int if requested
    if unsigned16bit:
        hdu.scale('int16', '', bzero=32768)
        hdu.header.add_history('Scaled to unsigned 16bit integer!')

    #update and verify the header
    hdu.header.add_history('Created by VISsim at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
    hdu.verify('fix')

    ofd.append(hdu)

    #write the actual file
    ofd.writeto(filename)


if __name__ == '__main__':
    data = np.ones((2066, 2048))
    data[1032:1033, 1022:1023] = 300000
    data[1000:1001, 500:501] = 1000000
    data[500:501, 1500:1501] = 5000000
    data[1500:1501, 700:701] = 50000000

    image = CCDBleeding(data)
    writeFITSfile(image, 'test.fits')