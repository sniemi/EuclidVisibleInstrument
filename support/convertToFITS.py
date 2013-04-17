"""
This script can be used to convert lab data to FITS files.

:requires: PyFITS
:requires: NumPy
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
import pyfits as pf
import os, time, sys
import glob as g
from support import files as fileIO
from optparse import OptionParser


def readBinaryFiles(file, dimensions=(100, 100), saveFITS=True, output='tmp.fits'):
    """
    This simple function reads data from a given file that is in the binary format in which
    the CCD lab measurements have been stored in. It reads in the data and optionally saves it to a
    FITS file format. The function also returns the data.

    :param file: name of the file to read
    :type file: str
    :param dimensions: dimensions of the image
    :type dimensions: tuple
    :param saveFITS: to control whether a FITS file should be written or not
    :type saveFITS: bool
    :param output: name of the output FITS file if saveFITS = True
    :type output: str

    :return: image
    :rtype: ndarray
    """
    fh = open(file, 'rb')
    #use numpy to read the binary format, the data is 16bit unsigned int
    a = np.fromfile(fh, dtype=np.uint16)
    fh.close()

    try:
        #note the transpose
        image = a.reshape(dimensions).T
    except:
        print 'Image shape as not expected'
        print a.shape
        return None

    if saveFITS:
        fileIO.writeFITS(image, output)

    return image


def convertAllBinsToFITS(suffix='.bin'):
    """
    Converts all binary files within the current working directory to FITS format.

    :return: None
    """
    for root, dirs, files in os.walk(os.getcwd()):
        print 'Root directory to process is %s \n' % root
        for f in files:
            #only process .bin files
            if f.endswith(suffix):
                tmp = root+'/'+f.replace(' ', '').replace(suffix, '.fits')
                #only process if the FITS file does not exist
                if not os.path.isfile(tmp):
                    input = root+'/'+f
                    print 'Processing file', input
                    i = readBinaryFiles(input, output=tmp)
                    if i is not None:
                        plotImage(i, tmp.replace('.fits', '.pdf'))


def plotImage(image, output):
    """
    A simple script to plot the imaging data.

    :param image: imaging data to be plotted.
    :type image: ndarray
    :param output: name of the output file e.g. test.pdf
    :type output: str
    :return: None
    """
    plt.figure(figsize=(12, 7))
    im = plt.imshow(image, origin='lower')
    c1 = plt.colorbar(im)
    c1.set_label('Image Scale')
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    convertAllBinsToFITS(suffix='.bim')
