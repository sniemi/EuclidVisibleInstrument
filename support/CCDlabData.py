"""

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
import os, time
import glob as g
from support import files as fileIO


def changePathNaming(folders='*', year=12):
    """
    Changes the folder names from the lab archive format to one
    which appears in ascending order when listing the file structure.

    :param folders: wild card identifier to find the folder
    :type folders: str
    :param year: the year to process
    :type year: int

    :return: None
    """
    #find all in the current working dir
    folders = g.glob(folders)

    for d in folders:
        if os.path.isdir(d):
            #check the stats
            stat = os.stat(d)
            created = os.stat(d).st_mtime
            asciiTime = time.asctime(time.gmtime(created))
            #print d, "is a dir  (created", asciiTime, ")"

            #rename the folder
            date, month = d.split()
            month_number = time.strptime(month, '%b').tm_mon
            #print month, date, month_number

            #so that folders are sorted in Unix type system
            if month_number < 10:
                new = '%i_0%i_%s' % (year, month_number, date)
            else:
                new = '%i_%i_%s' % (year, month_number, date)

            print d, new
            os.rename(d, new)


def readBinaryFiles(file, dimensions=(4196, 2072), saveFITS=True, output='tmp.fits'):
    """
    This simple function reads data from a given file that is in the binary format in which
    the CCD lab measurements have been stored in. It reads in the data and optionally saves it to a
    FITS file format. The function also returns the data

    :param file: name of the .bin file to read
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

    #four last numbers are zeros, throwing these out allows to reshape to the dimensions
    #there is still some onwated rows in the data, so we should remove the first four
    try:
        image = a[:-4].reshape(dimensions).T[4:, :]
    except:
        print 'Image shape as not expected'
        print a[:-4].shape
        return None

    if saveFITS:
        fileIO.writeFITS(image, output)

    return image


def convertAllBinsToFITS():
    """
    Converts all binary files within the current working directory to FITS format.

    :return: None
    """

    for root, dirs, files in os.walk(os.getcwd()):
        print 'Root directory to process is %s \n' % root
        for f in files:
            #only process .bin files that do not start with Euc
            if f.endswith('.bin') and not f.startswith('Euc'):
                tmp = root+'/'+f.replace(' ', '').replace('.bin', '.fits')
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

    #testing
    #img = readBinaryFiles('05 Sep_10_45_31s_Euclid.bin')
    #plotImage(img, 'test.pdf')

    convertAllBinsToFITS()