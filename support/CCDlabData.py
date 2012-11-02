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
import pyfits as pf
import os, time, sys
import glob as g
from support import files as fileIO
from optparse import OptionParser


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


def combineToFullCCD(fileEF, fileGH, output, evm2=False):
    """
    Combines imaging data from files containing EF and GH image areas.

    :param fileEF: name of the FITS file that contains EF image section
    :type fileEF: str
    :param fileGH: name of the FITS file that contains GH image section
    :type fileGH: str
    :param evm2: if EVM2 ROE board was used then the data need to be scaled
    :type evm2: bool

    :return: None
    """
    dataEF = pf.getdata(fileEF)
    dataGH = pf.getdata(fileGH)[::-1, ::-1]  #GH data needs to be rotated because of how the data have been recorded

    #remove two rows from data
    dataEF = dataEF[:-2, :]
    dataGH = dataGH[2:, :]

    #calculate some statistics
    print 'Statistics from %s' % fileEF
    Q0EF = dataEF[:, :2099]
    Q1EF = dataEF[:, 2098:]
    m0 = Q0EF.mean()
    m1 = Q1EF.mean()
    msk0 = (Q0EF < 1.1*m0) & (Q0EF > 0.9*m0)
    msk1 = (Q1EF < 1.1*m1) & (Q1EF > 0.9*m1)
    print 'Q0 median mean max min std clipped'
    print np.median(Q0EF), m0, Q0EF.max(), Q0EF.min(), Q0EF.std(), Q0EF[msk0].std()
    print 'Q1 median mean max min std clipped'
    print np.median(Q1EF), m1, Q1EF.max(), Q1EF.min(), Q1EF.std(), Q1EF[msk1].std()
    print 'Statistics from %s' % fileGH
    Q0GH = dataGH[:, :2099]
    Q1GH = dataGH[:, 2098:]
    m0 = Q0GH.mean()
    m1 = Q1GH.mean()
    msk0 = (Q0GH < 1.1*m0) & (Q0GH > 0.9*m0)
    msk1 = (Q1GH < 1.1*m1) & (Q1GH > 0.9*m1)
    print 'Q0 median mean max min std clipped'
    print np.median(Q0GH), m0, Q0GH.max(), Q0GH.min(), Q0GH.std(), Q0GH[msk0].std()
    print 'Q1 median mean max min std clipped'
    print np.median(Q1GH), m1, Q1GH.max(), Q1GH.min(), Q1GH.std(), Q1GH[msk1].std()

    if evm2:
            #this bias level is higher than anticipated with DM
            dataEF -= 2400
            dataGH -= 2400

    #stitch together
    CCD = np.vstack((dataEF, dataGH))

    #write out a FITS file
    fileIO.writeFITS(CCD, output)


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-c', '--combine', dest='combine', action='store_true',
                      help='Combine EF and GH data to a single file containing a full CCD.')
    parser.add_option('-p', '--process', dest='process', action='store_true',
                      help='Process the current working directory to convert binary files to FITS format.')
    parser.add_option('-e', '--ef', dest='ef',
                      help="Input file containing EF data if combining data", metavar='string')
    parser.add_option('-g', '--gh', dest='gh',
                      help="Input file containing GH data if combining data", metavar='string')
    parser.add_option('-o', '--output', dest='output',
                      help="Name of the output file if combining data", metavar='string')
    parser.add_option('-s', '--scale_evm2', dest='evm2', action='store_true',
                      help='Will rescale the data if EVM2 board was used.')

    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.process is None and opts.combine is None:
        processArgs(True)
        sys.exit(8)

    if opts.process:
        convertAllBinsToFITS()

    if opts.combine:
        if opts.output is None:
            print 'Setting the output to tmp.fits'
            opts.output = 'tmp.fits'
        combineToFullCCD(opts.ef, opts.gh, opts.output, opts.evm2)