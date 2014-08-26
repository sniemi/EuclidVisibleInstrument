"""
IO related functions.

:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: s.niemi@mucl.ac.uk
"""
import datetime, cPickle, os
import pyfits as pf
import numpy as np


def cPickleDumpDictionary(dictionary, output):
    """
    Dumps a dictionary of data to a cPickled file.

    :param dictionary: a Python data container does not have to be a dictionary
    :param output: name of the output file

    :return: None
    """
    out = open(output, 'wb')
    cPickle.dump(dictionary, out)
    out.close()


def cPicleRead(file):
    """
    Loads data from a pickled file.
    """
    return cPickle.load(open(file))


def cPickleDump(data, output):
    """
    Dumps data to a cPickled file.

    :param data: a Python data container
    :param output: name of the output file

    :return: None
    """
    out = open(output, 'wb')
    cPickle.dump(data, out)
    out.close()


def readFITSDataExcludeScanRegions(files, ext=1):
    """
    Reads in data from all the input files and subtracts the pre- and overscan regions if these were simulated.
    Takes into account which quadrant is being processed so that the extra regions are subtracted correctly.
    All information is taken from the FITS header.

    :param files: a list of files to open
    :type files: list or tuple
    :param ext: FITS extension
    :type ext: int

    :return: an array containing all data from all the files given
    :rtype: ndarray
    """
    data = []
    for i, file in enumerate(files):
        fh = pf.open(file, memmap=True)
        hdu = fh[ext].header

        try:
            overscan = hdu['OVERSCA']
        except:
            overscan = 'False'

        if 'True' in overscan:
            prescanx = hdu['PRESCANX']
            overscanx = hdu['OVRSCANX']
            quadrant = hdu['QUADRANT']

            if quadrant in (0, 2):
                data.append(fh[ext].data[:, prescanx: -overscanx])
            else:
                data.append(fh[ext].data[:, overscanx: -prescanx])
        else:
            data.append(fh[ext].data)
        fh.close()

    return np.asarray(data)


def writeFITS(data, output, overwrite=True, int=True):
    """
    Write out a FITS file using PyFITS. Will remove an existing file if overwrite=True.

    :param data: data to write to a FITS file
    :type data: ndarray
    :param output: name of the output file
    :type output: string
    :param overwrite: removes an existing file if present before writing a new one
    :type overwrite: bool
    :param int: whether or not to save the data scaled to 16bit unsigned integer values
    :type int: bool

    :return: None
    """
    if overwrite and os.path.isfile(output):
        os.remove(output)

    #create a new FITS file, using HDUList instance
    ofd = pf.HDUList(pf.PrimaryHDU())

    #new image HDU
    hdu = pf.ImageHDU(data=data)

    if int:
        hdu.scale('int16', '', bzero=32768)
        hdu.header.add_history('Scaled to unsigned 16bit integer!')

    #update and verify the header
    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
    hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
    hdu.verify('fix')

    ofd.append(hdu)

    #write the actual file
    ofd.writeto(output)
