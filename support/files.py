"""
IO related functions.
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


def readFITSDataExcludeScanRegions(files, ext=1):
    """
    Reads in data from all the input files.

    Subtracts the pre- and overscan regions if these were simulated. Takes into account
    which quadrant is being processed so that the extra regions are subtracted correctly.

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


def writeFITS(data, output):
    """
    Write out a FITS file using PyFITS.

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
