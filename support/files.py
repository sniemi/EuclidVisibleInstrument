import datetime, cPickle, os
import pyfits as pf


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
