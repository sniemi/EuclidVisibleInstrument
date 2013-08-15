"""
A simple script to convert Zemax text file output to FITS format.
Copies the meta to the header.

:requires: PyFITS
:requires: NumPy

:version: 0.1

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import numpy as np
import pyfits as pf
import datetime, codecs


def convertToFITS(filename, output, overwrite=True):
    """

    :param filename:
    :param output:
    :return:
    """
    f = codecs.open(filename, 'rb', 'utf-16')
    d = f.read().strip().encode('ascii','ignore').split()
    f.close()

    print d[:14]

    tmp = []
    lines = []
    for l in d:
        if 3 < len(l) < 50:
            lines.append(l.strip().encode('ascii','ignore'))
        else:
            tmp.append(l)

    print tmp

    #recode to ascii and split before converting to numpy array
    data = np.asarray(tmp, dtype=np.float32)
    print data.shape

    f.close()

    #create a new FITS file, using HDUList instance
    ofd = pf.HDUList(pf.PrimaryHDU())

    #new image HDU
    hdu = pf.ImageHDU(data=data)

    #write the info to the header
    for line in lines:
        hdu.header.add_history(line)

    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (s.niemi at ucl.ac.uk).')
    hdu.header.add_history('Created at %s' % (datetime.datetime.isoformat(datetime.datetime.now())))
    hdu.verify('fix')

    ofd.append(hdu)
    ofd.writeto(output, clobber=overwrite)


if __name__ == "__main__":
    convertToFITS('PSFfft.TXT', 'PSFfft.fits')