"""
File Conversions
================

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
import datetime, codecs, glob


def convertToFITS(filename, output, overwrite=True):
    """
    Converts Zemax output TXT file to FITS format. Stores the
    extra information from the Zemax output to the FITS header.

    :param filename: name of the Zemax output txt file
    :type filename: str
    :param output:  name of the output FITS file
    :type output: str
    :param overwrite: whether or not to overwrite the output file if exists
    :type overwrite: bool

    :return: None
    """
    d= []
    lines = []
    for tmp in codecs.open(filename, 'rb', 'utf16'):

        tmp = tmp.encode('ascii','ignore').strip()

        if 3 < len(tmp) < 100:
            lines.append(tmp)
        elif len(tmp) > 100:
            d.append([float(x) for x in tmp.strip().split()])


    #recode to ascii and split before converting to numpy array
    data = np.asarray(d, dtype=np.float32)

    if 'FFT' in lines[0]:
        #for some reason FFT output is flipped
        lines.append('Output rotated 180 with numpy.rot90(img, k=2)')
        data = np.rot90(data, k=2)
    print 'Shape:', data.shape

    #create a new FITS file, using HDUList instance
    ofd = pf.HDUList(pf.PrimaryHDU())

    #new image HDU
    hdu = pf.ImageHDU(data=data)

    #write the info to the header
    for line in lines:
        hdu.header.add_history(line)
        print line

    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (s.niemi at ucl.ac.uk).')
    hdu.header.add_history('Created at %s' % (datetime.datetime.isoformat(datetime.datetime.now())))
    hdu.verify('fix')

    ofd.append(hdu)
    ofd.writeto(output, clobber=overwrite)


if __name__ == "__main__":
    for f in glob.glob('*.TXT'):
        print '\n\nConverting %s' % f
        convertToFITS(f, f.replace('TXT', 'fits'))