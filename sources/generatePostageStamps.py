"""
Generating Postage Stamp Images
===============================

This simple script can be used to generate postage stamp images from a larger mosaic.
These images can then be used in the VIS simulator.

:requires: NumPy
:requires: PyFITS
:requires: VIS-PP

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.1
"""
import numpy as np
import pyfits as pf
from support import files as fileIO


def generaPostageStamps(filename, catalog, maglimit=22., output='galaxy'):
    """
    Generates postage stamp images from an input file given the input catalog position.
    The output files are saved to FITS files.

    :param filename: name of the FITS file from which the postage stamps are extracted
    :type filename: str
    :param catalog: name of the catalogue with x and y positions and magnitudes
    :type catalog: str
    :param maglimit: brighter galaxies than the given magnitude limit are extracted
    :type maglimit: float
    :param output: name of the postage stamp prefix (will add a running number to this)
    :type output: str

    :return: None
    """
    cat = np.loadtxt(catalog)
    xcoord = cat[:, 0]
    ycoord = cat[:, 1]
    mag = cat[:, 2]
    msk = mag < maglimit

    fh = pf.open(filename, mmap=True, memmap=True)
    ysize, xsize = fh[0].data.shape

    i = 0
    for x, y, mag in zip(xcoord[msk], ycoord[msk], mag[msk]):

        #postage stamp size
        sz = 0.2 ** ((mag - 22.) / 7.) * 50
        #cutout
        xmin = int(max(x - sz, 0))
        ymin = int(max(y - sz, 0))
        xmax = int(min(x + sz, xsize))
        ymax = int(min(y + sz, ysize))
        data = fh[0].data[ymin:ymax, xmin:xmax].copy()

        #renormalize the flux, try to cope with background
        data[data < 1e-4] = 0.0

        print data.max(), '%s%i.fits' % (output, i)
        if data.max() > 5:
            continue

        data /= data.sum()

        #savedata
        fileIO.writeFITS(data, '%s%i.fits' % (output, i), int=False)
        i +=1

    fh.close()


if __name__ == '__main__':
    inp = 'hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits'
    catalog = 'cosmos.cat'
    generaPostageStamps(inp, catalog)