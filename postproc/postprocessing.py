"""
Main code of the Euclid Visible Instrument Simulator

:requires: PyFITS
:requires: NumPy
:requires: cdm03 Fortran code

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import os, sys, datetime, math
import simulator.logger as lg
import pyfits as pf
import numpy as np
import cdm03


class PostProcessing():
    """
    Euclid Visible Instrument postprocessing class.
    """

    def __init__(self):
        """
        Class Constructor.
        """
        #setup logger
        self.log = lg.setUpLogger('PostProcessing.log')


    def loadFITS(self, filename, ext=0):
        """
        Loads data from a given FITS file and extension.

        :param filename: name of the FITS file
        :type filename: str
        :param ext: FITS header extension [default=0]
        :type ext: int

        :return: data, FITS header, xsize, ysize
        :rtype: dict
        """
        fh = pf.open(filename)
        data = fh[ext].data
        header = fh[ext].header

        self.log.info('Read data from %i extension of file %s' % (ext, filename))

        size = data.shape

        self.information = dict(data=data, header=header, ysize=size[0], xsize=size[1])
        return self.information


    def applyReadoutNoise(self, readout=4.5):
        """
        Applies readout noise. The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout).

        :param readout: readout noise [default = 4.5]
        :type readout: float

        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(readout),
                                 size=(self.information['ysize'], self.information['xsize']))
        self.log.info('Adding readout noise...')
        self.log.info('Sum of readnoise = %f' % np.sum(noise))
        self.information['data'] += noise


    def applyRadiationDamage(self, trapfile, dob=0.0, rdose=1.0e10):
        """

        """
        #read in trap information
        trapdata = np.loadtxt(trapfile)
        nt = trapdata[:, 0]
        sigma = trapdata[:, 1]
        taur = trapdata[:, 2]

        ntraps = len(nt)

        #call Fortran routine
        x = cdm03.cdm03(self.information['data'],
                        self.information['xsize'],
                        self.information['ysize'],sout,
                        iflip,jflip,
                        dob,
                        rdose,
                        ntraps,
                        nt,
                        sigma,
                        taur)


if __name__ == '__main__':
    pass