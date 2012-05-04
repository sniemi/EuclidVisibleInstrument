"""
Main code of the Euclid Visible Instrument Simulator

:requires: PyFITS
:requires: NumPy
:requires: cdm03 Fortran code

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1

:TODO: Need to probably define things in quadrants as the input might be a full CCD
"""
import os, sys, datetime, math
from optparse import OptionParser
import logger as lg
import pyfits as pf
import numpy as np
import cdm03


class PostProcessing():
    """
    Euclid Visible Instrument postprocessing class. This class allows
    to add radiation damage (as defined by the CDM03 model) and add
    readout noise to a simulated image.
    """

    def __init__(self):
        """
        Class Constructor.
        """
        #setup logger
        self.assumptions = {}
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


    def writeFITSfile(self, data, output):
        """
        Write out FITS files using PyFITS.

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

        #add some header keywords
        for key, value in self.assumptions.iteritems():
            try:
                hdu.header.update(key, value)
                #ugly hack to get around the problem that one cannot
                #input numpy arrays to headers...
            except:
                pass
                #hdu.header.update(key, str(value))

        #update and verify the header
        hdu.header.add_history(
            'Created by VISsim postprocessing tool at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(output)


    def discretisetoADUs(self, data, eADU=3.5, bias=1000.0):
        """
        Convert floating point arrays to integer arrays and convert to ADUs.

        :param data: data to be discretised to.
        :type data: ndarray
        :param eADU: conversion factor (from electrons to ADUs) [default=3.5]
        :type eADU: int
        :param bias: bias level in ADUs that will be added
        :type bias: float

        :return: discretised array in ADUs
        :rtype: ndarray
        """
        #convert to ADUs
        data /= eADU
        #add bias
        data += bias

        datai = data.astype(np.int)
        datai[datai > 2 ** 16 - 1] = 2 ** 16 - 1

        self.log.info('Maximum and total values of the image are %i and %i, respectively' % (np.max(datai),
                                                                                             np.sum(datai)))

        self.information['ADUs'] = datai
        self.assumptions.update(eADU=eADU)
        self.assumptions.update(bias=bias)
        return self.information


    def applyRadiationDamage(self, data, trapfile, dob=0.0, rdose=1.0e10, iquadrant=0):
        """
        Apply radian damage based on FORTRAN CDM03 model. The method assumes that
        input data covers only a single quadrant defined by the iquadrant integer.

        :param data:
        :type data: ndarray

        :param trapfile:
        :type trapfile: str

        :param dob:

        :param rdose: radiation dosage (over the full mission)
        :type rdose: float

        :param iquandrant: number of the quandrant to process:
        :type iquandrant: int

        cdm03 - Function signature:
          sout = cdm03(sinp,iflip,jflip,dob,rdose,in_nt,in_sigma,in_tr,[xdim,ydim,zdim])
        Required arguments:
          sinp : input rank-2 array('f') with bounds (xdim,ydim)
          iflip : input int
          jflip : input int
          dob : input float
          rdose : input float
          in_nt : input rank-1 array('d') with bounds (zdim)
          in_sigma : input rank-1 array('d') with bounds (zdim)
          in_tr : input rank-1 array('d') with bounds (zdim)
        Optional arguments:
          xdim := shape(sinp,0) input int
          ydim := shape(sinp,1) input int
          zdim := len(in_nt) input int
        Return objects:
          sout : rank-2 array('f') with bounds (xdim,ydim)

        :Note: Because Python/NumPy arrays are different row/column based, one needs
               to be extra careful here. NumPy.asfortranarray will be called to get
               an array laid out in Fortran order in memory.

        :Note: This is probably not the fastest way to do this, because it now requires
               transposing the arrays twice. However, at least the CTI tracks go to the
               right direction.

        """
        #read in trap information
        trapdata = np.loadtxt(trapfile)
        nt = trapdata[:, 0]
        sigma = trapdata[:, 1]
        taur = trapdata[:, 2]

        #call Fortran routine
        CTIed = cdm03.cdm03(np.asfortranarray(data.transpose()),
            iquadrant % 2, iquadrant / 2,
            dob, rdose,
            nt, sigma, taur,
            [self.information['xsize'], self.information['ysize'], len(nt)])
        self.information['CTIed'] = CTIed.transpose()

        self.assumptions.update(nt=nt)
        self.assumptions.update(taur=taur)
        self.assumptions.update(sigma=sigma)

        return self.information


    def applyReadoutNoise(self, data, readout=4.5):
        """
        Applies readout noise. The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout).

        :param data: input data to which the readout noise will be added to
        :type data: ndarray
        :param readout: readout noise [default = 4.5]
        :type readout: float

        :return: updated data, noise image, FITS header, xsize, ysize
        :rtype: dict
        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(readout),
            size=(self.information['ysize'], self.information['xsize']))

        self.log.info('Adding readout noise...')
        self.log.info('Sum of readnoise = %f' % np.sum(noise))

        self.information['readnoised'] = data + noise
        self.information['readonoise'] = noise

        self.assumptions.update(readnoise=readout)

        return self.information


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-i', '--inputfile', dest='inputfile',
        help="Name of the file to be processed", metavar="string")

    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.inputfile is None:
        processArgs(True)
        sys.exit(1)

    process = PostProcessing()
    info = process.loadFITS(opts.inputfile)
    info = process.applyRadiationDamage(info['data'], 'cdm_euclid.dat')
    info = process.applyReadoutNoise(info['CTIed'])
    info = process.discretisetoADUs(info['readnoised'])
    info = process.writeFITSfile(info['ADUs'], 'testCTI.fits')