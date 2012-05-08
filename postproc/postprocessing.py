"""
Tools to insert instrument specific features to a simulated image. Supports threading.

:requires: PyFITS
:requires: NumPy
:requires: CDM03 Fortran code

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.3
"""
import os, sys, datetime, time, math
import threading as t
import Queue as Q
import glob as g
import logger as lg
import pyfits as pf
import numpy as np
import cdm03


class PostProcessing(t.Thread):
    """
    Euclid Visible Instrument postprocessing class. This class allows
    to add radiation damage (as defined by the CDM03 model) and add
    readout noise to a simulated image.
    """

    def __init__(self, queue):
        """
        Class Constructor.
        """
        t.Thread.__init__(self)
        self.queue = queue

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

        return dict(data=data, header=header, ysize=size[0], xsize=size[1])


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

        assumptions= {}
        assumptions.update(eADU=eADU)
        assumptions.update(bias=bias)
        return datai, assumptions


    def radiateFullCCD(self, fullCCD, quads=[0,1,2,3], xsize=2048, ysize=2066):
        """
        This routine allows the whole CCD to be run through a radiation damage mode.
        The routine takes into account the fact that the amplifiers are in the corners
        of the CCD. The routine assumes that the CCD is using four amplifiers.

        :param fullCCD: image of containing the whole CCD
        :type fullCCD: ndarray
        :param quads: quadrants, numbered from lower left
        :type quads: list

        :return: radiation damaged image
        :rtype: ndarray
        """
        out = np.ones(fullCCD.shape)
        for i, quad in enumerate(quads):
            if i == 0:
                data = fullCCD[:ysize, :xsize].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=quad)
                out[:ysize, :xsize] = tmp.transpose()
            elif i == 1:
                data = fullCCD[:ysize, xsize:].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=quad)
                out[:ysize, xsize:] = tmp.transpose()
            elif i == 2:
                data = fullCCD[ysize:, :xsize].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=quad)
                out[ysize:, :xsize] = tmp.transpose()
            else:
                data = fullCCD[ysize:, xsize:].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=quad)
                out[ysize:, xsize:] = tmp.transpose()

        return out


    def applyRadiationDamage(self, data, trapfile='cdm_euclid.dat', dob=0.0, rdose=1.0e10, iquadrant=0):
        """
        Apply radian damage based on FORTRAN CDM03 model. The method assumes that
        input data covers only a single quadrant defined by the iquadrant integer.

        :param data: imaging data to which the CDM03 model will be applied to.
        :type data: ndarray

        :param trapfile: name of the file containing charge trap information [default=cdm_euclid.dat]
        :type trapfile: str

        :param dob:  diffuse (long term) optical background [e-/pixel/transit]
        :type dob: float

        :param rdose: radiation dosage (over the full mission)
        :type rdose: float

        :param iquandrant: number of the quadrant to process:
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

        #call Fortran routine, transpose the arrays
        CTIed = cdm03.cdm03(np.asfortranarray(data),
                            iquadrant % 2, iquadrant / 2,
                            dob, rdose,
                            nt, sigma, taur,
                            [data.shape[0], data.shape[1], len(nt)])

        return CTIed


    def applyReadoutNoise(self, data, readout=4.5):
        """
        Applies readout noise. The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout).

        :param data: input data to which the readout noise will be added to
        :type data: ndarray
        :param readout: readout noise [default = 4.5]
        :type readout: float

        :return: updated data, noise image
        :rtype: dict
        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(readout), size=data.shape)

        self.log.info('Adding readout noise...')
        self.log.info('Sum of readnoise = %f' % np.sum(noise))

        readnoised = data + noise
        readnoise = noise

        out = dict(readnoised=readnoised, readnoise=readnoise)

        return out


    def generateCTImap(self, CTIed, data):
        """
        Calculates a map showing the CTI effect. This map is being
        generated by dividing radiation damaged image with the
        """
        return CTIed / data


    def run(self):
        """
        The method threading will call.
        """
        while True:
            #grabs a file from queue
            filename = self.queue.get()

            file = filename.split('/')[-1].split('.fits')[0]

            print 'Started precessing %s\n' % filename
            start_time = time.time()

            info = self.loadFITS(filename)
            CTIed = self.radiateFullCCD(info['data'])
            noised = self.applyReadoutNoise(CTIed)
            datai, assumptions = self.discretisetoADUs(noised['readnoised'])
            CTImap = self.generateCTImap(CTIed, info['data'])

            self.writeFITSfile(datai, file+'CTI.fits')
            self.writeFITSfile(CTImap, file+'CTImap.fits')

            print '\nFinished processing %s.fits, took about %.0f minutes to run' % (file, -(start_time - time.time()) / 60.)

            #signals to queue job is done
            self.queue.task_done()


def main(input_files, cores=6):
    """
    Main driver function of the wrapper.
    """
    queue = Q.Queue()
    #spawn a pool of threads, and pass them queue instance
    for i in range(cores):
        th = PostProcessing(queue)
        th.setDaemon(True)
        th.start()

    for file in input_files:
        queue.put(file)

    #wait on the queue until everything has been processed
    queue.join()


if __name__ == '__main__':
    cores = 4
    inputs = g.glob('*.fits')

    #call the main function
    main(inputs, cores)

    print 'All done...'
