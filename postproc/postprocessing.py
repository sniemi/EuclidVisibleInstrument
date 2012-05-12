"""
A class to insert instrument specific features to a simulated image. Supports multiprocessing.

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.5
"""
import os, sys, datetime, time, math
import multiprocessing
import Queue
import glob as g
import logger as lg
import pyfits as pf
import numpy as np
import cdm03


class PostProcessing(multiprocessing.Process):
    """
    Euclid Visible Instrument postprocessing class. This class allows
    to add radiation damage (as defined by the CDM03 model) and add
    readout noise to a simulated image.
    """

    def __init__(self, values, work_queue, result_queue):
        """
        Class Constructor.
        """
        # base class initialization
        multiprocessing.Process.__init__(self)

        # job management queues
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False

        self.values = values

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
        size = data.shape

        self.log.info('Read data from %i extension of file %s' % (ext, filename))

        return dict(data=data, header=header, ysize=size[0], xsize=size[1])
    
    
    def cutoutRegion(self, data):
        """
        Cuts out a region from the imaging data. The cutout region is specified by
        xstart/stop and ystart/stop that are read out from the self.values dictionary.
        """
        out = data[self.values['ystart']:self.values['ystop'], self.values['xstart']:self.values['xstop']].copy()
        out[out > self.values['cutoff']] = 33e3
        return out
        

    def writeFITSfile(self, data, output, unsigned16bit=True):
        """
        Write out FITS files using PyFITS.

        :param data: data to write to a FITS file
        :type data: ndarray
        :param output: name of the output file
        :type output: string
        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool

        :return: None
        """
        if os.path.isfile(output):
            os.remove(output)

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=data)

        #convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)

        #add keywords
        for key, value in self.values.iteritems():
            hdu.header.update(key, value)

        #update and verify the header
        hdu.header.add_history('Created with VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.header.add_history('The following processing steps have been performed:')
        hdu.header.add_history('1)Original file has been cut to VIS size using xstart/stop and ystart/stop')
        hdu.header.add_history('2)Pixels with values greater than 65k were set to 33k (to prevent long trails)')
        hdu.header.add_history('3)CDM03 CTI model were applied to each quadrant separately')
        hdu.header.add_history('4)All four quadrants were combined to for a single CCD')
        hdu.header.add_history('5)Readnoise drawn from Normal distribution were added')
        hdu.header.add_history('6)Values were then converted from electrons to ADUs')
        hdu.header.add_history('7)Bias level were added')
        hdu.header.add_history('8)Image was saved to a FITS file in 16bit unsigned integer format')
        hdu.header.add_history('In addition to these steps also CTI corrected images were produced,')
        hdu.header.add_history('see CTIcorrected.fits')
        hdu.header.add_history('Contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk) if questions.')
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(output)


    def discretisetoADUs(self, data):
        """
        Convert floating point arrays to integer arrays and convert to ADUs.
        Adds bias level after converting to ADUs.

        :param data: data to be discretised to.
        :type data: ndarray

        :return: discretised array in ADUs
        :rtype: ndarray
        """
        #convert to ADUs
        data /= self.values['eADU']
        #add bias
        data += self.values['bias']

        datai = data.astype(np.int)
        datai[datai > 2 ** 16 - 1] = 2 ** 16 - 1

        self.log.info('Maximum and total values of the image are %i and %i, respectively' % (np.max(datai),
                                                                                             np.sum(datai)))

        return datai


    def radiateFullCCD(self, fullCCD, quads=(0,1,2,3), xsize=2048, ysize=2066):
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
        out = np.zeros(fullCCD.shape)

        for quad in quads:
            if quad == 0:
                data = fullCCD[ysize:, 0:xsize].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=2)
                out[ysize:, 0:xsize] = tmp.transpose()
            elif quad == 1:
                data = fullCCD[ysize:, xsize:].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=3)
                out[ysize:, xsize:] = tmp.transpose()
            elif quad == 2:
                data = fullCCD[0:ysize, 0:xsize].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=0)
                out[0:ysize, 0:xsize] = tmp.transpose()
            elif quad == 3:
                data = fullCCD[0:ysize, xsize:].copy().transpose()
                tmp = self.applyRadiationDamage(data, iquadrant=1)
                out[0:ysize, xsize:] = tmp.transpose()
            else:
                print 'ERROR -- too many quadrants!!'

        return out


    def applyRadiationDamage(self, data, iquadrant=0):
        """
        Apply radian damage based on FORTRAN CDM03 model. The method assumes that
        input data covers only a single quadrant defined by the iquadrant integer.

        :param data: imaging data to which the CDM03 model will be applied to.
        :type data: ndarray

        :param iquandrant: number of the quadrant to process
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

        :return: image that has been run through the CDM03 model
        :rtype: ndarray
        """
        #read in trap information
        trapdata = np.loadtxt(self.values['trapfile'])
        nt = trapdata[:, 0]
        sigma = trapdata[:, 1]
        taur = trapdata[:, 2]

        #call Fortran routine
        CTIed = cdm03.cdm03(np.asfortranarray(data),
                            iquadrant % 2, iquadrant / 2,
                            self.values['dob'], self.values['rdose'],
                            nt, sigma, taur,
                            [data.shape[0], data.shape[1], len(nt)])

        return CTIed


    def applyReadoutNoise(self, data):
        """
        Applies readout noise. The noise is drawn from a Normal (Gaussian) distribution.
        Mean = 0.0, and std = sqrt(readout).

        :param data: input data to which the readout noise will be added to
        :type data: ndarray

        :return: updated data, noise image
        :rtype: dict
        """
        noise = np.random.normal(loc=0.0, scale=math.sqrt(self.values['rnoise']), size=data.shape)

        self.log.info('Adding readout noise...')
        self.log.info('Sum of readnoise = %f' % np.sum(noise))

        readnoised = data + noise
        readnoise = noise

        out = dict(readnoised=readnoised, readnoise=readnoise)

        return out


    def generateCTImap(self, CTIed, originalData):
        """
        Calculates a map showing the CTI effect. This map is being
        generated by dividing radiation damaged image with the original
        data.

        :param CTIed: Radiation damaged image
        :type CTIed: ndarray
        :param originalData: Original image before any radiation damage
        :type originalData: ndarray

        :return: CTI map (ratio of radiation damaged image and original data)
        :rtype: ndarray
        """
        return CTIed / originalData


    def applyLinearCorrection(self, image):
        """
        Applies a linear correction after one forward readout through the CDM03 model.

        Bristow & Alexov (2003) algorithm further developed for HST data
        processing by Massey, Rhodes et al.

        :param image: radiation damaged image
        :type image: ndarray

        :return: corrected image after single forward readout
        :rtype: ndarray
        """
        return 2.*image - self.radiateFullCCD(image)


    def run(self):
        """
        This is the method that will be called when multiprocessing.
        """
        while not self.kill_received:
            # get a task from the queue
            try:
                filename = self.work_queue.get_nowait()
            except Queue.Empty:
                break

            file = filename.split('/')[-1].split('.fits')[0]

            print 'Started processing %s\n' % filename
            start_time = time.time()

            #calculate new frames
            info = self.loadFITS(filename)
            data = self.cutoutRegion(info['data'])
            CTIed = self.radiateFullCCD(data)
            noised = self.applyReadoutNoise(CTIed)
            datai = self.discretisetoADUs(noised['readnoised'])
            CTImap = self.generateCTImap(CTIed, data)

            #apply correction
            corrected = self.applyLinearCorrection(noised['readnoised'])
            CTImap2 = self.generateCTImap(corrected, data)

            #write some outputs
            self.writeFITSfile(datai, file+'CTI.fits')
            self.writeFITSfile(CTImap, file+'CTImap.fits', unsigned16bit=False)
            self.writeFITSfile(CTImap2, file+'CTIresidual.fits', unsigned16bit=False)
            self.writeFITSfile(corrected, file+'CTIcorrected.fits', unsigned16bit=False)

            # store the result, not really necessary in this case, but for info...
            str = '\nFinished processing %s.fits, took about %.1f minutes to run' % (file, -(start_time - time.time()) / 60.)
            self.result_queue.put(str)



if __name__ == '__main__':
    #how many processes to use?
    num_processes = 2

    #input values that are used in processing and save to the FITS headers
    values = {'rnoise' : 4.5, 'dob' : 0, 'rdose' : 3e10, 'trapfile' : 'cdm_euclid.dat', 'eADU' : 3.5,
              'bias' : 1000.0, 'beta' : 0.6, 'fwc' : 175000, 'vth' : 1.168e7, 't' : 1.024e-2, 'vg' : 6.e-11 ,
              'st' : 5.e-6, 'sfwc' : 730000., 'svg' : 1.0e-10, 'ystart' : 560, 'xstart' : 560, 'ystop' : 4692,
              'xstop' : 4656, 'cutoff' : 65000.0}

    #find all files to be processed
    inputs = g.glob('*.fits')

    # load up work queue
    work_queue = multiprocessing.Queue()
    for file in inputs:
        work_queue.put(file)

    # create a queue to pass to workers to store the results
    result_queue = multiprocessing.Queue()

    # spawn workers
    for file in inputs:
        worker = PostProcessing(values, work_queue, result_queue)
        worker.start()

    # collect the results off the queue
    results = []
    for file in inputs:
        print(result_queue.get())

    print 'All done...'
