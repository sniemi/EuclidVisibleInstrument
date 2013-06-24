"""
Charge Transfer Inefficiency
============================

This file contains a simple class to run a CDM03 CTI model developed by Alex Short (ESA).

This now contains both the official CDM03 and a new version that allows different trap
parameters in parallel and serial direction.

:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03bidir cdm03bidir.f90)

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.2
"""
import numpy as np
try:
    import cdm03bidir
except ImportError:
    print 'No CDM03bidir module available, please compile it: f2py -c -m cdm03bidir cdm03bidir.f90'


class CDM03bidir():
    """
    Class to run CDM03 CTI model, class Fortran routine to perform the actual CDM03 calculations.

    :param data: input data to be radiated
    :type data: ndarray
    :param input: input parameters
    :type input: dictionary
    :param log: instance to Python logging
    :type log: logging instance
    """
    def __init__(self, input, data, log=None):
        """
        Class constructor.

        :param data: input data to be radiated
        :type data: ndarray
        :param input: input parameters
        :type input: dictionary
        :param log: instance to Python logging
        :type log: logging instance
        """
        self.data = data
        self.values = dict(quads=(0,1,2,3), xsize=2048, ysize=2066, dob=0.0, rdose=8.0e9)
        self.values.update(input)
        self.log = log
        self._setupLogger()


    def _setupLogger(self):
        """
        Set up the logger.
        """
        self.logger = True
        if self.log is None:
            self.logger = False


    def radiateFullCCD(self):
        """
        This routine allows the whole CCD to be run through a radiation damage mode.
        The routine takes into account the fact that the amplifiers are in the corners
        of the CCD. The routine assumes that the CCD is using four amplifiers.

        There is an excess of .copy() calls, which should probably be cleaned up. However,
        given that I had problem with the Fortran code, I have kept the calls. If memory
        becomes an issue then this should be cleaned.

        :return: radiation damaged image
        :rtype: ndarray
        """
        ydim, xdim = self.data.shape
        out = np.zeros((xdim, ydim))

        #transpose the data, because Python has different convention than Fortran
        data = self.data.transpose().copy()

        for quad in self.values['quads']:
            if self.logger:
                self.log.info('Adding CTI to Q%i' % quad)

            if quad == 0:
                d = data[0:self.values['xsize'], 0:self.values['ysize']].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[0:self.values['xsize'], 0:self.values['ysize']] = tmp
            elif quad == 1:
                d = data[self.values['xsize']:, :self.values['ysize']].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[self.values['xsize']:, :self.values['ysize']] = tmp
            elif quad == 2:
                d = data[:self.values['xsize'], self.values['ysize']:].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[:self.values['xsize'], self.values['ysize']:] = tmp
            elif quad == 3:
                d = data[self.values['xsize']:, self.values['ysize']:].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[self.values['xsize']:, self.values['ysize']:] = tmp
            else:
                print 'ERROR -- too many quadrants!!'
                self.log.error('Too many quadrants! This method allows only four quadrants.')

        return out.transpose()


    def radiateFullCCD2(self):
         """
         This routine allows the whole CCD to be run through a radiation damage mode.
         The routine takes into account the fact that the amplifiers are in the corners
         of the CCD. The routine assumes that the CCD is using four amplifiers.

         There is an excess of .copy() calls, which should probably be cleaned up. However,
         given that I had problem with the Fortran code, I have kept the calls. If memory
         becomes an issue then this should be cleaned.

         :return: radiation damaged image
         :rtype: ndarray
         """
         ydim, xdim = self.data.shape
         out = np.empty((ydim, xdim))

         #transpose the data, because Python has different convention than Fortran
         data = self.data.copy()

         for quad in self.values['quads']:
             if self.logger:
                 self.log.info('Adding CTI to Q%i' % quad)

             if quad == 0:
                 d = data[:self.values['ysize'], :self.values['xsize']].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[:self.values['ysize'], :self.values['xsize']] = tmp
             elif quad == 1:
                 d = data[:self.values['ysize'], self.values['xsize']:].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[:self.values['ysize'], self.values['xsize']:] = tmp
             elif quad == 2:
                 d = data[self.values['ysize']:, :self.values['xsize']].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[self.values['ysize']:, :self.values['xsize']] = tmp
             elif quad == 3:
                 d = data[self.values['ysize']:, self.values['xsize']:].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[self.values['ysize']:, self.values['xsize']:] = tmp
             else:
                 print 'ERROR -- too many quadrants!!'
                 self.log.error('Too many quadrants! This method allows only four quadrants.')

         return out


    def applyRadiationDamage(self, data, iquadrant=0):
        """
        Apply radian damage based on FORTRAN CDM03 model. The method assumes that
        input data covers only a single quadrant defined by the iquadrant integer.

        :param data: imaging data to which the CDM03 model will be applied to.
        :type data: ndarray

        :param iquandrant: number of the quadrant to process
        :type iquandrant: int

        cdm03 - Function signature::

              sout = cdm03(sinp,iflip,jflip,dob,rdose,in_nt,in_sigma,in_tr,[xdim,ydim,zdim])
            Required arguments:
              sinp : input rank-2 array('d') with bounds (xdim,ydim)
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
              sout : rank-2 array('d') with bounds (xdim,ydim)

        .. Note:: Because Python/NumPy arrays are different row/column based, one needs
                  to be extra careful here. NumPy.asfortranarray will be called to get
                  an array laid out in Fortran order in memory. Before returning the
                  array will be laid out in memory in C-style (row-major order).

        :return: image that has been run through the CDM03 model
        :rtype: ndarray
        """""
        #read in trap information
        trapdata = np.loadtxt(self.values['parallelTrapfile'])
        nt_p = trapdata[:, 0]
        sigma_p = trapdata[:, 1]
        taur_p = trapdata[:, 2]

        trapdata = np.loadtxt(self.values['serialTrapfile'])
        nt_s = trapdata[:, 0]
        sigma_s = trapdata[:, 1]
        taur_s = trapdata[:, 2]

        iflip = iquadrant / 2
        jflip = iquadrant % 2

        if self.logger:
            self.log.info('nt_p=' + str(nt_p))
            self.log.info('nt_s=' + str(nt_s))
            self.log.info('sigma_p= ' + str(sigma_p))
            self.log.info('sigma_s= ' + str(sigma_s))
            self.log.info('taur_p= ' + str(taur_p))
            self.log.info('taur_s= ' + str(taur_s))
            self.log.info('dob=%f' % self.values['dob'])
            self.log.info('rdose=%e' % self.values['rdose'])
            self.log.info('xsize=%i' % data.shape[1])
            self.log.info('ysize=%i' % data.shape[0])
            self.log.info('quadrant=%i' % iquadrant)
            self.log.info('iflip=%i' % iflip)
            self.log.info('jflip=%i' % jflip)

        CTIed = cdm03bidir.cdm03(np.asfortranarray(data),
                                 iquadrant % 2, iquadrant / 2,
                                 self.values['dob'], self.values['rdose'],
                                 nt_p, sigma_p, taur_p,
                                 nt_s, sigma_s, taur_s,
                                 [data.shape[0], data.shape[1], len(nt_p), len(nt_s)])
        return np.asanyarray(CTIed)


class CDM03():
    """
    Class to run CDM03 CTI model, class Fortran routine to perform the actual CDM03 calculations.

    :param data: input data to be radiated
    :type data: ndarray
    :param input: input parameters
    :type input: dictionary
    :param log: instance to Python logging
    :type log: logging instance
    """
    def __init__(self, input, data, log=None):
        """
        Class constructor.

        :param data: input data to be radiated
        :type data: ndarray
        :param input: input parameters
        :type input: dictionary
        :param log: instance to Python logging
        :type log: logging instance
        """
        try:
            import cdm03
        except ImportError:
            print 'No CDM03 module available, please compile it: f2py -c -m cdm03 cdm03.f90'

        self.data = data
        self.values = dict(quads=(0,1,2,3), xsize=2048, ysize=2066, dob=0.0, rdose=8.0e9)
        self.values.update(input)
        self.log = log
        self._setupLogger()


    def _setupLogger(self):
        """
        Set up the logger.
        """
        self.logger = True
        if self.log is None:
            self.logger = False


    def radiateFullCCD(self):
        """
        This routine allows the whole CCD to be run through a radiation damage mode.
        The routine takes into account the fact that the amplifiers are in the corners
        of the CCD. The routine assumes that the CCD is using four amplifiers.

        There is an excess of .copy() calls, which should probably be cleaned up. However,
        given that I had problem with the Fortran code, I have kept the calls. If memory
        becomes an issue then this should be cleaned.

        :return: radiation damaged image
        :rtype: ndarray
        """
        ydim, xdim = self.data.shape
        out = np.zeros((xdim, ydim))

        #transpose the data, because Python has different convention than Fortran
        data = self.data.transpose().copy()

        for quad in self.values['quads']:
            if self.logger:
                self.log.info('Adding CTI to Q%i' % quad)

            if quad == 0:
                d = data[0:self.values['xsize'], 0:self.values['ysize']].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[0:self.values['xsize'], 0:self.values['ysize']] = tmp
            elif quad == 1:
                d = data[self.values['xsize']:, :self.values['ysize']].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[self.values['xsize']:, :self.values['ysize']] = tmp
            elif quad == 2:
                d = data[:self.values['xsize'], self.values['ysize']:].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[:self.values['xsize'], self.values['ysize']:] = tmp
            elif quad == 3:
                d = data[self.values['xsize']:, self.values['ysize']:].copy()
                tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                out[self.values['xsize']:, self.values['ysize']:] = tmp
            else:
                print 'ERROR -- too many quadrants!!'
                self.log.error('Too many quadrants! This method allows only four quadrants.')

        return out.transpose()


    def radiateFullCCD2(self):
         """
         This routine allows the whole CCD to be run through a radiation damage mode.
         The routine takes into account the fact that the amplifiers are in the corners
         of the CCD. The routine assumes that the CCD is using four amplifiers.

         There is an excess of .copy() calls, which should probably be cleaned up. However,
         given that I had problem with the Fortran code, I have kept the calls. If memory
         becomes an issue then this should be cleaned.

         :return: radiation damaged image
         :rtype: ndarray
         """
         ydim, xdim = self.data.shape
         out = np.empty((ydim, xdim))

         #transpose the data, because Python has different convention than Fortran
         data = self.data.copy()

         for quad in self.values['quads']:
             if self.logger:
                 self.log.info('Adding CTI to Q%i' % quad)

             if quad == 0:
                 d = data[:self.values['ysize'], :self.values['xsize']].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[:self.values['ysize'], :self.values['xsize']] = tmp
             elif quad == 1:
                 d = data[:self.values['ysize'], self.values['xsize']:].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[:self.values['ysize'], self.values['xsize']:] = tmp
             elif quad == 2:
                 d = data[self.values['ysize']:, :self.values['xsize']].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[self.values['ysize']:, :self.values['xsize']] = tmp
             elif quad == 3:
                 d = data[self.values['ysize']:, self.values['xsize']:].copy()
                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
                 out[self.values['ysize']:, self.values['xsize']:] = tmp
             else:
                 print 'ERROR -- too many quadrants!!'
                 self.log.error('Too many quadrants! This method allows only four quadrants.')

         return out


    def applyRadiationDamage(self, data, iquadrant=0):
        """
        Apply radian damage based on FORTRAN CDM03 model. The method assumes that
        input data covers only a single quadrant defined by the iquadrant integer.

        :param data: imaging data to which the CDM03 model will be applied to.
        :type data: ndarray

        :param iquandrant: number of the quadrant to process
        :type iquandrant: int

        cdm03 - Function signature::

              sout = cdm03(sinp,iflip,jflip,dob,rdose,in_nt,in_sigma,in_tr,[xdim,ydim,zdim])
            Required arguments:
              sinp : input rank-2 array('d') with bounds (xdim,ydim)
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
              sout : rank-2 array('d') with bounds (xdim,ydim)

        .. Note:: Because Python/NumPy arrays are different row/column based, one needs
                  to be extra careful here. NumPy.asfortranarray will be called to get
                  an array laid out in Fortran order in memory. Before returning the
                  array will be laid out in memory in C-style (row-major order).

        :return: image that has been run through the CDM03 model
        :rtype: ndarray
        """
        #read in trap information
        trapdata = np.loadtxt(self.values['trapfile'])
        nt = trapdata[:, 0]
        sigma = trapdata[:, 1]
        taur = trapdata[:, 2]

        iflip = iquadrant / 2
        jflip = iquadrant % 2

        if self.logger:
            self.log.info('nt=' + str(nt))
            self.log.info('sigma= ' + str(sigma))
            self.log.info('taur= ' + str(taur))
            self.log.info('dob=%f' % self.values['dob'])
            self.log.info('rdose=%e' % self.values['rdose'])
            self.log.info('xsize=%i' % data.shape[1])
            self.log.info('ysize=%i' % data.shape[0])
            self.log.info('quadrant=%i' % iquadrant)
            self.log.info('iflip=%i' % iflip)
            self.log.info('jflip=%i' % jflip)


        #call Fortran routine
        CTIed = cdm03.cdm03(np.asfortranarray(data),
                            iflip, jflip,
                            self.values['dob'], self.values['rdose'],
                            nt, sigma, taur)

        return np.asanyarray(CTIed)