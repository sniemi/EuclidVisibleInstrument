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

:version: 0.35
"""
import numpy as np
try:
    import cdm03bidir
    #import cdm03bidirTest as cdm03bidir  #for testing purposes only
except ImportError:
    print 'No CDM03bidir module available, please compile it: f2py -c -m cdm03bidir cdm03bidir.f90'

#try:
#    from numba import autojit
#    from numba import jit
#    from numba import double, int16
#except:
#    print 'No numba available!'


class CDM03bidir():
    """
    Class to run CDM03 CTI model, class Fortran routine to perform the actual CDM03 calculations.

     :param settings: input parameters
     :type settings: dict
     :param data: input data to be radiated
     :type data: ndarray
     :param log: instance to Python logging
     :type log: logging instance
    """
    def __init__(self, settings, data, log=None):
        """
        Class constructor.

        :param settings: input parameters
        :type settings: dict
        :param data: input data to be radiated
        :type data: ndarray
        :param log: instance to Python logging
        :type log: logging instance
        """
        self.data = data
        self.values = dict(quads=(0,1,2,3), xsize=2048, ysize=2066, dob=0.0, rdose=8.0e9)
        self.values.update(settings)
        self.log = log
        self._setupLogger()

        #default CDM03 settings
        self.params = dict(beta_p=0.6, beta_s=0.6, fwc=200000., vth=1.168e7, vg=6.e-11, t=20.48e-3,
                           sfwc=730000., svg=1.0e-10, st=5.0e-6, parallel=1., serial=1.)
        #update with inputs
        self.params.update(self.values)

        #read in trap information
        trapdata = np.loadtxt(self.values['parallelTrapfile'])
        if trapdata.ndim > 1:
            self.nt_p = trapdata[:, 0]
            self.sigma_p = trapdata[:, 1]
            self.taur_p = trapdata[:, 2]
        else:
            #only one trap species
            self.nt_p = [trapdata[0],]
            self.sigma_p = [trapdata[1],]
            self.taur_p = [trapdata[2],]

        trapdata = np.loadtxt(self.values['serialTrapfile'])
        if trapdata.ndim > 1:
            self.nt_s = trapdata[:, 0]
            self.sigma_s = trapdata[:, 1]
            self.taur_s = trapdata[:, 2]
        else:
            #only one trap species
            self.nt_s = [trapdata[0],]
            self.sigma_s = [trapdata[1],]
            self.taur_s = [trapdata[2],]

        #scale thibaut's values
        if 'thibaut' in self.values['parallelTrapfile']:
            self.nt_p /= 0.576  #thibaut's values traps / pixel
            self.sigma_p *= 1.e4 #thibaut's values in m**2
        if 'thibaut' in self.values['serialTrapfile']:
            self.nt_s *= 0.576 #thibaut's values traps / pixel  #should be division?
            self.sigma_s *= 1.e4 #thibaut's values in m**2


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
        iflip = iquadrant / 2
        jflip = iquadrant % 2

        params = [self.params['beta_p'], self.params['beta_s'], self.params['fwc'], self.params['vth'],
                  self.params['vg'], self.params['t'], self.params['sfwc'], self.params['svg'], self.params['st'],
                  self.params['parallel'], self.params['serial']]

        if self.logger:
            self.log.info('nt_p=' + str(self.nt_p))
            self.log.info('nt_s=' + str(self.nt_s))
            self.log.info('sigma_p= ' + str(self.sigma_p))
            self.log.info('sigma_s= ' + str(self.sigma_s))
            self.log.info('taur_p= ' + str(self.taur_p))
            self.log.info('taur_s= ' + str(self.taur_s))
            self.log.info('dob=%f' % self.values['dob'])
            self.log.info('rdose=%e' % self.values['rdose'])
            self.log.info('xsize=%i' % data.shape[1])
            self.log.info('ysize=%i' % data.shape[0])
            self.log.info('quadrant=%i' % iquadrant)
            self.log.info('iflip=%i' % iflip)
            self.log.info('jflip=%i' % jflip)

        CTIed = cdm03bidir.cdm03(np.asfortranarray(data),
                                 jflip, iflip,
                                 self.values['dob'], self.values['rdose'],
                                 self.nt_p, self.sigma_p, self.taur_p,
                                 self.nt_s, self.sigma_s, self.taur_s,
                                 params,
                                 [data.shape[0], data.shape[1], len(self.nt_p), len(self.nt_s), len(self.params)])
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



# class CDM03Python():
#     def __init__(self, input, data, log=None):
#         """
#         Class constructor.
#
#         :param data: input data to be radiated
#         :type data: ndarray
#         :param input: input parameters
#         :type input: dictionary
#         :param log: instance to Python logging
#         :type log: logging instance
#         """
#         self.data = data
#         self.values = dict(quads=(0, 1, 2, 3), xsize=2048, ysize=2066, dob=0.0, rdose=8.0e9)
#         self.values.update(input)
#         self.log = log
#         self._setupLogger()
#
#
#     def _setupLogger(self):
#         """
#         Set up the logger.
#         """
#         self.logger = True
#         if self.log is None:
#             self.logger = False
#
#
#     def radiateFullCCD(self):
#         """
#         This routine allows the whole CCD to be run through a radiation damage mode.
#         The routine takes into account the fact that the amplifiers are in the corners
#         of the CCD. The routine assumes that the CCD is using four amplifiers.
#
#         There is an excess of .copy() calls, which should probably be cleaned up. However,
#         given that I had problem with the Fortran code, I have kept the calls. If memory
#         becomes an issue then this should be cleaned.
#
#         :return: radiation damaged image
#         :rtype: ndarray
#         """
#         ydim, xdim = self.data.shape
#         out = np.zeros((xdim, ydim))
#
#         #transpose the data, because Python has different convention than Fortran
#         data = self.data.transpose().copy()
#
#         for quad in self.values['quads']:
#             if self.logger:
#                 self.log.info('Adding CTI to Q%i' % quad)
#
#             if quad == 0:
#                 d = data[0:self.values['xsize'], 0:self.values['ysize']].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[0:self.values['xsize'], 0:self.values['ysize']] = tmp
#             elif quad == 1:
#                 d = data[self.values['xsize']:, :self.values['ysize']].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[self.values['xsize']:, :self.values['ysize']] = tmp
#             elif quad == 2:
#                 d = data[:self.values['xsize'], self.values['ysize']:].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[:self.values['xsize'], self.values['ysize']:] = tmp
#             elif quad == 3:
#                 d = data[self.values['xsize']:, self.values['ysize']:].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[self.values['xsize']:, self.values['ysize']:] = tmp
#             else:
#                 print 'ERROR -- too many quadrants!!'
#                 self.log.error('Too many quadrants! This method allows only four quadrants.')
#
#         return out.transpose()
#
#
#     def radiateFullCCD2(self):
#         """
#         This routine allows the whole CCD to be run through a radiation damage mode.
#         The routine takes into account the fact that the amplifiers are in the corners
#         of the CCD. The routine assumes that the CCD is using four amplifiers.
#
#         There is an excess of .copy() calls, which should probably be cleaned up. However,
#         given that I had problem with the Fortran code, I have kept the calls. If memory
#         becomes an issue then this should be cleaned.
#
#         :return: radiation damaged image
#         :rtype: ndarray
#         """
#         ydim, xdim = self.data.shape
#         out = np.empty((ydim, xdim))
#
#         #transpose the data, because Python has different convention than Fortran
#         data = self.data.copy()
#
#         for quad in self.values['quads']:
#             if self.logger:
#                 self.log.info('Adding CTI to Q%i' % quad)
#
#             if quad == 0:
#                 d = data[:self.values['ysize'], :self.values['xsize']].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[:self.values['ysize'], :self.values['xsize']] = tmp
#             elif quad == 1:
#                 d = data[:self.values['ysize'], self.values['xsize']:].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[:self.values['ysize'], self.values['xsize']:] = tmp
#             elif quad == 2:
#                 d = data[self.values['ysize']:, :self.values['xsize']].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[self.values['ysize']:, :self.values['xsize']] = tmp
#             elif quad == 3:
#                 d = data[self.values['ysize']:, self.values['xsize']:].copy()
#                 tmp = self.applyRadiationDamage(d, iquadrant=quad).copy()
#                 out[self.values['ysize']:, self.values['xsize']:] = tmp
#             else:
#                 print 'ERROR -- too many quadrants!!'
#                 self.log.error('Too many quadrants! This method allows only four quadrants.')
#
#         return out
#
#
#     def applyRadiationDamage(self, data, nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, rdose=1.6e10, iquadrant=0):
#         """
#         Apply radian damage based on FORTRAN CDM03 model. The method assumes that
#         input data covers only a single quadrant defined by the iquadrant integer.
#
#         :param data: imaging data to which the CDM03 model will be applied to.
#         :type data: ndarray
#
#         :param iquandrant: number of the quadrant to process
#         :type iquandrant: int
#
#         :return: image that has been run through the CDM03 model
#         :rtype: ndarray
#         """
#         iflip = iquadrant / 2
#         jflip = iquadrant % 2
#
#         if self.logger:
#             self.log.info('nt_p=' + str(nt_p))
#             self.log.info('nt_s=' + str(nt_s))
#             self.log.info('sigma_p= ' + str(sigma_p))
#             self.log.info('sigma_s= ' + str(sigma_s))
#             self.log.info('taur_p= ' + str(taur_p))
#             self.log.info('taur_s= ' + str(taur_s))
#             self.log.info('dob=%f' % self.values['dob'])
#             self.log.info('rdose=%e' % self.values['rdose'])
#             self.log.info('xsize=%i' % data.shape[1])
#             self.log.info('ysize=%i' % data.shape[0])
#             self.log.info('quadrant=%i' % iquadrant)
#             self.log.info('iflip=%i' % iflip)
#             self.log.info('jflip=%i' % jflip)
#
#         #return run(data, nt_s, sigma_p, taur_p, nt_s, sigma_s, taur_s, iflip, jflip, True, True)
#         return run(data)#, [nt_s, sigma_p, taur_p, nt_s, sigma_s, taur_s, iflip, jflip, 1, 1])
#
#
# @autojit
# #@jit(double[:,:], double[:], double[:], double[:], double[:], double[:], double[:], int, int, int, int)
# #def run(image, nt_p, sigma_p, tr_p, nt_s, sigma_s, tr_s, iflip, jflip, parallel, serial):
# #@jit(argtypes=[double[:,:], [double[:], double[:], double[:], double[:], double[:], double[:], int16, int16, int16, int16]])
# #def run(image, params):
# #@jit(argtypes=double[:,:], restype=double[:,:])
# def run(image):
#     parallel = 'cdm_euclid_parallel.dat'
#     serial = 'cdm_euclid_serial.dat'
#     trapdata = np.loadtxt(parallel)
#     nt_p = trapdata[:, 0]
#     sigma_p = trapdata[:, 1]
#     tr_p = trapdata[:, 2]
#
#     trapdata = np.loadtxt(serial)
#     nt_s = trapdata[:, 0]
#     sigma_s = trapdata[:, 1]
#     tr_s = trapdata[:, 2]
#
#     iflip = 0
#     jflip = 0
#     parallel = True
#     serial = True
#
#     rdose = 8.0e9; dob = 0.0; beta_p = 0.6; beta_s = 0.6
#     fwc = 200000.; vth = 1.168e7; vg = 6.e-11; t = 20.48e-3
#     sfwc = 730000.; svg = 1.0e-10; st = 5.0e-6
#
#     # absolute trap density which should be scaled according to radiation dose
#     # (nt=1.5e10 gives approx fit to GH data for a dose of 8e9 10MeV equiv. protons)
#     nt_p = nt_p * rdose                    #absolute trap density [per cm**3]
#     nt_s = nt_s * rdose                    #absolute trap density [per cm**3]
#
#     #array sizes
#     ydim, xdim = image.shape
#     zdim_p = len(nt_p)
#     zdim_s = len(nt_s)
#
#     #work arrays
#     #s = np.zeros_like(image)
#     no = np.zeros_like(image, dtype=np.float64)
#     sno = np.zeros_like(image,dtype=np.float64)
#     sout = np.zeros_like(image,dtype=np.float64)
#
#     #flip data for Euclid depending on the quadrant being processed and
#     #rotate (j, i slip in s) to move from Euclid to Gaia coordinate system
#     #because this is what is assumed in CDM03 (EUCLID_TN_ESA_AS_003_0-2.pdf)
#     #for i in range(xdim):
#     #   for j in range(ydim):
#     #      s[j, i] = image[i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j)]
#     s = image.copy()
#
#     #add background electrons
#     s += dob
#
#     #apply FWC (anti-blooming)
#     msk = s > fwc
#     s[msk] = fwc
#
#     #start with parallel direction
#     if parallel:
#         print 'adding parallel'
#         alpha_p = t*sigma_p*vth*fwc**beta_p/2./vg
#         g_p = nt_p*2.*vg/fwc**beta_p
#
#         for i in range(ydim):
#             print i
#             gamm_p = g_p * i
#             for k in range(zdim_p):
#                 for j in range(xdim):
#                      nc = 0.
#
#                      if s[i, j] > 0.01:
#                          nc = max((gamm_p[k]*s[i,j]**beta_p - no[j,k])/(gamm_p[k]*s[i,j]**(beta_p - 1.) + 1.) *
#                                   (1.-np.exp(-alpha_p[k]*s[i,j]**(1.-beta_p))), 0.0)
#
#                      no[j,k] = no[j,k] + nc
#                      nr = no[j,k] * (1. - np.exp(-t/tr_p[k]))
#                      s[i,j] = s[i,j] - nc + nr
#                      no[j,k] = no[j,k] - nr
#
#     #now serial direction
#     if serial:
#         print 'adding serial'
#         alpha_s=st*sigma_s*vth*sfwc**beta_s/2./svg
#         g_s=nt_s*2.*svg/sfwc**beta_s
#
#         for j in range(xdim):
#             print j
#             gamm_s = g_s * j
#             for k in range(zdim_s):
#                 if tr_s[k] < t:
#                     for i in range(ydim):
#                         nc = 0.
#
#                         if s[i,j] > 0.01:
#                             nc = max((gamm_s[k]*s[i,j]**beta_s-sno[i,k])/(gamm_s[k]*s[i,j]**(beta_s-1.)+1.) *
#                                      (1.-np.exp(-alpha_s[k]*s[i,j]**(1.-beta_s))), 0.)
#
#                         sno[i,k] = sno[i,k] + nc
#                         nr = sno[i,k] * (1. - np.exp(-st/tr_s[k]))
#                         s[i,j] = s[i,j] - nc + nr
#                         sno[i,k] = sno[i,k] - nr
#
#
#     # We need to rotate back from Gaia coordinate system and
#     # flip data back to the input orientation
#     for i in range(ydim):
#        for j in range(xdim):
#           sout[i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j)] = s[j, i]
#
#     return sout