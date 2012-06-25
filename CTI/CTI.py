"""
Charge Transfer Inefficiency
============================

This file contains a simple class to run a CDM03 CTI model developed by Alex Short (ESA).

:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import numpy as np
try:
    import cdm03
except ImportError:
    print 'No CDM03 module available, please compile it: f2py -c -m cdm03 cdm03.f90'


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
        self.data = data
        self.values = dict(quads=(0,1,2,3), xsize=2048, ysize=2066, dob=0.0, rdose=3e10)
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

        if self.logger:
            self.log.info('nt=' + str(nt))
            self.log.info('sigma= ' + str(sigma))
            self.log.info('taur= ' + str(taur))
            self.log.info('dob=%f' % self.values['dob'])
            self.log.info('rdose=%e' % self.values['rdose'])

        #call Fortran routine
        CTIed = cdm03.cdm03(np.asfortranarray(data),
                            iquadrant % 2, iquadrant / 2,
                            self.values['dob'], self.values['rdose'],
                            nt, sigma, taur,
                            [data.shape[0], data.shape[1], len(nt)])

        return np.asanyarray(CTIed)


    def CDM03Python(self, data, nt, sigma, tr, quadrant, rdose):
        """
        Python adaptation of the FORTRAN code.

        .. Warning:: This is not properly written and should NOT be used.
        """
        dob = 0.0
        beta=0.6
        fwc=175000.
        vth=1.168e7
        t=1.024e-2
        vg=6.e-11
        st=5.e-6
        sfwc=730000.
        svg=1.0e-10

        ydim, xdim = data.shape
        zdim = len(nt)

        nt *= rdose

        #allocs
        no = np.zeros((ydim, zdim))
        sno = np.zeros((xdim,zdim))

        # flip data for Euclid depending on the quadrant being processed
        if quadrant == 1 or quadrant == 3:
            data = np.fliplr(data)
        if quadrant == 2 or quadrant == 3:
            data = np.flipud(data)

        #add background
        data = data + dob

        #anti-blooming
        data[data > fwc] = fwc

        #parallel direction
        alpha = t * sigma * vth * fwc**beta / 2. / vg
        g = nt * 2. * vg / fwc**beta

        for i in range(xdim):
            gamm = g * i
            for k in range(zdim):
                for j in range(ydim):
                    nc = 0.

                    if data[j, i] > 0.01:
                        div = (gamm[k]*data[i,j]**(beta-1.)+1.)*(1.-np.exp(-alpha[k]*data[i,j]**(1.-beta)))
                        nc = gamm[k]*data[i,j]**beta - no[j,k] / div
                        if nc < 0:
                            nc = 0.

                    no[j,k] = no[j,k] + nc
                    nr = no[j,k] * (1. - np.exp(-t/tr[k]))
                    data[i,j] = data[i,j] - nc + nr
                    no[j,k] = no[j,k] - nr


        #now serial direction
        alpha=st*sigma*vth*sfwc**beta/2./svg
        g=nt*2.*svg/sfwc**beta

        for j in range(xdim):
            gamm = g * j
            for k in range(zdim):
                 if tr[k] < t:
                    for i in range(ydim):
                        nc = 0.

                        if data[i,j] > 0.01:
                            nc = gamm[k]*data[i,j]**beta-sno[i,k] / (gamm[k]*data[i,j]**(beta-1.)+1.)*(1.-np.exp(-alpha[k]*data[i,j]**(1.-beta)))
                            if nc < 0.0:
                                nc = 0.

                        sno[i,k] += nc
                        nr = sno[i,k] * (1. - np.exp(-st/tr[k]))
                        data[i,j] = data[i,j] - nc + nr
                        sno[i,k] = sno[i,k] - nr


        if quadrant == 1 or quadrant == 3:
            data = np.fliplr(data)
        if quadrant == 2 or quadrant == 3:
            data = np.flipud(data)


        return data