"""
This script can be used to test the CDM03 CTI model.

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import matplotlib
matplotlib.use('PDF')
import os, sys, datetime
import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import cdm03


def radiateFullCCD(fullCCD, quads=(0,1,2,3), xsize=2048, ysize=2066):
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
    ydim, xdim = fullCCD.shape
    out = np.zeros((xdim, ydim))

    #transpose the data, because Python has different convention
    #than Fortran
    data = fullCCD.transpose().copy()

    for quad in quads:
        if quad == 0:
            d = data[:xsize, :ysize].copy()
            print 'Q0', d.shape
            tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[:xsize, :ysize] = tmp
        elif quad == 1:
            d = data[xsize:, :ysize].copy()
            print 'Q1', d.shape
            tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[xsize:, :ysize] = tmp
        elif quad == 2:
            d = data[:xsize, ysize:].copy()
            print 'Q2', d.shape
            tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[:xsize, ysize:] = tmp
        elif quad == 3:
            d = data[xsize:, ysize:].copy()
            print 'Q3', d.shape
            tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[xsize:, ysize:] = tmp

        else:
            print 'ERROR -- too many quadrants!!'

    return out.transpose()


def applyRadiationDamage(data, iquadrant=0, rdose=3.0e10):
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
    trapdata = np.loadtxt('cdm_euclid.dat')
    nt = trapdata[:, 0]
    sigma = trapdata[:, 1]
    taur = trapdata[:, 2]

    CTIed = cdm03.cdm03(np.asfortranarray(data),
                        iquadrant%2, iquadrant/2,
                        0.0, rdose,
                        nt, sigma, taur,
                        [data.shape[0], data.shape[1], len(nt)])
    return np.asanyarray(CTIed)


def writeFITSfile(data, output, unsigned16bit=True):
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
        hdu.header.add_history('Scaled to unsigned 16bit integer!')

    #update and verify the header
    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
    hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
    hdu.verify('fix')

    ofd.append(hdu)

    #write the actual file
    ofd.writeto(output)


def plotProfiles(vertical, horizontal, lines, len=50, width=9, xsize=2048, ysize=2066):
    """

    """
    #quadrants
    Q0v = vertical[:ysize, :xsize].copy()
    Q1v = vertical[0:ysize, xsize:].copy()
    Q2v = vertical[ysize:, 0:xsize].copy()
    Q3v = vertical[ysize:, xsize:].copy()
    Q0h = horizontal[:ysize, :xsize].copy()
    Q1h = horizontal[0:ysize, xsize:].copy()
    Q2h = horizontal[ysize:, 0:xsize].copy()
    Q3h = horizontal[ysize:, xsize:].copy()

    #average the profile and flip Q1 and 3 over for vertical and 2 and 3 for horizontal
    profileQ0v = np.average(Q0v, axis=0)
    profileQ1v = np.average(Q1v, axis=0)[::-1]
    profileQ2v = np.average(Q2v, axis=0)
    profileQ3v = np.average(Q3v, axis=0)[::-1]
    profileQ0h = np.average(Q0h, axis=1)
    profileQ1h = np.average(Q1h, axis=1)
    profileQ2h = np.average(Q2h, axis=1)[::-1]
    profileQ3h = np.average(Q3h, axis=1)[::-1]

    #generate the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(profileQ0v[lines['xstart1']+width: lines['xstart1']+width+len], ls='-', label='Q0 serial CTI')
    #ax.semilogy(profileQ1v[lines['xstart1']+width: lines['xstart1']+width+len], ls=':', label='Q1 serial CTI')
    #ax.semilogy(profileQ2v[lines['xstart1']+width: lines['xstart1']+width+len], ls='-.', label='Q2 serial CTI')
    #ax.semilogy(profileQ3v[lines['xstart1']+width: lines['xstart1']+width+len], ls='--', label='Q3 serial CTI')

    ax.semilogy(profileQ0h[lines['ystart1']+width: lines['ystart1']+width+len], ls='-', label='Q0 parallel CTI')
    #ax.semilogy(profileQ1h[lines['ystart1']+width: lines['ystart1']+width+len], ls=':', label='Q1 parallel CTI')
    #ax.semilogy(profileQ2h[lines['ystart1']+width: lines['ystart1']+width+len], ls='-.', label='Q2 parallel CTI')
    #ax.semilogy(profileQ3h[lines['ystart1']+width: lines['ystart1']+width+len], ls='--', label='Q3 parallel CTI')

    #measurements
    plt.semilogy([4,5,6,7,8,9,10], [50, 27, 16, 11, 9, 8, 7.5], 'y*', label='Alex parallel')
    #plt.semilogy([2,160], [10,1], 'k--', lw=2.5, label='Alex serial')

    plt.ylim(1e-1, 1e5)
    plt.xlim(0, len)
    plt.xlabel('Pixels')
    plt.ylabel('Electrons (1e = 1ADU)')
    plt.legend()
    plt.savefig('CTItest.pdf')


if __name__ == '__main__':
    #amount of charge
    chargeInjection = 44000.

    #set up the charge injection chargeInjection positions
    lines = dict(xstart1=1590, xstop1=1601, ystart1=1590, ystop1=1601,
                 xstart2=2495, xstop2=2506, ystart2=2531, ystop2=2542)

    #create two CCD files
    CCDhor = np.zeros((4132, 4096), dtype=np.float32)
    CCDver = np.zeros((4132, 4096), dtype=np.float32)

    #add horizontal charge injection lines
    CCDhor[lines['ystart1']:lines['ystop1'],:] = chargeInjection
    CCDhor[lines['ystart2']:lines['ystop2'],:] = chargeInjection

    #add vertical charge injection lines
    CCDver[:, lines['xstart1']:lines['xstop1']] = chargeInjection
    CCDver[:, lines['xstart2']:lines['xstop2']] = chargeInjection

    #write output files
    writeFITSfile(CCDhor, 'ChargeInjectionsHorizontal.fits', unsigned16bit=False)
    writeFITSfile(CCDver, 'ChargeInjectionsVertical.fits', unsigned16bit=False)

    #radiate the CCDs
    CCDCTIhor = radiateFullCCD(CCDhor)
    CCDCTIver = radiateFullCCD(CCDver)

    #write output files
    writeFITSfile(CCDCTIhor, 'ChargeInjectionsHorizontalCTI.fits', unsigned16bit=False)
    writeFITSfile(CCDCTIver, 'ChargeInjectionsVerticalCTI.fits', unsigned16bit=False)

    #plot profiles
    plotProfiles(CCDCTIver, CCDCTIhor, lines)