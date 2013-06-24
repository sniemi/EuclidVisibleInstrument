"""
This script can be used to test the CDM03 CTI model.

:requires: PyFITS
:requires: NumPy
:requires: CDM03 (FORTRAN code, f2py -c -m cdm03 cdm03.f90)

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.3
"""
import os, datetime, time
import numpy as np
import pyfits as pf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cdm03bidir


def parallelMeasurements(filename='CCD204_05325-03-02_Hopkinson_EPER_data_200kHz_one-output-mode_1.6e10-50MeV.txt',
                         datafolder='/Users/smn2/EUCLID/CTItesting/data/', gain1=1.17, limit=105, returnScale=False):
    """

    :param filename:
    :param datafolder:
    :param gain1:
    :param limit:
    :return:
    """
    tmp = np.loadtxt(datafolder + filename, usecols=(0, 5)) #5 = 152.55K
    ind = tmp[:, 0]
    values = tmp[:, 1]
    values *= gain1
    if returnScale:
        return ind, values
    else:
        values = values[ind > -5.]
        values = np.abs(values[:limit])
        return values


def serialMeasurements(filename='CCD204_05325-03-02_Hopkinson-serial-EPER-data_200kHz_one-output-mode_1.6e10-50MeV.txt',
        datafolder='/Users/smn2/EUCLID/CTItesting/data/', gain1=1.17, limit=105, returnScale=False):
    """

    :param filename:
    :param datafolder:
    :param gain1:
    :param limit:
    :return:
    """
    tmp = np.loadtxt(datafolder + filename, usecols=(0, 6)) #6 = 152.55K
    ind = tmp[:, 0]
    values = tmp[:, 1]
    values *= gain1
    if returnScale:
        return ind, values
    else:
        values = values[ind > -5.]
        values = np.abs(values[:limit])
        return values


def radiateFullCCD(fullCCD, quads=(0,1,2,3), xsize=2048, ysize=2066, bidir=True):
    """
    This routine allows the whole CCD to be run through a radiation damage mode.
    The routine takes into account the fact that the amplifiers are in the corners
    of the CCD. The routine assumes that the CCD is using four amplifiers.

    :param fullCCD: image of containing the whole CCD
    :type fullCCD: ndarray
    :param file: name of the CDM03 parameter file
    :type file: str
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
            if bidir:
                tmp = applyRadiationDamageBiDir(d, iquadrant=quad).copy()
            else:
                tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[:xsize, :ysize] = tmp
        elif quad == 1:
            d = data[xsize:, :ysize].copy()
            print 'Q1', d.shape
            if bidir:
                tmp = applyRadiationDamageBiDir(d, iquadrant=quad).copy()
            else:
                tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[xsize:, :ysize] = tmp
        elif quad == 2:
            d = data[:xsize, ysize:].copy()
            print 'Q2', d.shape
            if bidir:
                tmp = applyRadiationDamageBiDir(d, iquadrant=quad).copy()
            else:
                tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[:xsize, ysize:] = tmp
        elif quad == 3:
            d = data[xsize:, ysize:].copy()
            print 'Q3', d.shape
            if bidir:
                tmp = applyRadiationDamageBiDir(d, iquadrant=quad).copy()
            else:
                tmp = applyRadiationDamage(d, iquadrant=quad).copy()
            out[xsize:, ysize:] = tmp

        else:
            print 'ERROR -- too many quadrants!!'

    return out.transpose()


def applyRadiationDamage(data, file='cdm_euclid.dat', iquadrant=0, rdose=1.6e10):
    """
    Apply radian damage based on FORTRAN CDM03 model. The method assumes that
    input data covers only a single quadrant defined by the iquadrant integer.

    :param data: imaging data to which the CDM03 model will be applied to.
    :type data: ndarray
    :param file: name of the CDM03 parameter file
    :type file: str
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
    import cdm03

    #read in trap information
    print 'Trap file = ', file
    trapdata = np.loadtxt(file)
    nt = trapdata[:, 0]
    sigma = trapdata[:, 1]
    taur = trapdata[:, 2]

    #CTIed = cdm03.cdm03(np.asfortranarray(data),
    CTIed = cdm03.cdm03(data,
                        iquadrant%2, iquadrant/2,
                        0.0, rdose,
                        nt, sigma, taur,
                        [data.shape[0], data.shape[1], len(nt)])
    return np.asanyarray(CTIed)


def applyRadiationDamageBiDir(data, f1='cdm_euclid_parallel.dat', f2='cdm_euclid_serial.dat',
                              iquadrant=0, rdose=1.6e10, sdob=0.0):
    """
    Note: no transpose

    :return: image that has been run through the CDM03 model
    :rtype: ndarray
    """
    #read in trap information
    print 'Trap files = %s and %s' % (f1, f2)
    trapdata = np.loadtxt(f1)
    nt_p = trapdata[:, 0]
    sigma_p = trapdata[:, 1]
    taur_p = trapdata[:, 2]

    trapdata = np.loadtxt(f2)
    nt_s = trapdata[:, 0]
    sigma_s = trapdata[:, 1]
    taur_s = trapdata[:, 2]

    CTIed = cdm03bidir.cdm03(data,
                             iquadrant%2, iquadrant/2,
                             sdob, rdose,
                             nt_p, sigma_p, taur_p,
                             nt_s, sigma_s, taur_s,
                             [data.shape[0], data.shape[1], len(nt_p), len(nt_s)])
    return np.asanyarray(CTIed)


def applyRadiationDamageBiDir2(data, nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, iquadrant=0, rdose=1.6e10, sdob=0.0):
    """

    :param data:
    :param nt_p:
    :param sigma_p:
    :param taur_p:
    :param nt_s:
    :param sigma_s:
    :param taur_s:
    :param iquadrant:
    :param rdose:

    :return:
    """
    #read in trap information
    CTIed = cdm03bidir.cdm03(data.transpose(),
                             iquadrant%2, iquadrant/2,
                             sdob, rdose,
                             nt_p, sigma_p, taur_p,
                             nt_s, sigma_s, taur_s,
                             [data.shape[1], data.shape[0], len(nt_p), len(nt_s)])
    return CTIed.transpose()


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
    hdu.header.add_history('If questions, please contact Sami-Matias Niemi (s.niemi at ucl.ac.uk).')
    hdu.header.add_history('This file has been created with the VISsim Python Package at %s' \
                           % datetime.datetime.isoformat(datetime.datetime.now()))
    hdu.verify('fix')

    ofd.append(hdu)

    #write the actual file
    ofd.writeto(output)


def plotTrail(data, measurements, parallel=True, output='CTItest.pdf'):
    """

    :param data: input data showing simulated CTI trail
    :param parallel: whether the input data is for parallel or serial CTI
    :param output: name of the output file

    :return: None
    """
    #generate the plot
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    ax1 = fig.add_subplot(111)

    if parallel:
        ax1.set_title('Parallel CTI')
    else:
        ax1.set_title('Serial CTI')

    ax1.semilogy(np.arange(len(data))-5, data, ls='-', c='g', label='Simulation')
    ax1.semilogy(np.arange(len(measurements))-4, measurements, 'rs', ms=2, label='Measurements')
    ax1.axvline(x=0, ls='--', c='k')

    ax1.set_ylim(0.1, 60000)
    ax1.set_xlim(-5, 100)

    ax1.set_ylabel('Photoelectrons')
    ax1.set_xlabel('Pixels')
    ax1.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig(output)
    plt.close()


def plotProfiles(vertical, horizontal, lines, len=200, width=9, xsize=2048, ysize=2066, output='CTItest.pdf'):
    """

    :param vertical:
    :param horizontal:
    :param lines:
    :param len:
    :param width:
    :param xsize:
    :param ysize:
    :param output:
    :return:
    """
    #quadrants
    Q0v = vertical[:ysize, :xsize].copy()
    #Q1v = vertical[0:ysize, xsize:].copy()
    #Q2v = vertical[ysize:, 0:xsize].copy()
    #Q3v = vertical[ysize:, xsize:].copy()
    Q0h = horizontal[:ysize, :xsize].copy()
    #Q1h = horizontal[0:ysize, xsize:].copy()
    #Q2h = horizontal[ysize:, 0:xsize].copy()
    #Q3h = horizontal[ysize:, xsize:].copy()

    #average the profile and flip Q1 and 3 over for vertical and 2 and 3 for horizontal
    profileQ0v = np.average(Q0v, axis=0)
    #profileQ1v = np.average(Q1v, axis=0)[::-1]
    #profileQ2v = np.average(Q2v, axis=0)
    #profileQ3v = np.average(Q3v, axis=0)[::-1]
    profileQ0h = np.average(Q0h, axis=1)
    #profileQ1h = np.average(Q1h, axis=1)
    #profileQ2h = np.average(Q2h, axis=1)[::-1]
    #profileQ3h = np.average(Q3h, axis=1)[::-1]

    #generate the plot
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    #ax2.semilogy(profileQ0v[lines['xstart1']+width: lines['xstart1']+width+len], ls='-', c='g', label='Q0 serial CTI')
    ax2.semilogy(profileQ0v[lines['xstart1']+width+1:], ls='-', c='g', label='Q0 serial CTI')
    #ax1.semilogy(profileQ0h[lines['ystart1']+width: lines['ystart1']+width+len], ls='-', c='g', label='Q0 parallel CTI')
    ax1.semilogy(profileQ0h[lines['ystart1']+width+1:], ls='-', c='g', label='Q0 parallel CTI')

    #measurements
    #plt.semilogy([4,5,6,7,8,9,10], [50, 27, 16, 11, 9, 8, 7.5], 'y*', label='Alex parallel')
    #plt.semilogy([2,160], [10,1], 'k--', lw=2.5, label='Alex serial')
    datafolder = '/Users/smn2/EUCLID/CTItesting/data/'
    gain1 = 1.17
    parallel1 = np.loadtxt(
        datafolder + 'CCD204_05325-03-02_Hopkinson_EPER_data_200kHz_one-output-mode_1.6e10-50MeV.txt',
        usecols=(0, 5)) #5 = 152.55K
    serial1 = np.loadtxt(
        datafolder + 'CCD204_05325-03-02_Hopkinson-serial-EPER-data_200kHz_one-output-mode_1.6e10-50MeV.txt',
        usecols=(0, 6)) #6 = 152.55
    #convert to electrons
    parallel1[:, 1] *= gain1
    serial1[:, 1] *= gain1

    ax1.set_title('Parallel CTI')
    ax2.set_title('Serial CTI')

    ax1.semilogy(parallel1[:, 0], parallel1[:, 1], 'bo', ms=2, label='152.55K')
    ax2.semilogy(serial1[:, 0], serial1[:, 1], 'rs', ms=2, label='152.5K')

    ax1.axvline(x=0, ls='--', c='k')
    ax2.axvline(x=0, ls='--', c='k')

    ax1.set_ylim(0.1, 60000)
    ax2.set_ylim(0.1, 60000)

    ax1.set_xlim(-10, 200)
    ax2.set_xlim(-10, 200)

    ax2.set_xlabel('?Pixels?')
    ax1.set_ylabel('Photoelectrons')
    ax2.set_ylabel('Photoelectrons')
    ax1.legend(fancybox=True, shadow=True, numpoints=1)
    ax2.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig(output)
    plt.close()


def plotTestData(datafolder='/Users/smn2/EUCLID/CTItesting/data/', gain1=1.17):
    """

    :return:
    """
    parallel1 = np.loadtxt(datafolder+'CCD204_05325-03-02_Hopkinson_EPER_data_200kHz_one-output-mode_1.6e10-50MeV.txt',
                           usecols=(0, 5)) #5 = 152.55K
    serial1 = np.loadtxt(datafolder+'CCD204_05325-03-02_Hopkinson-serial-EPER-data_200kHz_one-output-mode_1.6e10-50MeV.txt',
                         usecols=(0, 6)) #6 = 152.55
    #convert to electrons
    parallel1[:, 1] *= gain1
    serial1[:, 1] *= gain1

    print np.max(parallel1[:, 1]), np.max(serial1[:, 1])

    fig = plt.figure()
    fig.suptitle('CCD204 05325-03-02 Hopkinson EPER at 200kHz, 1.6e10 at 50MeV')
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title('Parallel CTI')
    ax2.set_title('Serial CTI')

    ax1.semilogy(parallel1[:, 0], parallel1[:, 1], 'bo', ms=2, label='152.55K')
    ax2.semilogy(serial1[:, 0], serial1[:, 1], 'rs', ms=2, label='152.5K')

    ax1.axvline(x=0, ls='--', c='k')
    ax2.axvline(x=0, ls='--', c='k')

    ax1.set_ylim(0.1, 60000)
    ax2.set_ylim(0.1, 60000)

    ax1.set_xlim(-10, 900)
    ax2.set_xlim(-10, 330)

    #ax1.set_xlabel('Pixels')
    ax2.set_xlabel('?')
    ax1.set_ylabel('Photoelectrons')
    ax2.set_ylabel('Photoelectrons')
    ax1.legend(fancybox=True, shadow=True, numpoints=1)
    ax2.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('testData.pdf')
    plt.close()


def currentValues(chargeInjection=43500.):
    #set up the charge injection chargeInjection positions
    lines = dict(xstart1=577, xstop1=588, ystart1=1064, ystop1=1075,
                 xstart2=4096 - 588, xstop2=4096 - 577, ystart2=4132 - 1075, ystop2=4132 - 1064)

    #create two CCD files
    CCDhor = np.zeros((4132, 4096), dtype=np.float32)
    CCDver = np.zeros((4132, 4096), dtype=np.float32)

    #add horizontal charge injection lines
    CCDhor[lines['ystart1']:lines['ystop1'], :] = chargeInjection
    CCDhor[lines['ystart2']:lines['ystop2'], :] = chargeInjection

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


def fitParallelBayesian(lines, chargeInjection=43500., test=False):
    """

    :param chargeInjection:
    :return:
    """
    import pymc

    #measurements
    values = parallelMeasurements()

    #create a quadrant
    CCDhor = np.zeros((2066, 5), dtype=np.float32)

    #add horizontal charge injection lines
    CCDhor[lines['ystart1']:lines['ystop1'], :] = chargeInjection

    nt1 = pymc.distributions.Uniform('nt1', 0.0, 100.0, value=40.)
    nt2 = pymc.distributions.Uniform('nt2', 0.0, 100.0, value=1.2)
    nt3 = pymc.distributions.Uniform('nt3', 0.0, 100.0, value=1.)
    nt4 = pymc.distributions.Uniform('nt4', 0.0, 100.0, value=1.)
    nt5 = pymc.distributions.Uniform('nt5', 0.0, 100.0, value=0.2)
    sigma1 = pymc.distributions.Uniform('sigma1', 1.0e-20, 1.0e-5, value=2.0e-13)
    sigma2 = pymc.distributions.Uniform('sigma2', 1.0e-20, 1.0e-5, value=2.0e-13)
    sigma3 = pymc.distributions.Uniform('sigma3', 1.0e-20, 1.0e-5, value=5.0e-15)
    sigma4 = pymc.distributions.Uniform('sigma4', 1.0e-20, 1.0e-5, value=1.0e-16)
    sigma5 = pymc.distributions.Uniform('sigma5', 1.0e-20, 1.0e-5, value=1.0e-18)
    tau1 = pymc.distributions.Uniform('tau1', 1.0e-8, 1.0e3, value=8.0e-7)
    tau2 = pymc.distributions.Uniform('tau2', 1.0e-8, 1.0e3, value=3.0e-4)
    tau3 = pymc.distributions.Uniform('tau3', 1.0e-8, 1.0e3, value=2.0e-3)
    tau4 = pymc.distributions.Uniform('tau4', 1.0e-8, 1.0e3, value=2.0e-2)
    tau5 = pymc.distributions.Uniform('tau5', 1.0e-8, 1.0e3, value=1.0)

    @pymc.deterministic(plot=False, trace=False)
    def model(x=CCDhor, nt1=nt1, nt2=nt2, nt3=nt3, nt4=nt4, nt5=nt5,
              sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4, sigma5=sigma5,
              tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4, tau5=tau5):
        #serial fixed
        nt_s = [20, 10, 2.]
        sigma_s = [6e-20, 1.13e-14, 5.2e016]
        taur_s = [2.38e-2, 1.7e-6, 2.2e-4]
        #parallel fit values
        sigma_p = [sigma1, sigma2, sigma3, sigma4, sigma5]
        taur_p = [tau1, tau2, tau3, tau4, tau5]
        nt_p = [nt1, nt2, nt3, nt4, nt5]

        tmp = applyRadiationDamageBiDir2(x.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)[1245:1350, 0]
        return tmp

    #likelihood function, note that an inverse weighting has been applied... this could be something else too
    y = pymc.distributions.Normal('y', mu=model, tau=1./values**2, value=values, observed=True, trace=False)

    #store the model to a dictionary
    d = {'nt1': nt1,
         'nt2': nt2,
         'nt3': nt3,
         'nt4': nt4,
         'nt5': nt5,
         'sigma1': sigma1,
         'sigma2': sigma2,
         'sigma3': sigma3,
         'sigma4': sigma4,
         'sigma5': sigma5,
         'tau1': tau1,
         'tau2': tau2,
         'tau3': tau3,
         'tau4': tau4,
         'tau5': tau5,
         'f': model,
         'y': y}

    print 'Will start running a chain...'
    start = time.time()
    R = pymc.MCMC(d)

    if test:
        R.sample(100)
    else:
        R.sample(iter=20000, burn=5000, thin=5)

    R.write_csv("parallel.csv", variables=['nt1', 'nt2', 'nt3', 'nt4', 'nt5',
                                           'sigma1', 'sigma2', 'sigma3', 'sigma4', 'sigma5',
                                           'tau1', 'tau2', 'tau3', 'tau4', 'tau5'])

    R.nt1.summary()
    R.nt2.summary()
    R.nt3.summary()
    R.nt4.summary()
    R.nt5.summary()
    R.sigma1.summary()
    R.sigma2.summary()
    R.sigma3.summary()
    R.sigma4.summary()
    R.sigma5.summary()
    R.tau1.summary()
    R.tau2.summary()
    R.tau3.summary()
    R.tau4.summary()
    R.tau5.summary()

    #generate plots
    pymc.Matplot.plot(R)
    pymc.Matplot.summary_plot(R)

    Rs = R.stats()
    print 'Finished MCMC in %e seconds' % (time.time() - start)

    #output mean values
    print Rs['nt1']['mean']
    print Rs['nt2']['mean']
    print Rs['nt3']['mean']
    print Rs['nt4']['mean']
    print Rs['nt5']['mean']
    print Rs['sigma1']['mean']
    print Rs['sigma2']['mean']
    print Rs['sigma3']['mean']
    print Rs['sigma4']['mean']
    print Rs['sigma5']['mean']
    print Rs['tau1']['mean']
    print Rs['tau2']['mean']
    print Rs['tau3']['mean']
    print Rs['tau4']['mean']
    print Rs['tau5']['mean']

    #show results
    nt = [Rs['nt1']['mean'], Rs['nt2']['mean'], Rs['nt3']['mean'], Rs['nt4']['mean'], Rs['nt5']['mean']]
    sigma = [Rs['sigma1']['mean'], Rs['sigma2']['mean'], Rs['sigma3']['mean'], Rs['sigma4']['mean'], Rs['sigma5']['mean']]
    taur = [Rs['tau1']['mean'], Rs['tau2']['mean'], Rs['tau3']['mean'], Rs['tau4']['mean'], Rs['tau5']['mean']]
    nt_s = [20, 10, 2.]
    sigma_s = [6e-20, 1.13e-14, 5.2e016]
    taur_s = [2.38e-2, 1.7e-6, 2.2e-4]

    CCD = np.zeros((2066, 5), dtype=np.float32)
    #add horizontal charge injection lines
    CCD[lines['ystart1']:lines['ystop1'], :] = chargeInjection

    CCDCTIhor = applyRadiationDamageBiDir2(CCD.copy(), nt, sigma, taur, nt_s, sigma_s, taur_s)
    writeFITSfile(CCDCTIhor, 'ChargeInjectionsHorizontalCTIfinalBayesian.fits', unsigned16bit=False)

    #plot trails
    parallelValues = parallelMeasurements()
    profile = np.average(CCDCTIhor, axis=1)
    plotTrail(profile[lines['ystop1']-6:], parallelValues, output='FinalparallelValuesBayesian.pdf')

    #write out results
    fh = open('cdm_euclid_parallel_new_bayesian.dat', 'w')
    for a, b, c in zip(nt, sigma, taur):
        fh.write('%e %e %e\n' % (a, b, c))
    fh.close()


def fitSerialBayesian(lines, chargeInjection=43500., test=False):
    """

    :param chargeInjection:
    :return:
    """
    import pymc

    #measurements
    values = serialMeasurements()

    #create a quadrant
    CCD = np.zeros((2066, 2048), dtype=np.float32)

    #add horizontal charge injection lines
    CCD[:, lines['xstart1']:lines['xstop1']] = chargeInjection

    nt1 = pymc.distributions.Uniform('nt1', 0.0, 100.0, value=40.)
    nt2 = pymc.distributions.Uniform('nt2', 0.0, 100.0, value=1.2)
    nt3 = pymc.distributions.Uniform('nt3', 0.0, 100.0, value=1.)
    nt4 = pymc.distributions.Uniform('nt4', 0.0, 100.0, value=1.)
    nt5 = pymc.distributions.Uniform('nt5', 0.0, 100.0, value=0.2)
    sigma1 = pymc.distributions.Uniform('sigma1', 1.0e-20, 1.0e-5, value=2.0e-13)
    sigma2 = pymc.distributions.Uniform('sigma2', 1.0e-20, 1.0e-5, value=2.0e-13)
    sigma3 = pymc.distributions.Uniform('sigma3', 1.0e-20, 1.0e-5, value=5.0e-15)
    sigma4 = pymc.distributions.Uniform('sigma4', 1.0e-20, 1.0e-5, value=1.0e-16)
    sigma5 = pymc.distributions.Uniform('sigma5', 1.0e-20, 1.0e-5, value=1.0e-18)
    tau1 = pymc.distributions.Uniform('tau1', 1.0e-8, 1.0e3, value=8.0e-7)
    tau2 = pymc.distributions.Uniform('tau2', 1.0e-8, 1.0e3, value=3.0e-4)
    tau3 = pymc.distributions.Uniform('tau3', 1.0e-8, 1.0e3, value=2.0e-3)
    tau4 = pymc.distributions.Uniform('tau4', 1.0e-8, 1.0e3, value=2.0e-2)
    tau5 = pymc.distributions.Uniform('tau5', 1.0e-8, 1.0e3, value=1.0)

    @pymc.deterministic(plot=False, trace=False)
    def model(x=CCD, nt1=nt1, nt2=nt2, nt3=nt3, nt4=nt4, nt5=nt5,
              sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4, sigma5=sigma5,
              tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4, tau5=tau5):
        #fit serial values
        nt_s = [sigma1, sigma2, sigma3, sigma4, sigma5]
        sigma_s = [tau1, tau2, tau3, tau4, tau5]
        taur_s = [nt1, nt2, nt3, nt4, nt5]
        #keep parallel fixed
        sigma_p = [2.2e-13, 2.2e-13, 4.72e-15, 1.37e-16, 2.78e-17, 1.93e-17, 6.39e-18]
        taur_p = [8.2e-07, 3.0e-04, 2.0e-03, 2.5e-02, 1.24e-01, 1.67e+01, 4.96e+02]
        nt_p = [4.0e+01, 1.2e+00, 5.82587756e+02, 1.14724258, 3.13617389e-01, 2.07341804, 4.29146077e-07]

        tmp = applyRadiationDamageBiDir2(x.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)[10, 592:697]

        return tmp

    #likelihood function, note that an inverse weighting has been applied... this could be something else too
    y = pymc.distributions.Normal('y', mu=model, tau=1./values**2, value=values, observed=True, trace=False)

    #store the model to a dictionary
    d = {'nt1': nt1,
         'nt2': nt2,
         'nt3': nt3,
         'nt4': nt4,
         'nt5': nt5,
         'sigma1': sigma1,
         'sigma2': sigma2,
         'sigma3': sigma3,
         'sigma4': sigma4,
         'sigma5': sigma5,
         'tau1': tau1,
         'tau2': tau2,
         'tau3': tau3,
         'tau4': tau4,
         'tau5': tau5,
         'f': model,
         'y': y}

    print 'Will start running a chain...'
    start = time.time()
    R = pymc.MCMC(d)

    if test:
        R.sample(100)
    else:
        R.sample(iter=5000, burn=100, thin=1)

    R.write_csv("parallel.csv", variables=['nt1', 'nt2', 'nt3', 'nt4', 'nt5',
                                           'sigma1', 'sigma2', 'sigma3', 'sigma4', 'sigma5',
                                           'tau1', 'tau2', 'tau3', 'tau4', 'tau5'])

    R.nt1.summary()
    R.nt2.summary()
    R.nt3.summary()
    R.nt4.summary()
    R.nt5.summary()
    R.sigma1.summary()
    R.sigma2.summary()
    R.sigma3.summary()
    R.sigma4.summary()
    R.sigma5.summary()
    R.tau1.summary()
    R.tau2.summary()
    R.tau3.summary()
    R.tau4.summary()
    R.tau5.summary()

    #generate plots
    pymc.Matplot.plot(R)
    pymc.Matplot.summary_plot(R)

    Rs = R.stats()
    print 'Finished MCMC in %e seconds' % (time.time() - start)

    #output mean values
    print Rs['nt1']['mean']
    print Rs['nt2']['mean']
    print Rs['nt3']['mean']
    print Rs['nt4']['mean']
    print Rs['nt5']['mean']
    print Rs['sigma1']['mean']
    print Rs['sigma2']['mean']
    print Rs['sigma3']['mean']
    print Rs['sigma4']['mean']
    print Rs['sigma5']['mean']
    print Rs['tau1']['mean']
    print Rs['tau2']['mean']
    print Rs['tau3']['mean']
    print Rs['tau4']['mean']
    print Rs['tau5']['mean']

    #show results
    nt_s = [Rs['nt1']['mean'], Rs['nt2']['mean'], Rs['nt3']['mean'], Rs['nt4']['mean'], Rs['nt5']['mean']]
    sigma_s = [Rs['sigma1']['mean'], Rs['sigma2']['mean'], Rs['sigma3']['mean'], Rs['sigma4']['mean'], Rs['sigma5']['mean']]
    taur_s = [Rs['tau1']['mean'], Rs['tau2']['mean'], Rs['tau3']['mean'], Rs['tau4']['mean'], Rs['tau5']['mean']]
    nt_p = [20, 10, 2.]
    sigma_p = [6e-20, 1.13e-14, 5.2e016]
    taur_p = [2.38e-2, 1.7e-6, 2.2e-4]

    CCD = np.zeros((2066, 5), dtype=np.float32)
    #add horizontal charge injection lines
    CCD[:, lines['xstart1']:lines['xstop1']] = chargeInjection

    CCDCTI= applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)
    writeFITSfile(CCDCTI, 'ChargeInjectionsVerticalCTIfinalBayesian.fits', unsigned16bit=False)

    #plot trails
    serialValues = serialMeasurements()
    profile = np.average(CCDCTI, axis=1)
    plotTrail(profile[lines['xstop1'] - 1:], serialValues, parallel=False, output='FinalserialValuesBayesian.pdf')

    #write out results
    fh = open('cdm_euclid_serial_new_bayesian.dat', 'w')
    for a, b, c in zip(nt_s, sigma_s, taur_s):
        fh.write('%e %e %e\n' % (a, b, c))
    fh.close()


def fitParallelLSQ(lines, chargeInjection=43500.):
    """
    Fit parallel CTI using Least Squares.

    :param chargeInjection:
    :return:
    """
    def fitparallel(x, nt1, nt2, nt3, nt4, nt5, nt6, nt7):
        #serial fixed
        nt_s = [20, 10, 2.]
        sigma_s = [6e-20, 1.13e-14, 5.2e016]
        taur_s = [2.38e-2, 1.7e-6, 2.2e-4]

        #keep sigma and taur fixed for parallel
        sigma_p = [2.2e-13, 2.2e-13, 4.72e-15, 1.37e-16, 2.78e-17, 1.93e-17, 6.39e-18]
        taur_p = [8.2e-07, 3.0e-04, 2.0e-03, 2.5e-02, 1.24e-01, 1.67e+01, 4.96e+02]

        #params that are being fit
        nt_p = np.abs(np.asarray([nt1, nt2, nt3, nt4, nt5, nt6, nt7]))

        tmp = applyRadiationDamageBiDir2(x.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)[1245:1350, 0]

        return tmp

    #get parallel measurements
    parallelValues = parallelMeasurements()

    #trap parameters: parallel
    f1 = 'cdm_euclid_parallel.dat'
    trapdata = np.loadtxt(f1)
    nt_p = trapdata[:, 0]
    sigma_p = trapdata[:, 1]
    taur_p = trapdata[:, 2]

    nt_s = [20, 10, 2.]
    sigma_s = [6e-20, 1.13e-14, 5.2e016]
    taur_s = [2.38e-2, 1.7e-6, 2.2e-4]

    #create a quadrant
    #CCDhor = np.zeros((2066, 2048), dtype=np.float32)
    CCDhor = np.zeros((2066, 5), dtype=np.float32)

    #add horizontal charge injection lines
    CCDhor[lines['ystart1']:lines['ystop1'], :] = chargeInjection
    writeFITSfile(CCDhor, 'ChargeInjectionsHorizontal.fits', unsigned16bit=False)

    #radiate CTI to plot initial set trails
    CCDCTIhor = applyRadiationDamageBiDir2(CCDhor.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)
    writeFITSfile(CCDCTIhor, 'ChargeInjectionsHorizontalCTIinitial.fits', unsigned16bit=False)

    #plot trails
    profile = np.average(CCDCTIhor, axis=1)
    plotTrail(profile[lines['ystop1']-6:], parallelValues, output='IntialparallelValues.pdf')

    #initial guess
    initials = nt_p

    #do fitting
    popt, cov, info, mesg, ier = curve_fit(fitparallel, CCDhor.copy(), parallelValues,
                                           p0=initials, full_output=True)
    popt = np.abs(popt)
    print info
    print mesg
    #print ier
    print popt
    print
    print popt / initials

    #plot results
    CCDCTIhor = applyRadiationDamageBiDir2(CCDhor.copy(), popt, sigma_p, taur_p, nt_s, sigma_s, taur_s)
    writeFITSfile(CCDCTIhor, 'ChargeInjectionsHorizontalCTIfinal2.fits', unsigned16bit=False)

    #plot trails
    profile = np.average(CCDCTIhor, axis=1)
    plotTrail(profile[lines['ystop1']-6:], parallelValues, output='FinalparallelValues.pdf')

    #write out results
    fh = open('cdm_euclid_parallel_new.dat', 'w')
    for a, b, c in zip(popt, sigma_p, taur_p):
        fh.write('%e %e %e\n' % (a, b, c))
    fh.close()


def fitSerialLSQ(lines, chargeInjection=43500.):
    """
    Fit parallel CTI using Least Squares.

    :param chargeInjection:
    :return:
    """
    def fit(x, nt1, nt2, nt3):
        #serial fixed
        sigma_s = [6e-20, 1.13e-14, 5.2e-16]
        taur_s = [2.38e-2, 1.7e-6, 2.2e-4]

        #parallel fixed
        sigma_p = [2.2e-13, 2.2e-13, 4.72e-15, 1.37e-16, 2.78e-17, 1.93e-17, 6.39e-18]
        taur_p = [8.2e-07, 3.0e-04, 2.0e-03, 2.5e-02, 1.24e-01, 1.67e+01, 4.96e+02]
        nt_p = [4.0e+01, 1.2e+00, 5.82587756e+02, 1.14724258, 3.13617389e-01, 2.07341804, 4.29146077e-07]

        #params that are being fit
        nt_s = np.abs(np.asarray([nt1, nt2, nt3]))

        tmp = applyRadiationDamageBiDir2(x.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)[10, 592:697]

        return tmp

    #get parallel measurements
    serialValues = serialMeasurements()

    #trap parameters: parallel
    f = 'cdm_euclid_parallel.dat'
    trapdata = np.loadtxt(f)
    nt_p = trapdata[:, 0]
    sigma_p = trapdata[:, 1]
    taur_p = trapdata[:, 2]

    f = 'cdm_euclid_serial.dat'
    trapdata = np.loadtxt(f)
    nt_s = trapdata[:, 0]
    sigma_s = trapdata[:, 1]
    taur_s = trapdata[:, 2]

    #create a quadrant
    CCD = np.zeros((2066, 2048), dtype=np.float32)

    #add horizontal charge injection lines
    CCD[:, lines['xstart1']:lines['xstop1']] = chargeInjection
    writeFITSfile(CCD, 'ChargeInjectionsVertical.fits', unsigned16bit=False)

    #radiate CTI to plot initial set trails
    CCDCTI = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s)
    writeFITSfile(CCDCTI, 'ChargeInjectionsVerticalCTIinitial.fits', unsigned16bit=False)

    #plot trails
    profile = np.average(CCDCTI, axis=0)
    plotTrail(profile[lines['xstop1']-6:], serialValues, parallel=False, output='IntialserialValues.pdf')

    #initial guess
    initials = nt_s

    #do fitting
    popt, cov, info, mesg, ier = curve_fit(fit, CCD.copy(), serialValues, p0=initials, full_output=True)
    popt = np.abs(popt)
    print info
    print mesg
    #print ier
    print popt
    print
    print popt / initials

    #plot results
    CCDCTI = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, popt, sigma_s, taur_s)
    writeFITSfile(CCDCTI, 'ChargeInjectionsVerticalCTIfinal2.fits', unsigned16bit=False)

    #plot trails
    profile = np.average(CCDCTI, axis=0)
    plotTrail(profile[lines['xstop1']-6:], serialValues, parallel=False, output='FinalserialValues.pdf')

    #write out results
    fh = open('cdm_euclid_serial_new.dat', 'w')
    for a, b, c in zip(popt, sigma_s, taur_s):
        fh.write('%e %e %e\n' % (a, b, c))
    fh.close()


def deriveTrails(parallel, serial, chargeInjection=44500.,
                 lines=dict(ystart1=1064, ystop1=1075, xstart1=577, xstop1=588), thibaut=False):
    """
    Derive CTI trails in both parallel and serial direction separately given the input data files

    :param parallel: name of the parallel CTI file
    :param serial:  name of the serial CTI file
    :param chargeInjection: number of electrons to inject
    :param lines: positions of the charge injectino lines
    :param thibaut: whether or not to scale the capture cross sections from m**2 to cm**2

    :return:
    """

    trapdata = np.loadtxt(parallel)
    nt_p = trapdata[:, 0]
    sigma_p = trapdata[:, 1]
    taur_p = trapdata[:, 2]

    trapdata = np.loadtxt(serial)
    nt_s = trapdata[:, 0]
    sigma_s = trapdata[:, 1]
    taur_s = trapdata[:, 2]

    if thibaut:
        sigma_p *= 10000.  #thibault's values in m**2
        sigma_s *= 10000.

    #create a quadrant
    CCD = np.zeros((2066, 2048), dtype=np.float32)

    #add horizontal charge injection lines
    CCD[lines['ystart1']:lines['ystop1'], :] = chargeInjection
    if thibaut:
        writeFITSfile(CCD.copy(), 'ChargeHT.fits', unsigned16bit=False)
    else:
        writeFITSfile(CCD.copy(), 'ChargeH.fits', unsigned16bit=False)

    #radiate CTI to plot initial set trails
    if thibaut:
        CCDCTIhor = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, rdose=8.e9,
                                               sdob=0.0)#sdob=17.74)
        writeFITSfile(CCDCTIhor, 'CTIHT.fits', unsigned16bit=False)
    else:
        CCDCTIhor = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, rdose=1.6e10)
        writeFITSfile(CCDCTIhor, 'CTIH.fits', unsigned16bit=False)

    profileParallel = np.average(CCDCTIhor, axis=1)

    #now serial
    CCD = np.zeros((2066, 2048), dtype=np.float32)

    #add horizontal charge injection lines
    CCD[:, lines['xstart1']:lines['xstop1']] = chargeInjection
    if thibaut:
        writeFITSfile(CCD, 'ChargeVT.fits', unsigned16bit=False)
    else:
        writeFITSfile(CCD, 'ChargeV.fits', unsigned16bit=False)

    #radiate CTI to plot initial set trails
    if thibaut:
        CCDCTI = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, rdose=8.e9,
                                            sdob=0.0)#sdob=17.74)
        writeFITSfile(CCDCTI, 'CTIVT.fits', unsigned16bit=False)
    else:
        CCDCTI = applyRadiationDamageBiDir2(CCD.copy(), nt_p, sigma_p, taur_p, nt_s, sigma_s, taur_s, rdose=1.6e10)
        writeFITSfile(CCDCTI, 'CTIV.fits', unsigned16bit=False)

    #plot trails
    profileSerial = np.average(CCDCTI, axis=0)

    return profileParallel, profileSerial


def testThibautResults(derive=False, lines=dict(ystart1=1064, ystop1=1249, xstart1=577, xstop1=596)):
    """

    :return:
    """
    #get parallel measurements
    indp, parallelValues = parallelMeasurements(returnScale=True)
    inds, serialValues = serialMeasurements(returnScale=True)

    #mask out irrelevant values
    mskp = indp > -5
    msks = inds > -5
    indp = indp[mskp][:100]
    inds = inds[msks][:250]
    parallelValues = parallelValues[mskp][:100]
    serialValues = serialValues[msks][:250]

    #rescale
    indp += 185
    inds += 10

    if derive:
        parallelT, serialT = deriveTrails('cdm_thibaut_parallel.dat', 'cdm_thibaut_serial.dat', thibaut=True)
        parallel, serial = deriveTrails('cdm_euclid_parallel.dat', 'cdm_euclid_serial.dat')
    else:
        tmp = pf.getdata('CTIHT.fits')
        parallelT = np.average(tmp, axis=1)
        tmp = pf.getdata('CTIVT.fits')
        serialT = np.average(tmp, axis=0)
        tmp = pf.getdata('CTIH.fits')
        parallel = np.average(tmp, axis=1)
        tmp = pf.getdata('CTIV.fits')
        serial = np.average(tmp, axis=0)

    #cutout right region
    shift = 5
    profileParallel = parallelT[lines['ystop1']- shift:lines['ystop1']- shift + len(indp)]
    profileParallelM = parallel[lines['ystop1'] - shift:lines['ystop1'] - shift + len(indp)]
    profileSerial = serialT[lines['xstop1'] - shift: lines['xstop1'] - shift + len(inds)]
    profileSerialM = serial[lines['xstop1'] - shift: lines['xstop1'] - shift + len(inds)]

    #set up the charge injection chargeInjection positions
    fig = plt.figure()
    fig.suptitle('CCD204 05325-03-02 Hopkinson EPER at 200kHz, with 20.48ms, 8e9 at 10MeV')
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title('Parallel CTI')
    ax2.set_title('Serial CTI')

    ax1.semilogy(indp, parallelValues, 'bo', ms=3, label='152.55K')
    ax1.semilogy(indp, profileParallel, 'r-', label='Thibaut')
    ax1.semilogy(indp, profileParallelM, 'y-', label='MSSL')

    ax2.semilogy(inds, serialValues, 'bs', ms=3, label='152.5K')
    ax2.semilogy(inds, profileSerial, 'r-', label='Thibaut')
    ax2.semilogy(inds, profileSerialM, 'y-', label='MSSL')

    ax1.set_ylim(1., 60000)
    ax2.set_ylim(1., 60000)

    ax1.set_xlim(180, 250)
    ax2.set_xlim(0, 220)

    ax2.set_xlabel('Pixels')
    ax1.set_ylabel('Photoelectrons')
    ax2.set_ylabel('Photoelectrons')
    ax1.legend(fancybox=True, shadow=True, numpoints=1)
    ax2.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('CompareCDM03values.pdf')
    plt.close()


if __name__ == '__main__':
    #locations of the charge injection lines
    lines = dict(ystart1=1064, ystop1=1250, xstart1=577, xstop1=597)

    #compare MSSL and Thibaut's charge trails
    #parallelT, serialT = deriveTrails('cdm_thibaut_parallel.dat', 'cdm_thibaut_serial.dat', thibaut=True, lines=lines)
    #on may need to change cdm03 model for this
    #parallel, serial = deriveTrails('cdm_euclid_parallel.dat', 'cdm_euclid_serial.dat', lines=lines)
    #testThibautResults(lines=lines)

    #plot EPER test data
    #plotTestData()

    #fit new trap parameters -- LSQ doesn't work too well, perhaps many local minima
    #fitParallelLSQ(lines)
    #fitSerialLSQ(lines)

    fitParallelBayesian(lines)
    #fitParallelBayesian(lines, test=True)
    fitSerialBayesian(lines)
    #fitSerialBayesian(lines, test=True)

    #plot the current trails
    #currentValues()  #add lines input
