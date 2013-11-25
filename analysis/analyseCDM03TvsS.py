import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
import pyfits as pf
import numpy as np
from analysis import shape
from support import logger as lg
from support import files as fileIO
from CTI import CTI
import pprint, sys, traceback


def ThibautsCDM03params():
    return dict(beta_p=0.29, beta_s=0.12, fwc=200000., vth=1.62E+07,
                t=2.10E-02, vg=7.20E-11, st=5.00E-06, sfwc=1450000., svg=3.00E-10)


def MSSLCDM03params():
    return dict(beta_p=0.29, beta_s=0.12, fwc=200000., vth=1.168e7,
                t=20.48e-3, vg=6.e-11, st=5.0e-6, sfwc=730000., svg=1.20E-10)


def TestCDM03params():
    return dict(beta_p=0.5, beta_s=0.5, fwc=200000., vth=162457.31182144422,
                t=2.0e-2, vg=7.20e-11, st=5.0e-6, sfwc=200000., svg=3.0e-10)


def addCTI(file, locx=200, locy=200, bcgr=72.2, thibautCDM03=False, test=False,
           beta=True, serial=1, parallel=0, quadrant=0, single=True):
    """
    Add CTI trails to a FITS file or input data.
    """
    #trap parameters: parallel2
    if thibautCDM03:
        f1 = '/Users/sammy/EUCLID/vissim-python/data/cdm_thibaut_parallel.dat'
        f2 = '/Users/sammy/EUCLID/vissim-python/data/cdm_thibaut_serial.dat'
        params = ThibautsCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.e9, serial=serial, parallel=parallel))
    elif test:
        if single:
            f1 = '/Users/sammy/EUCLID/CTItesting/Reconciliation/singletrap/cdm_test_parallel.dat'
            f2 = '/Users/sammy/EUCLID/CTItesting/Reconciliation/singletrap/cdm_test_serial.dat'
        else:
            f1 = '/Users/sammy/EUCLID/CTItesting/Reconciliation/multitrap/cdm_test_parallel.dat'
            f2 = '/Users/sammy/EUCLID/CTItesting/Reconciliation/multitrap/cdm_test_serial.dat'
        params = TestCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8e9, serial=serial, parallel=parallel))
    else:
        f1 = '/Users/sammy/EUCLID/vissim-python/data/cdm_euclid_parallel.dat'
        f2 = '/Users/sammy/EUCLID/vissim-python/data/cdm_euclid_serial.dat'
        params = MSSLCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.e9, serial=serial, parallel=parallel))
        if beta:
            params.update(dict(beta_p=0.6, beta_s=0.6))

    #load data
    if type(file) is str:
        nocti = pf.getdata(file)
    else:
        nocti = file.copy()

    #place it on canvas
    if locx is not None:
        tmp = np.zeros((2066, 2048))
        ysize, xsize = nocti.shape
        ysize /= 2
        xsize /= 2
        tmp[locy-ysize:locy+ysize, locx-xsize:locx+xsize] = nocti.copy()
    else:
        tmp = nocti

    #add background
    if bcgr is not None:
        tmp += bcgr

    #run CDM03
    c = CTI.CDM03bidir(params, [])
    tmp = c.applyRadiationDamage(tmp.T, iquadrant=quadrant).T

    #make a cutout
    if locx is not None:
        CTIdata = tmp[locy-ysize:locy+ysize, locx-xsize:locx+xsize]
    else:
        CTIdata = tmp

    #remove background
    if bcgr is not None:
        CTIdata -= bcgr
        CTIdata[CTIdata < 0.] = 0.

    return CTIdata


def testShapeMeasurementAlgorithms(log, sigma=0.75, iterations=3, weighted=True,
                                   fixedPosition=False, fixedX=None, fixedY=None):
    #Thibauts data
    folder = '//Users/sammy/EUCLID/CTItesting/Reconciliation/'
    wcti = pf.getdata(folder + 'damaged_image_parallel.fits')
    wo = pf.getdata(folder +
                    'galaxy_100mas_dist2_q=0.9568_re=22.2670_theta=-1.30527_norm=1000_dx=0.2274_dy=0.2352.fits')


    #reset settings
    settings = dict(sigma=sigma, iterations=iterations, weighted=weighted, fixedX=fixedX, fixedY=fixedY,
                    fixedPosition=fixedPosition)

    #calculate shapes
    sh = shape.shapeMeasurement(wcti, log, **settings)
    wctiresults = sh.measureRefinedEllipticity()

    sh = shape.shapeMeasurement(wo, log, **settings)
    woresults = sh.measureRefinedEllipticity()
    #remove one key not needed...
    #woresults.pop('GaussianWeighted', None)
    #pprint.pprint(woresults)

    if fixedPosition == True:
        print '\n\n\n\n%i iterations, keeping centroid fixed' % iterations
    else:
        print '\n\n\n\n%i iterations' % iterations

    #Thibaut's results from his email
    x = 73.41129368592757
    y = 84.48016119109027
    e1 = 0.8130001725751526
    e2 = 0.004147873150093767
    e = 0.8130107535936392
    r2 = 68.45385021546944
    xn = 83.82895826068
    yn = 84.504735271
    e1n = 0.026406
    e2n = 0.031761186
    en = 0.04130482605
    r2n = 11.4263310

    print 'Without CTI:'
    print 'Parameter    Thibaut        SMN             Delta [S-T]'
    print 'X            %.3f        %.3f           %.8f' % (xn,  woresults['centreX'], woresults['centreX'] - xn)
    print 'Y            %.3f        %.3f           %.8f' % (yn, woresults['centreY'], woresults['centreY'] - yn)
    print 'e_1           %.5f       %.5f         %.8f' % (e1n, woresults['e1'], woresults['e1'] - e1n)
    print 'e_2           %.5f       %.5f          %.8f' % (e2n, woresults['e2'], woresults['e2'] - e2n)
    print 'e             %.5f       %.5f          %.8f' % (en, woresults['ellipticity'], woresults['ellipticity'] - en)
    print 'R**2         %.2f         %.2f             %.8f' % (r2n, woresults['R2'], woresults['R2'] - r2n)

    print '\nWith CTI:'
    print 'Parameter    Thibaut        SMN             Delta [S-T]'
    print 'X            %.3f        %.3f            %.8f' % (x,  wctiresults['centreX'], wctiresults['centreX'] - x)
    print 'Y            %.3f        %.3f            %.8f' % (y, wctiresults['centreY'], wctiresults['centreY'] - y)
    print 'e_1           %.5f       %.5f         %.8f' % (e1, wctiresults['e1'], wctiresults['e1'] - e1)
    print 'e_2           %.5f       %.5f          %.8f' % (e2, wctiresults['e2'], wctiresults['e2'] - e2)
    print 'e             %.5f       %.5f         %.8f' % (e, wctiresults['ellipticity'], wctiresults['ellipticity'] - e)
    print 'R**2         %.2f         %.2f           %.8f' % (r2, wctiresults['R2'], wctiresults['R2'] - r2)


def testCTIinclusion(log, sigma=0.75, iterations=3, weighted=True,
                                   fixedPosition=False, fixedX=None, fixedY=None):
    #reset settings
    settings = dict(sigma=sigma, iterations=iterations, weighted=weighted, fixedX=fixedX, fixedY=fixedY,
                    fixedPosition=fixedPosition)

    #Thibauts data
    folder = '//Users/sammy/EUCLID/CTItesting/Reconciliation/'
    wcti = pf.getdata(folder + 'damaged_image_parallel.fits')
    wocti = pf.getdata(folder +
                       'galaxy_100mas_dist2_q=0.9568_re=22.2670_theta=-1.30527_norm=1000_dx=0.2274_dy=0.2352.fits')

    wocti /= np.max(wocti)
    wocti *= 420.

    sh = shape.shapeMeasurement(wcti, log, **settings)
    wctiresults = sh.measureRefinedEllipticity()

    #include CTI with my recipe
    ctiMSSL = addCTI(wocti.copy()).T
    ctiThibault = addCTI(wocti.copy(), thibautCDM03=True).T

    sh = shape.shapeMeasurement(ctiMSSL, log, **settings)
    wMSSLctiresults = sh.measureRefinedEllipticity()

    sh = shape.shapeMeasurement(ctiThibault, log, **settings)
    wThibautctiresults = sh.measureRefinedEllipticity()

    fileIO.writeFITS(ctiThibault, 'tmp2.fits', int=False)
    fileIO.writeFITS(wcti/ctiThibault, 'tmp3.fits', int=False)

    for key in wctiresults:
        tmp1 = wctiresults[key] - wMSSLctiresults[key]
        tmp2 = wctiresults[key] - wThibautctiresults[key]
        if 'Gaussian' in key:
            print key, np.max(np.abs(tmp1)), np.max(np.abs(tmp2))
        else:
            print key, tmp1, tmp2


def fullQuadrantTestSingleTrapSpecies():
    """

    """
    #Thibauts data
    folder = '//Users/sammy/EUCLID/CTItesting/Reconciliation/singletrap/'
    wocti = pf.getdata(folder + 'no_cti.fits')

    #include CTI with my recipe
    ctiMSSLp = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=1, serial=-1, quadrant=2, test=True)
    ctiMSSLs = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=-1, serial=1, quadrant=2, test=True)
    ctiMSSLps = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=1, serial=1, quadrant=2, test=True)

    #save images
    fileIO.writeFITS(ctiMSSLp, 'singletrap/ctiMSSLp.fits', int=False)
    fileIO.writeFITS(ctiMSSLs, 'singletrap/ctiMSSLs.fits', int=False)
    fileIO.writeFITS(ctiMSSLps, 'singletrap/ctiMSSLps.fits', int=False)

    #load Thibaut's data
    Tp = pf.getdata('singletrap/p_cti.fits')
    Ts = pf.getdata('singletrap/s_cti.fits')
    Tps = pf.getdata('singletrap/ps_cti.fits')

    #ratio images
    rp = ctiMSSLp/Tp
    rs = ctiMSSLs/Ts
    rps = ctiMSSLps/Tps
    fileIO.writeFITS(rp, 'singletrap/ctiMSSLdivTp.fits', int=False)
    fileIO.writeFITS(rs, 'singletrap/ctiMSSLdivTs.fits', int=False)
    fileIO.writeFITS(rps, 'singletrap/ctiMSSLdivTps.fits', int=False)
    print 'Parallel Ratio [max, min]:', rp.max(), rp.min()
    print 'Serial Ratio [max, min]:', rs.max(), rs.min()
    print 'Serial+Parallel Ratio [max, min]:', rps.max(), rps.min()

    print 'Checking arrays, parallel'
    np.testing.assert_array_almost_equal(ctiMSSLp, Tp, decimal=7, err_msg='', verbose=True)
    print 'Checking arrays, serial'
    np.testing.assert_array_almost_equal(ctiMSSLs, Ts, decimal=7, err_msg='', verbose=True)
    print 'Checking arrays, serial + parallel'
    np.testing.assert_array_almost_equal(ctiMSSLps, Tps, decimal=7, err_msg='', verbose=True)


def fullQuadrantTestMultiTrapSpecies():
    """

    """
    #Thibauts data
    folder = '//Users/sammy/EUCLID/CTItesting/Reconciliation/multitrap/'
    wocti = pf.getdata(folder + 'no_cti.fits')

    #include CTI with my recipe
    ctiMSSLp = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=1, serial=-1, quadrant=2, test=True, single=False)
    ctiMSSLs = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=-1, serial=1, quadrant=2, test=True, single=False)
    ctiMSSLps = addCTI(wocti.copy(), locx=None, bcgr=None, parallel=1, serial=1, quadrant=2, test=True, single=False)

    #save images
    fileIO.writeFITS(ctiMSSLp, 'multitrap/ctiMSSLp.fits', int=False)
    fileIO.writeFITS(ctiMSSLs, 'multitrap/ctiMSSLs.fits', int=False)
    fileIO.writeFITS(ctiMSSLps, 'multitrap/ctiMSSLps.fits', int=False)

    #load Thibaut's data
    Tp = pf.getdata('multitrap/p_cti.fits')
    Ts = pf.getdata('multitrap/s_cti.fits')
    Tps = pf.getdata('multitrap/ps_cti.fits')

    #ratio images
    rp = ctiMSSLp/Tp
    rs = ctiMSSLs/Ts
    rps = ctiMSSLps/Tps
    fileIO.writeFITS(rp, 'multitrap/ctiMSSLdivTp.fits', int=False)
    fileIO.writeFITS(rs, 'multitrap/ctiMSSLdivTs.fits', int=False)
    fileIO.writeFITS(rps, 'multitrap/ctiMSSLdivTps.fits', int=False)
    print 'Parallel Ratio [max, min]:', rp.max(), rp.min()
    print 'Serial Ratio [max, min]:', rs.max(), rs.min()
    print 'Serial+Parallel Ratio [max, min]:', rps.max(), rps.min()

    print 'Checking arrays, parallel'
    np.testing.assert_array_almost_equal(ctiMSSLp, Tp, decimal=6, err_msg='', verbose=True)
    print 'Checking arrays, serial'
    np.testing.assert_array_almost_equal(ctiMSSLs, Ts, decimal=6, err_msg='', verbose=True)
    print 'Checking arrays, serial + parallel'
    np.testing.assert_array_almost_equal(ctiMSSLps, Tps, decimal=6, err_msg='', verbose=True)


if __name__ == '__main__':
    log = lg.setUpLogger('CTItesting.log')

    #testShapeMeasurementAlgorithms(log)
    #testShapeMeasurementAlgorithms(log, iterations=10)
    #testShapeMeasurementAlgorithms(log, iterations=500)
    #testShapeMeasurementAlgorithms(log, iterations=1, fixedPosition=True, fixedX=83.829, fixedY=84.504, sigma=0.1)

    #testCTIinclusion(log)

    #when running these, remember to change CTI.py to include import cdm03bidirTest as cdm03bidir
    try:
        print '\n\n\nSingle Trap Species'
        fullQuadrantTestSingleTrapSpecies()
    except AssertionError:
        _,_,tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
    try:
        print '\n\n\nMultiple Trap Species'
        fullQuadrantTestMultiTrapSpecies()
    except AssertionError:
        _,_,tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format