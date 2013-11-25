"""
Testing the CTI Correction Algorithm
====================================

This script can be used to test the CTI correction algorithm performance.

:requires: NumPy
:requires: PyFITS
:requires: matplotlib

:version: 0.3

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
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
import glob as g
import pyfits as pf
import numpy as np
import cPickle, os, datetime, shutil
from multiprocessing import Pool
from analysis import shape
from support import logger as lg
from support import files as fileIO
from CTI import CTI


def ThibautsCDM03params():
    return dict(beta_p=0.29, beta_s=0.12, fwc=200000., vth=1.62E+07,
                t=2.10E-02, vg=7.20E-11, st=5.00E-06, sfwc=1450000., svg=3.00E-10)


def MSSLCDM03params():
    return dict(beta_p=0.29, beta_s=0.12, fwc=200000., vth=1.168e7,
                t=20.48e-3, vg=6.e-11, st=5.0e-6, sfwc=730000., svg=1.20E-10)


def testCTIcorrection(log, files, sigma=0.75, iterations=4, xcen=1900, ycen=1900, side=20):
    """
    Calculates PSF properties such as ellipticity and size from data without CTI and from
    CTI data.

    :param log: python logger instance
    :type log: instance
    :param files: a list of files to be processed
    :type files: list
    :param sigma: size of the Gaussian weighting function
    :type sigma: float
    :param iterations: the number of iterations for the moment based shape estimator
    :type iterations: int
    :param xcen: x-coordinate of the object centre
    :type xcen: int
    :param ycen: y-coordinate of the object centre
    :type ycen: int
    :param side: size of the cutout around the centre (+/- side)
    :type side: int

    :return: ellipticity and size
    :rtype: dict
    """
    settings = dict(sigma=sigma, iterations=iterations)

    eclean = []
    e1clean = []
    e2clean = []
    R2clean = []
    eCTI = []
    e1CTI = []
    e2CTI = []
    R2CTI = []
    for file in files:
        #load no cti data
        nocti = pf.getdata(file.replace('CTI', 'nocti'))[ycen-side:ycen+side, xcen-side:xcen+side]
        #subtract background
        nocti -= 27.765714285714285
        nocti[nocti < 0.] = 0.  #remove negative numbers

        #load CTI data
        CTI = pf.getdata(file)[ycen-side:ycen+side, xcen-side:xcen+side]
        CTI[CTI < 0.] = 0. #remove negative numbers

        sh = shape.shapeMeasurement(nocti, log, **settings)
        results = sh.measureRefinedEllipticity()

        eclean.append(results['ellipticity'])
        e1clean.append(results['e1'])
        e2clean.append(results['e2'])
        R2clean.append(results['R2'])

        sh = shape.shapeMeasurement(CTI, log, **settings)
        results = sh.measureRefinedEllipticity()

        eCTI.append(results['ellipticity'])
        e1CTI.append(results['e1'])
        e2CTI.append(results['e2'])
        R2CTI.append(results['R2'])

    results = {'eclean' : np.asarray(eclean),
               'e1clean' : np.asarray(e1clean),
               'e2clean' : np.asarray(e2clean),
               'R2clean' : np.asarray(R2clean),
               'eCTI' : np.asarray(eCTI),
               'e1CTI' : np.asarray(e1CTI),
               'e2CTI' : np.asarray(e2CTI),
               'R2CTI' : np.asarray(R2CTI)}

    #save to a file
    fileIO.cPickleDumpDictionary(results, 'results.pk')

    return results


def testCTIcorrectionNonoise(log, files, output, sigma=0.75, iterations=4):
    """
    Calculates PSF properties such as ellipticity and size from data w/ and w/o CTI.

    :param log: python logger instance
    :type log: instance
    :param files: a list of files to be processed
    :type files: list
    :param sigma: size of the Gaussian weighting function
    :type sigma: float
    :param iterations: the number of iterations for the moment based shape estimator
    :type iterations: int

    :return: ellipticity and size
    :rtype: dict
    """
    eclean = []
    e1clean = []
    e2clean = []
    R2clean = []
    xclean = []
    yclean = []
    eCTI = []
    e1CTI = []
    e2CTI = []
    R2CTI = []
    xCTI = []
    yCTI = []
    eCTIfixed = []
    e1CTIfixed = []
    e2CTIfixed = []
    R2CTIfixed = []
    xCTIfixed = []
    yCTIfixed = []

    fh = open(output.replace('pk', 'csv'), 'w')
    fh.write('#file, delta_e, delta_e1, delta_e2, delta_R2, delta_x, delta_y\n')
    for f in files:
        print 'Processing: ', f

        #reset settings
        settings = dict(sigma=sigma, iterations=iterations)

        #load no cti data
        nocti = pf.getdata(f.replace('CUT', 'CUTnoctinonoise'))

        #load CTI data
        CTI = pf.getdata(f)

        sh = shape.shapeMeasurement(nocti, log, **settings)
        results = sh.measureRefinedEllipticity()

        eclean.append(results['ellipticity'])
        e1clean.append(results['e1'])
        e2clean.append(results['e2'])
        R2clean.append(results['R2'])
        xclean.append(results['centreX'])
        yclean.append(results['centreY'])

        #CTI, fitted centroid
        sh = shape.shapeMeasurement(CTI.copy(), log, **settings)
        results2 = sh.measureRefinedEllipticity()

        eCTI.append(results2['ellipticity'])
        e1CTI.append(results2['e1'])
        e2CTI.append(results2['e2'])
        R2CTI.append(results2['R2'])
        xCTI.append(results2['centreX'])
        yCTI.append(results2['centreY'])

        #fixed centroid
        settings['fixedPosition'] = True
        settings['fixedX'] = results['centreX']
        settings['fixedY'] = results['centreY']
        settings['iterations'] = 1
        sh = shape.shapeMeasurement(CTI.copy(), log, **settings)
        results3 = sh.measureRefinedEllipticity()

        eCTIfixed.append(results3['ellipticity'])
        e1CTIfixed.append(results3['e1'])
        e2CTIfixed.append(results3['e2'])
        R2CTIfixed.append(results3['R2'])
        xCTIfixed.append(results3['centreX'])
        yCTIfixed.append(results3['centreY'])

        text = '%s,%e,%e,%e,%e,%e,%e\n' % (f, results['ellipticity'] - results2['ellipticity'],
              results['e1'] - results2['e1'], results['e2'] - results2['e2'], results['R2'] - results2['R2'],
              results['centreX'] - results2['centreX'], results['centreY'] - results2['centreY'])
        fh.write(text)
        print text

    fh.close()

    results = {'eclean' : np.asarray(eclean),
               'e1clean' : np.asarray(e1clean),
               'e2clean' : np.asarray(e2clean),
               'R2clean' : np.asarray(R2clean),
               'xclean' : np.asarray(xclean),
               'yclean' : np.asarray(yclean),
               'eCTI' : np.asarray(eCTI),
               'e1CTI' : np.asarray(e1CTI),
               'e2CTI' : np.asarray(e2CTI),
               'R2CTI' : np.asarray(R2CTI),
               'xCTI' : np.asarray(xCTI),
               'yCTI' : np.asarray(yCTI),
               'eCTIfixed': np.asarray(eCTIfixed),
               'e1CTIfixed': np.asarray(e1CTIfixed),
               'e2CTIfixed': np.asarray(e2CTIfixed),
               'R2CTIfixed': np.asarray(R2CTIfixed),
               'xCTIfixed': np.asarray(xCTIfixed),
               'yCTIfixed': np.asarray(yCTIfixed)}

    #save to a file
    fileIO.cPickleDumpDictionary(results, output)

    return results


def plotResults(results):
    """
    Plot the CTI correction algorithm results.

    :param results: CTI test results
    :return: None
    """
    e = results['eclean'] - results['eCTI']
    e1 = results['e1clean'] - results['e1CTI']
    e2 = results['e2clean'] - results['e2CTI']

    print 'Delta e, e_1, e_2:', np.mean(e), np.mean(e1), np.mean(e2)
    print 'std e, e_1, e_2:', np.std(e), np.std(e1), np.std(e2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(e, bins=15, label='$e$', alpha=0.5)
    ax.hist(e1, bins=15, label='$e_{2}$', alpha=0.5)
    ax.hist(e2, bins=15, label='$e_{1}$', alpha=0.5)
    ax.set_xlabel(r'$\delta e$ [no CTI - CDM03 corrected]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('ellipticityDelta.pdf')
    plt.close()

    r2 = (results['R2clean'] - results['R2CTI'])/results['R2clean']
    print 'delta R2 / R2: mean, std ', np.mean(r2), np.std(r2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(r2, bins=15, label='$R^{2}$')
    ax.set_xlabel(r'$\frac{\delta R^{2}}{R^{2}_{ref}}$ [no CTI - CDM03 corrected]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('sizeDelta.pdf')
    plt.close()


def plotResultsNoNoise(inputfile, title, bins=10):
    """
    Plot the CTI correction algorithm results.

    :return: None
    """
    path = datetime.datetime.now().isoformat()
    os.mkdir(path)
    path += '/'

    results = cPickle.load(open(inputfile))
    #copy input to the path
    try:
        shutil.copy2(inputfile, path+inputfile)
    except:
        pass

    print '\n\n\n\nFitted centre:'

    e = results['eclean'] - results['eCTI']
    e1 = results['e1clean'] - results['e1CTI']
    e2 = results['e2clean'] - results['e2CTI']
    x = results['xclean'] - results['xCTI']
    y = results['yclean'] - results['yCTI']
    r2 = (results['R2clean'] - results['R2CTI']) / results['R2clean']
    meane = np.mean(e)
    meane1 = np.mean(e1)
    meane2 = np.mean(e2)
    meanx = np.mean(x)
    meany = np.mean(y)
    meanr2 = np.mean(r2)

    print 'Delta e, e_1, e_2:', meane, meane1, meane2
    #print 'std e, e_1, e_2:', np.std(e), np.std(e1), np.std(e2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(e, bins=bins, color='b', label='$e$', alpha=0.5)
    ax.hist(e1, bins=bins, color='r', label='$e_{1}$', alpha=0.5)
    ax.hist(e2, bins=bins, color='g', label='$e_{2}$', alpha=0.5)
    ax.axvline(x=meane, color='b', label='%.2e' % meane)
    ax.axvline(x=meane1, color='r', label='%.2e' % meane1)
    ax.axvline(x=meane2, color='g', label='%.2e' % meane2)
    ax.set_xlabel(r'$\delta e$ [w/o - w/ CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'ellipticityDeltaFittedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(e1, e2, s=8, color='r', marker='o', alpha=0.5, label='w/o - w/ CTI')
    ax.set_xlabel(r'$\delta e_{1}$')
    ax.set_ylabel(r'$\delta e_{2}$')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'ellipticityFittedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(results['e1clean'], results['e2clean'], s=8, color='k', marker='s', alpha=0.1, label='no CTI')
    ax.scatter(results['e1CTI'], results['e2CTI'], s=8, color='r', marker='o', alpha=0.4, label='CTI')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$e_{1}$')
    ax.set_ylabel(r'$e_{2}$')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'e1vse2FittedCentre.pdf')
    plt.close()

    print 'delta R2 / R2: mean, std ', meanr2, np.std(r2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(r2, bins=bins, color='b', label='$R^{2}$')
    ax.axvline(x=meanr2,color='b', label='%.2e' % meanr2)
    ax.set_xlabel(r'$\frac{\delta R^{2}}{R^{2}_{ref}}$ [w/o - w CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'sizeDeltaFittedCentre.pdf')
    plt.close()

    print 'delta x: mean, std ', meanx, np.std(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(x, bins=bins, color='b', label='X Centre')
    ax.axvline(x=meanx,color='b', label='%.2e' % meanx)
    ax.set_xlabel(r'$\delta X - X_{CTI}$ [w/o - w CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'xDeltaFittedCentre.pdf')
    plt.close()

    print 'delta y: mean, std ', meany, np.std(y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(y, bins=bins, color='b', label='Y Centre')
    ax.axvline(x=meany,color='b', label='%.2e' % meany)
    ax.set_xlabel(r'$\delta Y - Y_{CTI}$ [w/o - w CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'yDeltaFittedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(x, y, s=15, color='k', marker='s', alpha=0.5, label='w/o - w/ CTI')
    ax.set_xlabel(r'$\delta X$')
    ax.set_ylabel(r'$\delta Y$')
    plt.legend(shadow=True, fancybox=True, scatterpoints=1)
    plt.savefig(path+'coordinatesFittedCentre.pdf')
    plt.close()

    print '\n\n\n\nFixed centre:'

    e = results['eclean'] - results['eCTIfixed']
    e1 = results['e1clean'] - results['e1CTIfixed']
    e2 = results['e2clean'] - results['e2CTIfixed']
    x = results['xclean'] - results['xCTIfixed']
    y = results['yclean'] - results['yCTIfixed']
    r2 = (results['R2clean'] - results['R2CTIfixed']) / results['R2clean']
    meane = np.mean(e)
    meane1 = np.mean(e1)
    meane2 = np.mean(e2)
    meanx = np.mean(x)
    meany = np.mean(y)
    meanr2 = np.mean(r2)

    print 'Delta e, e_1, e_2:', meane, meane1, meane2
    #print 'std e, e_1, e_2:', np.std(e), np.std(e1), np.std(e2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(e, bins=bins, color='b', label='$e$', alpha=0.5)
    ax.hist(e1, bins=bins, color='r', label='$e_{1}$', alpha=0.5)
    ax.hist(e2, bins=bins, color='g', label='$e_{2}$', alpha=0.5)
    ax.axvline(x=meane, color='b', label='%.2e' % meane)
    ax.axvline(x=meane1, color='r', label='%.2e' % meane1)
    ax.axvline(x=meane2, color='g', label='%.2e' % meane2)
    ax.set_xlabel(r'$\delta e$ [w/o - w/ CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'ellipticityDeltaFixedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(e1, e2, s=8, color='r', marker='o', alpha=0.5, label='w/o - w/ CTI')
    ax.set_xlabel(r'$\delta e_{1}$')
    ax.set_ylabel(r'$\delta e_{2}$')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'ellipticityFixedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(results['e1clean'], results['e2clean'], s=8, color='k', marker='s', alpha=0.1, label='no CTI')
    ax.scatter(results['e1CTIfixed'], results['e2CTIfixed'], s=8, color='r', marker='o', alpha=0.4, label='CTI')
    ax.set_xlabel(r'$e_{1}$')
    ax.set_ylabel(r'$e_{2}$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'e1vse2FixedCentre.pdf')
    plt.close()

    print 'delta R2 / R2: mean, std ', meanr2, np.std(r2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(r2, bins=bins, color='b', label='$R^{2}$')
    ax.axvline(x=meanr2, color='b', label='%.2e' % meanr2)
    ax.set_xlabel(r'$\frac{\delta R^{2}}{R^{2}_{ref}}$ [w/o - w CTI]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'sizeDeltaFixedCentre.pdf')
    plt.close()

    print 'delta x: mean, std ', meanx, np.std(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(x, bins=bins, color='b', label='X Centre')
    ax.axvline(x=meanx, color='b', label='%.2e' % meanx)
    ax.set_xlabel(r'$X - X_{CTI}$')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'xDeltaFixedCentre.pdf')
    plt.close()

    print 'delta y: mean, std ', meany, np.std(y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.hist(y, bins=bins, color='b', label='Y Centre')
    ax.axvline(x=meany, color='b', label='%.2e' % meany)
    ax.set_xlabel(r'$Y - Y_{CTI}$')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig(path+'yDeltaFixedCentre.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(x, y, s=15, color='k', marker='s', alpha=0.5, label='w/o - w/ CTI')
    ax.set_xlabel(r'$\delta X$')
    ax.set_ylabel(r'$\delta Y$')
    plt.legend(shadow=True, fancybox=True, scatterpoints=1)
    plt.savefig(path+'coordinatesFixedCentre.pdf')
    plt.close()


def cutoutRegions(files, xcen=1900, ycen=1900, side=140):
    """

    :param files:
    :param xcen:
    :param ycen:
    :param side:
    :return:
    """
    print 'Generating postage stamp images'
    for f in files:
        print 'Processing: ', f

        #load no cti data
        fh = pf.open('noctinonoise' + f)
        data = fh[1].data
        #subtract background
        #print '%.15f' % np.median(pf.getdata('noctinonoise' + f)[100:1000, 100:1000])
        d = data[ycen - side:ycen + side, xcen - side:xcen + side]
        d -= 103.258397270204995
        d[d < 0.] = 0.
        fh[1].data = d
        fh.writeto('CUTnoctinonoise' + f, clobber=True)

        #load CTI data
        fh = pf.open(f)
        data = fh[1].data
        #subtract background
        #print '%.15f' % np.median(pf.getdata('noctinonoise' + f)[100:1000, 100:1000])
        d = data[ycen - side:ycen + side, xcen - side:xcen + side]
        d -= 103.181143639616520
        d[d < 0.] = 0.
        fh[1].data = d
        fh.writeto('CUT' + f, clobber=True)


def useThibautsData(log, output, bcgr=72.2, sigma=0.75, iterations=4, loc=1900, galaxies=1000,
                    datadir='/Users/smn2/EUCLID/CTItesting/uniform/',
                    thibautCDM03=False, beta=False, serial=1, parallel=1):
    """
    Test the impact of CTI in case of no noise and no correction.

    :param log: logger instance
    :param bcgr: background in electrons for the CTI modelling
    :param sigma: size of the weighting function for the quadrupole moment
    :param iterations: number of iterations in the quadrupole moments estimation
    :param loc: location to which the galaxy will be placed [default=1900]
    :param galaxies: number of galaxies to use (< 10000)
    :param datadir: directory pointing to the galaxy images

    :return:
    """
    files = g.glob(datadir + '*.fits')
    #pick randomly
    files = np.random.choice(files, galaxies, replace=False)

    #trap parameters: parallel
    if thibautCDM03:
        f1 = '/Users/smn2/EUCLID/vissim-python/data/cdm_thibaut_parallel.dat'
        f2 = '/Users/smn2/EUCLID/vissim-python/data/cdm_thibaut_serial.dat'
        params = ThibautsCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.0e9, serial=serial, parallel=parallel))
    else:
        f1 = '/Users/smn2/EUCLID/vissim-python/data/cdm_euclid_parallel.dat'
        f2 = '/Users/smn2/EUCLID/vissim-python/data/cdm_euclid_serial.dat'
        params = MSSLCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.0e9, serial=serial, parallel=parallel))
        if beta:
            params.update(dict(beta_p=0.6, beta_s=0.6))

    print f1, f2

    #store shapes
    eclean = []
    e1clean = []
    e2clean = []
    R2clean = []
    xclean = []
    yclean = []
    eCTI = []
    e1CTI = []
    e2CTI = []
    R2CTI = []
    xCTI = []
    yCTI = []
    eCTIfixed = []
    e1CTIfixed = []
    e2CTIfixed = []
    R2CTIfixed = []
    xCTIfixed = []
    yCTIfixed = []

    fh = open(output.replace('.pk', '.csv'), 'w')
    fh.write('#files: %s and %s\n' % (f1, f2))
    for key in params:
        print key, params[key]
        fh.write('# %s = %s\n' % (key, str(params[key])))
    fh.write('#file, delta_e, delta_e1, delta_e2, delta_R2, delta_x, delta_y\n')
    for f in files:
        print 'Processing: ', f

        #load data
        nocti = pf.getdata(f)

        #scale to SNR about 10 (average galaxy, a single exposure)
        nocti /= np.sum(nocti)
        nocti *= 1500.

        #place it on canvas
        tmp = np.zeros((2066, 2048))
        ysize, xsize = nocti.shape
        ysize /= 2
        xsize /= 2
        tmp[loc-ysize:loc+ysize, loc-xsize:loc+xsize] = nocti.copy()

        #add background
        tmp += bcgr

        #run CDM03
        c = CTI.CDM03bidir(params, [])
        tmp = c.applyRadiationDamage(tmp.copy().transpose()).transpose()

        #remove background and make a cutout
        CTIdata = tmp[loc-ysize:loc+ysize, loc-xsize:loc+xsize]
        CTIdata -= bcgr
        CTIdata[CTIdata < 0.] = 0.

        #write files
        #fileIO.writeFITS(nocti, f.replace('.fits', 'noCTI.fits'), int=False)
        #fileIO.writeFITS(CTI, f.replace('.fits', 'CTI.fits'), int=False)

        #reset settings
        settings = dict(sigma=sigma, iterations=iterations)

        #calculate shapes
        sh = shape.shapeMeasurement(nocti.copy(), log, **settings)
        results = sh.measureRefinedEllipticity()

        eclean.append(results['ellipticity'])
        e1clean.append(results['e1'])
        e2clean.append(results['e2'])
        R2clean.append(results['R2'])
        xclean.append(results['centreX'])
        yclean.append(results['centreY'])

        #CTI, fitted centroid
        sh = shape.shapeMeasurement(CTIdata.copy(), log, **settings)
        results2 = sh.measureRefinedEllipticity()

        eCTI.append(results2['ellipticity'])
        e1CTI.append(results2['e1'])
        e2CTI.append(results2['e2'])
        R2CTI.append(results2['R2'])
        xCTI.append(results2['centreX'])
        yCTI.append(results2['centreY'])

        #fixed centroid
        settings['fixedPosition'] = True
        settings['fixedX'] = results['centreX']
        settings['fixedY'] = results['centreY']
        settings['iterations'] = 1
        sh = shape.shapeMeasurement(CTIdata.copy(), log, **settings)
        results3 = sh.measureRefinedEllipticity()

        eCTIfixed.append(results3['ellipticity'])
        e1CTIfixed.append(results3['e1'])
        e2CTIfixed.append(results3['e2'])
        R2CTIfixed.append(results3['R2'])
        xCTIfixed.append(results3['centreX'])
        yCTIfixed.append(results3['centreY'])

        text = '%s,%e,%e,%e,%e,%e,%e\n' % (f, results['ellipticity'] - results2['ellipticity'],
                                           results['e1'] - results2['e1'], results['e2'] - results2['e2'],
                                           results['R2'] - results2['R2'],
                                           results['centreX'] - results2['centreX'],
                                           results['centreY'] - results2['centreY'])
        fh.write(text)
        print text

    fh.close()

    results = {'eclean': np.asarray(eclean),
               'e1clean': np.asarray(e1clean),
               'e2clean': np.asarray(e2clean),
               'R2clean': np.asarray(R2clean),
               'xclean': np.asarray(xclean),
               'yclean': np.asarray(yclean),
               'eCTI': np.asarray(eCTI),
               'e1CTI': np.asarray(e1CTI),
               'e2CTI': np.asarray(e2CTI),
               'R2CTI': np.asarray(R2CTI),
               'xCTI': np.asarray(xCTI),
               'yCTI': np.asarray(yCTI),
               'eCTIfixed': np.asarray(eCTIfixed),
               'e1CTIfixed': np.asarray(e1CTIfixed),
               'e2CTIfixed': np.asarray(e2CTIfixed),
               'R2CTIfixed': np.asarray(R2CTIfixed),
               'xCTIfixed': np.asarray(xCTIfixed),
               'yCTIfixed': np.asarray(yCTIfixed)}

    #save to a file
    fileIO.cPickleDumpDictionary(results, output)

    return results


def addCTI(file, loc=1900, bcgr=72.2, thibautCDM03=False, beta=True, serial=1, parallel=1):
    """
    Add CTI trails to a FITS file or input data.
    """
    #trap parameters: parallel
    if thibautCDM03:
        f1 = '/Users/sammy/EUCLID/vissim-python/data/cdm_thibaut_parallel.dat'
        f2 = '/Users/sammy/EUCLID/vissim-python/data/cdm_thibaut_serial.dat'
        params = ThibautsCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.0e9, serial=serial, parallel=parallel))
    else:
        f1 = '/Users/sammy/EUCLID/vissim-python/data/cdm_euclid_parallel.dat'
        f2 = '/Users/sammy/EUCLID/vissim-python/data/cdm_euclid_serial.dat'
        params = MSSLCDM03params()
        params.update(dict(parallelTrapfile=f1, serialTrapfile=f2, rdose=8.0e9, serial=serial, parallel=parallel))
        if beta:
            params.update(dict(beta_p=0.6, beta_s=0.6))

    print f1, f2

    #load data
    if type(file) is str:
        nocti = pf.getdata(file)
    else:
        nocti = file.copy()

    #place it on canvas
    tmp = np.zeros((2066, 2048))
    ysize, xsize = nocti.shape
    ysize /= 2
    xsize /= 2
    tmp[loc-ysize:loc+ysize, loc-xsize:loc+xsize] = nocti.copy()

    #add background
    tmp += bcgr

    #run CDM03
    c = CTI.CDM03bidir(params, [])
    tmp = c.applyRadiationDamage(tmp)

    #remove background and make a cutout
    CTIdata = tmp[loc-ysize:loc+ysize, loc-xsize:loc+xsize]
    CTIdata -= bcgr
    CTIdata[CTIdata < 0.] = 0.

    return CTIdata


def simpleTest(log, sigma=0.75, iterations=50):
    #Thibauts data
    folder = '/Users/sammy/EUCLID/CTItesting/uniform/'
    wcti = pf.getdata(folder +
                      'galaxy_100mas_dist2_q=0.5078_re=6.5402_theta=0.91895_norm=1000_dx=0.3338_dy=0.0048CTI.fits')
    wocti = pf.getdata(folder +
                       'galaxy_100mas_dist2_q=0.5078_re=6.5402_theta=0.91895_norm=1000_dx=0.3338_dy=0.0048noCTI.fits')

    #reset settings
    settings = dict(sigma=sigma, iterations=iterations)

    #calculate shapes
    sh = shape.shapeMeasurement(wcti, log, **settings)
    wctiresults = sh.measureRefinedEllipticity()

    sh = shape.shapeMeasurement(wocti, log, **settings)
    woctiresults = sh.measureRefinedEllipticity()

    #include CTI with my recipe
    ctiMSSL = addCTI(wocti.copy())
    ctiThibault = addCTI(wocti.copy(), thibautCDM03=True)

    sh = shape.shapeMeasurement(ctiMSSL, log, **settings)
    wMSSLctiresults = sh.measureRefinedEllipticity()

    sh = shape.shapeMeasurement(ctiThibault, log, **settings)
    wThibautctiresults = sh.measureRefinedEllipticity()

    fileIO.writeFITS(ctiMSSL, 'tmp1.fits', int=False)
    fileIO.writeFITS(ctiThibault, 'tmp2.fits', int=False)
    fileIO.writeFITS(wcti/ctiMSSL, 'tmp3.fits', int=False)

    for key in wctiresults:
        tmp1 = wctiresults[key] - wMSSLctiresults[key]
        tmp2 = wctiresults[key] - wThibautctiresults[key]
        if 'Gaussian' in key:
            print key, np.max(np.abs(tmp1)), np.max(np.abs(tmp2))
        else:
            print key, tmp1, tmp2


if __name__ == '__main__':
    log = lg.setUpLogger('CTItesting.log')

    #simple test with Thibaut's files
    simpleTest(log)

    #use Thibaut's input galaxies
    galaxies = 800
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsDataP.pk', galaxies=galaxies, serial=-1)
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsDatab6P.pk', beta=True, galaxies=galaxies, serial=-1)
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsDataThibautsCDM03P.pk', thibautCDM03=True, galaxies=galaxies, serial=-1)
    #plotResultsNoNoise('resultsNoNoiseThibautsDataP.pk', 'MSSL CDM03 Parameters (beta=0.29, 0.12) (parallel only)')
    #plotResultsNoNoise('resultsNoNoiseThibautsDatab6P.pk', 'MSSL CDM03 Parameters (beta=0.6, 0.6) (parallel only)')
    #plotResultsNoNoise('resultsNoNoiseThibautsDataThibautsCDM03P.pk', 'Thibaut CDM03 Parameters (parallel only)')
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsData.pk', galaxies=galaxies)
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsDatab6.pk', beta=True, galaxies=galaxies)
    #thibaut = useThibautsData(log, 'resultsNoNoiseThibautsDataThibautsCDM03.pk', thibautCDM03=True, galaxies=galaxies)
    #plotResultsNoNoise('resultsNoNoiseThibautsData.pk', 'MSSL CDM03 Parameters (beta=0.29, 0.12)')
    #plotResultsNoNoise('resultsNoNoiseThibautsDatab6.pk', 'MSSL CDM03 Parameters (beta=0.6, 0.6)')
    #plotResultsNoNoise('resultsNoNoiseThibautsDataThibautsCDM03.pk', 'Thibaut CDM03 Parameters')

    #cut out regions
    #cutoutRegions(g.glob('Q0_00_00stars*.fits'))
    #cutoutRegions(g.glob('Q0_00_00galaxy*.fits'))

    #use the cutouts -- stars
    #results = testCTIcorrectionNonoise(log, g.glob('CUTQ*stars*.fits'), 'resultsNoNoiseStars.pk', iterations=4)
    #plotResultsNoNoise('resultsNoNoiseStars.pk')

    #galaxies
    #results = testCTIcorrectionNonoise(log, g.glob('CUTQ*galaxy*.fits'), 'resultsNoNoiseGalaxies.pk', iterations=4)
    #plotResultsNoNoise('resultsNoNoiseGalaxies.pk')

    #results = testCTIcorrection(log, g.glob('CTIQ0_00_00stars*'), iterations=8, side=25)
    #plotResults(results)
