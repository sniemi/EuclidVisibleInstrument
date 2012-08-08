"""
Flat Field Calibration
======================

This simple script can be used to study the number of flat fields required to meet the VIS calibration requirements.

The following requirements related to the bias calibration has been taken from GDPRD.

R-GDP-CAL-0:


R-GDP-CAL-0:


:requires: PyFITS
:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools, glob
from analysis import shape
from support import logger as lg
from support import surfaceFitting as sf
from support import bleedingtest as write
from support import files as fileIO


def test():
    """

    """
    #find all FITS files and load data
    files = glob.glob('Q0*flatfield*.fits')[:5]
    data = fileIO.readFITSDataExcludeScanRegions(files)

    #check that the sizes match and median combine
    if len(set(x.shape for x in data))  > 1:
        print 'ERROR'
    else:
        #median combine and scale to electrons
        median = np.median(data, axis=0) * 3.5

    #fit surface to the median and normalize it out
    #m = sf.polyfit2d(xx.ravel(), yy.ravel(), median.ravel(), order=order)
    # Evaluate it on a rectangular grid
    #fitted = sf.polyval2d(xx, yy, m)

    ysize, xsize = median.shape
    xx, yy = np.meshgrid(np.linspace(0, xsize, xsize), np.linspace(0, ysize, ysize))

    #plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx, yy, median, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('Flat Field Counts [electrons]')
    plt.savefig('MedianFlat.png')
    plt.close()

    #load the lamp profile that went in
    lamp = pf.getdata('data/VIScalibrationUnitflux.fits')
    pixvar = median / lamp

    #plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx, yy, pixvar, rstride=100, cstride=100, alpha=0.6, cmap=cm.jet)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('Flat Field Counts [electrons]')
    plt.savefig('PixelFlat.png')
    plt.close()

    real = pf.getdata('data/VISFlatField2percent.fits')
    res = pixvar / real * 100
    print np.mean(res), np.min(res), np.max(res), np.std(res)

    #plot
    im = plt.imshow(res, origin='lower')
    c1 = plt.colorbar(im)
    c1.set_label('Derived / Input * 100 [per cent]')
    plt.xlabel('Y [pixels]')
    plt.ylabel('X [pixels]')
    plt.savefig('ResidualFlatField2D.png')
    plt.close()



if __name__ == '__main__':
    run = True

    #start the script
    log = lg.setUpLogger('flatfieldCalibration.log')
    log.info('Testing flat fielding calibration...')

    test()

#    if run:
#        pass
#    else:
#        resultsDelta = cPickle.load(open('flatfieldResultsDelta.pk'))
#        resultsSigma = cPickle.load(open('flatfieldResultsSigma.pk'))
#
#    plotNumberOfFramesSigma(resultsSigma)
#    plotNumberOfFramesDelta(resultsDelta)

    log.info('Run finished...\n\n\n')
