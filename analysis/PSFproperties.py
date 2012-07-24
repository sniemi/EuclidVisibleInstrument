"""
This script can be used to plot some PSF properties such as ellipticity and size as a function of the focal plane position.

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
from matplotlib.mlab import griddata
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pyfits as pf
import numpy as np
import math, datetime, cPickle, itertools, re, glob
from analysis import shape
from support import logger as lg
from support import files as fileIO


def readData(file):
    """
    Reads in the data from a given FITS file.
    """
    return pf.getdata(file)


def parseName(file):
    """
    Parse information from the input file name.

    Example name::

        detector_jitter-1_TOL05_MC_T0074_50arcmin2_grid_Nim=16384x16384_pixsize=1.000um_lbda=800nm_fieldX=-0.306_fieldY=1.042.fits
    """
    xpos = float(re.compile('fieldX=([-+]?[0-9]*\.?[0-9]*)').findall(file)[0])
    ypos = float(re.compile('fieldY=([-+]?[0-9]*\.?[0-9]*)').findall(file)[0])
    lbda = float(re.compile('lbda=([0-9]*\.?[0-9]*)').findall(file)[0])
    pixsize = float(re.compile('pixsize=([0-9]*\.?[0-9]*)').findall(file)[0])

    out = dict(xpos=xpos, ypos=ypos, lbda=lbda, pixsize=pixsize)
    return out


def measureChars(data, info, log):
    """
    Measure ellipticity, R2, FWHM etc.
    """
    #settings = dict(pixelSize=info['pixsize'], sampling=info['pixsize']/12.)
    settings = dict(sampling=info['pixsize']/12.)
    sh = shape.shapeMeasurement(data.copy(), log, **settings)
    results = sh.measureRefinedEllipticity()

    out = dict(ellipticity=results['ellipticity'], e1=results['e1'], e2=results['e2'], R2=results['R2'])

    return out


def generatePlots(filedata, interactive=False):
    """
    Generate a simple plot showing some results.
    """
    x = []
    y = []
    e = []
    R2 = []
    e1 = []
    e2 = []
    for key, value in filedata.iteritems():
        x.append(value['info']['xpos'])
        y.append(value['info']['ypos'])
        e.append(value['values']['ellipticity'])
        e1.append(value['values']['e1'])
        e2.append(value['values']['e2'])
        R2.append(value['values']['R2'])

    x = np.asarray(x)
    y = np.asarray(y)
    e = np.asarray(e)
    R2 = np.asarray(R2) / 1.44264123086 #denominator is R_ref

    #coordinate vectors
    xi = np.linspace(np.min(x), np.max(x))
    yi = np.linspace(np.min(y), np.max(y))

    #data grids
    Z = griddata(x, y, e, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)

    #ellipticity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 225)

    plt.title('PSF ellipticity over full VIS FoV')
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=3, cstride=3, cmap=cm.jet, vmin=0.02, vmax=0.07, shade=True)

    ax.set_zlim(0.02, 0.07)
    ax.set_xlabel('FoV X [deg]', linespacing=3.2)
    ax.set_ylabel('FoV Y [deg]', linespacing=3.2)
    ax.w_zaxis.set_label_text(r'Ellipticity $e$', fontdict={'rotation' : 50})

    if interactive:
        plt.show()
    else:
        plt.savefig('ellipticity.png')
        plt.close()

    #same with Mayvi
    #s = mlab.surf(X, Y, Z, colormap='Spectral')
    #mlab.savefig('FoVEllipticity.pdf')

    #R2
    Z = griddata(x, y, R2, xi, yi, interp='linear')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.title(r'PSF wings $\left ( \frac{R}{R_{ref}} \right )^{2}$ over full VIS FoV')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.5, cmap=cm.jet, vmin=3.4, vmax=3.7)

    ax.set_zlim(3.4, 3.7)
    ax.set_xlabel('FoV X [deg]', linespacing=3.2)
    ax.set_ylabel('FoV Y [deg]', linespacing=3.2)
    ax.w_zaxis.set_label_text(r'$\left ( \frac{R}{R_{ref}} \right )^{2}$', linespacing=3.2, rotation='vertical')

    ax.azim = 225

    if interactive:
        plt.show()
    else:
        plt.savefig('R2.png')
        plt.close()

    #vector plot of e components
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('VIS System PSF $e_{1,2}$')

    #Q = ax.quiver(x, y, -np.asarray(e1), e2, color='k', headwidth=1.5, headlength=3.5)
    Q = ax.quiver(x, y, e1, e2, color='k', headwidth=1.5, headlength=3.5)
    ax.quiverkey(Q, 0.9, 0.95, 0.1, r'$e_{i}$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold'})
    ax.set_xlabel('FoV X [deg]')
    ax.set_ylabel('FoV Y [deg]')

    ax.set_xlim(ax.get_xlim()[0]*0.9, ax.get_xlim()[1]*1.1)
    ax.set_ylim(ax.get_ylim()[0]*0.9, ax.get_ylim()[1]*1.1)

    if interactive:
        plt.show()
    else:
        plt.savefig('ecomponents.png')
        plt.close()


if __name__ == '__main__':
    run = False
    outfile = 'PSFdata.pk'

    #start the script
    log = lg.setUpLogger('PSFproperties.log')


    #derive results for each file
    if run:
        log.info('Deriving PSF properties...')

        #find files
        fls = glob.glob('/Volumes/disk_xray10/smn2/euclid/PSFs/detector_jitter-1_TOL05_MC_T0133_Nim=*.fits')

        txt = 'Processing %i files...' % (len(fls))
        print txt
        log.info(txt)

        filedata = {}
        for file in fls:
            data = readData(file)
            info = parseName(file)
            values = measureChars(data, info, log)
            filedata[file] = dict(info=info, values=values)
            txt = 'File %s processed...' % file
            print txt
            log.info(txt)

        #save data
        fileIO.cPickleDumpDictionary(filedata, outfile)
    else:
        filedata = cPickle.load(open(outfile))

    #generate plots
    generatePlots(filedata)

    log.info('Run finished...\n\n\n')
