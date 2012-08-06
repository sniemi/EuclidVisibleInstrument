"""
Generates an idealised flat field surface representing the calibration unit flux output

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from support import files


def generateIdealisedFlatFieldSurface(numdata=2066, floor=1e5, xsize=2048, ysize=2066):
    """

    """
    # generate random data
    x = (np.random.random(numdata) - 0.5)
    y = (np.random.random(numdata) - 0.5)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), xsize), np.linspace(y.min(), y.max(), ysize))
    surface = (-1.4*xx*xx - 1.6*yy*yy - 1.5*xx*yy)*floor*0.09 + floor        #about 9-10 per cent range

    #cutout extra
    surface = surface[:ysize, :xsize]
    x, y = np.meshgrid(np.arange(0, xsize, 1), np.arange(0, ysize, 1))

    print np.max(surface), np.min(surface), np.mean(surface)

    #plot 3D
    fig = plt.figure()
    plt.title('VIS Flat Fielding: Idealised Calibration Unit Flux')
    ax = Axes3D(fig)
    ax.plot_surface(x, y, surface, rstride=100, cstride=100, alpha=0.5, cmap=cm.jet)
    ax.set_zlabel('electrons')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig('flatfield.pdf')
    plt.close()

    #save to file
    files.writeFITS(surface, 'VIScalibrationUnitflux.fits')


if __name__ == '__main__':
    generateIdealisedFlatFieldSurface()