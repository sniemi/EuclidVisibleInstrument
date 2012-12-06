"""

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
import matplotlib.animation as animation
import pyfits as pf
import numpy as np


def visualiseWavelengthDependency2D(d1, d2, d3, outname, logscale=True):
    """
    """
    plt.subplots(ncols=3, figsize=(18, 7))
    ax1 = plt.subplot(1, 3, 1, frame_on=False)
    ax2 = plt.subplot(1, 3, 2, frame_on=False)
    ax3 = plt.subplot(1, 3, 3, frame_on=False)

    ax1.imshow(d1, origin='lower')
    ax2.imshow(d2, origin='lower')
    ax3.imshow(d3, origin='lower')

    if logscale:
        ax1.set_title(r'$\lambda = 400$nm, logscale')
        ax2.set_title(r'$\lambda = 550$nm, logscale')
        ax3.set_title(r'$\lambda = 800$nm, logscale')
    else:
        ax1.set_title(r'$\lambda = 400$nm, linscale')
        ax2.set_title(r'$\lambda = 550$nm, linscale')
        ax3.set_title(r'$\lambda = 800$nm, linscale')

    #turn of ticks
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

    plt.savefig(outname)
    plt.close()


def visualiseWavelengthDependency3D(d1, d2, d3, outname, PSF=True):
    """
    """
    stopy, stopx = d1.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

    #3D comparison
    fig = plt.figure(1, figsize=(25, 12))
    rect1 = fig.add_subplot(1, 3, 1, frame_on=False, visible=False).get_position()
    ax1 = Axes3D(fig, rect1)
    rect2 = fig.add_subplot(1, 3, 2, frame_on=False, visible=False).get_position()
    ax2 = Axes3D(fig, rect2)
    rect3 = fig.add_subplot(1, 3, 3, frame_on=False, visible=False).get_position()
    ax3 = Axes3D(fig, rect3)

    ax1.plot_surface(X, Y, d1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.plot_surface(X, Y, d2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.plot_surface(X, Y, d3, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax1.set_title(r'$\lambda = 400$nm')
    ax2.set_title(r'$\lambda = 550$nm')
    ax3.set_title(r'$\lambda = 800$nm')

    if PSF:
        ax1.set_zlim(0., 0.95)
        ax2.set_zlim(0., 0.95)
        ax3.set_zlim(0., 0.95)
    else:
        ax1.set_zlim(-0.1, 0.1)
        ax2.set_zlim(-0.1, 0.1)
        ax3.set_zlim(-0.1, 0.1)

    #turn of ticks
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

    plt.savefig(outname)
    plt.close()


def visualiseBasisWavelengthDependency():
    d1400 = pf.getdata('PSF400nm/PCAbasis001.fits')
    d1550 = pf.getdata('PSF550nm/PCAbasis001.fits')
    d1800 = pf.getdata('PSF800nm/PCAbasis001.fits')

    d2400 = pf.getdata('PSF400nm/PCAbasis002.fits')
    d2550 = pf.getdata('PSF550nm/PCAbasis002.fits')
    d2800 = pf.getdata('PSF800nm/PCAbasis002.fits')

    d3400 = pf.getdata('PSF400nm/PCAbasis003.fits')
    d3550 = pf.getdata('PSF550nm/PCAbasis003.fits')
    d3800 = pf.getdata('PSF800nm/PCAbasis003.fits')

    d4400 = pf.getdata('PSF400nm/PCAbasis004.fits')
    d4550 = pf.getdata('PSF550nm/PCAbasis004.fits')
    d4800 = pf.getdata('PSF800nm/PCAbasis004.fits')

    stopy, stopx = d1400.shape
    X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

    #3D comparison
    fig = plt.figure(1, figsize=(18, 18))
    rect = fig.add_subplot(4, 3, 1, frame_on=False, visible=False).get_position()
    ax1 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 2, frame_on=False, visible=False).get_position()
    ax2 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 3, frame_on=False, visible=False).get_position()
    ax3 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 4, frame_on=False, visible=False).get_position()
    ax4 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 5, frame_on=False, visible=False).get_position()
    ax5 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 6, frame_on=False, visible=False).get_position()
    ax6 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 7, frame_on=False, visible=False).get_position()
    ax7 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 8, frame_on=False, visible=False).get_position()
    ax8 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 9, frame_on=False, visible=False).get_position()
    ax9 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 10, frame_on=False, visible=False).get_position()
    ax10 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 11, frame_on=False, visible=False).get_position()
    ax11 = Axes3D(fig, rect)
    rect = fig.add_subplot(4, 3, 12, frame_on=False, visible=False).get_position()
    ax12 = Axes3D(fig, rect)

    ax1.plot_surface(X, Y, d1400, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.plot_surface(X, Y, d1550, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.plot_surface(X, Y, d1800, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax4.plot_surface(X, Y, d2400, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax5.plot_surface(X, Y, d2550, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax6.plot_surface(X, Y, d2800, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax7.plot_surface(X, Y, d3400, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax8.plot_surface(X, Y, d3550, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax9.plot_surface(X, Y, d3800, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax10.plot_surface(X, Y, d4400, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax11.plot_surface(X, Y, d4550, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax12.plot_surface(X, Y, d4800, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax1.set_title(r'$\lambda = 400$nm')
    ax2.set_title(r'$\lambda = 550$nm')
    ax3.set_title(r'$\lambda = 800$nm')

    ax1.set_zlim(-0.1, 0.1)
    ax2.set_zlim(-0.1, 0.1)
    ax3.set_zlim(-0.1, 0.1)
    ax4.set_zlim(-0.1, 0.1)
    ax5.set_zlim(-0.1, 0.1)
    ax6.set_zlim(-0.1, 0.1)
    ax7.set_zlim(-0.1, 0.1)
    ax8.set_zlim(-0.1, 0.1)
    ax9.set_zlim(-0.1, 0.1)
    ax10.set_zlim(-0.1, 0.1)
    ax11.set_zlim(-0.1, 0.1)
    ax12.set_zlim(-0.1, 0.1)

    #turn of ticks
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

    plt.savefig('BasisSetComparison3D.pdf')
    plt.close()



if __name__ == '__main__':
    d1 = pf.getdata('PSF400nm/mean.fits')
    d2 = pf.getdata('PSF550nm/mean.fits')
    d3 = pf.getdata('PSF800nm/mean.fits')
    visualiseWavelengthDependency2D(np.log10(d1), np.log10(d2), np.log10(d3), 'MeanPSFComparison2D.pdf')
    visualiseWavelengthDependency3D(d1, d2, d3, 'MeanPSFComparison3D.pdf')

    d1 = pf.getdata('PSF400nm/PCAbasis003.fits')
    d2 = pf.getdata('PSF550nm/PCAbasis003.fits')
    d3 = pf.getdata('PSF800nm/PCAbasis003.fits')
    visualiseWavelengthDependency2D(d1, d2, d3, 'PCA1PSFComparison2D.pdf', logscale=False)
    visualiseWavelengthDependency3D(d1, d2, d3, 'PCA1PSFComparison3D.pdf', PSF=False)

    visualiseBasisWavelengthDependency()
