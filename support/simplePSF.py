"""
Simple script to draw a pupil image and then take a Fourier transform to generate a PSF.

Created on Mar 4, 2010

:author: Sami-Matias Niemi
"""
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def drawCircle(pupil, radius, x, y, value = 1):
    rad = np.sqrt(x**2 + y**2)
    mask = rad < radius
    pupil[mask] = value
    return pupil


def drawWFC3UVIS():
    size = 1000
    rad = 490
    SpiderRadius = size * 0.011
    
    pupil = np.zeros((size, size))
    cx, cy = np.shape(pupil)[1]/2, np.shape(pupil)[0]/2
    y, x = np.indices(pupil.shape)
    xc = x - cx
    yc = y - cy
    
    pupil = drawCircle(pupil, rad, xc, yc)
    #secondary obsc.
    pupil = drawCircle(pupil, rad*0.33, xc, yc, value = 0)
    #OTA mirror pads
    pupil = drawCircle(pupil, rad*0.065, xc, yc-0.8921*cy, value = 0)
    pupil = drawCircle(pupil, rad*0.065, xc+0.7555*cx, yc+0.4615*cy, value = 0)
    pupil = drawCircle(pupil, rad*0.065, xc-0.7606*cx, yc+0.4564*cy, value = 0)
    #Spiders
    #pupil = drawRectangle(pupil, xc, yc, SpiderRadius*rad, 2.1*rad)
    #pupil = drawRectangle(pupil, xc, yc, 2.1*rad, SpiderRadius*rad)
    return pupil


def derivePSF(pupil):
    """
    Derives a PSF from a pupil image.

    :param pupil: pupil image
    :type pupil: ndarray

    :return: PSF
    :rtype: ndarray
    """
    F1 = fftpack.fft2(pupil)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)
    # Calculate a 2D power spectrum
    psd2D = np.abs(F2)**2

    return psd2D


def plot(pupil):
    """

    :param pupil:
    :return:
    """
    PSF = derivePSF(pupil)
    PSF /= np.max(PSF)

    #make plot
    plt.subplots(figsize=(12, 6))
    ax1 = plt.subplot(1, 3, 1, frame_on=False)
    ax2 = plt.subplot(1, 3, 2, frame_on=False)
    ax3 = plt.subplot(1, 3, 3, frame_on=False)

    ax1.imshow(pupil, origin='lower', cmap=cm.gray)
    ax2.imshow(np.log10(PSF), origin='lower')
    ax3.imshow(np.log10(PSF), origin='lower')

    ax3.set_xlim(450, 550)
    ax3.set_ylim(450, 550)

    ax1.set_title('Pupil Image')
    ax2.set_title('PSF, log scale')
    ax3.set_title('PSF, log scale, zoomed=in')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.savefig('PSF.pdf')
    plt.close()

if __name__ == '__main__':
    pupil = drawWFC3UVIS()
    plot(pupil)
