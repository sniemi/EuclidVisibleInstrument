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
from matplotlib import animation
from matplotlib.patches import Ellipse
import pyfits as pf
import numpy as np
from support import logger as lg
from analysis import shape


def plotfile(filename='/Users/sammy/EUCLID/vissim-python/data/psf1x.fits',
             sigma=0.75, iterations=4,
             out='test.pdf', scale=False, log=False, zoom=30):
    """
    Calculate ellipticity from a given input file using quadrupole moments and
    plot the data.
    """
    settings = dict(sigma=sigma, iterations=iterations)

    l = lg.setUpLogger('CTItesting.log')

    data = pf.getdata(filename)
    if scale:
        data /= np.max(data)
        data *= 1.e5

    sh = shape.shapeMeasurement(data, l, **settings)
    results = sh.measureRefinedEllipticity()

    fig, axarr = plt.subplots(1, 2, sharey=True)
    ax1 = axarr[0]
    ax2 = axarr[1]
    fig.subplots_adjust(wspace=0)

    if log:
        ax1.set_title(r'$\log_{10}$(Image)')
        ax2.set_title(r'$\log_{10}$(Gaussian Weighted)')
    else:
        ax1.set_title(r'Image')
        ax2.set_title(r'Gaussian Weighted')


    #no ticks on the right hand side plot
    plt.setp(ax2.get_yticklabels(), visible=False)

    if log:
        im1 = ax1.imshow(np.log10(data), origin='lower')
        im2 = ax2.imshow(np.log10(results['GaussianWeighted']), origin='lower')
    else:
        im1 = ax1.imshow(data, origin='lower')
        im2 = ax2.imshow(results['GaussianWeighted'], origin='lower')

    ang = 0.5 * np.arctan(results['e2']/results['e1'])

    e = Ellipse(xy=(results['centreX']-1, results['centreY']-1),
                width=results['a'], height=results['b'], angle=ang,
                facecolor='none', ec='k', lw=2)
    fig.gca().add_artist(e)

    if zoom is not None:
        ax1.set_xlim(results['centreX']-zoom-1, results['centreX']+zoom)
        ax2.set_xlim(results['centreX']-zoom-1, results['centreX']+zoom)
        ax1.set_ylim(results['centreY']-zoom-1, results['centreY']+zoom)
        ax2.set_ylim(results['centreY']-zoom-1, results['centreY']+zoom)

    plt.savefig(out)
    plt.close()


def shapeMovie(filename='/Users/sammy/EUCLID/CTItesting/Reconciliation/damaged_image_parallel.fits',
               sigma=0.75, scale=False, zoom=30, frames=20):
    #settings = dict(sigma=sigma, iterations=1)
    settings = dict(sigma=sigma, iterations=1, fixedPosition=True, fixedX=85.0, fixedY=85.)

    l = lg.setUpLogger('CTItesting.log')

    data = pf.getdata(filename)
    if scale:
        data /= np.max(data)
        data *= 1.e5

    sh = shape.shapeMeasurement(data, l, **settings)
    results = sh.measureRefinedEllipticity()
    ang = 0.5 * np.arctan(results['e2']/results['e1'])

    fig, axarr = plt.subplots(1, 2, sharey=True)
    ax1 = axarr[0]
    ax2 = axarr[1]
    fig.subplots_adjust(wspace=0)

    ax1.set_title(r'Image w/ CTI')
    ax2.set_title(r'Gaussian Weighted')

    #no ticks on the right hand side plot
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax1.imshow(data, origin='lower')
    ax2.imshow(results['GaussianWeighted'], origin='lower')

    if zoom is not None:
        ax1.set_xlim(results['centreX']-zoom-1, results['centreX']+zoom)
        ax2.set_xlim(results['centreX']-zoom-1, results['centreX']+zoom)
        ax1.set_ylim(results['centreY']-zoom-1, results['centreY']+zoom)
        ax2.set_ylim(results['centreY']-zoom-1, results['centreY']+zoom)

    text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, color='white')

    e = Ellipse(xy=(results['centreX']-1, results['centreY']-1),
                width=results['a'], height=results['b'], angle=ang,
                facecolor='none', ec='white', lw=2)

    def init():
        # initialization function: plot the background of each frame
        ax2.imshow([[], []])
        fig.gca().add_artist(e)
        text.set_text(' ')
        return ax2, text, e

    def animate(i):
        #settings = dict(sigma=sigma, iterations=i+1)
        settings = dict(sigma=sigma, iterations=i+1, fixedPosition=True, fixedX=85.0, fixedY=85.)
        sh = shape.shapeMeasurement(data, l, **settings)
        results = sh.measureRefinedEllipticity()

        text.set_text(r'%i iterations, $e \sim %.4f$' % (i+1, results['ellipticity']))

        ax2.imshow(results['GaussianWeighted'], origin='lower')
        ang = 0.5 * np.arctan(results['e2']/results['e1'])

        e.center = (results['centreX']-1, results['centreY']-1)
        e.width = results['a']
        e.height = results['b']
        e.angle = ang

        return ax2, text, e

    #note that the frames defines the number of times animate functions is being called
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=2, blit=True)
    anim.save('shapeMovie.mp4', fps=0.7)


if __name__ == '__main__':

    #plotfile(filename='/Users/sammy/EUCLID/CTItesting/Reconciliation/damaged_image_parallel.fits',
    #         out='ctitest.pdf', iterations=3)

    shapeMovie(sigma=0.2)
    #shapeMovie(sigma=0.35, frames=50)