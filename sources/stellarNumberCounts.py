"""
This file provides simple functions to calculate the integrated stellar number counts
as a function of limiting magnitude and galactic coordinates.

:requires: NumPy
:requires: matplotlib

:version: 0.1

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
import numpy as np
import matplotlib.pyplot as plt


def bahcallSoneira(magnitude, longitude, latitude, constants):
    """
    Implemented Equation B1 from Bahcall and Soneira 1980 (1980ApJS...44...73B).

    Note that the values return do not necessarily agree with the Table 6 of the paper values.
    Mostly the integrated number of stars agree within 15%, which is the error quoted
    in the paper.

    :param magnitude: limiting magnitude
    :type magnitude:float
    :param longitude: galactic longitude in degrees
    :type longitude: float
    :param latitude: galactic latitude in degrees
    :type latitude: float

    :return: Number of stars per square degree
    """
    #rename variables for easy reference and convert coordinates to radians
    m = magnitude
    l = np.deg2rad(longitude)
    b = np.deg2rad(latitude)
    C1 = constants['C1']
    C2 = constants['C2']
    beta = constants['beta']
    alpha = constants['alpha']
    delta = constants['delta']
    lam = constants['lam']
    eta = constants['eta']
    kappa = constants['kappa']

    #magnitude dependent values
    if m <= 12:
        mu = 0.03
        gamma = 0.36
    elif 12 < m < 20:
        mu = 0.0075*(m - 12) + 0.03
        gamma = 0.04*(12 - m) + 0.36
    else:
        mu = 0.09
        gamma = 0.04

    #position dependency
    sigma = 1.45 - 0.2*np.cos(b) * np.cos(l)

    #precompute the delta mags
    dm = m - constants['mstar']
    dm2 = m - constants['mdagger']

    #split the equation to two parts
    D1 = (C1*10**(beta*dm)) / ((1. + 10**(alpha*dm))**delta) / ((np.sin(b)*(1 - mu/np.tan(b)*np.cos(l)))**(3 - 5*gamma))
    D2 = (C2*10**(eta*dm2)) / ((1. + 10**(kappa*dm2))**lam) / ((1 - np.cos(b)*np.cos(l))**sigma)

    #final counts
    D = D1 + D2

    return D


def integratedCountsVband():
    """
    Returns constant values for the integrated number counts in the V-band.

    :return: constants to be used when calculating the integrated number counts.
    :rtype: dict
    """
    return dict(C1=925., alpha=-0.132, beta=0.035, delta=3., mstar=15.75,
                C2=1050., kappa=-0.18, eta=0.087, lam=2.5, mdagger=17.5)


def _skyProjectionPlot(maglimit, b, l, z, blow, bhigh, llow, lhigh, bnum, lnum):
    """
    Generate a sky projection plot.

    :param maglimit:
    :param b:
    :param l:
    :param z:
    :return:
    """
    from kapteyn import maputils

    header = {'NAXIS': 2,
              'NAXIS1': len(l),
              'NAXIS2': len(b),
              'CTYPE1': 'GLON',
              'CRVAL1': llow,
              'CRPIX1': 0,
              'CUNIT1': 'deg',
              'CDELT1': float(bhigh-blow)/bnum,
              'CTYPE2': 'GLAT',
              'CRVAL2': blow,
              'CRPIX2': 0,
              'CUNIT2': 'deg',
              'CDELT2': float(lhigh-llow)/lnum}

    fig = plt.figure(figsize=(12, 11))
    frame1 = fig.add_axes([0.1,0.5,0.85, 0.44])
    frame2 = fig.add_axes([0.1,0.07,0.85, 0.4])

    #generate image
    f = maputils.FITSimage(externalheader=header, externaldata=np.log10(z))
    im1 = f.Annotatedimage(frame1)

    h = header.copy()
    h['CTYPE1'] = 'RA---CAR'
    h['CTYPE2'] = 'DEC--CAR'
    h['CRVAL1'] = 0
    h['CRVAL2'] = 0

    # Get an estimate of the new corners
    x = [0]*5
    y = [0]*5
    x[0], y[0] = f.proj.toworld((1, 1))
    x[1], y[1] = f.proj.toworld((len(l), 1))
    x[2], y[2] = f.proj.toworld((len(l), len(b)))
    x[3], y[3] = f.proj.toworld((1, len(b)))
    x[4], y[4] = f.proj.toworld((len(l)/2., len(b)))

    # Create a dummy object to calculate pixel coordinates
    # in the new system. Then we can find the area in pixels
    # that corresponds to the area in the sky.
    f2 = maputils.FITSimage(externalheader=h)
    px, py = f2.proj.topixel((x,y))
    pxlim = [int(min(px))-10, int(max(px))+10]
    pylim = [int(min(py))-10, int(max(py))+10]

    reproj = f.reproject_to(h, pxlim_dst=pxlim, pylim_dst=pylim)

    grat1 = im1.Graticule(skyout='Galactic', starty=blow, deltay=10, startx=llow, deltax=20)

    colorbar = im1.Colorbar(orientation='horizontal')
    colorbar.set_label(label='log10(Stars per sq deg)', fontsize=18)

    im1.Image()
    im1.plot()

    im2 = reproj.Annotatedimage(frame2)
    grat2 = im2.Graticule()

    im2.Image()
    im2.plot()

    title = r'Integrated Number Density of Stars $V \leq %.1f$' % (maglimit)
    frame1.set_title(title, y=1.02)

    plt.savefig('stellarD%i.pdf' % maglimit)
    plt.close()


def skyNumbers(maglimit=20, blow=20., bhigh=90., llow=0., lhigh=360., bnum=71, lnum=361, plot=True):
    """
    Calculate the integrated stellar number counts in a grid of galactic coordinates.
    Plot the results in two projections.

    :param maglimit: magnitude limit
    :type maglimit: int or float
    :param blow: lower limit for the galactic latitude
    :type blow: float
    :param bhigh: upper limit for the galactic latitude
    :type bhigh: float
    :param llow: lower limit for the galactic longitude
    :type llow: float
    :param lhigh: upper limit of the galacti longitude:
    :type lhigh: float
    :param bnum: number of galactic latitude grid points
    :type bnum: int
    :param lnum: number of galactic longitude grid points
    :type lnum: int
    :param plot: whether or not to generate sky coverage plots
    :type plot: bool

    :return: grid of galactic coordinates and the number of stars in the grid
    """
    Nvconst = integratedCountsVband()

    b = np.linspace(blow, bhigh, num=bnum)
    l = np.linspace(llow, lhigh, num=lnum)

    counts = np.vectorize(bahcallSoneira)

    ll, bb = np.meshgrid(l, b)

    z = counts(maglimit, ll, bb, Nvconst)

    #plot
    if plot:
        _skyProjectionPlot(maglimit, b, l, z, blow, bhigh, llow, lhigh, bnum, lnum)

    return l, b, z


if __name__ == '__main__':
    #constants for V-band
    Nvconst = integratedCountsVband()

    skyNumbers(maglimit=10)
    skyNumbers(maglimit=15)
    skyNumbers(maglimit=18)
    skyNumbers(maglimit=20)
    skyNumbers(maglimit=22)
    skyNumbers(maglimit=24)
    skyNumbers(maglimit=26)
    skyNumbers(maglimit=29)

    #testing
    #print bahcallSoneira(22, 90, 20, Nvconst)
    #print bahcallSoneira(22, 90, 30, Nvconst)
    #print bahcallSoneira(22, 90, 50, Nvconst)
    #print bahcallSoneira(22, 90, 90, Nvconst)