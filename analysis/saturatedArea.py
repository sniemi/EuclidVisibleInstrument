"""
Saturated Imaging Area
======================

This scripts can be used to study the imaging area that is impacted by bright stars.


:requires: NumPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.2

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
import numpy as np
import math, bisect
from support.VISinstrumentModel import VISinformation
from sources import stellarNumberCounts


def pixelsImpacted(magnitude, exptime=565, pixelFractions=(0.65, 0.4, 0.35, 0.18, 0.09, 0.05),
                   star=False, lookup=True):
    """

    This potentially overestimates because does not consider the fact that bleeding is along the column
    and hence some saturated pixels may be double counted.
    """
    if lookup:
        #use a lookup table
        data = [(0, 311609),
                (1, 251766),
                (2, 181504),
                (3, 119165),
                (4, 75173),
                (5, 46298),
                (6, 28439),
                (7, 18181),
                (8, 12491),
                (9, 7552),
                (10, 4246),
                (11, 1652),
                (12, 636),
                (13, 247),
                (14, 93),
                (15, 29),
                (16, 8),
                (17, 2),
                (18, 1),
                (19, 0),
                (20, 0)]
        data.sort()

        pos = bisect.bisect_left(data, (magnitude - 0.99,))
        return data[pos][1]
    else:
        #try to calculate
        info = VISinformation()
        zp = info['zeropoint']
        fw = info['fullwellcapacity']

        electrons = 10**(-.4*(magnitude - zp)) * exptime

        mask = 0
        for x in pixelFractions:
            mask += np.round(electrons * x / fw - 0.4)  #0.4 as we don't want to mask if say 175k pixels...

        if star:
            mask += (20*20)

        if mask > 2000**2:
            mask = 2000**2

        return mask


def areaImpacted(magnitudes=np.arange(0, 20., 1.), offset=0.5, star=False):
    """

    """
    s = 0.1

    Nvconst = stellarNumberCounts.integratedCountsVband()

    print '\n mag     b       l       stars      CCD        area'
    for b in [20, 25, 30, 50, 90]:
        for l in [0, 90, 180]:
            area = 0
            prev = 0
            for ml in magnitudes:
                m = s * math.ceil(float(ml + offset) / s)
                n = stellarNumberCounts.bahcallSoneira(m, l, b, Nvconst)
                n -= prev #subtract the number of stars in the previous bin

                ccd = n * 49.6 / 3600
                covering = pixelsImpacted(m, star=star)
                area += (ccd * covering) / (4096 * 4132.) * 100.

                prev = n  #store the number of stars in the current bin

                if area > 100:
                    area = 100.


            txt = '%.1f     %2d     %3d     %.2f        %.1f        %.3f' % (m, b, l, n, ccd, area)
            print txt

            if b == 90:
                #no need to do more than once... l is irrelevant
                break

    print '\n\n\nIntegrated Area Loss:'
    blow=20
    bhigh=90
    llow=0
    lhigh=180
    bnum=71
    lnum=181

    prev = 0
    for i, ml in enumerate(magnitudes):
        m = s * math.ceil(float(ml + offset) / s)
        l, b, counts = stellarNumberCounts.skyNumbers(m, blow, bhigh, llow, lhigh, bnum, lnum)

        counts -= prev
        prev = counts.copy()

        #average
        stars = np.mean(counts)

        ccd = stars * 49.6 / 3600
        covering = pixelsImpacted(m, star=star)
        area = (ccd * covering) / (4096 * 4132.) * 100.

        if area > 100:
            area = 100.

        print 'magnitude = %.1f, average = %.5f, max = %.5f' % (ml, stars, np.max(counts))
        print '%i stars per square degree will mean %i stars per CCD and thus an area loss of %.4f per cent' % \
              (stars, ccd, area)

        if i < 1:
            z = counts * covering / (4096 * 4132. * (4096 * 0.1 * 4132 * 0.1 / 60. / 60.)) * 100.
        else:
            z += counts * covering / (4096 * 4132. * (4096 * 0.1 * 4132 * 0.1 / 60. / 60.)) * 100.

    msk = z > 100
    z[msk] = 100.

    _areaLossPlot(m, b, l, z, blow, bhigh, llow, lhigh, bnum, lnum,
                  'Masking of Saturated Pixels From All Stars', 'AreaLoss')


def _areaLossPlot(maglimit, b, l, z, blow, bhigh, llow, lhigh, bnum, lnum, title, output):
    """
    Generate a plot showing the area loss as a function of galactic coordinates for given magnitude limit.

    :param maglimit:
    :param b:
    :param l:
    :param z:
    :param blow:
    :param bhigh:
    :param llow:
    :param lhigh:
    :param bnum:
    :param lnum:

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

    fig = plt.figure(figsize=(12, 7))
    frame1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    #generate image
    f = maputils.FITSimage(externalheader=header, externaldata=z)
    im1 = f.Annotatedimage(frame1)

    grat1 = im1.Graticule(skyout='Galactic', starty=blow, deltay=10, startx=llow, deltax=20)

    colorbar = im1.Colorbar(orientation='horizontal')
    colorbar.set_label(label=r'Imaging Area Lost Because of Saturated Pixels [\%]', fontsize=18)

    im1.Image()
    im1.plot()

    title += r' $V \leq %.1f$' % maglimit
    frame1.set_title(title, y=1.02)

    plt.savefig(output + '%i.pdf' % maglimit)
    plt.close()


def _test():
    pixels = np.vectorize(pixelsImpacted)
    mag = np.arange(0, 19., 1) + 0.4
    rs = pixels(mag)
    for m, val in zip(mag, rs):
        print m, val

    print '\n\n\n0 mag'
    n = stellarNumberCounts.bahcallSoneira(1, 0, 20, stellarNumberCounts.integratedCountsVband())
    print 'objects    perCCD      areaLoss'
    print n, n * 49.6 / 3600 , n * pixelsImpacted(1) * 49.6 / 3600 / (4096 * 4132.) * 100.

    print '\n10 mag'
    n = stellarNumberCounts.bahcallSoneira(10, 0, 20, stellarNumberCounts.integratedCountsVband())
    print 'objects    perCCD      areaLoss'
    print n, n * 49.6 / 3600 , n * pixelsImpacted(10) * 49.6 / 3600 / (4096 * 4132.) * 100.

    print '\n18 mag'
    n = stellarNumberCounts.bahcallSoneira(18, 0, 20, stellarNumberCounts.integratedCountsVband())
    print 'objects    perCCD      areaLoss'
    print n, n * 49.6 / 3600 , n * pixelsImpacted(18) * 49.6 / 3600 / (4096 * 4132.) * 100.

    for m in mag:
        n = stellarNumberCounts.bahcallSoneira(m, 0, 20, stellarNumberCounts.integratedCountsVband())
        print m, n * 49.6 / 3600


if __name__ == '__main__':
    _test()

    #areaImpacted()
