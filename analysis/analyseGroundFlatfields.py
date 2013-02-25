"""
A simple script to analyse ground/lab flat fields.
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
import pyfits as pf
import numpy as np
import glob as g
import cPickle
from support import files as fileIO
import scipy.optimize as optimize


def makeFlat(files):
    shape = pf.getdata(files[0]).shape
    summed = np.zeros(shape)

    for file in files:
        data = pf.getdata(file)

        prescan = data[11:2056, 9:51].mean()
        overscan = data[11:2056, 4150:4192].mean()

        Q0 = data[:, 51:2098]
        Q1 = data[:, 2098:4145]

        #subtract the bias levels
        Q0 -= prescan
        Q1 -= overscan

        data[:, 51:2098] = Q0
        data[:, 2098:4145] = Q1

        fileIO.writeFITS(data, file.replace('.fits', 'biasremoved.fits'), int=False)

        summed += data

    summed /= summed[11:2056, 56:4131].mean()

    #write out FITS file
    fileIO.writeFITS(summed, 'combined.fits', int=False)

    avg = np.average(np.asarray([pf.getdata(file) for file in files]), axis=0)
    avg /= avg[11:2056, 56:4131].mean()
    fileIO.writeFITS(avg, 'averaged.fits', int=False)

    avg = np.median(np.asarray([pf.getdata(file) for file in files]), axis=0)
    avg /= avg[11:2056, 56:4131].mean()
    fileIO.writeFITS(avg, 'median.fits', int=False)


def measureNoise(data, size, file, gain=3.5, flat='combined.fits', debug=False):
    """
    Measure average signal level and variance in several patches within a single image.

    .. Warning:: One must flat field the data before calculating the variance. Hence, the results are uncertain.
                 It is better to use a pairwise analysis if at least two exposures at a given flux level is
                 available.

    :param data:
    :param size:
    :return:
    """
    #move to electrons
    data *= gain

    #means of prescan and overscan
    prescan = data[11:2056, 9:51].mean()
    overscan = data[11:2056, 4150:4192].mean()

    #take out pre and overscan
    #x should start from 55 and go to 2090 for first Q
    #from 2110 to 4130 for the second Q
    # y should range from 10 to 2055 to have clean area...
    Q0 = data[:, 51:2098].copy()
    Q1 = data[:, 2098:4145].copy()

    if debug:
        print prescan, overscan

    #subtract the bias levels
    Q0 -= prescan
    Q1 -= overscan

    #load a flat and remove-pixel-to-pixel variation due to the flat...
    flat = pf.getdata(flat)
    Q0 /= flat[:, 51:2098]
    Q1 /= flat[:, 2098:4145]
    data[:, 51:2098] = Q0
    data[:, 2098:4145] = Q1
    fileIO.writeFITS(data, file.replace('.fits', 'flattened.fits'), int=False)

    Q0 = data[11:2056, 56:2091].copy()
    Q1 = data[11:2056, 2111:4131].copy()

    #number of pixels in new areas
    Q0y, Q0x = Q0.shape
    Q1y, Q1x = Q1.shape

    #number of patches
    Q0y = int(np.floor(Q0y / size))
    Q0x = int(np.floor(Q0x / size))
    Q1y = int(np.floor(Q1y / size))
    Q1x = int(np.floor(Q1x / size))

    flux = []
    variance = []
    for i in range(Q0y):
        for j in range(Q0x):
            minidy = i*int(size)
            maxidy = minidy + int(size)
            minidx = j*int(size)
            maxidx = minidx + int(size)
            patch = Q0[minidy:maxidy, minidx:maxidx]
            avg = np.mean(patch)
            var = np.var(patch)
            #filter out stuff too close to saturation
            if avg < 300000 and var/avg < 2.5 and avg/var < 2.5:
                flux.append(avg)
                variance.append(var)

    for i in range(Q1y):
        for j in range(Q1x):
            minidy = i * int(size)
            maxidy = minidy + int(size)
            minidx = j * int(size)
            maxidx = minidx + int(size)
            patch = Q1[minidy:maxidy, minidx:maxidx]
            avg = np.mean(patch)
            var = np.var(patch)
            #filter out stuff too close to saturation
            if avg < 300000 and var/avg < 2.5 and avg/var < 2.5:
                flux.append(avg)
                variance.append(var)

    flux = np.asarray(flux)
    variance = np.asarray(variance)

    print file, np.mean(flux), np.mean(variance)

    results = dict(flux=flux, variance=variance)

    return results


def measureNoiseRandomPositions(data, size, file, flat='combined.fits', rands=50, debug=False):
    """
    Measure average signal level and variance in several patches within a single image

    :param data:
    :param size:
    :return:
    """
    #means of prescan and overscan
    prescan = data[11:2056, 9:51].mean()
    overscan = data[11:2056, 4150:4192].mean()

    #take out pre and overscan
    #x should start from 55 and go to 2090 for first Q
    #from 2110 to 4130 for the second Q
    # y should range from 10 to 2055 to have clean area...
    Q0 = data[:, 51:2098].copy()
    Q1 = data[:, 2098:4145].copy()

    if debug:
        print prescan, overscan

    #subtract the bias levels
    Q0 -= prescan
    Q1 -= overscan

    #load a flat and remove-pixel-to-pixel variation due to the flat...
    flat = pf.getdata(flat)
    Q0 /= flat[:, 51:2098]
    Q1 /= flat[:, 2098:4145]
    data[:, 51:2098] = Q0
    data[:, 2098:4145] = Q1
    fileIO.writeFITS(data, file.replace('.fits', 'flattened.fits'), int=False)

    Q0 = data[11:2056, 56:2091].copy()
    Q1 = data[11:2056, 2111:4131].copy()

    #number of pixels in new areas
    Q0y, Q0x = Q0.shape
    Q1y, Q1x = Q1.shape

    h = size / 2.
    Q0y -= h
    Q0x -= h
    Q1y -= h
    Q1x -= h

    xpos = np.random.random_integers(h, min(Q0x, Q1x), size=rands)
    ypos = np.random.random_integers(h, min(Q0y, Q1y), size=rands)

    flux = []
    variance = []
    for i in xpos:
        for j in ypos:
            patch = Q0[j-h:j+h, i-h:i+h]
            avg = np.mean(patch)
            var = np.var(patch)
            #filter out stuff too close to saturation
            if avg < 62000:# and var/avg < 2.0 and avg/var < 2.0:
                flux.append(avg)
                variance.append(var)
                #print avg, var

    for i in xpos:
        for j in ypos:
            patch = Q1[j-h:j+h, i-h:i+h]
            avg = np.mean(patch)
            var = np.var(patch)
            #filter out stuff too close to saturation
            if avg < 62000:# and var/avg < 2.0 and avg/var < 2.0:
                flux.append(avg)
                variance.append(var)

    flux = np.asarray(flux)
    variance = np.asarray(variance)

    results = dict(flux=flux, variance=variance)

    return results


def plotAutocorrelation(data, output='Autocorrelation.pdf'):
    """

    :param data:
    :return:
    """
    import acor

    fig = plt.figure()
    plt.subplots_adjust(left=0.15)
    plt.title(r'CCD273-84-2-F15, Serial number: 11312-14-01')
    ax = fig.add_subplot(111)

    for file, values in data.iteritems():
        flux = np.mean(values['flux'])
        variance = values['variance']

        tau, mean, sigma = acor.acor(variance)

        ax.plot(flux, mean, 'bo')


    ax.set_xlabel(r'$ \left < \mathrm{Signal} \right > \quad [e^{-}]$')
    ax.set_ylabel('mean')

    #plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    plt.savefig(output)
    plt.close()


def plotResults(data, size, pairwise=True, output='FlatfieldFullwellEstimate.pdf'):
    """

    :param data:
    :return:
    """
    size = int(size)

    fig = plt.figure()
    plt.subplots_adjust(left=0.15)
    plt.title(r'CCD273-84-2-F15, Serial number: 11312-14-01')
    ax = fig.add_subplot(111)

    mf = np.asarray([])
    mv = np.asarray([])
    for file, values in data.iteritems():
        flux = values['flux']
        variance = values['variance']
        mflux = np.mean(flux)
        mvar = np.mean(variance)

        mf = np.hstack((mf, flux))
        mv = np.hstack((mv, variance))

        ax.plot(flux, variance, 'r.', alpha=0.05)
        ax.plot(np.median(flux), np.median(variance), 'bo')
        ax.errorbar(mflux, mvar, yerr=np.std(variance), marker='s', ecolor='green',
                    mfc='green', mec='green', ms=5, mew=1)

    ax.plot([-1,], [-1,], 'r.', label='data')
    ax.plot([-1,], [-1,], 'bo', label='median')
    ax.errorbar([-1,], [-1,], marker='s', ecolor='green', mfc='green', mec='green', ms=5, mew=1, label='mean')

    msk = mf < 220000
    z = np.polyfit(mf[msk], mv[msk], 2)
    ev = np.poly1d(z)
    x = np.linspace(0, 250000, 100)

    #second order but no intercept
    fitfunc = lambda p, x: p[0]*x*(1 - p[1]*x)
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p1, success = optimize.leastsq(errfunc, [1.0, 1e-6], args=(mf[msk], mv[msk]))
    y2 = fitfunc(p1, x)

    ax.plot(x, ev(x), 'k-', lw=2, label='2nd order fit')
    ax.plot(x, y2, 'y:', lw=2, label='2nd order fit, no intercept')
    txt = r'$y = %.3e \times x^{2} + %.4f x + %.3f$' % (z[0], z[1], z[2])
    txt2 = r'$y = %.4f x (1 - %.3e \times x)$' % (p1[0], p1[1])
    ax.text(0.3, 0.1, txt, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            alpha=0.5, size='small')
    ax.text(0.3, 0.15, txt2, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
        alpha=0.5, size='small')

    ax.plot([0, 250000], [0, 250000], 'm--', lw=1.5, label='shot noise')

    ax.set_xlim(0, 240000)
    ax.set_ylim(0, 170000)

    ax.set_xlabel(r'$ \left < \mathrm{Signal}_{%i \times %i} \right > \quad [e^{-}]$' % (size, size))
    if pairwise:
        ax.set_ylabel(r'$\frac{1}{2}\sigma^{2}(\Delta \mathrm{Signal}_{%i \times %i}) \quad [(e^{-})^{2}]$' % (size, size))
    else:
        ax.set_ylabel(r'$\sigma^{2}(\mathrm{Signal}_{%i \times %i}) \quad [(e^{-})^{2}]$' % (size, size))

    plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    plt.savefig(output)
    plt.close()


def plotResultsRowColumn(data, pairwise=True, output='FlatfieldFullwellEstimateRowColumn.pdf'):
    """

    :param data:
    :return:
    """
    #diff plot
    fig = plt.figure()
    plt.subplots_adjust(left=0.15)
    plt.title(r'CCD273-84-2-F15, Serial number: 11312-14-01')
    ax = fig.add_subplot(111)

    for file, values in data.iteritems():
        rowvariance = values['rowvariance']
        rowflux = values['rowflux']
        rowmflux = np.mean(rowflux)
        rowmvar = np.mean(rowvariance)

        columnflux = values['columnflux']
        columnmflux = np.mean(columnflux)
        columnvariance = values['columnvariance']
        columnmvar = np.mean(columnvariance)

        flux = (rowmflux+columnmflux)/2.

        ax.plot(flux, rowmvar / columnmvar, 'bo')
        #ax.plot(flux, np.median(rowvariance) / np.median(columnvariance), 'rs')

    ax.plot([-1, ], [-1, ], 'bo', label='mean')
    #ax.plot([-1, ], [-1, ], 'rs', label='median')

    ax.plot([0, 250000], [1, 1], 'k--', lw=1.5)

    ax.set_xlim(0, 240000)
    #ax.set_ylim(0.9, 1.1)
    ax.set_ylim(0.99, 1.01)

    ax.set_ylabel(r'$\frac{\sigma^{2}_{row}}{\sigma^{2}_{column}}$')
    ax.set_xlabel(r'$ \left < \mathrm{Signal} \right > \quad [e^{-}]$')

    plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    plt.savefig('RowColumnDifference.pdf')
    plt.close()

    #fits
    fig = plt.figure()
    plt.subplots_adjust(left=0.15)
    plt.title(r'CCD273-84-2-F15, Serial number: 11312-14-01')
    ax = fig.add_subplot(111)

    rowmf = np.asarray([])
    rowmv = np.asarray([])
    columnmf = np.asarray([])
    columnmv = np.asarray([])
    for file, values in data.iteritems():
        rowflux = values['rowflux']
        rowvariance = values['rowvariance']
        rowmflux = np.mean(rowflux)
        rowmvar = np.mean(rowvariance)
        rowmf = np.hstack((rowmf, rowflux))
        rowmv = np.hstack((rowmv, rowvariance))

        columnflux = values['columnflux']
        columnvariance = values['columnvariance']
        columnmflux = np.mean(columnflux)
        columnmvar = np.mean(columnvariance)
        columnmf = np.hstack((columnmf, columnflux))
        columnmv = np.hstack((columnmv, columnvariance))        

        ax.plot(np.median(rowflux), np.median(rowvariance), 'ro')
        ax.errorbar(rowmflux, rowmvar, yerr=np.std(rowvariance), marker='s', ecolor='red',
                    mfc='red', mec='red', ms=5, mew=1)

        ax.plot(np.median(columnflux), np.median(columnvariance), 'bo')
        ax.errorbar(columnmflux, columnmvar, yerr=np.std(columnvariance), marker='s', ecolor='blue',
                    mfc='blue', mec='blue', ms=5, mew=1)

    ax.plot([-1,], [-1,], 'ro', label='row median')
    ax.errorbar([-1,], [-1,], marker='s', ecolor='red', mfc='red', mec='red', ms=5, mew=1, label='row mean')

    ax.plot([-1,], [-1,], 'bo', label='column median')
    ax.errorbar([-1,], [-1,], marker='s', ecolor='blue', mfc='blue', mec='blue', ms=5, mew=1, label='column mean')

    msk = rowmf < 170000
    x = np.linspace(0, 250000, 100)

    #second order but no intercept
    fitfunc = lambda p, x: p[0]*x*(1 - p[1]*x)
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    p1, success = optimize.leastsq(errfunc, [1.0, 1e-6], args=(rowmf[msk], rowmv[msk]))
    y2 = fitfunc(p1, x)

    ax.plot(x, y2, 'r-', lw=2, label='2nd order fit (row)')
    txt2 = r'$y_{row} = %.4f x (1 - %.3e \times x)$' % (p1[0], p1[1])
    ax.text(0.3, 0.15, txt2, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
        alpha=0.5, size='small')
    
    msk = columnmf < 170000

    #second order but no intercept
    p2, success = optimize.leastsq(errfunc, [1.0, 1e-6], args=(columnmf[msk], columnmv[msk]))
    y3 = fitfunc(p2, x)

    ax.plot(x, y3, 'b--', lw=2, label='2nd order fit (column)')
    txt2 = r'$y_{column} = %.4f x (1 - %.3e \times x)$' % (p2[0], p2[1])
    ax.text(0.3, 0.1, txt2, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
        alpha=0.5, size='small')

    ax.plot([0, 250000], [0, 250000], 'k--', lw=1.5, label='shot noise')

    ax.set_xlim(0, 240000)
    ax.set_ylim(0, 170000)

    ax.set_xlabel(r'$ \left < \mathrm{Signal} \right > \quad [e^{-}]$')
    if pairwise:
        ax.set_ylabel(r'$\frac{1}{2}\sigma^{2}(\Delta \mathrm{Signal}) \quad [(e^{-})^{2}]$')
    else:
        ax.set_ylabel(r'$\sigma^{2}(\Delta \mathrm{Signal}) \quad [(e^{-})^{2}]$')

    plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    plt.savefig(output)
    plt.close()


def findPairs():
    """

    :return:
    """
    files = g.glob('05*Euclid.fits')

    for file in files:
        data = pf.getdata(file)
        c1 = np.average(data[1800:1810, 200:210])
        print file, c1


def pairwiseNoise(pairs, gain=3.5, size=100.0, simple=False):
    """
    Calculates the mean flux within a region of size * size and the variance from the difference image.
    The variance of the difference image is divided by 2, given that var(x-y) = var(x) + var(y).

    The calculations are performed in electrons so that variance = noise**2 should be equal to the mean counts
    if no correlated noise and other effects are present. This would be the case of pure shot noise.

    :return:
    """
    results = {}
    for f1, f2 in pairs:
        #move from ADUs to electrons
        d1 = pf.getdata(f1) * gain
        d2 = pf.getdata(f2) * gain

        if simple:
            #pre/overscans
            prescan1 = d1[11:2056, 9:51].mean()
            overscan1 = d1[11:2056, 4150:4192].mean()
            prescan2 = d2[11:2056, 9:51].mean()
            overscan2 = d2[11:2056, 4150:4192].mean()

            #define quadrants and subtract the bias levels
            Q10 = d1[11:2051, 58:2095].copy() - prescan1
            Q20 = d2[11:2051, 58:2095].copy() - prescan2
            Q11 = d1[11:2051, 2110:4132].copy() - overscan1
            Q21 = d2[11:2051, 2110:4132].copy() - overscan2
        else:
            y1, x1 = d1.shape
            #subtract over/prescan row-by-row to minimise the any bias variation in column direction
            for row in range(y1):
                prescan1 = np.median(d1[row, 9:48])
                prescan2 = np.median(d2[row, 9:48])
                d1[row, :2099] -= prescan1
                d2[row, :2099] -= prescan2

            for row in range(y1):
                overscan1 = np.median(d1[row, 4152:4190])
                overscan2 = np.median(d2[row, 4152:4190])
                d1[row, 2100:] -= overscan1
                d2[row, 2100:] -= overscan2

            #define quadrants; usable image area Q0
            Q10 = d1[11:2051, 58:2095].copy()
            Q20 = d2[11:2051, 58:2095].copy()
            #Q1
            Q11 = d1[11:2051, 2110:4132].copy()
            Q21 = d2[11:2051, 2110:4132].copy()

        #number of pixels in new areas
        Q10y, Q10x = Q10.shape
        Q11y, Q11x = Q11.shape

        #number of patches
        Q0y = int(np.floor(Q10y / size))
        Q0x = int(np.floor(Q10x / size))
        Q1y = int(np.floor(Q11y / size))
        Q1x = int(np.floor(Q11x / size))

        flux = []
        variance = []
        for i in range(Q0y):
            for j in range(Q0x):
                minidy = i * int(size)
                maxidy = minidy + int(size)
                minidx = j * int(size)
                maxidx = minidx + int(size)
                patch1 = np.ravel(Q10[minidy:maxidy, minidx:maxidx])
                patch2 = np.ravel(Q20[minidy:maxidy, minidx:maxidx])
                avg1 = np.mean(patch1)
                avg2 = np.mean(patch2)

                diff = patch1.copy() - patch2.copy()
                var = np.var(diff)

                #filter out stuff if the averages are too far off
                if avg1-avg2 < 100:
                    flux.append((avg1+avg2)/2.)
                    variance.append(var/2.)
                    #variance.append(var/np.sqrt(2.))

        for i in range(Q1y):
            for j in range(Q1x):
                minidy = i * int(size)
                maxidy = minidy + int(size)
                minidx = j * int(size)
                maxidx = minidx + int(size)
                patch1 = np.ravel(Q11[minidy:maxidy, minidx:maxidx])
                patch2 = np.ravel(Q21[minidy:maxidy, minidx:maxidx])
                avg1 = np.mean(patch1)
                avg2 = np.mean(patch2)

                diff = patch1.copy() - patch2.copy()
                var = np.var(diff)

                #filter out stuff if the averages are too far off
                if avg1-avg2 < 100:
                    flux.append((avg1+avg2)/2.)
                    variance.append(var/2.)
                    #variance.append(var/np.sqrt(2.))

        flux = np.asarray(flux)
        variance = np.asarray(variance)

        results[f1] = dict(flux=flux, variance=variance)
        print f1, flux.mean(), variance.mean()

    return results


def pairwiseNoiseRowColumns(pairs, gain=3.5):
    """
    Calculates the mean flux within a row/column and the variance from the difference image.
    The variance of the difference image is divided by 2, given that var(x-y) = var(x) + var(y).

    The calculations are performed in electrons so that variance = noise**2 should be equal to the mean counts
    if no correlated noise and other effects are present. This would be the case of pure shot noise.

    :return:
    """
    print 'file     , column flux, row flux, column variance, row variance'

    results = {}

    for f1, f2 in pairs:
        #move from ADUs to electrons
        d1 = pf.getdata(f1) * gain
        d2 = pf.getdata(f2) * gain

        y1, x1 = d1.shape

        #subtract over/prescan row-by-row to minimise the any bias variation in column direction
        for row in range(y1):
            prescan1 = np.median(d1[row, 9:48])
            prescan2 = np.median(d2[row, 9:48])
            d1[row, :2099] -= prescan1
            d2[row, :2099] -= prescan2

        for row in range(y1):
            #not really an overscan but prescan to a different node
            overscan1 = np.median(d1[row, 4152:4190])
            overscan2 = np.median(d2[row, 4152:4190])
            d1[row, 2100:] -= overscan1
            d2[row, 2100:] -= overscan2

        #define quadrants; usable image area Q0
        Q10 = d1[11:2051, 58:2095].copy()
        Q20 = d2[11:2051, 58:2095].copy()
        #Q1
        Q11 = d1[11:2051, 2110:4132].copy()
        Q21 = d2[11:2051, 2110:4132].copy()

        #number of pixels in newly defined areas
        Q10y, Q10x = Q10.shape
        Q11y, Q11x = Q11.shape

        #data containers
        rowflux = []
        rowvariance = []
        columnflux = []
        columnvariance = []

        #loop over rows in Q0
        for row in range(Q10y):
            patch1 = Q10[row, :]
            patch2 = Q20[row, :]
            avg1 = np.mean(patch1)
            avg2 = np.mean(patch2)

            diff = patch1.copy() - patch2.copy()
            var = np.var(diff)

            #filter out stuff if the averages are too far off
            if avg1-avg2 < 200:
                rowflux.append((avg1+avg2)/2.)
                rowvariance.append(var/2.)

        #loop over columns in Q0
        for column in range(Q10x):
            patch1 = Q10[:, column]
            patch2 = Q20[:, column]
            avg1 = np.mean(patch1)
            avg2 = np.mean(patch2)

            diff = patch1.copy() - patch2.copy()
            var = np.var(diff)

            #filter out stuff if the averages are too far off
            if avg1-avg2 < 200:
                columnflux.append((avg1+avg2)/2.)
                columnvariance.append(var/2.)

        #loop over rows in Q1
        for row in range(Q11y):
            patch1 = Q11[row, :]
            patch2 = Q21[row, :]
            avg1 = np.mean(patch1)
            avg2 = np.mean(patch2)

            diff = patch1.copy() - patch2.copy()
            var = np.var(diff)

            #filter out stuff if the averages are too far off
            if avg1-avg2 < 200:
                rowflux.append((avg1+avg2)/2.)
                rowvariance.append(var/2.)

        #loop over columns in Q1
        for column in range(Q11x):
            patch1 = Q11[:, column]
            patch2 = Q21[:, column]
            avg1 = np.mean(patch1)
            avg2 = np.mean(patch2)

            diff = patch1.copy() - patch2.copy()
            var = np.var(diff)

            #filter out stuff if the averages are too far off
            if avg1-avg2 < 200:
                columnflux.append((avg1+avg2)/2.)
                columnvariance.append(var/2.)

        rowflux = np.asarray(rowflux)
        rowvariance = np.asarray(rowvariance)

        columnflux = np.asarray(columnflux)
        columnvariance = np.asarray(columnvariance)

        results[f1] = dict(rowflux=rowflux, rowvariance=rowvariance,
                           columnflux=columnflux, columnvariance=columnvariance)

        #print f1, np.median(columnflux), np.median(rowflux), np.median(columnvariance), np.median(rowvariance)
        print f1, np.mean(columnflux), np.mean(rowflux), np.mean(columnvariance), np.mean(rowvariance)

    return results


def plotDetectorCounts():
    """

    :return:
    """
    #files = g.glob('05*Euclid.fits')
    #files = g.glob('05*Euclidflattened.fits')
    files = g.glob('05*Euclidbiasremoved.fits')

    fig = plt.figure(1)
    plt.title(r'CCD273-84-2-F15, Serial number: 11312-14-01')
    ax = fig.add_subplot(111)

    for file in files:
        data = pf.getdata(file)
        c1 = np.average(data[1800:1810, 200:210])
        c2 = np.average(data[1000:1010, 1800:1810])
        c3 = np.average(data[1000:1010, 2300:2310])
        c4 = np.average(data[200:210, 3800:3810])

        plt.figure(2)
        im = plt.imshow(data, origin='lower', extent=[0, 4100, 0, 2070])
        plt.plot([205, 1805, 2305, 3805], [1805, 1005, 1005, 205], 'rs')
        plt.text(200, 1810, 'C1')
        plt.text(1800, 1010, 'C2')
        plt.text(2300, 1010, 'C3')
        plt.text(3800, 210, 'C4')
        plt.xlim(0, 4100)
        plt.ylim(0, 2070)
        c = plt.colorbar(im)
        c.set_label('Image Scale')
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.savefig(file.replace('.fits', '.png'))
        plt.close()

        del data

        plt.plot(c1, c1 / c2, 'bo')
        plt.plot(c4, c4 / c3, 'rs')

    plt.plot(c1, c1 / c2, 'bo', label='C1 vs C1 / C2')
    plt.plot(c4, c4 / c3, 'rs', label='C4 vs C4 / C3')

    ax.set_xlim(0, 65000)

    ax.set_xlabel('Counts [C1 or C4] [ADU]')
    ax.set_ylabel('Delta Counts [C1/C2 or C4/C3] [ADU]')

    plt.legend(shadow=True, fancybox=True, numpoints=1)
    plt.savefig('gradient.pdf')


def simulatePoissonProcess(max=200000, size=200):
    """
    Simulate a Poisson noise process.

    :param max:
    :param size:
    :return: None
    """
    #for non-linearity
    from support import VISinstrumentModel

    size = int(size)

    fluxlevels = np.linspace(1000, max, 50)

    #readnoise
    readnoise = np.random.normal(loc=0, scale=4.5, size=(size, size))
    #PRNU
    prnu = np.random.normal(loc=1.0, scale=0.02, size=(size, size))

    fig = plt.figure(1)
    plt.title(r'Simulation: $%i \times %s$ region' % (size, size))
    plt.subplots_adjust(left=0.14)

    ax = fig.add_subplot(111)

    for flux in fluxlevels:
        d1 = np.random.poisson(flux, (size, size))*prnu + readnoise
        d2 = np.random.poisson(flux, (size, size))*prnu + readnoise
        fx = (np.average(d1) + np.average(d2)) / 2.
        ax.plot(fx, np.var(d1-d2)/2., 'bo')

        d1 = np.random.poisson(flux, (size, size))*prnu + readnoise
        d2 = np.random.poisson(flux, (size, size))*prnu + readnoise
        #d1nonlin = VISinstrumentModel.CCDnonLinearityModelSinusoidal(d1, 0.1, phase=0.5, multi=1.5)
        #d2nonlin = VISinstrumentModel.CCDnonLinearityModelSinusoidal(d2, 0.1, phase=0.5, multi=1.5)
        d1nonlin = VISinstrumentModel.CCDnonLinearityModel(d1)
        d2nonlin = VISinstrumentModel.CCDnonLinearityModel(d2)
        fx = (np.average(d1) + np.average(d2)) / 2.
        ax.plot(fx, np.var(d1nonlin-d2nonlin)/2., 'rs')

        d1 = np.random.poisson(flux, (size, size))*prnu*1.05 + readnoise #5% gain change
        d2 = np.random.poisson(flux, (size, size))*prnu + readnoise
        fx = (np.average(d1) + np.average(d2)) / 2.
        ax.plot(fx, np.var(d1 - d2) / 2., 'mD')

    ax.plot([-1, ], [-1, ], 'bo', label='data (linear)')
    ax.plot([-1, ], [-1, ], 'rs', label='data (non-linear)')
    ax.plot([-1, ], [-1, ], 'mD', label='data (gain change)')

    ax.plot([0, max], [0, max], 'k-', lw=1.5, label='shot noise')

    ax.set_xlim(0, max)
    ax.set_ylim(0, max)

    ax.set_xlabel(r'$ \left < \mathrm{Signal}_{%i \times %i} \right > \quad [e^{-}]$' % (size, size))
    ax.set_ylabel(r'$\frac{1}{2}\sigma^{2}(\Delta \mathrm{Signal}) \quad [(e^{-})^{2}]$')

    plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    plt.savefig('Simulation.pdf')
    plt.close()


def simulatePoissonProcessRowColumn(max=200000, size=200, short=True, Gaussian=False):
    """

    :param max:
    :param size:
    :return:
    """
    from scipy import ndimage

    fluxlevels = np.linspace(1000, max, 30)

    #readnoise
    readnoise = np.random.normal(loc=0, scale=4.5, size=(size, size))
    #PRNU
    prnu = np.random.normal(loc=1.0, scale=0.02, size=(size, size))

    fig = plt.figure(1)
    plt.title(r'Simulation: $%i \times %s$ region' % (size, size))
    plt.subplots_adjust(left=0.14)

    ax = fig.add_subplot(111)

    #correlation coefficient
    val = 1.455e-6 / 2.

    for flux in fluxlevels:
        d1 = np.random.poisson(flux, (size, size)) * prnu + readnoise
        d2 = np.random.poisson(flux, (size, size)) * prnu + readnoise

        #convolution
        if ~Gaussian:
            kernel = np.array([[0,val*flux,0],[0,(1-val),0],[0,val*flux,0]])
            d1 = ndimage.convolve(d1, kernel)
            d2 = ndimage.convolve(d2, kernel)

        #gaussian smoothing
        if Gaussian:
            if short:
               d1 = ndimage.filters.gaussian_filter(d1, [2, 0])
               d2 = ndimage.filters.gaussian_filter(d2, [2, 0])
            else:
               d2 = ndimage.filters.gaussian_filter(d2, [15, 0])
               d1 = ndimage.filters.gaussian_filter(d1, [15, 0])

        #change the correlation in row direction
        #for column in range(size):
        #    for row in range(size-2):
        #        d1[row+1, column] = (d1[row, column] + d1[row+1, column] + d1[row+2, column]) / 3.
        #        d2[row+1, column] = (d2[row, column] + d2[row+1, column] + d2[row+2, column]) / 3.

        #calculate correlation in row/column direction
        rowvar = []
        rowfx = []
        for row1, row2 in zip(d1, d2):
            var = np.var(row1 - row2) / 2.
            fx = (np.average(row1) + np.average(row2)) / 2.
            rowvar.append(var)
            rowfx.append(fx)

        ax.plot(rowfx, rowvar, 'b.', alpha=0.1)
        ax.plot(np.average(np.asarray(rowfx)), np.average(np.asarray(rowvar)), 'bo')
        #ax.plot(np.median(np.asarray(rowfx)), np.median(np.asarray(rowvar)), 'bo')

        colvar = []
        colfx = []
        for column1, column2 in zip(d1.T, d2.T):
            var = np.var(column1 - column2) / 2.
            fx = (np.average(column1) + np.average(column2)) / 2.
            colvar.append(var)
            colfx.append(fx)

        ax.plot(colfx, colvar, 'r.', alpha=0.1)
        ax.plot(np.average(np.asarray(colfx))+2000, np.average(np.asarray(colvar)), 'rs')
        #ax.plot(np.average(np.median(colfx)), np.median(np.asarray(colvar)), 'rs')

    #save d1 to a FITS file...
    if short:
        fileIO.writeFITS(d1, 'correlatedNoiseShort.fits', int=False)
    else:
        fileIO.writeFITS(d1, 'correlatedNoiseLong.fits', int=False)

    ax.plot([0, ], [0, ], 'bo', label='row')
    ax.plot([0, ], [0, ], 'rs', label='column')

    ax.plot([0, max], [0, max], 'k-', lw=1.5, label='shot noise')

    ax.set_xlim(0, max)
    ax.set_ylim(0, max)

    ax.set_xlabel(r'$ \left < \mathrm{Signal}_{%i} \right > \quad [e^{-}]$' % size)
    ax.set_ylabel(r'$\frac{1}{2}\sigma^{2}(\Delta \mathrm{Signal}) \quad [(e^{-})^{2}]$')

    plt.legend(shadow=True, fancybox=True, loc='upper left', numpoints=1)
    if short:
        plt.savefig('SimulationRowColShort.pdf')
    else:
        plt.savefig('SimulationRowColLong.pdf')
    plt.close()


def analyseCorrelationFourier(file1='05Sep_14_35_00s_Euclid.fits', file2='05Sep_14_36_31s_Euclid.fits',
                              small=False, shift=False):
    """

    :param file1:
    :param file2:
    :param small:
    :param shift:
    :return:
    """
    from scipy import fftpack

    d1 = pf.getdata(file1)
    d2 = pf.getdata(file2)

    #pre/overscans
    #prescan1 = d1[11:2056, 9:51].mean()
    overscan1 = d1[11:2056, 4150:4192].mean()
    #prescan2 = d2[11:2056, 9:51].mean()
    overscan2 = d2[11:2056, 4150:4192].mean()

    #define quadrants and subtract the bias levels
    #Q10 = d1[11:2051, 58:2095].copy() - prescan1
    #Q20 = d2[11:2051, 58:2095].copy() - prescan2
    Q11 = d1[11:2051, 2110:4132].copy() - overscan1
    Q21 = d2[11:2051, 2110:4132].copy() - overscan2

    #small region
    if small:
        Q11 = Q11[500:700, 500:700].copy()
        Q21 = Q21[500:700, 500:700].copy()

    #Fourier analysis: calculate 2D power spectrum and take a log
    if shift:
        fourierSpectrum1 = np.log10(np.abs(fftpack.fftshift(fftpack.fft2(Q11)))**2)
        fourierSpectrum2 = np.log10(np.abs(fftpack.fftshift(fftpack.fft2(Q21)))**2)
    else:
        fourierSpectrum1 = np.log10(np.abs(fftpack.fft2(Q11))**2)
        fourierSpectrum2 = np.log10(np.abs(fftpack.fft2(Q21))**2)
    #difference image
    diff = (Q11 - Q21).copy()
    if shift:
        fourierSpectrumD = np.log10(np.abs(fftpack.fftshift(fftpack.fft2(diff)))**2)
    else:
        fourierSpectrumD = np.log10(np.abs(fftpack.fft2(diff))**2)

    #plot images
    fig = plt.figure(figsize=(12,7))
    plt.suptitle('Fourier Analysis of Flat-field Data')
    plt.suptitle('Original Image', x=0.3, y=0.26)
    plt.suptitle(r'$\log_{10}$(2D Power Spectrum)', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(Q11, origin='lower')
    i2 = ax2.imshow(fourierSpectrum1, origin='lower')#, vmin=9, vmax=16)
    if small:
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.1e', ticks=[46000, 47000, 48000, 49000])
    else:
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.1e', ticks=[25000, 35000, 45000, 55000])
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    if small:
        plt.savefig('Fourier1.pdf')
    else:
        plt.savefig('Fourier1Full.pdf')
    plt.close()

    fig = plt.figure(figsize=(12,7))
    plt.suptitle('Fourier Analysis of Flat-field Data')
    plt.suptitle('Original Image', x=0.3, y=0.26)
    plt.suptitle(r'$\log_{10}$(2D Power Spectrum)', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(Q21, origin='lower')
    i2 = ax2.imshow(fourierSpectrum2, origin='lower')#, vmin=9, vmax=16)
    if small:
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.1e', ticks=[46000, 47000, 48000, 49000])
    else:
        plt.colorbar(i1, ax=ax1, orientation='horizontal', format='%.1e', ticks=[25000, 35000, 45000, 55000])
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    if small:
        plt.savefig('Fourier2.pdf')
    else:
        plt.savefig('Fourier2Full.pdf')
    plt.close()

    fig = plt.figure(figsize=(12,7))
    plt.suptitle('Fourier Analysis of Flat-field Data')
    plt.suptitle('Difference Image', x=0.3, y=0.26)
    plt.suptitle(r'$\log_{10}$(2D Power Spectrum)', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(diff, origin='lower', vmin=-400, vmax=400)
    if small:
        i2 = ax2.imshow(fourierSpectrumD, origin='lower')#, vmin=7, vmax= 12)
    else:
        i2 = ax2.imshow(fourierSpectrumD, origin='lower')#, vmin=9, vmax=12)
    plt.colorbar(i1, ax=ax1, orientation='horizontal', ticks=[-400, -200, 0, 200, 400])
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    if small:
        plt.savefig('FourierDifference.pdf')
    else:
        plt.savefig('FourierDifferenceFull.pdf')
    plt.close()


def autocorr(data):
    """

    :param data:
    :return:
    """
    #dataFT = np.fft.fft(data, axis=1)
    dataFT = np.fft.fft(data)
    #dataAC = np.fft.ifft(dataFT * np.conjugate(dataFT), axis=1).real
    dataAC = np.fft.ifft(dataFT * np.conjugate(dataFT)).real
    return dataAC


def analyseAutocorrelation(file1='05Sep_14_35_00s_Euclid.fits', file2='05Sep_14_36_31s_Euclid.fits'):
    """

    :param file1:
    :param file2:
    :return:
    """
    d1 = pf.getdata(file1)
    d2 = pf.getdata(file2)

    #pre/overscans
    #prescan1 = d1[11:2056, 9:51].mean()
    overscan1 = d1[11:2056, 4150:4192].mean()
    #prescan2 = d2[11:2056, 9:51].mean()
    overscan2 = d2[11:2056, 4150:4192].mean()

    #define quadrants and subtract the bias levels
    #Q10 = d1[11:2051, 58:2095].copy() - prescan1
    #Q20 = d2[11:2051, 58:2095].copy() - prescan2
    Q11 = d1[11:2051, 2110:4132].copy() - overscan1
    Q21 = d2[11:2051, 2110:4132].copy() - overscan2

    #small region
    Q11 = Q11[500:700, 500:700].copy()
    Q21 = Q21[500:700, 500:700].copy()

    #autocorrelation
    fourierSpectrum1 = np.log10(autocorr(Q11))
    fourierSpectrum2 = np.log10(autocorr(Q21))
    #difference image
    diff = (Q11 - Q21).copy()
    fourierSpectrumD = autocorr(diff)

    #plot images
    fig = plt.figure(figsize=(12, 7))
    plt.suptitle('Autocorrelation Analysis of Flat-field Data')
    plt.suptitle('Original Image', x=0.3, y=0.26)
    plt.suptitle(r'$\log_{10}$(Autocorrelation)', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(Q11, origin='lower')
    i2 = ax2.imshow(fourierSpectrum1, origin='lower')
    plt.colorbar(i1, ax=ax1, orientation='horizontal')
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    plt.savefig('Autocorr1.pdf')
    plt.close()

    fig = plt.figure(figsize=(12, 7))
    plt.suptitle('Autocorrelation Analysis of Flat-field Data')
    plt.suptitle('Original Image', x=0.3, y=0.26)
    plt.suptitle(r'$\log_{10}$(Autocorrelation)', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(Q21, origin='lower')
    i2 = ax2.imshow(fourierSpectrum2, origin='lower')
    plt.colorbar(i1, ax=ax1, orientation='horizontal')
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    plt.savefig('Autocorr2.pdf')
    plt.close()

    fig = plt.figure(figsize=(12, 7))
    plt.suptitle('Autocorrelation Analysis of Flat-field Data')
    plt.suptitle('Difference Image', x=0.3, y=0.26)
    plt.suptitle(r'Autocorrelation', x=0.75, y=0.26)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    i1 = ax1.imshow(diff, origin='lower', vmin=-400, vmax=400)
    i2 = ax2.imshow(fourierSpectrumD, origin='lower')
    plt.colorbar(i1, ax=ax1, orientation='horizontal')
    plt.colorbar(i2, ax=ax2, orientation='horizontal')
    plt.savefig('AutocorrDifference.pdf')
    plt.close()


if __name__ == '__main__':
    size = 200.

    analyseCorrelationFourier()
    analyseCorrelationFourier(small=True)
    analyseAutocorrelation()

    #makeFlat(files)
    #plotDetectorCounts()
    #findPairs()
    #
    # pairs = [('05Sep_14_57_00s_Euclid.fits', '05Sep_14_58_27s_Euclid.fits'),
    #          ('05Sep_14_41_10s_Euclid.fits', '05Sep_14_43_30s_Euclid.fits'),
    #          ('05Sep_14_25_05s_Euclid.fits', '05Sep_14_26_30s_Euclid.fits'),
    #          ('05Sep_15_00_09s_Euclid.fits', '05Sep_15_02_21s_Euclid.fits'),
    #          ('05Sep_14_45_07s_Euclid.fits', '05Sep_14_46_28s_Euclid.fits'),
    #          ('05Sep_14_27_58s_Euclid.fits', '05Sep_14_30_22s_Euclid.fits'),
    #          ('05Sep_14_09_15s_Euclid.fits', '05Sep_14_10_38s_Euclid.fits'),
    #          ('05Sep_15_03_51s_Euclid.fits', '05Sep_15_05_18s_Euclid.fits'),
    #          ('05Sep_14_47_56s_Euclid.fits', '05Sep_14_49_25s_Euclid.fits'),
    #          ('05Sep_14_31_56s_Euclid.fits', '05Sep_14_33_23s_Euclid.fits'),
    #          ('05Sep_14_13_32s_Euclid.fits', '05Sep_14_14_57s_Euclid.fits'),
    #          ('05Sep_15_06_50s_Euclid.fits', '05Sep_15_08_18s_Euclid.fits'),
    #          ('05Sep_14_50_59s_Euclid.fits', '05Sep_14_52_31s_Euclid.fits'),
    #          ('05Sep_14_35_00s_Euclid.fits', '05Sep_14_36_31s_Euclid.fits')]
    #
    # #simulation
    # simulatePoissonProcess(size=size)
    # simulatePoissonProcessRowColumn()
    # simulatePoissonProcessRowColumn(short=False)
    #
    # #pixel region
    # output = pairwiseNoise(pairs, size=size)
    # fileIO.cPickleDumpDictionary(output, 'data.pk')
    #
    # output = cPickle.load(open('data.pk'))
    # plotAutocorrelation(output)
    # plotResults(output, size)
    #
    # #row-column
    # out = pairwiseNoiseRowColumns(pairs)
    # fileIO.cPickleDumpDictionary(out, 'dataRowColumn.pk')
    #
    # out = cPickle.load(open('dataRowColumn.pk'))
    # plotResultsRowColumn(out)
    #
    # out = {}
    # for file in g.glob('05*Euclid.fits'):
    #    data = pf.getdata(file)
    #    results = measureNoise(data, size, file)
    #    try:
    #        print file, np.median(results['flux']), np.median(results['variance'])
    #    except:
    #        print 'No useful data in ', file
    #        continue
    #    out[file] = results
    # fileIO.cPickleDumpDictionary(out, 'dataOLD.pk')
    # out = cPickle.load(open('dataOLD.pk'))
    # plotResults(out, size, pairwise=False, output='FullwellEstimateOLD.pdf')
