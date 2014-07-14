"""
A simple script to analyse ground/lab flat fields.

This script has been written to analyse the importance of the spectral width of the input light on the PRNU recovery.

:author: Sami-Matias Niemi
:version: 0.2
"""
import matplotlib
#matplotlib.use('pdf')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['image.interpolation'] = 'none'
import matplotlib.pyplot as plt
import pyfits as pf
import numpy as np
import glob as g
from support import files as fileIO
from scipy import signal
from scipy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Box2DKernel
from skimage.measure import structural_similarity as ssim
from multiprocessing import Pool
import sys, math
from fitting import splineFitting


def subtractBias(data):
    """
    Subtract ADC offset using the pre- and overscan information for each quadrant.
    """
    prescanL = data[3:2200, 3:51].mean()
    prescanH = data[2350:4590, 3:51].mean()
    overscanL = data[3:2200, 4550:4598].mean()
    overscanH = data[2350:4590, 4550:4598].mean()

    Q0 = data[:2300, :2300]
    Q2 = data[2300:, :2300]
    Q1 = data[:2300, 2300:]
    Q3 = data[2300:, 2300:]

    #subtract the bias levels
    Q0 -= prescanL
    Q2 -= prescanH
    Q1 -= overscanL
    Q3 -= overscanH

    data[:2300, :2300] = Q0
    data[2300:, :2300] = Q2
    data[:2300, 2300:] = Q1
    data[2300:, 2300:] = Q3

    return data


def makeFlat(files, output, gain=1.):
    """
    Combine flat fields
    """
    d = []

    for file in files:
        data = subtractBias(pf.getdata(file)) * gain  #this would reserve tons of memory when multiprocessing
        #data = subtractBias(pf.open(file,  memmap=True)[0].data) * gain
        fileIO.writeFITS(data, file.replace('.fits', 'biasremoved.fits'), int=False)

        d.append(data)

    d = np.asarray(d)

    #write out FITS file
    avg = np.average(d, axis=0)
    fileIO.writeFITS(avg, output+'averaged.fits', int=False)

    med = np.median(d, axis=0)
    fileIO.writeFITS(med, output+'median.fits', int=False)

    return avg, med


def normaliseFlat(data, output, order=5, mask=True, method='boxcar'):
    """
    Normalise each quadrant separately. If limit set use to to generate a mask.
    """
    #split to quadrants
    Q0 = data[7:2052, 57:2098].copy()
    Q2 = data[2543:4592, 57:2098].copy()
    Q1 = data[7:2052, 2505:4545].copy()
    Q3 = data[2543:4592, 2505:4545].copy()
    Qs = [Q0, Q1, Q2, Q3]

    res = []
    for tmp in Qs:
        if mask:
            print 'Using masked 2D arrays (not applied in spline fitting)...'
            # median = np.median(tmp)
            # msk = (tmp > median*0.88) & (tmp < 40000.)
            # #note the inversion of the mask before applying, as for numpy masked arrays True means masking
            # #while in my selection above True means good data
            # t = np.ma.MaskedArray(tmp, mask=~msk)
            t = sigma_clip(tmp, sig=3.) #this can be used to generate automatically a masked array
        else:
            print 'No 2D masking applied...'
            t = tmp.copy()

        if method is 'surface':
            print 'Fitting a surface to model the illumination profile'
            #meshgrid representing data
            x, y = np.mgrid[:t.shape[0], :t.shape[1]]

            #fit a polynomial 2d surface to remove the illumination profile
            p_init = models.Polynomial2D(degree=order)
            f = fitting.NonLinearLSQFitter()
            p = f(p_init, x, y, t)

            #normalize data and save it to res list
            tmp /= p(x, y)

        elif method is 'boxcar':
            size = 15 #this is very small, so will probably smooth out some actual PRNU, but needed to remove dust specs
            print 'Using a boxcar smoothed image to model the illumination profile'
            #will have to convert masked array to NaN array as convolve does not support masks
            t = t.filled(np.nan)
            box_2D_kernel = Box2DKernel(size)
            if size > 50:
                model = convolve_fft(t, box_2D_kernel)
            else:
                model = convolve(t, box_2D_kernel) #faster for small kernels
            tmp /= model

        elif method is 'spline':
            spacing = 27
            print 'Fitting 1D splines to each row to model the illumination profile'

            for i, line in enumerate(tmp):
                #Initializes the instance with dummy xnodes
                Spline = splineFitting.SplineFitting([0, ])

                #filter dead pixels from the data
                y = line.copy()
                median = np.median(y)
                y = y[y > median*0.92]  #this is pretty aggressive masking, but needed because of no dead pixel map
                x = np.arange(len(y))

                #Median filter the data
                medianFiltered = signal.medfilt(y, 25)

                #Spline nodes and initial guess for y positions from median filtered
                xnods = np.arange(0, len(y), spacing)
                ynods = medianFiltered[xnods]
                #Updates dummy xnodes in Spline instance with real deal
                Spline.xnodes = xnods

                #Do the fitting
                fittedYnodes, success = Spline.doFit(ynods, x, y)

                #normalize the line with the fit
                tmp[i, :] /= Spline.fitfunc(np.arange(len(line)), fittedYnodes)

        else:
            print 'No fitting method selected, will exit...'
            sys.exit(-9)

        res.append(tmp)
        print np.mean(tmp), np.median(tmp), np.std(tmp)

    #save out
    out = np.zeros_like(data)
    out[7:2052, 57:2098] = res[0]
    out[7:2052, 2505:4545] = res[1]
    out[2543:4592, 57:2098] = res[2]
    out[2543:4592, 2505:4545] = res[3]

    fileIO.writeFITS(out, output+'FlatField%s.fits' % (method), int=False)

    return out


def __generateFlats(key, files):
    """
    Actual calls to generate flat fields.
    Stack the flats first and then normalise.
    """
    print key
    avg, med = makeFlat(files, key)
    normed = normaliseFlat(med, key, method='surface')
    return normed


def _generateFlats(key, files):
    """
    Actual calls to generate flat fields.
    Normalise the flats first and then stack.
    """
    size = 15 #this is very small, so will probably smooth out some actual PRNU, but needed to remove dust specs
    print key
    d = []
    for file in files:
        print file
        data = subtractBias(pf.getdata(file))
        fileIO.writeFITS(data, file.replace('.fits', 'biasremoved.fits'), int=False)

        #split to quadrants
        Q0 = data[7:2052, 57:2098].copy()
        Q2 = data[2543:4592, 57:2098].copy()
        Q1 = data[7:2052, 2505:4545].copy()
        Q3 = data[2543:4592, 2505:4545].copy()
        Qs = [Q0, Q1, Q2, Q3]

        res = []
        for tmp in Qs:
            t = sigma_clip(tmp, sig=3.) #this can be used to generate automatically a masked array

            print 'Using a boxcar smoothed image to model the illumination profile'
            #will have to convert masked array to NaN array as convolve does not support masks
            t = t.filled(np.nan)
            box_2D_kernel = Box2DKernel(size)
            if size > 50:
                model = convolve_fft(t, box_2D_kernel)
            else:
                model = convolve(t, box_2D_kernel) #faster for small kernels

            tmp /= model

            res.append(tmp)
            print np.mean(tmp), np.median(tmp), np.std(tmp)

        #save out
        out = np.zeros_like(data)
        out[7:2052, 57:2098] = res[0]
        out[7:2052, 2505:4545] = res[1]
        out[2543:4592, 57:2098] = res[2]
        out[2543:4592, 2505:4545] = res[3]
        d.append(out)

    #median combine
    d = np.asarray(d)

    #write out FITS file
    avg = np.average(d, axis=0)
    fileIO.writeFITS(avg, key+'averagedBC.fits', int=False)

    med = np.median(d, axis=0)
    fileIO.writeFITS(med, key+'medianBC.fits', int=False)

    return med


def generateFlats(args):
    """
    A wrapper to generate flat fields simultaneously at different wavelengths.
    A hack required as Pool does not accept multiple arguments.
    """
    return _generateFlats(*args)


def flats(processes=6):
    """
    Generates normalised flats at several wavelengths. Use all input files.
    """
    #search for the right files
    files = findFiles()

    #generate flats using multiprocessing
    pool = Pool(processes=processes)
    pool.map(generateFlats, [(key, files[key]) for key in files.keys()])


def generateFlatsSingle(args):
    """
    A wrapper to generate flat fields simultaneously at different wavelengths.
    A hack required as Pool does not accept multiple arguments.
    """
    return __generateFlats(*args)


def flatsSingle(processes=6):
    """
    Generates normalised flats at several wavelengths. Use all input files.
    """
    #search for the right files
    files = findFiles()

    #generate flats using multiprocessing
    pool = Pool(processes=processes)
    pool.map(generateFlatsSingle, [(key, files[key]) for key in files.keys()])


def findFiles():
    """

    """
    #wave = 609, 709, 809, 909, 959
    #fwhm = 3, 6, 9, 12, 15
    out = dict(f600nm3=g.glob('3nm/band*_3nm_609*_00??.fits'),
               f600nm6=g.glob('6nm/band*_6nm_609*_00??.fits'),
               f600nm9=g.glob('9nm/band*_9nm_609*_00??.fits'),
               f600nm12=g.glob('12nm/band*_12nm_609*_00??.fits'),
               f600nm15=g.glob('15nm/band*_15nm_609*_00??.fits'),
               f700nm3=g.glob('3nm/band*_3nm_709*_00??.fits'),
               f700nm6=g.glob('6nm/band*_6nm_709*_00??.fits'),
               f700nm9=g.glob('9nm/band*_9nm_709*_00??.fits'),
               f700nm12=g.glob('12nm/band*_12nm_709*_00??.fits'),
               f700nm15=g.glob('15nm/band*_15nm_709*_00??.fits'),
               f800nm3=g.glob('3nm/band*_3nm_809*_00??.fits'),
               f800nm6=g.glob('6nm/band*_6nm_809*_00??.fits'),
               f800nm9=g.glob('9nm/band*_9nm_809*_00??.fits'),
               f800nm12=g.glob('12nm/band*_12nm_809*_00??.fits'),
               f800nm15=g.glob('15nm/band*_15nm_809*_00??.fits'),
               f900nm3=g.glob('3nm/band*_3nm_909*_00??.fits'),
               f900nm6=g.glob('6nm/band*_6nm_909*_00??.fits'),
               f900nm9=g.glob('9nm/band*_9nm_909*_00??.fits'),
               f900nm12=g.glob('12nm/band*_12nm_909*_00??.fits'),
               f900nm15=g.glob('15nm/band*_15nm_909*_00??.fits'),
               f950nm3=g.glob('3nm/band*_3nm_959*_00??.fits'),
               f950nm6=g.glob('6nm/band*_6nm_959*_00??.fits'),
               f950nm9=g.glob('9nm/band*_9nm_959*_00??.fits'),
               f950nm12=g.glob('12nm/band*_12nm_959*_00??.fits'),
               f950nm15=g.glob('15nm/band*_15nm_959*_00??.fits'))

    return out


def normalisedCrosscorrelation(image1, image2):
    """
    Calculates the normalised cross-correlation between two input images (2D arrays).
    """
    dist_ncc = np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2))) / \
                     ((image1.size - 1) * np.std(image1) * np.std(image2))
    return dist_ncc


def correlateBC(xmin=80, xmax=2000, ymin=40, ymax=2000):
    for wave in [600, 700, 800, 900, 950]:
        print 'Wavelength: %i nm' % wave
        #load data
        data = {}
        for file in g.glob('f%inm*medianBC.fits' % wave):
            fh = pf.open(file)
            width = int(file.replace('medianBC.fits', '').split('nm')[1])
            data[width] = fh[1].data
            fh.close()

        cr = []
        for i, wave1 in enumerate(sorted(data.keys())):
            for j, wave2 in enumerate(sorted(data.keys())):
                tmp1 = data[wave1][ymin:ymax, xmin:xmax].copy()
                tmp2 = data[wave2][ymin:ymax, xmin:xmax].copy()

                 # calculate the difference and its norms
                diff = tmp1 - tmp2  # elementwise for scipy arrays
                m_norm = np.sum(np.abs(diff))  # Manhattan norm
                z_norm = norm(diff.ravel(), 0)  # Zero norm

                #cor = np.corrcoef(tmp1, tmp2)
                dist_ncc = normalisedCrosscorrelation(tmp1, tmp2)

                print wave1, wave2
                print "Manhattan norm:", m_norm, "/ per pixel:", m_norm/tmp1.size
                print "Zero norm:", z_norm, "/ per pixel:", z_norm*1./tmp1.size
                print "Normalized cross-correlation:", dist_ncc

                cr.append(dist_ncc)

        #data containers, make a 2D array of the cross-correlations
        wx = [x for x in sorted(data.keys())]
        wy = [y for y in sorted(data.keys())]
        cr = np.asarray(cr).reshape(len(wx), len(wy))

        fig = plt.figure()
        plt.title('Normalized Cross-Correlation (PRNU @ %i nm)' % wave)
        ax = fig.add_subplot(111)

        plt.pcolor(cr, cmap='Greys', vmin=0.95, vmax=1.)
        plt.colorbar()

        #change the labels and move ticks to centre
        ticks = np.arange(len(wx)) + 0.5
        plt.xticks(ticks, wx)
        plt.yticks(ticks, wy)
        ax.xaxis.set_ticks_position('none') #remove the tick marks
        ax.yaxis.set_ticks_position('none') #remove the tick marks

        ax.set_xlabel('FWHM [nm]')
        ax.set_ylabel('FWHM [nm]')

        plt.savefig('Crosscorrelation%iBoxcarInd.pdf' % wave)
        plt.close()



def correlateSurface(xmin=80, xmax=2000, ymin=40, ymax=2000):
    for wave in [600, 700, 800, 900, 950]:
        print 'Wavelength: %i nm' % wave
        #load data
        data = {}
        for file in g.glob('f%inm*FlatFieldsurface.fits' % wave):
            fh = pf.open(file)
            width = int(file.replace('FlatFieldsurface.fits', '').split('nm')[1])
            data[width] = fh[1].data
            fh.close()

        cr = []
        for i, wave1 in enumerate(sorted(data.keys())):
            for j, wave2 in enumerate(sorted(data.keys())):
                tmp1 = data[wave1][ymin:ymax, xmin:xmax].copy()
                tmp2 = data[wave2][ymin:ymax, xmin:xmax].copy()

                 # calculate the difference and its norms
                diff = tmp1 - tmp2  # elementwise for scipy arrays
                m_norm = np.sum(np.abs(diff))  # Manhattan norm
                z_norm = norm(diff.ravel(), 0)  # Zero norm

                #cor = np.corrcoef(tmp1, tmp2)
                dist_ncc = normalisedCrosscorrelation(tmp1, tmp2)

                print wave1, wave2
                print "Manhattan norm:", m_norm, "/ per pixel:", m_norm/tmp1.size
                print "Zero norm:", z_norm, "/ per pixel:", z_norm*1./tmp1.size
                print "Normalized cross-correlation:", dist_ncc

                cr.append(dist_ncc)

        #data containers, make a 2D array of the cross-correlations
        wx = [x for x in sorted(data.keys())]
        wy = [y for y in sorted(data.keys())]
        cr = np.asarray(cr).reshape(len(wx), len(wy))

        fig = plt.figure()
        plt.title('Normalized Cross-Correlation (PRNU @ %i nm)' % wave)
        ax = fig.add_subplot(111)

        plt.pcolor(cr, cmap='Greys', vmin=0.95, vmax=1.)
        plt.colorbar()

        #change the labels and move ticks to centre
        ticks = np.arange(len(wx)) + 0.5
        plt.xticks(ticks, wx)
        plt.yticks(ticks, wy)
        ax.xaxis.set_ticks_position('none') #remove the tick marks
        ax.yaxis.set_ticks_position('none') #remove the tick marks

        ax.set_xlabel('FWHM [nm]')
        ax.set_ylabel('FWHM [nm]')

        plt.savefig('Crosscorrelation%iSurface.pdf' % wave)
        plt.close()


def mse(x, y):
    """
    Mean Square Error (MSE)
    """
    return np.linalg.norm(x - y)


def structuralSimilarity(xmin=80, xmax=2000, ymin=40, ymax=2000, smooth=0.):
    """
    Adapted from:
    http://scikit-image.org/docs/0.9.x/auto_examples/plot_ssim.html#example-plot-ssim-py
    """
    for wave in [600, 700, 800, 900, 950]:
        print 'Wavelength: %i nm' % wave
        #load data
        # data = {}
        # for file in g.glob('f%inm*FlatFieldsurface.fits' % wave):
        #     fh = pf.open(file)
        #     width = int(file.replace('FlatFieldsurface.fits', '').split('nm')[1])
        #     data[width] = fh[1].data
        #     fh.close()
        data = {}
        for file in g.glob('f%inm*medianBC.fits' % wave):
            fh = pf.open(file)
            width = int(file.replace('medianBC.fits', '').split('nm')[1])
            data[width] = fh[1].data
            fh.close()

        ref = data[15][ymin:ymax, xmin:xmax].copy()
        if smooth > 1:
            ref = gaussian_filter(ref, smooth)

        number_of_subplots = math.ceil(len(data.keys())/2.)

        fig = plt.figure(figsize=(13, 13))
        plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.95, bottom=0.05)

        #loop over data from shortest wavelength to the longest
        wavearray = []
        msearray = []
        ssiarray = []
        for i, w in enumerate(sorted(data.keys())):
            tmp = data[w][ymin:ymax, xmin:xmax].copy()

            #Gaussian smooth to enhance structures for plotting
            if smooth > 1:
                tmp = gaussian_filter(tmp, smooth)

            ms = mse(tmp, ref)
            #careful with the win_size, can take up to 16G of memory if set to e.g. 19
            ssi = ssim(tmp, ref, dynamic_range=tmp.max() - tmp.min(), win_size=9)

            ax = plt.subplot(number_of_subplots, 2, i+1)
            im = ax.imshow(gaussian_filter(tmp, 2), interpolation='none', origin='lower', vmin=0.999, vmax=1.001)

            ax.set_title(r'$\lambda =$ ' + str(int(wave)) + 'nm, FWHM = ' + str(int(w)) + ';' + ' MSE: %.2f, SSIM: %.3f' % (ms, ssi))

            plt.axis('off')

            print w, ms, ssi
            wavearray.append(int(w))
            msearray.append(ms)
            ssiarray.append(ssi)

        cbar = plt.colorbar(im, cax=fig.add_axes([0.65, 0.14, 0.25, 0.03], frameon=False),
                            ticks=[0.999, 1, 1.001], format='%.3f', orientation='horizontal')
        cbar.set_label('Normalised Pixel Values')

        plt.savefig('StructuralSimilarities%i.png' % wave)
        plt.close()

        fig = plt.figure()
        plt.title(r'Mean Squared Error Wrt. FWHM$ = 15$nm')
        ax = fig.add_subplot(111)
        ax.plot(wavearray, msearray, 'bo')
        #ax.set_xlim(2, 16)
        #ax.set_ylim(-0.03, 6.)
        ax.set_ylabel('MSE')
        ax.set_xlabel('Wavelength [nm]')
        plt.savefig('MSEwave%i.pdf' % wave)
        plt.close()


if __name__ == '__main__':
    #for testings
    #test = dict(f800=g.glob('3nm/band*_3nm_809*_00??.fits'))
    #_generateFlats('f800', test['f800'])

    #generate flats from all available data
    #flats(processes=2)
    #flatsSingle(processes=2)

    #analysis
    #correlateBC()
    #correlateSurface()
    structuralSimilarity()