# -*- coding: utf-8 -*-
"""
Cosmic Rays
===========

This scripts derives simple cosmic ray statististics from Gaia BAM data.
Note that the Gaia BAM data are binned 4 x 1 leading to pixel geometries
that are 120 x 10 microns. One can derive a statistical correction to
take into account the binning. Further corrections are needed to scale
e.g. to VIS CCD273 pixels, which are 12 x 12 microns. Thus, the results
will necessarily contain some uncertainties.

:requires: pyfits (tested with 3.3)
:requires: numpy (tested with 1.9.2)
:requires: scipy (tested with 0.15.1)
:requires: matplotlib (tested with 1.4.3)
:requires: skimage (scikit-image, tested with 0.10.1)
:requires: sklearn (scikit-learn, tested with 0.15.2)
:requires: statsmodels (tested with 0.6.1)
:requires: vissim-python

:author: Sami-Matias Niemi
:contact: s.niemi@icloud.com

:version: 1.0
"""
import matplotlib
matplotlib.use('pdf')
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
import scipy as sp
import scipy.interpolate as interpolate
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import ndimage
import glob as g
from support import logger as lg


def findFiles(log, fileID='/Users/sammy/EUCLID/CCD273/CR/data/BAM_0000*.fits'):
    """
    Find all files that match the ID.
    
    :param log: a logger instance
    :type log: instance
    :param fileID: identification with a wild card to find the files
    :type fileID: str
    
    :return: a list containing all files matching the wild card.
    :rtype: lst
    """
    files = g.glob(fileID)
    msg = 'Found %i files' % len(files)
    print msg
    log.info(msg)
    return  files   
    

def readData(log, files):
    """
    Read data and gather information from the header from all files given.

    :param log: a logger instance
    :type log: instance    
    :param files: a list of FITS file names
    :type files: lst
    
    :return: NumPy array contaning pixel data and a list containing dictionaries that hold header information
    :rtype: ndarray, lst
    """
    info = []
    data = []
    for f in files:
        fh = pf.open(f)
        hdr = fh[0].header
        exptime = float(hdr['EXP_TIME'].split()[0])
        tditime = float(hdr['TRO_TIME'].split()[0])
        gain = float(hdr['CONVGAIN'].split()[0])
        pixels = hdr['PIX_GEOM'].split()[::2]
        binningx, binningy = hdr['BINNING'].strip().split('x')
        binningx = int(binningx)
        binningy = int(binningy)

        info.append(dict(exptime=exptime, tditime=tditime, gain=gain, pixels=pixels,
                         binningx=binningx, binningy=binningy))
                         
        data.append(fh[0].data.astype(np.float64))
        
        fh.close()
        
    np.asarray(data)
    log.info('files read')
    return data, info


def preProcessData(log, data, info):
    """
    Removes the first line, transposes the array, and subtracts the median (derived ADC offset).
    
    :param log: a logger instance
    :type log: instance    
    :param data: a list of pixel data arrays to process
    :type data: lst
    :param info: a list of dictionaries that contain the header information
    :type info: lst
    
    :return: list of pixel data arrays
    :rtype: lst
    """
    out = []
    for d, i in zip(data, info):
        #remove first line
        d = d[1:, :].T
        
        #remove median, this is assumed to correspond to the average ADC offset
        med = np.median(d)
        msg = 'Subtracting %.1f from the data' % med
        print msg
        log.info(msg)
        d -= med
        
        #convert to electrons
        msg = 'Multiplying with gain of %.3f' % i['gain']
        print msg
        log.info(msg)
        d *= i['gain']
                
        out.append(d)
    return out
    
    
def _drawFromCumulativeDistributionFunction(cpdf, x, number):
    """
    Draw a number of random x values from a cumulative distribution function.

    :param cpdf: cumulative distribution function
    :type cpdf: numpy array
    :param x: values of the abscissa
    :type x: numpy array
    :param number: number of draws
    :type number: int

    :return: randomly drawn x value
    :rtype: ndarray
    """
    luck = np.random.random(number)
    tck = interpolate.splrep(cpdf, x)
    out = interpolate.splev(luck, tck)
    return out


def _findCosmicRays(log, array, info, output, sigma=3.5, correctLengths=True):
    """
    Find all cosmic rays from data. A simple threshold above the noise level is used.
    All pixels above the given sigma limit are assumed to be cosmic ray events.
    
    :param log: a logger instance
    :type log: instance    
    :param array: pixel data array
    :type array: ndarray
    :param info: dictionary containg the header information of the pixel data
    :type info: dict
    :param output: name of the output file
    :type output: str
    :param sigma: the thershold (std) above which the cosmic rays are identified
    :type sigma: float
    :param correctLenghts: whether or not correct for the binning.
    :type correctLengths: bool
    
    :return: a list containing CR labels, tracks, energies, and fluence
    :rtype: lst
    """
    #find all pixels above a threshold
    thresholded = array > array.std()*sigma
    CRdata = array[thresholded]
    #label the pixels
    labels, numb = ndimage.label(thresholded)
    print 'Found %i cosmic rays' % numb    
    #find locations    
    locations = ndimage.measurements.find_objects(labels)

    if correctLengths:
        print 'Trying to correct for the track lengths, loading a track length PDF'
        #read in the PDF of lengths
        data = np.loadtxt('trackLengthPDF.txt', delimiter=' ')
        pix = data[:, 0]
        PDF = data[:, 1]
        #convert to CDF
        dx = pix[1] - pix[0] #assume equal size steps
        cdf = np.cumsum(PDF*dx)    
    
    #count the track lengths and energies
    tracks = []
    energy = []
    for loc in locations:
        pixels = array[loc]     
        num = pixels.size

        #TODO: check that this is correct, tno sure...
        #correct for the fact that the data heavily binned
        if correctLengths:        
            if num == 1:
                #if single pixel event, then make a correction to the track length
                tmp = _drawFromCumulativeDistributionFunction(cdf, pix, num)
                num = tmp[0] / info['binningx']
            else:
                #multiple pixels, need to know direction
                x, y = pixels.shape #we transposed the array earlier when loading data
                if x > 1:
                    #need to draw a correction for each pixel covered
                    tmp = _drawFromCumulativeDistributionFunction(cdf, pix, x)
                    x = np.sum(tmp)                                
                #total number of pixels covered
                num = x + y
            
        #store information
        tracks.append(num)
        energy.append(pixels.sum())

    #convert to NumPy array
    tracks = np.asarray(tracks)
    energy = np.asarray(energy)
    
    #calculate statitics
    sm = float(tracks.sum())
    rate = sm / (info['exptime'] + info['tditime']) /array.size  #not sure if it should be half of the TDI time
    fluence = rate / (float(info['pixels'][0])*info['binningx']*float(info['pixels'][1])*info['binningy']) / 1e-8 / tracks.mean()

    if correctLengths: 
        print 'The longest track covers %i unbinned pixels' % tracks.max()
        print 'Average track length is %.1f unbinned pixels' % tracks.mean()
        print 'In total, %i unbinned pixels were affected, i.e. %.1f per cent' % (sm, 100.*sm/array.size)
        print 'The rate of cosmic rays is %.2e CR / second / unbinned pixel' % rate
        print 'The fluence of cosmic rays is %.1f events / second / cm**2' % fluence
        print 'Most energetic cosmic ray deposited %.1f photoelectrons' % energy.max()
        print 'Average track energy is %.1f photoelectrons' % energy.mean()

    else:
        print 'The longest track covers %i binned pixels' % tracks.max()
        print 'Average track length is %.1f binned pixels' % tracks.mean()
        print 'In total, %i binned pixels were affected, i.e. %.1f per cent' % (sm, 100.*sm/array.size)
        print 'The rate of cosmic rays is %.2e CR / second / binned pixel' % rate
        print 'The fluence of cosmic rays is %.1f events / second / cm**2' % fluence
        print 'Most energetic cosmic ray deposited %.1f photoelectrons' % energy.max()
        print 'Average track energy is %.1f photoelectrons' % energy.mean()

    #plot simple histogram of the pixel values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.imshow(array, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=500)
    ax1.axis('off')
    ax2.set_title('Pixel Values')
    ax2.hist(array.ravel(), bins=np.linspace(0, 500., 100), normed=True)
    ax2.set_xlabel('Energy [e$^{-}$]')
    plt.savefig(output+'histogram.png')
    plt.close()
    
    #threshold image
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))    
    ax1.set_title('Data')
    ax2.set_title('Found CRs')
    ax1.imshow(array, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=2000)
    ax2.imshow(labels, cmap=plt.cm.jet, interpolation='none', vmin=1)
    ax1.axis('off')    
    ax2.axis('off')
    plt.savefig(output+'thresholded.png')
    plt.close()
    
    #energy and track lengths
    plt.title(output.replace('_', '\_')) #needed for LaTeX
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(energy, bins=20, normed=True)
    ax2.hist(tracks, bins=20, normed=True)
    ax1.set_xlabel('Energy [e$^{-}$]')
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('Track Lengths [pix]')
    plt.savefig(output+'CRs.png')
    plt.close()

    return labels, tracks, energy, fluence
    

def analyseData(log, files, data, info):
    """
    Analyse all BAM data held in files.
    
    :param log: a logger instance
    :type log: instance    
    :param files: a list of file names
    :type files: lst
    :param data: a list of pixel data
    :type data: lst
    :parama info: a list of meta data dictionaries
    :type info: dict
    
    :return: None
    """
    allD = []
    for f, d, i in zip(files, data, info):
        msg = 'Processing: ', f
        out = f.split('/')[-1].replace('.fits', '')
        print msg
        log.info(msg)
        labels, tracks, energy, fluence = _findCosmicRays(log, d, i, out)
        allD.append([d, labels, tracks, energy, fluence])
        
    #pull out the information from the individual files and join to a single array
    tracks = np.concatenate(np.asarray([x[2] for x in allD]))
    energies = np.concatenate(np.asarray([x[3] for x in allD]))
    fluences = np.asarray([x[4] for x in allD])    

    #scale the track lengths to VIS 12 micron square pixels, assumes that the tracks
    #are unbinned lengths
    tracks *= ((float(info[0]['pixels'][0]) * float(info[0]['pixels'][1])) / (12.*12.))
    
    print '\n\n\nCR fluences in events / cm**2 / second (min, max, average, std):'
    print fluences.min(), fluences.max(), fluences.mean(), fluences.std()
    
    #take a log10, better for visualisation and comparison against Stardust modelling
    tracks = np.log10(tracks)
    energies = np.log10(energies)    
    
    #histogram bins
    esample = np.linspace(0., 7, 30)
    tsample = np.linspace(0., 3.5, 20)
 
    #KDE for energy
    d2d = energies[:, np.newaxis]
    x_gride = np.linspace(0.0, esample.max(), 500)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde_skl.fit(d2d)
    log_pdfe = kde_skl.score_samples(x_gride[:, np.newaxis])
    epdf = np.exp(log_pdfe)
    np.savetxt('CRenergyPDF.txt', np.vstack([x_gride, epdf]).T)

    #KDE for track lengts
    d2d = tracks[:, np.newaxis]
    x_gridt = np.linspace(0.0, tsample.max(), 500)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=0.05)
    kde_skl.fit(d2d)
    log_pdft = kde_skl.score_samples(x_gridt[:, np.newaxis])
    tpdf = np.exp(log_pdft)
    np.savetxt('CRtrackPDF.txt', np.vstack([x_gridt, tpdf]).T)

    #energy and track lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(energies, bins=esample, normed=True, alpha=0.2)
    ax1.plot(x_gride, epdf, lw=3, c='r')
    ax2.hist(tracks, bins=tsample, normed=True, alpha=0.2)
    ax2.plot(x_gridt, tpdf, lw=3, c='r')    
    ax1.set_xlabel('$\log_{10}(\Sigma$Energy [e$^{-}$]$)$')
    ax1.set_xlim(2.5, 7.)
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('$\log_{10}($Track Lengths [pix]$)$')
    plt.savefig('CRPDFs.png')
    plt.close()
    
    
def generateBAMdatagridImage():
    """
    Generates an example plot showing the Gaia BAM detector geometry
    and binning used. A simple illustration.
    
    :return: None
    """
    #grid
    x = np.linspace(0, 30*10, 11) #30 micron pixels
    y = np.linspace(0, 10*10, 11) #10 micron pixels


    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.title('Gaia BAM Data: CR Tracks')

    for v in y:        
        ax.axhline(y=v)
    for v in x:
        ax.axvline(x=v)
        
    #binning area 1
    plt.fill_between([0, 120], y1=[10, 10], y2=[20, 20], facecolor='r', alpha=0.1)
    plt.text(60, 14, 'Binned Pixel N+1', horizontalalignment='center', verticalalignment='center')
    plt.fill_between([0, 120], y1=[20, 20], y2=[30, 30], facecolor='g', alpha=0.1)
    plt.text(60, 24, 'Binned Pixel N+2', horizontalalignment='center', verticalalignment='center')
    plt.fill_between([0, 120], y1=[30, 30], y2=[40, 40], facecolor='m', alpha=0.1)
    plt.text(60, 34, 'Binned Pixel N+3', horizontalalignment='center', verticalalignment='center')

    #CR examples
    plt.fill_between([120, 240], y1=[70, 70], y2=[80, 80], facecolor='k', alpha=0.1)
    plt.plot([120, 240], [70, 80], 'k:', lw=1.)
    plt.text(230, 73, '4 Pix', horizontalalignment='center', verticalalignment='center')
    plt.fill_between([120, 240], y1=[60, 60], y2=[70, 70], facecolor='k', alpha=0.1)
    plt.plot([120, 190], [60, 70], 'k:', lw=1.)
    plt.text(230, 63, '3 Pix', horizontalalignment='center', verticalalignment='center')
    plt.fill_between([120, 240], y1=[50, 50], y2=[60, 60], facecolor='k', alpha=0.1)
    plt.plot([120, 170], [50, 60], 'k:', lw=1.)
    plt.text(230, 53, '2 Pix', horizontalalignment='center', verticalalignment='center')
    plt.fill_between([120, 240], y1=[40, 40], y2=[50, 50], facecolor='k', alpha=0.1)
    plt.plot([120, 140], [40, 50], 'k:', lw=1.)
    plt.text(230, 43, '1 Pix', horizontalalignment='center', verticalalignment='center')
    
    #possible CRs
    plt.fill_between([120, 240], y1=[20, 20], y2=[30, 30], facecolor='k', alpha=0.1)
    for xval in np.linspace(-60, 60, 31):
        plt.plot([180, 180+xval], [20, 30], 'k:', lw=0.5)
    for yval in np.linspace(20, 30, 11):
        plt.plot([180, 120], [20, yval], 'k:', lw=0.5)
        plt.plot([180, 240], [20, yval], 'k:', lw=0.5)      
      
    ax.set_xticks(x)    
    ax.set_yticks(y)    
    plt.xlabel('Physical X [microns]')     
    plt.ylabel('Physical Y [microns]')     
    plt.axis('scaled')
    plt.xlim(0, 300)
    plt.ylim(0, 100)
    plt.savefig('BAMccdGrid.pdf')
    plt.close()
    

def deriveCumulativeFunctionsforBinning(xbin=4, ybin=1, xsize=1, ysize=1, mc=100000, dx=0.1):
    """
    Because the original BAM data are binned 4 x 1, we do not know the track lengths.
    One can try to derive a statistical correction by randomizing the position and the
    angle a cosmic ray may have arrived. This function derives a probability density
    function for the track lengths by Monte Carloing over the random locations and 
    angles.
    
    :param xbin: number of pixels binned in x-direction
    :type xbin: int
    :param ybin: number of pixels binned in y-direction
    :type ybin: int
    :param xsize: how many binned pixels to use in the derivation
    :type xsize: int
    :param ysize:how many binned pixels to use in the derivation
    :type ysize: int
    :param mc: number of random realisations to generate
    :type mc: int
    :param dx: size of the steps to adopt when deriving CDF
    :type dx: float
    
    :return: None
    """
    #pixel sizes, unbinned
    ys = ysize * ybin
    xs = xsize * xbin
    #random location, random angle
    xloc = np.random.random(size=mc)*xbin
    yloc = np.random.random(size=mc)*ybin    
    angle = np.deg2rad(np.random.random(size=mc)*90)
    
    #maximum distance in y direction is either location or size - location
    ymax = np.maximum(yloc, ys - yloc)
    xmax = np.maximum(xloc, xs - xloc)
    
    #x length is the minimum of travel distance or the distance from the edge, but no smaller than 1
    xtravel = np.minimum(ymax / np.tan(angle), xmax)
    xtravel = np.maximum(xtravel, 1)
    
    #covering full pixels, but no more than xs
    covering = np.minimum(np.ceil(xtravel), xs)
    
    print 'Track lengths (mean, median, std):'
    print covering.mean(), np.median(covering), covering.std()
    
    #PDF for track lengths
    d2d = covering[:, np.newaxis]
    x_grid = np.linspace(0.0, xs+0.5, 1000)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=0.4)
    kde_skl.fit(d2d)
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    PDF = np.exp(log_pdf)
    #save to file
    np.savetxt('trackLengthPDF.txt', np.vstack([x_grid, PDF]).T)
    #get CDF using cumulative sum of the PDF
    dx = x_grid[1] - x_grid[0]
    CD = np.cumsum(PDF*dx)

    #derive empirical CDF and add unity stopping point
    CDF = ECDF(covering)
    CDF.y.put(-1, 1)
    CDF.x.put(-1, xs+1)
    CDFx = CDF.x
    CDFy = CDF.y
    
    #plot
    bins = np.arange(0, xs+1)+0.5
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.title('Probabilistic Track Lengths: Binning %i x %i' % (xbin, ybin))
    plt.plot(x_grid, PDF, lw=2, c='r', label='PDF')
    plt.plot(CDFx-0.5, CDFy, lw=1.5, c='m', alpha=0.7, label='CDF')
    plt.plot(x_grid, CD, lw=1.4, c='r', ls='--')
    plt.hist(covering, bins, normed=True, facecolor='green', alpha=0.35)
    ax.set_xticks(bins+0.5)
    plt.xlim(0.5, xs+0.5)
    plt.ylim(0, 1.05)
    plt.xlabel('Track Length')
    plt.ylabel('PDF / CDF')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('TrackLengths.pdf')
    plt.close()
    

def runAll(deriveCDF=True, examplePlot=True):
    """
    Run all steps from finding suitable Gaia BAM files to analysing them.
    
    :return: None
    """
    log = lg.setUpLogger('analyse.log')
    log.info('\n\nStarting to analyse')

    if deriveCDF: deriveCumulativeFunctionsforBinning()
    if examplePlot: generateBAMdatagridImage()

    files = findFiles(log)
    data, info = readData(log, files)
    data = preProcessData(log, data, info)
    analyseData(log, files, data, info)


if __name__ == '__main__':
    runAll()