# -*- coding: utf-8 -*-
"""
Cosmic Rays
===========

This scripts derives simple cosmic ray statististics from Gaia BAM data.

:requires: pyfits (tested with 3.3)
:requires: numpy (tested with 1.9.1)
:requires: scipy (tested with 0.15.1)
:requires: matplotlib (tested with 1.4.2)
:requires: skimage (scikit-image, tested with 0.10.1)
:requires: sklearn (scikit-learn, tested with 0.15.2)
:requires: vissim-python

:author: Sami-Matias Niemi (s.niemi@ucl.ac.uk)
:version: 0.5
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
from sklearn.neighbors import KernelDensity
from scipy import ndimage
import glob as g
from support import logger as lg


def findFiles(log, fileID='/Users/sammy/EUCLID/CCD273/CR/data/BAM_0000*.fits'):
    """
    """
    files = g.glob(fileID)
    msg = 'Found %i files' % len(files)
    print msg
    log.info(msg)
    return  files   
    

def readData(log, files):
    """
    Read data and gather information from the header.
    """
    info = []
    data = []
    for f in files:
        fh = pf.open(f)
        hdr = fh[0].header
        exptime = float(hdr['EXP_TIME'].split()[0])
        tditime = float(hdr['TRO_TIME'].split()[0])
        gain = float(hdr['CONVGAIN'].split()[0])
        pixels = hdr['PIX_GEOM']
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


def preProcessData(log, data, info, rebin=True):
    """
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
        
        if rebin:
            msg = 'Rebinning by %i x %i' % (i['binningx'], i['binningy'])
            print msg
            log.info(msg)
            #TODO: not sure if this is correct...
            sm = d.sum()
            d = ndimage.zoom(d, (i['binningx'], i['binningy']), order=1) #bilinear
            d = d*(sm / d.sum())
        
        out.append(d)
    return out


def _findCosmicRays(log, array, info, output, sigma=2.):
    """
    
    :param sigma: the thershold (std) above which the cosmic rays are identified
    :type sigma: float
    """
    #find all pixels above a threshold
    thresholded = array > array.std()*sigma
    CRdata = array[thresholded]
    #label the pixels
    labels, numb = ndimage.label(thresholded)
    print 'Found %i cosmic rays' % numb    
    
    #count the track lengths
    tracks = []
    energy = []
    for x in range(numb):
        selected = labels[labels == x+1]
        num = selected.shape[0]
        enrg = selected.sum()
        tracks.append(num)
        energy.append(enrg)

    tracks = np.asarray(tracks)
    energy = np.asarray(energy)
    sm = float(tracks.sum())
    rate = sm / info['exptime'] /array.size

    print 'The longest track covers %i pixels' % tracks.max()
    print 'Average track length is %.1f pixels' % tracks.mean()
    print 'In total, %i pixels were affected, i.e. %.1f per cent' % (sm, 100.*sm/array.size)
    print 'The rate of cosmic rays is %.2e CR / second / pixel' % rate
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
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

    return labels, tracks, energy
    

def analyseData(log, files, data, info):
    """
    """
    allD = []
    for f, d, i in zip(files, data, info):
        msg = 'Processing: ', f
        out = f.split('/')[-1].replace('.fits', '')
        print msg
        log.info(msg)
        labels, tracks, energy = _findCosmicRays(log, d, i, out)
        allD.append([d, labels, tracks, energy])
        
    #pull out the information from the individual files and join to a single array
    tracks = np.concatenate(np.asarray([x[2] for x in allD]))
    energies = np.concatenate(np.asarray([x[3] for x in allD]))
    
    #histogram bins
    esample = np.linspace(0, 3000, 50)
    tsample = np.linspace(1, 60, 30)
 
    #KDE for energy
    d2d = energies[:, np.newaxis]
    x_gride = np.linspace(0.0, esample.max(), 1000)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=40.)
    kde_skl.fit(d2d)
    log_pdfe = kde_skl.score_samples(x_gride[:, np.newaxis])

    #KDE for track lengts
    d2d = tracks[:, np.newaxis]
    x_gridt = np.linspace(0.0, tsample.max(), 500)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=1.5)
    kde_skl.fit(d2d)
    log_pdft = kde_skl.score_samples(x_gridt[:, np.newaxis])

    #energy and track lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(energies, bins=esample, normed=True, alpha=0.2)
    ax1.plot(x_gride, np.exp(log_pdfe), lw=3, c='r')
    ax2.hist(tracks, bins=tsample, normed=True, alpha=0.2)
    ax2.plot(x_gridt, np.exp(log_pdft), lw=3, c='r')    
    ax1.set_xlabel('$\Sigma$Energy [e$^{-}$]')
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('Track Lengths [pix]')
    plt.savefig('CRPDFs.png')
    plt.close()
    

def runAll():
    """
    """
    log = lg.setUpLogger('analyse.log')
    log.info('\n\nStarting to analyse')

    files = findFiles(log)
    data, info = readData(log, files)
    data = preProcessData(log, data, info)
    analyseData(log, files, data, info)


if __name__ == '__main__':
    runAll()