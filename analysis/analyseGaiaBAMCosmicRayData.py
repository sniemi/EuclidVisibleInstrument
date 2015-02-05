# -*- coding: utf-8 -*-
"""
Cosmic Rays
===========

This scripts derives simple cosmic ray statististics from Gaia BAM data.

:requires: pyfits
:requires: numpy
:requires: scipy
:requires: matplotlib
:requires: vissim-python
:requires: skimage (scikit-image)

:author: Sami-Matias Niemi (s.niemi@ucl.ac.uk)
:version: 0.2
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
import glob as g
from support import logger as lg
from skimage.feature import blob_dog, blob_log, blob_doh



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
    """
    data = np.asarray([pf.getdata(file).astype(np.float64) for file in files])
    log.info('files read')
    return data


def preProcessData(log, data):
    """
    """
    out = []
    for d in data:
        #remove first line
        d = d[1:, :].T
        #remove median, this is assumed to correspond to the average ADC offset
        med = np.median(d)
        msg = 'Subtracting %.1f from the data' % med
        print msg
        log.info(msg)
        d -= med
        out.append(d)
    return out


def _findCosmicRays(log, array, output, sigma=2.):
    """
    """
    #find all pixels above a threshold
    thresholded = array > array.std()*sigma
    CRdata = array[thresholded]
    #label the pixels
    labels, numb = sp.ndimage.label(thresholded)
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

    print 'The longest track is %i pixels' % tracks.max()
    print 'Average track length is %.1f pixels' % tracks.mean()

    #plot simple histogram of the pixel values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    ax1.imshow(array, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=500)
    ax1.axis('off')
    ax2.hist(array.ravel(), bins=np.linspace(0, 500., 100), normed=True)
    plt.savefig(output+'histogram.png')
    plt.close()
    
    #threshold image
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    ax1.set_title('Data')
    ax2.set_title('Found CRs')
    ax1.imshow(array, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=2000)
    ax2.imshow(labels, cmap=plt.cm.jet, interpolation='none', vmin=1)
    ax1.axis('off')    
    ax2.axis('off')
    plt.savefig(output+'thresholded.png')
    plt.close()
    
    #energy and track lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(energy, bins=20, normed=True)
    ax2.hist(tracks, bins=20, normed=True)
    ax1.set_xlabel('Energy [DN]')
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('Track Lengths [pix]')
    plt.savefig(output+'CRs.png')
    plt.close()


#    blobs = blob_log(array, min_sigma=0.7, max_sigma=5., num_sigma=5)
##    blobs = blob_dog(array, min_sigma=0.7, max_sigma=5., sigma_ratio=1.5)
##    blobs = blob_doh(array, min_sigma=0.7, max_sigma=5., num_sigma=5)
#
#    print blobs
#    print blobs.shape
#    
#    fig, ax = plt.subplots(1, 1)
#    ax.set_title('Cosmic Ray Detection')
#    ax.imshow(array, interpolation='none')
#    for blob in blobs:
#        y, x, r = blob
#        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
#        ax.add_patch(c)
#    
#    plt.savefig(output+'CRids.png')
#    plt.close()

    return labels, tracks, energy

    

def analyseData(log, files, data):
    """
    """
    allD = []
    for f, d in zip(files, data):
        msg = 'Processing: ', f
        out = f.split('/')[-1].replace('.fits', '')
        print msg
        log.info(msg)
        labels, tracks, energy = _findCosmicRays(log, d, out)
        allD.append([d, labels, tracks, energy])
        
    tracks = [x[2] for x in allD]
    energies = [x[3] for x in allD]
    
    #energy and track lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(energies, bins=np.linspace(0, 2000, 25), normed=True)
    ax2.hist(tracks, bins=np.linspace(1, 40, 20), normed=True)
    ax1.set_xlabel('Energy [DN]')
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
    data = readData(log, files)
    data = preProcessData(log, data)
    analyseData(log, files, data)


if __name__ == '__main__':
    runAll()