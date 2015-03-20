# -*- coding: utf-8 -*-
"""
Cosmic Rays
===========

This scripts derives simple cosmic ray statististics from Gaia RVS data.

:requires: pyfits (tested with 3.3)
:requires: numpy (tested with 1.9.2)
:requires: scipy (tested with 0.15.1)
:requires: matplotlib (tested with 1.4.3)
:requires: astropy (tested with 1.01)
:requires: sklearn (scikit-learn, tested with 0.15.2)
:requires: vissim-python

:author: Sami-Matias Niemi
:contact: s.niemi@icloud.com

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
import pandas as pd
from scipy import ndimage
from astropy.modeling import models, fitting
from astropy.modeling.rotations import Rotation2D
from sklearn.neighbors import KernelDensity
import glob as g
from support import files as fileIO


def convertFilesToFITS(folder='CFSData_16/', outFolder='fits/'):
    """
    Each file is a transit/spectra.  So you process each file separately to get 13000 spectra/images.
    //1260 is the number of lines in a file, you might want to do a count(numberoflines) to really make certain.
    
    double [][]flux = new double[10][1260]
    double [][]fluxErr = new double[10][1260]
    //remember to skip first line i.e. header
    for(int i = 0;i < 1260;i++) {
       flux[0][i] = column1
       flux[1][i] = column7
       flux[2][i] = column9
       flux[3][i] = column11
       flux[4][i] = column13
       flux[5][i] = column15
       flux[6][i] = column17
       flux[7][i] = column19
       flux[8][i] = column21
       flux[8][i] = column23
    
       fluxErr[0][i] = column2
       fluxErr[1][i] = column10
       fluxErr[2][i] = column12
       ....
    
       //you can see if column 25 exists to give you a hint of the location of the CR.
       //See my description on the shifting to find the ac and al location.
    """
    files = g.glob(folder + '*.dat')
    
    for filename in files:
        print 'Converting %s to FITS format' % filename
        #data = pd.read_table(filename, header=0, sep=' ')
        data = np.loadtxt(filename, skiprows=1)
        new = [data[:, 1], data[:, 7], data[:, 9], data[:, 11], data[:, 13],
               data[:, 15], data[:, 17], data[:, 19], data[:, 21], data[:, 23]]
                      
        #convert to 2D
        new = np.asarray(new)
        
        #write to fits
        output = outFolder + filename.replace(folder, '').replace('.dat', '.fits')
        print 'Saving the data to %s' % output
        fileIO.writeFITS(new, output, int=False)
        
        
def medianCombineAllFiles(folder='fits/'):
    """
    Median combine all FITS files to form a "composite spectrum" that can be
    subtracted from the data to have more or less that background and any
    cosmic rays.
    """
    files = g.glob(folder + '*.fits')
    data = []
    for filename in files:
        fh = pf.open(filename, memmap=False)
        d = fh[1].data
        data.append(d)
        fh.close()
    data = np.asarray(data)
    print data.shape
    med = np.median(data, axis=0)
    fileIO.writeFITS(med, 'medianCombined.fits', int=False)
    

def scaleAndSubtract(files, combined='medianCombined.fits'):
    """
    Scale the combined image to the peak of the file and subtract it from the data.
    Store to a new FITS file with "sub" appended.

    :param files: a list of filenames to process
    :type files: lst
    :param combined: name of the file to subtract from the raw data
    :type combined: str
    
    :return: None
    """
    subtract = pf.getdata(combined)
    subtract /= np.max(subtract)
    fileIO.writeFITS(subtract, combined.replace('.fits', 'Normalized.fits'), int=False)
    for filename in files:
        data = pf.getdata(filename)
        peak = np.max(data)
        s = subtract.copy()*peak
        data -= s
        print 'Processing %s with a peak of %.1f' % (filename, peak)
        fileIO.writeFITS(data, filename.replace('.fits', 'sub.fits'), int=False)
        

def fitPolynomialAndSubtract(files, degree=3, subtract=True, twoD=False):
    """
    Fit a polynomial surface to individual spectrum and then either subtract or
    divide the raw data with the fit to normalise. In principle second order
    fit would be enough, but because the spectrum is tilted one should either
    fit the rotation or go to a higher order model.
    
    In reality the tilt of the spectrum is field of view and detector dependent.
    See for example, GAIA-C6-TN-OPM-PPA-006-D, figure 16. We can assume that on
    average the tilt is about 2 pixels when averaging over a macrosample. This
    would give an average tilt of:
    In SMN [146]: np.rad2deg(np.tan(2/1155.))
    Out SMN [146]: 0.099213570180474012
    One could assume a tilt of 0.1degrees. The problem with this is that it is often
    incorrect. The best would be to build a compound model and fit simultaneously
    a polynomial surface with a rotation. This is not supported in Astropy at the
    moment (or at least I couldn't make it work).
    
    Another option is to fit a 1D polynomial along each row. This can be chosen by
    setting twoD to False. Doing the normalisation each row separately allows us
    to bypass at least some of the complications arising from the rotated spectrum.
    Nonetheless, this does not seem to work too well either...
    
    :param files: a list of filenames to process
    :type files: lst
    :param degree: the degree of the 2D polynomial
    :type degree: int
    :param subtract: whether to subtract or divide the data with the model
    :type subtract: bool
    :param twoD: whether to use 2D or 1D fit
    :type twoD: bool
    
    :return: None
    """    
    #http://astropy.readthedocs.org/en/latest/modeling/compound-models.html
    class RotatedPolynomial(Rotation2D | models.Polynomial2D(degree=degree)):
        """
        Rotated polynomial surface
        """
    
    for filename in files:
        data = pf.getdata(filename)
        
        if twoD:
            #rotate the data a fixed angle
            #data = ndimage.interpolation.rotate(data, -0.1, reshape=False, order=1, mode='nearest')        
    
            #grid
            ysize, xsize = data.shape
            y, x = np.mgrid[:ysize, :xsize]
            
            #fit simple polymial model
            p_init = models.Polynomial2D(degree=degree)
            #p_init = Rotation2D | models.Polynomial2D     #composite model, not fittable :-(
            fit_p = fitting.LevMarLSQFitter()
            fit = fit_p(p_init, x, y, data)    
            model = Rotation2D | fit
            rotated = model(0.1)
            fitted = rotated(x, y)
            #fitted = fit(x, y)
        else:
            model = []
            for i, line in enumerate(data):
                #fit a model for each line                
                x = np.arange(len(line))
                p_init = models.Polynomial1D(degree=degree)
                fit_p = fitting.LevMarLSQFitter()
                fit = fit_p(p_init, x, line)
                f = fit(x)
                model.append(f)

            fitted = np.asarray(model)

        #normalise by either subtracting or dividing
        if subtract:
            data -= fitted
        else:
            data /= fitted

        print 'Processing %s with fit' % (filename)
        print fit
        fileIO.writeFITS(fitted, filename.replace('.fits', 'Model.fits'), int=False)            
        fileIO.writeFITS(data, filename.replace('.fits', 'Norm.fits'), int=False)            
            
   
def _findCosmicRays(array, output, sigma=5., gain=1.):
    """
    Find all cosmic rays from data. A simple threshold above the noise level is used.
    All pixels above the given sigma limit are assumed to be cosmic ray events.
    
    :param array: pixel data array
    :type array: ndarray
    :param output: name of the output file
    :type output: str
    :param sigma: the thershold (std) above which the cosmic rays are identified
    :type sigma: float
    :param gain: gain factor to be used to multiply the pixel values (pre-processed data already gain corrected)
    :type gain: float
    
    :return: a list containing CR labels, tracks, energies, and fluence
    :rtype: lst
    """
    #find all pixels above a threshold
    thresholded = array > array.std()*sigma
    CRdata = array[thresholded]
    #label the pixels
    labels, numb = ndimage.label(thresholded)
    print 'Found %i cosmic rays' % numb

    #if no CRs found, then return zero arrays so that fluence calculations take these into account    
    if numb < 1:
        return np.asarray([0,]), np.asarray([0,]), np.asarray([0,]), np.asarray([0,])
        
    #find locations    
    locations = ndimage.measurements.find_objects(labels)

    #count the track lengths and energies
    tracks = []
    energy = []
    for loc in locations:
        pixels = array[loc]     
        num = pixels.size
        #store information
        tracks.append(num)
        energy.append(pixels.sum())

    #convert to NumPy array
    tracks = np.asarray(tracks)
    energy = np.asarray(energy) * gain
    
    if tracks.max() < 2:
        #a single pixel event, likely to be a noise spike rather than CR
        return np.asarray([0,]), np.asarray([0,]), np.asarray([0,]), np.asarray([0,])        
        
    #calculate statitics
    sm = float(tracks.sum())
    rate = sm / (4.42) /array.size
    fluence = rate / (10.*30.) / 1e-8 / tracks.mean()

    print 'The longest track covers %i pixels' % tracks.max()
    print 'Average track length is %.1f pixels' % tracks.mean()
    print 'In total, %i binned pixels were affected, i.e. %.1f per cent' % (sm, 100.*sm/array.size)
    print 'The rate of cosmic rays is %.2e CR / second / binned pixel' % rate
    print 'The fluence of cosmic rays is %.1f events / second / cm**2' % fluence
    print 'Most energetic cosmic ray deposited %.1f photoelectrons' % energy.max()
    print 'Average track energy is %.1f photoelectrons' % energy.mean()

    return labels, tracks, energy, fluence
    

def analyseData(files):
    """
    """
    allD = []
    for filename in files:
        print 'processing %s' % (filename)
        
        fh = pf.open(filename, memmap=False)
        d = fh[1].data
        fh.close()
        
        out = _findCosmicRays(d, filename.replace('.fits', ''))

        if out is not None:
            labels, tracks, energy, fluence = out    
            allD.append([d, labels, tracks, energy, fluence])

    #pull out the information from the individual files and join to a single array
    tracks = np.concatenate(np.asarray([x[2] for x in allD]))
    energies = np.concatenate(np.asarray([x[3] for x in allD]))
    fluences = np.asarray([x[4] for x in allD])    

    #scale the track lengths to VIS 12 micron square pixels, assumes that the tracks
    #are unbinned lengths from pixel scale of 10 x 30 microns
    tracks *= ((10.*30.) / (12.*12.))
    
    print '\n\n\nCR fluences in events / cm**2 / second (min, max, average, std):'
    print fluences.min(), fluences.max(), fluences.mean(), fluences.std()
    
    #take a log10, better for visualisation and comparison against Stardust modelling
    tracks = np.log10(tracks)
    tracks = tracks[np.isfinite(tracks)] #because return zeros if no CR found, then after log -inf
    energies = np.log10(energies) 
    energies = energies[np.isfinite(energies)] #because return zeros if no CR found, then after log -inf
    
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
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=0.1)
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

    
if __name__ == '__main__':
    for folder in g.glob('CFS*'):
        print 'processing folder', folder
        convertFilesToFITS(folder=folder+'/')
    fitPolynomialAndSubtract(g.glob('fits/*CR.fits'))
    analyseData(g.glob('fits/*CRNorm.fits'))