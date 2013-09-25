"""
This file provides simple functions to calculate wavelength dependent effects.

The functions can also be used to estimate the Weak Lensing Channel ghosts as a function
of spectral type.

:requires: NumPy
:requires: matplotlib
:requires: pysynphot

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
import pysynphot as S


def _VISbandpass(scale=0.8, area=10000.):
    """
    Returns Weak Lensing Channel bandpass objects.

    Sets the primary mirror collecting area to 1m**2. This affects the count rate calcuations.

    :param scale: scale the throughput to EoL situation [default=0.8]
    :type scale: float
    :param area: collecting area of the primary mirror [default=10000.]
    :type area: float

    :return: BoL and EoL bandpass objects
    """
    data = np.loadtxt('/Users/sammy/EUCLID/throughputs/VIS.txt')

    bp = S.ArrayBandpass(wave=data[:, 0], throughput=data[:, 1], waveunits='angstrom', name='VIS')

    bpEoL = S.ArrayBandpass(wave=data[:, 0].copy(), throughput=data[:, 1].copy()*scale,
                            waveunits='angstrom', name='VIS EoL')

    #set the primary mirror collecting area to 1m**2, effects the count rate estimates
    bp.primary_area = area
    bpEoL.primary_area = area

    return bp, bpEoL


def _VISbandpassGhost(scale=0.8, area=10000.):
    """
    Returns Weak Lensing Channel ghost objects.

    Sets the primary mirror collecting area to 1m**2. This affects the count rate calcuations.

    :param scale: scale the throughput to EoL situation [default=0.8]
    :type scale: float
    :param area: collecting area of the primary mirror [default=10000.]
    :type area: float

    :return: BoL and EoL bandpass objects
    """
    data = np.loadtxt('/Users/sammy/EUCLID/throughputs/ghost.txt')

    bp = S.ArrayBandpass(wave=data[:, 0], throughput=data[:, 1], waveunits='angstrom', name='VIS')

    bpEoL = S.ArrayBandpass(wave=data[:, 0].copy(), throughput=data[:, 1].copy()*scale,
                            waveunits='angstrom', name='VIS EoL')

    #set the primary mirror collecting area to 1m**2, effects the count rate estimates
    bp.primary_area = area
    bpEoL.primary_area = area

    return bp, bpEoL



def throughputs(output='throughputs.pdf'):
    """
    Plot throughputs, compares to HST WFC3 UVIS F600LP

    :param output: name of the output file
    :type output: str

    :return: None
    """
    #comparison
    bp1 = S.ObsBandpass('wfc3,uvis2,f600lp')

    #VIS
    bp, bpEoL = _VISbandpass()

    #ghost
    bpG, bpEoLG = _VISbandpassGhost()

    #plot
    plt.semilogy(bp1.wave/10., bp1.throughput, 'r-', label='WFC3 F600LP')
    plt.semilogy(bp.wave/10., bp.throughput, 'b-', label='VIS Best Estimate')
    plt.semilogy(bpEoL.wave/10., bpEoL.throughput, 'g--', label='VIS EoL Req.')
    plt.semilogy(bpG.wave/10., bpG.throughput, 'm-', label='VIS Ghost')
    plt.semilogy(bpEoLG.wave/10., bpEoLG.throughput, 'y-.', label='VIS Ghost EoL')
    plt.xlim(230, 1100)
    plt.xlabel(r'Wavelength [nm]')
    plt.ylabel(r'Total System Throughput')
    plt.legend(shadow=True, fancybox=True, loc='best')
    plt.savefig(output)
    plt.close()


def testFlatSpectrum(mag=18):
    """
    Test the pysynphot flat spectra, how to make flat in lambda and nu.

    :param mag:
    :return:
    """
    unitflux1 = S.FlatSpectrum(mag, fluxunits='abmag')                 #F_lam
    unitflux2 = S.FlatSpectrum(mag, fluxunits='abmag', waveunits='Hz') #F_nu

    #unitflux2.convert(S.units.Angstrom)

    plt.plot(unitflux1.wave, unitflux1.flux, 'r-')
    plt.plot(unitflux2.waveunits.ToAngstrom(unitflux2.wave), unitflux2.flux, 'b--')
    #plt.plot(unitflux2.wave, unitflux2.flux, 'b--')

    plt.xlim(3000, 11000)
    #plt.ylim(17.5, 18.5)
    plt.savefig('flatSpectra.pdf')
    plt.close()


def flatSpectrum(mag=18):
    unitflux = S.FlatSpectrum(mag, fluxunits='abmag')

    #observing bandpass
    bp1 = S.ObsBandpass('wfc3,uvis2,f600lp')
    #VIS
    bp, bpEoL = _VISbandpass()

    #observations
    obs1 = S.Observation(unitflux, bp1)
    obs2 = S.Observation(unitflux, bp)
    obsEoL = S.Observation(unitflux, bpEoL)

    #converts
    obs1.convert('counts')
    obs2.convert('counts')
    obsEoL.convert('counts')

    print 'Count rates in e/s (WFC3 vs VIS):'
    print obs1.countrate(range=[3500, 11000]) #source countrate e/s (all counts, no aperture assumed)
    print obs2.countrate(range=[3500, 11000]) #source countrate e/s (all counts, no aperture assumed)
    print obsEoL.countrate(range=[3500, 11000]) #source countrate e/s (all counts, no aperture assumed)

    print 'Count rate VIS in-band / total:'
    print obs2.countrate(range=[5500, 9000]) / obs2.countrate()

    #print 'integrated ?'
    #print obs1.integrate() #
    #print obs2.integrate() #

    #bin widths
    bw1 = np.diff(obs1.binwave)
    bw2 = np.diff(obs2.binwave)

    #plot
    plt.title(r'Flat Spectrum $F_{\lambda}$ 18 mag$_{AB}$ [photons cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]')
    plt.plot(obs1.binwave[1:], obs1.binflux[1:]/bw1, 'r-', label='WFC3 F600LP')
    plt.plot(obs2.binwave[1:], obs2.binflux[1:]/bw2, 'b-', label='VIS Best Estimate')
    plt.plot(obsEoL.binwave[1:], obsEoL.binflux[1:]/bw2, 'g--', label='VIS EoL Req.')
    plt.xlim(3000, 11000)
    plt.xlabel(r'Wavelength [\AA]')
    plt.ylabel(r'Counts per Wavelength Unit')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('comparisonFlatspectrum.pdf')
    plt.close()


def G2star():
    #object flux
    G2 = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_26.fits')

    #observing bandpass
    bp1 = S.ObsBandpass('wfc3,uvis2,f600lp')
    #VIS
    bp, bpEoL = _VISbandpass()

    #observations
    obs1 = S.Observation(G2, bp1)
    obs2 = S.Observation(G2, bp)
    obsEoL = S.Observation(G2, bpEoL)
    obs1.convert('counts')
    obs2.convert('counts')
    obsEoL.convert('counts')

    print 'effective wavelength [AA]'
    print obs2.efflam()  #effective wavelength

    print 'Countrate in e/s:'
    #source countrate e/s (all counts, no aperture assumed)
    print obs2.countrate(range=[3500, 11000])
    print obsEoL.countrate(range=[3500, 11000])
    print obs2.countrate(range=[5500, 9000])
    print obsEoL.countrate(range=[5500, 9000])

    print 'Count rate VIS in-band / total:'
    print obs2.countrate(range=[5500, 9000]) / obs2.countrate()

    #bin widths
    bw1 = np.diff(obs1.binwave)
    bw2 = np.diff(obs2.binwave)

    #plot
    plt.title('G2 Star (Pickels\_uk\_26)')
    plt.plot(obs1.binwave[1:], obs1.binflux[1:]/bw1, 'r-', label='WFC3 F600LP')
    plt.plot(obs2.binwave[1:], obs2.binflux[1:]/bw2, 'b-', label='VIS Best Estimate')
    plt.plot(obsEoL.binwave[1:], obsEoL.binflux[1:]/bw2, 'g--', label='VIS EoL Req.')
    plt.xlim(3000, 11000)
    plt.xlabel(r'Wavelength [\AA]')
    plt.ylabel(r'Counts per Wavelength Unit')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('comparisonG2.pdf')


def ghostCalculations(sourceSpectrum, title, output):
    """
    Weak Lensing Channel Ghost calculations.

    :param sourceSpectrum: pysynphot spectrum object
    :type sourceSpectrum: object
    :param title: title of the plot
    :type title: str
    :param output: name of the output file
    :type output: str

    :return: pysynphot observation object for VIS and Ghost
    :rtype: list of objects
    """
    bp, bpEoL = _VISbandpass()
    bpG, bpEoLG = _VISbandpassGhost()

    #BoL, the best estimate
    obs = S.Observation(sourceSpectrum, bp)
    obsG = S.Observation(sourceSpectrum, bpG)

    #convert to counts and derive the count rate in e/s
    obs.convert('counts')
    obsG.convert('counts')
    c = obs.countrate()
    cG = obsG.countrate()

    print 'effective stimulation in magnitude (AB)'
    print obs.effstim('abmag'), obsG.effstim('abmag')

    print 'effective wavelength'
    print obs.efflam(), obs.efflam()

    print 'source vs. ghost count rates [e/s]'
    print c, cG

    print 'ghost count rate / total count rate = %e' % (cG / c)
    print 'ghost count rate / total count rate [1750 dilution] = %e' % (cG / c / 1750.)

    #bin widths
    if 'lat' in title:
        #Flat spectrum requires different scaling...
        scale = 1.e5
    else:
        scale = 1.e7

    #binflux is wavelength bin dependent...
    bw1 = np.diff(obs.binwave) * scale
    bw2 = np.diff(obsG.binwave) * scale

    #make a plot
    plt.title(title)
    plt.semilogy(obs.binwave[1:], obs.binflux[1:]/bw1, 'b-', label='VIS Best Estimate')
    plt.semilogy(obsG.binwave[1:], obsG.binflux[1:]/bw2, 'r--', label='VIS Ghost')
    plt.xlim(3000, 11000)
    plt.ylim(1e-7, 10.)
    plt.xlabel(r'Wavelength [\AA]')
    plt.ylabel(r'Normalised Counts per Wavelength Unit')
    plt.legend(shadow=True, fancybox=True, loc='best')
    plt.savefig(output)
    plt.close()

    return obs, obsG


def ghostResults():
    """
    Calculate the VIS channel ghost contribution.

    Synphot table for the Pickles stellar library:
    http://www.stsci.edu/hst/HST_overview/documents/synphot/AppA_Catalogs5.html

    :return:
    """
    #G2V
    print '\n\n\nG2V:'
    G2 = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_26.fits')
    obsBoL2, obsBoLG2 = ghostCalculations(G2, 'G2V Star (Pickels\_uk\_26)', 'G2Ghost.pdf')

    #Flat
    print '\n\n\nFlat F_lam:'
    unitflux = S.FlatSpectrum(18.)
    obsBoL, obsBoLG = ghostCalculations(unitflux,
                                        r'Flat Spectrum $F_{\lambda}$ [photons cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]',
                                        'FlatGhost.pdf')

    #O5V
    print '\n\n\nO5V:'
    O5V = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_1.fits')
    obsBoL2, obsBoLG2 = ghostCalculations(O5V, 'O5V Star (Pickels\_uk\_1)', 'O5Ghost.pdf')

    #G2IV
    print '\n\n\nG2IV:'
    sp = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_54.fits')
    obsBoL2, obsBoLG2 = ghostCalculations(sp, 'G2IV Star (Pickels\_uk\_54)', 'G2IVGhost.pdf')

    #G5III
    print '\n\n\nG5III:'
    sp = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_73.fits')
    obsBoL2, obsBoLG2 = ghostCalculations(sp, 'G5III Star (Pickels\_uk\_73)', 'G5IIIGhost.pdf')

    #K2I
    print '\n\n\nK2I:'
    sp = S.FileSpectrum('/Users/sammy/synphot/pickles/dat_uvk/pickles_uk_128.fits')
    obsBoL2, obsBoLG2 = ghostCalculations(sp, 'K2I Star (Pickels\_uk\_128)', 'K2IGhost.pdf')


if __name__ == '__main__':
    #testFlatSpectrum()
    throughputs()
    #flatSpectrum()
    #G2star()

    ghostResults()