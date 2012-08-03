import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from analysis import ETC
from sources import createObjectCatalogue as cr


def plotDist():
    galaxies = np.loadtxt('data/cdf_galaxies.dat')
    gmodel = np.loadtxt('data/galaxy_model.dat')
    cdfstars = np.loadtxt('data/cdf_stars.dat')
    stars = np.loadtxt('data/stars.dat')
    besancon = np.loadtxt('data/besanc.dat')
    metcalfe = np.loadtxt('data/metcalfe.dat')
    shao = np.loadtxt('data/shao.dat')

    UDF = np.loadtxt('catalog0.dat', usecols=(2,3))
    st = UDF[:,0][UDF[:,1] < 1]
    gal = UDF[:,0][UDF[:,1] > 7]
    print '%i stars and %i galaxies in the catalog' % (len(st), len(gal))

    bins = np.arange(5.0, 31.5, 0.7)
    df = bins[1] - bins[0]
    weight = 1./(2048*2*2066*2.*0.1*0.1 * 7.71604938e-8) #how many square degrees one CCD is on sky

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(gal, bins=bins, log=True, alpha=0.4, weights=np.ones(gal.size)*weight/df, label='Catalog: galaxies')
    ax.hist(st, bins=bins, log=True, alpha=0.3, weights=np.ones(st.size)*weight/df, label='Catalog: stars')
    ax.semilogy(galaxies[:,0], galaxies[:,1], label='cdf_galaxies.dat')
    #ax.semilogy(gmodel[:,0], gmodel[:,1], ls='--', label='Galaxy model from Excel spreadsheet')
    ax.semilogy(shao[:,0], shao[:,4]*3600, ls=':', label='Shao et al. 2009')
    ax.semilogy(cdfstars[:,0], cdfstars[:,1], label='cdf_stars.dat')
    ax.semilogy(stars[:,0], stars[:,1], label='Star Dist (30deg)')
    ax.semilogy(stars[:,0], stars[:,2], label='Star Dist (60deg)')
    ax.semilogy(stars[:,0], stars[:,3], label='Star Dist (90deg)')
    #ax.semilogy(besancon[:,0], besancon[:,1], ls='--', label='Besancon')
    ax.semilogy(metcalfe[:,0], metcalfe[:,4], ls = '-.', label='Metcalfe')
    ax.set_xlabel('AB')
    ax.set_ylabel('N [sq deg]')
    ax.set_xlim(3, 30)
    ax.set_ylim(1e-2, 1e7)

    plt.legend(shadow=True, fancybox=True, loc='upper left')
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize='xx-small')

    plt.savefig('Distributions.pdf')


def plotSNR(deg=60, kdes=True):
    CCDs = 1000

    #bins = np.linspace(0, 1500, 301)
    bins = np.linspace(0, 1500, 70)
    df = bins[1] - bins[0]
    weight = 1. / (2048 * 2 * 2066 * 2 * 0.1 * 0.1 * 7.71604938e-8 * CCDs) / df #how many square degrees on sky

    #cumulative distribution of stars for different galactic latitudes
    if deg == 30:
        sfudge = 0.82
        tmp = 1
    elif deg == 60:
        sfudge = 0.78
        tmp = 2
    else:
        #90 deg
        sfudge = 0.74
        tmp = 3

    #stars
    d = np.loadtxt('data/stars.dat', usecols=(0, tmp))
    stmags = d[:, 0]
    stcounts = d[:, 1]

    #fit a function and generate finer sample
    z = np.polyfit(stmags, np.log10(stcounts), 4)
    p = np.poly1d(z)
    starmags = np.arange(1, 30.2, 0.2)
    starcounts = 10**p(starmags)

    cpdf = (starcounts - np.min(starcounts))/ (np.max(starcounts) - np.min(starcounts))
    starcounts /=  3600. #convert to square arcseconds
    nstars = int(np.max(starcounts) * 110 * sfudge) * CCDs
    magStars = cr.drawFromCumulativeDistributionFunction(cpdf, starmags, nstars)
    SNRsStars = ETC.SNR(ETC.VISinformation(), magnitude=magStars, exposures=1, galaxy=False)

    #calculate Gaussian KDE with statsmodels package (for speed)
    if kdes:
        kn = SNRsStars[SNRsStars < 1600]
        kdeStars = KDE(kn)
        kdeStars.fit(adjust=10)
        nst = kn.size / 10. / 1.34

    #galaxies
    #cumulative distribution of galaxies
    d = np.loadtxt('data/cdf_galaxies.dat', usecols=(0, 1))
    gmags = d[:, 0]
    gcounts = d[:, 1]
    nums = int(np.max(gcounts) / 3600. * 110 * CCDs)
    z = np.polyfit(gmags, np.log10(gcounts), 4)
    p = np.poly1d(z)
    galaxymags = np.arange(10.0, 30.2, 0.2)
    galaxycounts = 10**p(galaxymags)
    cumulative = (galaxycounts - np.min(galaxycounts))/ (np.max(galaxycounts) - np.min(galaxycounts))
    mag = cr.drawFromCumulativeDistributionFunction(cumulative, galaxymags, nums)
    SNRsGalaxies = ETC.SNR(ETC.VISinformation(), magnitude=mag, exposures=1)

    #calculate Gaussian KDE, this time with scipy to save memory, and evaluate it
    if kdes:
        pos = np.linspace(1, 1610, num=300)
        kn = SNRsGalaxies[SNRsGalaxies < 1600]
        kdegal = gaussian_kde(kn)
        gals = kdegal(pos)
        ngl = kn.size / 10. / 1.34

    #make a plot
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Euclid Visible Instrument')
    hist1 = ax.hist(SNRsStars, bins=bins, alpha=0.2, log=True, weights=[weight,]*len(magStars),
                    label='Stars [60 deg]', color='r')
    hist2 = ax.hist(SNRsGalaxies, bins=bins, alpha=0.2, log=True, weights=[weight,]*len(mag),
                    label='Galaxies', color='blue')

    if kdes:
        ax.plot(kdeStars.support, kdeStars.density*nst, 'r-', label='Gaussian KDE (stars)')
        ax.plot(pos, gals*ngl, 'b-', label='Gaussian KDE (galaxies)')

    ax.set_ylim(1,1e6)
    ax.set_xlim(0, 1500)
    ax.set_xlabel('Signal-to-Noise Ratio [assuming a single 565s exposure]')
    ax.set_ylabel('Number of Objects [deg$^{-2}$]')

    plt.text(0.8, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('SNRtheoretical.pdf')

    #output to file
    fh = open('SNRsSTARS.txt', 'w')
    fh.write('#These values are for stars at 60deg (%s)\n' % txt)
    fh.write('#SNR number_of_stars\n')
    fh.write('#bin_centre per_square_degree\n')
    for a, b in zip(hist1[0], hist1[1]):
        fh.write('%i %f\n' %(b+df/2., a))
    fh.close()
    fh = open('SNRsGALAXIES.txt', 'w')
    fh.write('#These values are for galaxies (%s)\n' % txt)
    fh.write('#SNR number_of_galaxies\n')
    fh.write('#bin_centre per_square_degree\n')
    for a, b in zip(hist2[0], hist2[1]):
        fh.write('%i %f\n' %(b+df/2., a))
    fh.close()


def plotSNRfromCatalog():
    #read in data
    catalog = np.loadtxt('catalog0.dat', usecols=(2,3))
    st = catalog[:,0][catalog[:,1] < 1] #stars, magnitudes
    weight = 1./(2048*2*2066*2.*0.1*0.1 * 7.71604938e-8) #how many square degrees one CCD is on sky
    SNRs = ETC.SNR(ETC.VISinformation(), magnitude=st)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(SNRs, bins=18, alpha=0.5, log=True, weights=[weight,]*len(st))
    ax.set_xlabel('SNR [assuming 3*565 seconds]')
    ax.set_ylabel('N [sq deg]')
    plt.savefig('SNRs.pdf')


if __name__ == '__main__':
    #plotDist()
    plotSNR()