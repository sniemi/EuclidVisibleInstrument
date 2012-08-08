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
    weight = 1./(2048*2*2066*2.*0.1*0.1 * 7.71604938e-8) / df #how many square degrees one CCD is on sky

    print 'Between magnitudes 18 and 22:'
    print (st[(st > 18) & (st < 22)]).size
    print 'Between magnitudes 18 and 23:'
    print (st[(st > 18) & (st < 23)]).size

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(gal, bins=bins, log=True, alpha=0.4, weights=np.ones(gal.size)*weight, label='Catalog: galaxies',
            cumulative=True)
    ax.hist(st, bins=bins, log=True, alpha=0.3, weights=np.ones(st.size)*weight, label='Catalog: stars',
            cumulative=True)
    ax.semilogy(galaxies[:,0], galaxies[:,1], label=r'cdfgalaxies.dat')
    #ax.semilogy(gmodel[:,0], gmodel[:,1], ls='--', label='Galaxy model from Excel spreadsheet')
    ax.semilogy(shao[:,0], shao[:,4]*3600, ls=':', label='Shao et al. 2009')
    ax.semilogy(cdfstars[:,0], cdfstars[:,1], label='cdfstars.dat')
    ax.semilogy(stars[:,0], stars[:,1], label='Stars (30deg)')
    ax.semilogy(stars[:,0], stars[:,2], label='Stars (60deg)')
    ax.semilogy(stars[:,0], stars[:,3], label='Stars (90deg)')
    #ax.semilogy(besancon[:,0], besancon[:,1], ls='--', label='Besancon')
    ax.semilogy(metcalfe[:,0], metcalfe[:,4], ls = '-.', label='Metcalfe')
    ax.set_xlabel(r'$M_{AB}$')
    ax.set_ylabel(r'Cumulative Number of Objects [deg$^{-2}$]')
    ax.set_xlim(3, 30)
    ax.set_ylim(1e-2, 1e7)

    plt.legend(shadow=True, fancybox=True, loc='upper left')
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize='xx-small')

    plt.savefig('Distributions.pdf')


def plotSNR(deg=60, kdes=True):
    CCDs = 1000
    fudge = 47.0

    bins = np.linspace(0, 800, 161)
    #bins = np.linspace(0, 800, 81)
    df = bins[1] - bins[0]
    weight = 1. / (2048 * 2 * 2066 * 2 * 0.1 * 0.1 * 7.71604938e-8 * CCDs) / df #how many square degrees on sky

    #cumulative distribution of stars for different galactic latitudes
    if deg == 30:
        tmp = 1
        sfudge = 0.79
    elif deg == 60:
        tmp = 2
        sfudge = 0.79
    else:
        tmp = 3
        sfudge = 0.78

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
    nstars = int(np.max(starcounts) * fudge * sfudge) * CCDs
    magStars = cr.drawFromCumulativeDistributionFunction(cpdf, starmags, nstars)
    SNRsStars = ETC.SNR(ETC.VISinformation(), magnitude=magStars, exposures=1, galaxy=False)

    print 'Assuming Galactic Lattitude = %i deg' % deg
    print 'Number of stars within a pointing (36CCDs) with 70 < SNR < 700 (single 565s exposure):', \
            int((SNRsStars[(SNRsStars > 70) & (SNRsStars < 700)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 60 < SNR < 80 (single 565s exposure):', \
            int((SNRsStars[(SNRsStars > 60) & (SNRsStars < 80)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 690 < SNR < 710 (single 565s exposure):', \
            int((SNRsStars[(SNRsStars > 690) & (SNRsStars < 710)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 18 < mag < 22 (single 565s exposure):', \
            int((SNRsStars[(magStars > 18) & (magStars < 22)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 18 < mag < 23 (single 565s exposure):', \
            int((SNRsStars[(magStars > 18) & (magStars < 23)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 17.9 < mag < 18.1 (single 565s exposure):', \
            int((SNRsStars[(magStars > 17.9) & (magStars < 18.1)]).size * 36. / CCDs)
    print 'Number of stars within a pointing (36CCDs) with 21 < mag < 23 (single 565s exposure):', \
            int((SNRsStars[(magStars > 21) & (magStars < 23)]).size * 36. / CCDs)

    #simple magnitude distribution plot for stars
    stars = np.loadtxt('data/stars.dat')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(magStars, bins=25, cumulative=True, log=True, alpha=0.3, weights=np.ones(magStars.size)*weight*df,
           label='Random Draws')
    ax.semilogy(stars[:,0], stars[:,1], label='Stars (30deg)')
    ax.semilogy(stars[:,0], stars[:,2], label='Stars (60deg)')
    ax.semilogy(stars[:,0], stars[:,3], label='Stars (90deg)')
    ax.set_xlabel(r'$M_{AB}$')
    ax.set_ylabel(r'Cumulative Number of Objects [deg$^{-2}$]')
    plt.legend(shadow=True, fancybox=True, loc='upper left')
    plt.savefig('stars%ideg.pdf' % deg)
    plt.close()

    #calculate Gaussian KDE with statsmodels package (for speed)
    if kdes:
        kn = SNRsStars[SNRsStars < 800]
        kdeStars = KDE(kn)
        kdeStars.fit(adjust=5)
        nst = kn.size / 10. / 1.38

    #galaxies
    #cumulative distribution of galaxies
    d = np.loadtxt('data/cdf_galaxies.dat', usecols=(0, 1))
    gmags = d[:, 0]
    gcounts = d[:, 1]
    nums = int(np.max(gcounts) / 3600. * fudge * CCDs)
    z = np.polyfit(gmags, np.log10(gcounts), 4)
    p = np.poly1d(z)
    galaxymags = np.arange(10.0, 30.2, 0.2)
    galaxycounts = 10**p(galaxymags)
    cumulative = (galaxycounts - np.min(galaxycounts))/ (np.max(galaxycounts) - np.min(galaxycounts))
    mag = cr.drawFromCumulativeDistributionFunction(cumulative, galaxymags, nums)
    SNRsGalaxies = ETC.SNR(ETC.VISinformation(), magnitude=mag, exposures=1)

    #calculate Gaussian KDE, this time with scipy to save memory, and evaluate it
    if kdes:
        kn = SNRsGalaxies[SNRsGalaxies < 800]
        #pos = np.linspace(1, 810, num=70)
        #kdegal = gaussian_kde(kn)
        #gals = kdegal(pos)
        #ngl = kn.size #/ df
        kdeGalaxy = KDE(kn)
        kdeGalaxy.fit(adjust=50)
        ngl = kn.size / 10. / 1.38

    #make a plot
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Euclid Visible Instrument')
    hist1 = ax.hist(SNRsStars, bins=bins, alpha=0.2, log=True, weights=[weight,]*len(magStars),
                    label='Stars [%i deg]' %deg, color='r')
    hist2 = ax.hist(SNRsGalaxies, bins=bins, alpha=0.2, log=True, weights=[weight,]*len(mag),
                    label='Galaxies', color='blue')

    if kdes:
        ax.plot(kdeStars.support, kdeStars.density*nst, 'r-', label='Gaussian KDE (stars)')
        #ax.plot(pos, gals*ngl, 'b-', label='Gaussian KDE (galaxies)')
        ax.plot(kdeGalaxy.support, kdeGalaxy.density*ngl, 'b-', label='Gaussian KDE (galaxies)')

    ax.set_ylim(1,1e5)
    ax.set_xlim(0, 800)
    ax.set_xlabel('Signal-to-Noise Ratio [assuming a single 565s exposure]')
    ax.set_ylabel('Number of Objects [deg$^{-2}$ dex$^{-1}$]')

    plt.text(0.8, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('SNRtheoretical%ideg.pdf' % deg)

    mid = df / 2.
    #output to file
    fh = open('SNRsSTARS%ideg.txt' % deg, 'w')
    fh.write('#These values are for stars at %ideg (%s)\n' % (deg, txt))
    fh.write('#SNR number_of_stars  N\n')
    fh.write('#bin_centre per_square_degree per_pointing\n')
    for a, b in zip(hist1[0], hist1[1]):
        fh.write('%i %f %f\n' %(b+mid, a*df, a*df*0.496))
    fh.close()
    fh = open('SNRsGALAXIES.txt', 'w')
    fh.write('#These values are for galaxies (%s)\n' % txt)
    fh.write('#SNR number_of_galaxies   N\n')
    fh.write('#bin_centre per_square_degree per_pointing\n')
    for a, b in zip(hist2[0], hist2[1]):
        fh.write('%i %f %f\n' %(b+mid, a*df, a*df*0.496))
    fh.close()


def plotSNRfromCatalog():
    #read in data
    catalog = np.loadtxt('catalog0.dat', usecols=(2,3))
    st = catalog[:,0][catalog[:,1] < 1] #stars, magnitudes
    gal = catalog[:,0][catalog[:,1] > 7] #galaxies, mags

    bins = np.linspace(0, 700, 31)
    df = bins[1] - bins[0]

    weight = 1./(2048*2*2066*2.*0.1*0.1 * 7.71604938e-8) / df #how many square degrees one CCD is on sky
    SNRs = ETC.SNR(ETC.VISinformation(), magnitude=st, exposures=1, galaxy=False)
    SNRg = ETC.SNR(ETC.VISinformation(), magnitude=gal, exposures=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(SNRs, bins=bins, alpha=0.5, log=True, weights=[weight,]*len(st), label='Stars')
    ax.hist(SNRg, bins=bins, alpha=0.2, log=True, weights=[weight,]*len(gal), label='Galaxies')
    ax.set_xlabel('SNR [assuming 565s exposure]')
    ax.set_ylabel(r'N [deg$^{-2}$ dex$^{-1}$]')
    plt.legend(shadow=True, fancybox=True)
    plt.savefig('SNRs.pdf')
    plt.close()


if __name__ == '__main__':
    #plot from a generated catalog assumed to be named "catalog0.dat"
    plotDist()
    plotSNRfromCatalog()

    #generates a new catalog on fly and plots SNRs
    #plotSNR(deg=30)
    #plotSNR(deg=60)
    #plotSNR(deg=90)