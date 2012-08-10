import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
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


def plotSNR(deg=60, kdes=True, log=False):
    CCDs = 1000
    fudge = 47.0

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

    #calculate Gaussian KDE with statsmodels package (for speed)
    if kdes:
        kn = SNRsStars[SNRsStars < 1000]
        kdeStars = KDE(kn)
        kdeStars.fit(adjust=2)
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
    magGalaxies = cr.drawFromCumulativeDistributionFunction(cumulative, galaxymags, nums)
    SNRsGalaxies = ETC.SNR(ETC.VISinformation(), magnitude=magGalaxies, exposures=1)

    #calculate Gaussian KDE, this time with scipy to save memory, and evaluate it
    if kdes:
        kn = SNRsGalaxies[SNRsGalaxies < 1000]
        #pos = np.linspace(1, 810, num=70)
        #kdegal = gaussian_kde(kn)
        #gals = kdegal(pos)
        #ngl = kn.size #/ df
        kdeGalaxy = KDE(kn)
        kdeGalaxy.fit(adjust=10)
        ngl = kn.size / 10. / 1.38

    #histogram binning and weighting
    bins = np.linspace(0., 1000., 101)
    df = bins[1] - bins[0]
    weight = 1. / (2048 * 2 * 2066 * 2 * 0.1 * 0.1 * 7.71604938e-8 * CCDs) / df
    weightsS = np.ones(magStars.size)*weight
    weightsG = np.ones(magGalaxies.size)*weight

    #simple magnitude distribution plot for stars
    stars = np.loadtxt('data/stars.dat')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(magStars, bins=30, cumulative=True, log=True, alpha=0.3, weights=weightsS*df,
           label='Random Draws')
    ax.semilogy(stars[:,0], stars[:,1], label='Stars (30deg)')
    ax.semilogy(stars[:,0], stars[:,2], label='Stars (60deg)')
    ax.semilogy(stars[:,0], stars[:,3], label='Stars (90deg)')
    ax.set_xlabel(r'$M_{AB}$')
    ax.set_ylabel(r'Cumulative Number of Objects [deg$^{-2}$]')
    plt.legend(shadow=True, fancybox=True, loc='upper left')
    plt.savefig('stars%ideg.pdf' % deg)
    plt.close()

    #make a plot
    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())
    ax = host_subplot(111, axes_class=AA.Axes)

    hist1 = ax.hist(SNRsStars, bins=bins, alpha=0.2, log=True, weights=weightsS,
                    label='Stars [%i deg]' %deg, color='r')
    hist2 = ax.hist(SNRsGalaxies, bins=bins, alpha=0.2, log=True, weights=weightsG,
                    label='Galaxies', color='blue')

    if kdes:
        ax.plot(kdeStars.support, kdeStars.density*nst, 'r-', label='Gaussian KDE (stars)')
        #ax.plot(pos, gals*ngl, 'b-', label='Gaussian KDE (galaxies)')
        ax.plot(kdeGalaxy.support, kdeGalaxy.density*ngl, 'b-', label='Gaussian KDE (galaxies)')

    #calculate magnitude scale, top-axis
    if log:
        mags = np.asarray([17, 18, 19, 20, 21, 22, 23, 24])
        SNRs = ETC.SNR(ETC.VISinformation(), magnitude=mags, exposures=1, galaxy=False)
    else:
        mags = np.asarray([17, 17.5, 18, 18.5, 19, 20, 21, 22.5])
        SNRs = ETC.SNR(ETC.VISinformation(), magnitude=mags, exposures=1, galaxy=False)

    ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
    ax2.set_xticks(SNRs)
    ax2.set_xticklabels([str(tmp) for tmp in mags])
    ax2.set_xlabel('$M(R+I)_{AB}$ [mag]')
    ax2.axis['right'].major_ticklabels.set_visible(False)

    ax.set_ylim(1e-1, 1e5)

    ax.set_ylabel('Number of Objects [deg$^{-2}$ dex$^{-1}$]')
    ax.set_xlabel('Signal-to-Noise Ratio [assuming a single 565s exposure]')

    plt.text(0.8, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)
    plt.legend(shadow=True, fancybox=True)

    if log:
        ax.set_xscale('log')
        plt.savefig('SNRtheoretical%ideglog.pdf' % deg)
    else:
        ax.set_xlim(1, 1e3)
        plt.savefig('SNRtheoretical%ideglin.pdf' % deg)

    plt.close()

    #write output
    if not log:
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
    #plotSNR(deg=30, kdes=False)
    #plotSNR(deg=60, kdes=False)
    #plotSNR(deg=90, kdes=False)

    #plotSNR(deg=30, kdes=False, log=True)
    #plotSNR(deg=60, kdes=False, log=True)
    #plotSNR(deg=90, kdes=False, log=True)