import numpy as np
import matplotlib.pyplot as plt

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
weight = 1./(2048*2*2066*2.*0.1*0.1 * 7.71604938e-8) #how many square degrees one CCD is on sky

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(gal, bins=18, log=True, alpha=0.4, weights=[weight,]*len(gal), label='Catalog: galaxies')
ax.hist(st, bins=18, log=True, alpha=0.3, weights=[weight,]*len(st), label='Catalog: stars')
ax.semilogy(galaxies[:,0], galaxies[:,1], label='cdf_galaxies.dat')
#ax.semilogy(gmodel[:,0], gmodel[:,1], ls='--', label='Galaxy model from Excel spreadsheet')
ax.semilogy(shao[:,0], shao[:,4]*3600, ls=':', label='Shao et al. 2009')
ax.semilogy(cdfstars[:,0], cdfstars[:,1], label='cdf_stars.dat')
ax.semilogy(stars[:,0], stars[:,1], label='Star Dist (30deg)')
ax.semilogy(stars[:,0], stars[:,2], label='Star Dist (60deg)')
ax.semilogy(stars[:,0], stars[:,3], label='Star Dist (90deg)')
#ax.semilogy(besancon[:,0], besancon[:,1], ls='--', label='Besancon')
ax.semilogy(metcalfe[:,0], metcalfe[:,4], ls = '-.', label='Metcalfe')
ax.set_xlabel('AB?')
ax.set_ylabel('N [sq deg]')
ax.set_xlim(3, 30)
ax.set_ylim(1e-2, 1e7)

plt.legend(shadow=True, fancybox=True, loc='upper left')
leg = plt.gca().get_legend()
ltext  = leg.get_texts() 
plt.setp(ltext, fontsize='xx-small')

plt.savefig('Distributions.pdf')
