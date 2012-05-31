"""

:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data = np.loadtxt('cartesianMatched.csv', delimiter=',', skiprows=1)
    flag = data[:,3]
    mags = data[:,4]
    cnts = data[:,6]
    magi = data[:,14]

    #print np.mean(mags[(mags > 18) & (mags < 23)]/magi[(mags > 18) & (mags < 23)])

    msk = ~(flag > 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('1.0 pixel cartesian matching')
    ax.plot(magi, mags/magi, 'bo', label='All')
    ax.plot(magi[msk], mags[msk]/magi[msk], 'rs', label='Flag=0')
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Magnitude Ratio (extracted / input)')
    ax.plot([14,26], [1.0, 1.0], 'g--')
    ax.set_xlim(14, 26)
    ax.set_ylim(0.95, 1.05)

    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Magnitudes10.pdf')
    plt.close()


    #plot count test
    cntsin = 565 * 10**(-0.4*(magi[msk] - 25.579883922997453)) / 3.5
    #cntsin = 565 * 10**(-0.4*magi[msk]) / 3.5 * 1.7059e10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(cntsin, cnts[msk]/cntsin, 'bo', label='Flag=0')
    ax.plot([1e2,1e6], [1.0, 1.0], 'g--')
    ax.set_xlabel('Input Counts')
    ax.set_ylabel('Count Ratio (extracted / input)')
    ax.set_ylim(0.9, 1.1)
    ax.set_xlim(1e2, 1e6)

    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Counts10.pdf')
    plt.close()