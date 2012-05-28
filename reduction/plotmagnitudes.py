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
    magi = data[:,11]

    #print np.mean(mags[(mags > 18) & (mags < 23)]/magi[(mags > 18) & (mags < 23)])

    msk = ~(flag > 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('2.5 pixel cartesian matching')
    ax.plot(magi, mags/magi, 'bo', label='All')
    ax.plot(magi[msk], mags[msk]/magi[msk], 'rs', label='Flag=0')
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Magnitude Ratio (extraced / input)')
    ax.plot([14,26], [1.0, 1.0], 'g--')
    ax.set_xlim(14, 26)
    ax.set_ylim(0.95, 1.05)

    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Magnitudes25.pdf')