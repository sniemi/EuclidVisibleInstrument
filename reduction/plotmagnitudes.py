"""

:requires: NumPy
:requires: matplotlib

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(filename, rad=1.0):
    """
    Simple comparison plot. Assumes that the input is
    a matched catalog.
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    flag = data[:, 2]
    mags = data[:, 3]
    cnts = data[:, 5]
    classification = data[:, 10]
    magi = data[:, 13]
    machingrad = data[:, -1]
    cntsin = 565 * 10**(-0.4*(magi - 25.579883922997453)) / 3.5
    #cntsin = 565 * 10**(-0.4*magi) / 3.5 * 1.7059e10

    #info
    print np.mean(mags[(mags > 18) & (mags < 23)] - magi[(mags > 18) & (mags < 23)])

    #write out x and y of those with large mag diff
    fh = open('largediff.txt', 'w')
    msk = (mags - magi) < -0.6
    for a, b in zip(x[msk], y[msk]):
        fh.write('%f %f\n' % (a, b))
    fh.close()

    #masks
    msk = ~(flag > 0)
    stars = (classification < 0.6) & (classification > -0.6)

    #plot magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('%.2f pixel cartesian matching' % rad)
    ax.plot(magi, mags - magi, 'bo', label='All')
    ax.plot(magi[msk], mags[msk] - magi[msk], 'rs', label='Flag=0')
    ax.plot(magi[stars], mags[stars] - magi[stars], 'g*', label='Stars')
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Delta Magnitude (extracted - input)')
    ax.plot([14,26], [0.0, 0.0], 'g--')
    ax.set_xlim(14, 26)
    ax.set_ylim(-2.0, 1.5)
    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Magnitudes%s.pdf' % str(rad).replace('.', ''))
    plt.close()

    #plot matching radius
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('%.2f pixel cartesian matching' % rad)
    ax.plot(machingrad, mags - magi, 'bo', label='All')
    ax.plot(machingrad[msk], mags[msk] - magi[msk], 'rs', label='Flag=0')
    ax.plot(machingrad[stars], mags[stars] - magi[stars], 'g*', label='Stars')
    ax.set_xlabel('Matching Distance [pixels]')
    ax.set_ylabel('Delta Magnitude (extracted - input)')
    ax.plot([0.0, rad+0.02], [0.0, 0.0], 'g--')
    ax.set_ylim(-2.0, 1.5)
    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Matching%s.pdf' % str(rad).replace('.', ''))
    plt.close()

    #plot counts
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(cntsin[msk], cnts[msk]/cntsin[msk], 'bo', label='Flag=0')
    ax.plot([1e2,1e6], [1.0, 1.0], 'g--')
    ax.set_xlabel('Input Counts')
    ax.set_ylabel('Count Ratio (extracted / input)')
    ax.set_ylim(0.5, 2.0)
    ax.set_xlim(1e2, 1e6)
    plt.legend(fancybox=True, shadow=True, numpoints=1)
    plt.savefig('Counts%s.pdf' % str(rad).replace('.', ''))
    plt.close()



if __name__ == '__main__':
    plot('0.5pix.csv', rad=0.5)
    plot('1.0pix.csv', rad=1.0)
    plot('1.5pix.csv', rad=1.5)
