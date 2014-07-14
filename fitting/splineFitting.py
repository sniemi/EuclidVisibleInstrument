"""
Example how to spline B-spline to fake data.

:requires: Python 2.5 or later (no 3.x compatible)
:requires: NumPy
:requires: SciPy

TESTED:
Python 2.5.1
NumPy: 1.4.0.dev7576
SciPy: 0.7.1
matplotlib 1.0.svn

HISTORY:
Created on November 26, 2009

:version: 0.1: test release (SMN)

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import numpy as N
import scipy.signal as SS
import scipy.interpolate as I
import scipy.optimize as O
import pylab as P
import math

__author__ = 'Sami-Matias Niemi (s.niemi@ucl.ac.uk)'
__version__ = '0.1'

class SplineFitting:
    '''
    Fits a B-spline representation of 1-D curve.
    Uses Levenberg-Marquardt algorithm for minimizing
    the sum of squares.
    '''

    def __init__(self, xnodes, spline_order=3):
        self.xnodes = xnodes
        self.k = spline_order

    def _fakeData(self):
        x = N.linspace(1, 1024, 1024)
        y = self._gety(x, 2.5, 1.3, 0.5, 10)
        yn = y + 0.25 * N.random.normal(size=len(x))
        return x, yn

    def _gety(self, x, a, b, c, d):
        return a * N.exp(-b * x) + c * N.log(d * x ** 2)

    def fitfunc(self, x, ynodes):
        '''
        Function that is fitted.
        This can be changed to whatever function.
        Note that ynodes can then be a list of parameters.

        :return: 1-D B-spline value at each x.
        '''
        return I.splev(x, I.splrep(self.xnodes, ynodes, k=self.k))

    def errfunc(self, ynodes, x, y):
        '''
        Error function.

        :return: fit - ydata
        '''
        return self.fitfunc(x, ynodes) - y

    def doFit(self, ynodes, x, y):
        '''
        Return the point which minimizes the sum of squares of M (non-linear)
        equations in N unknowns given a starting estimate, x0, using a
        modification of the Levenberg-Marquardt algorithm.

        :return: fitted parameters, error/success message
        '''
        return O.leastsq(self.errfunc, ynodes, args=(x, y))

if __name__ == '__main__':
    '''
    Executed if ran from a command line.
    '''
    #Initializes the instance with dummy xnodes
    Spline = SplineFitting([0, ])

    #Makes some faked data
    x, y = Spline._fakeData()

    #Median filter the data
    medianFiltered = SS.medfilt(y, 7)

    #Spline nodes and initial guess for y positions from median filtered
    xnods = N.arange(0, 1050, 50)
    ynods = medianFiltered[xnods]
    #Updates dummy xnodes in Spline instance with read deal
    Spline.xnodes = xnods

    #Do the fitting
    fittedYnodes, success = Spline.doFit(ynods, x, y)

    #We can check how good the fit is
    chi2 = N.sum(N.power(Spline.errfunc(fittedYnodes, x, y), 2))
    dof = len(ynods) - 1.
    crit = (math.sqrt(2 * (dof - 1.)) + 1.635) ** 2 #only valid for large dofs
    print 'Chi**2 %6.2f vs %6.2f' % (chi2, crit)
    #note that there is also chisquare in scipy.stats which
    #could be used to evaluate p-values...

    #Lets plot the data for visual inspection
    fig = P.figure()

    left, width = 0.1, 0.8
    rect1 = [left, 0.3, width, 0.65]
    rect2 = [left, 0.1, width, 0.2]

    ax1 = fig.add_axes(rect2)  #left, bottom, width, height
    ax2 = fig.add_axes(rect1)

    ax2.plot(x, y, label='Noisy data')
    ax2.plot(x, medianFiltered, 'y-', label='Median Filtered', lw=2)
    ax2.plot(x, Spline.fitfunc(x, ynods), 'm-', label='Initial Spline', lw=2)
    ax2.plot(x, Spline.fitfunc(x, fittedYnodes), 'r-', label='Fitted Spline', lw=2)
    ax2.plot(xnods, ynods, 'go', label='Initial Spline nodes')
    ax2.plot(xnods, fittedYnodes, 'gs', label='Fitted Spline nodes')

    ax1.axhline(0)
    ax1.plot(x, SS.medfilt((y - Spline.fitfunc(x, ynods)), 55), 'm-', label='Initial guess residuals')
    ax1.plot(x, SS.medfilt((y - Spline.fitfunc(x, fittedYnodes)), 55), 'r-', label='Fitted residuals')

    ax1.set_xlim(0, 1000)
    ax2.set_xlim(0, 1000)

    ax2.set_xticklabels([])
    ax2.set_yticks(ax2.get_yticks()[1:])
    ax1.set_yticks(ax1.get_yticks()[::2])

    ax1.set_ylabel('Residuals')
    ax2.set_ylabel('Arbitrary Counts')
    ax1.set_xlabel('Pixels')

    try:
        #IRAFDEV has too old matplotlib...
        ax2.legend(numpoints=1, loc='best')
    except:
        ax2.legend(loc='best')
    P.savefig('SplineFitting.pdf')
