"""
Functions to find a centroid of an object, such as a PSF.

:requires: NumPy

:version: 0.1

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import numpy as np


def fwcentroid(image, checkbox=1, maxiterations=30, threshold=1e-5, halfwidth=35, verbose=False):
    """ Implement the Floating-window first moment centroid algorithm
        chosen for JWST target acquisition.

        See JWST-STScI-001117 and JWST-STScI-001134 for details.

        This code makes no attempt to vectorize or optimize for speed;
        it's pretty much just a straight verbatim implementation of the
        IDL-like pseudocode provided in JWST-STScI-001117


        Parameters
        ----------
        image : array_like
            image to centroid
        checkbox : int
            size of moving checkbox for initial peak pixel guess. Default 1
        halfwidth : int
            Half width of the centroid box size (less 1). Specify as a scalar, or a tuple Xhalfwidth, Yhalfwidth.
            Empirical tests suggest this parameter should be at *least* the PSF FWHM for convergence,
            preferably some small factor larger
        maxiterations : int
            Max number of loops. Default 30
        threshold : float
            Position threshold for convergence

        Returns
        --------
        (ycen, xcen) : float tuple
            Measured centroid position. Note that this is returned in Pythonic
            Y,X order for use as array indices, etc.


        -Marshall Perrin 2011-02-11
    """
    if hasattr(halfwidth, '__iter__'):
        XHW, YHW = halfwidth[0:2]
    else:
        XHW, YHW = halfwidth, halfwidth

    # Determine starting peak location
    if checkbox > 1:
        raise NotImplemented("Checkbox smoothing not done yet")
    else:
        # just use brightest pixel
        w = np.where(image == image.max())
        YPEAK, XPEAK = w[0][0], w[1][0]
        if verbose: print "Peak pixels are %d, %d" % (XPEAK, YPEAK)


    # Calculate centroid for first iteration
    SUM = 0.0
    XSUM = 0.0
    XSUM2 = 0.0
    XSUM3 = 0.0
    YSUM = 0.0
    YSUM2 = 0.0
    YSUM3 = 0.0
    CONVERGENCEFLAG = False

    for i in np.arange( 2*XHW+1)+ XPEAK-XHW :
        for j in np.arange( 2*YHW+1) +YPEAK-YHW :
            XLOC = i
            YLOC = j
            SUM += image[j,i]
            XSUM += XLOC * image[j,i]
            XSUM2 += XLOC**2 * image[j,i]
            XSUM3 += XLOC**3 * image[j,i]
            YSUM += YLOC * image[j,i]
            YSUM2 += YLOC**2 * image[j,i]
            YSUM3 += YLOC**3 * image[j,i]
    XCEN = XSUM / SUM
    XMOMENT2 = XSUM2 / SUM
    XMOMENT3 = XSUM3 / SUM
    YCEN = YSUM / SUM
    YMOMENT2 = YSUM2 / SUM
    YMOMENT3 = YSUM3 / SUM

    oldXCEN = XCEN
    oldYCEN = YCEN

    if verbose:
        print( "After initial calc, cent pos is  (%f, %f)" % (XCEN, YCEN))

    # Iteratively calculate centroid until solution converges,
    # use more neighboring pixels and apply weighting:
    for k in range(maxiterations):
        SUM = 0.0
        XSUM = 0.0
        XSUM2 = 0.0
        XSUM3 = 0.0
        YSUM = 0.0
        YSUM2 = 0.0
        YSUM3 = 0.0
        for i in np.arange( 2*(XHW+1)+1)+ int(oldXCEN)-(XHW+1) :
            for j in np.arange( 2*(YHW+1)+1) +int(oldYCEN)-(YHW+1) :
                #stop()
                #-- Calculate weights
                #Initialize weights to zero:
                XWEIGHT = 0
                YWEIGHT = 0
                #Adjust weights given distance from current centroid:
                XOFF = np.abs(i - oldXCEN)
                YOFF = np.abs(j - oldYCEN)
                #If within original centroid box, set the weight to one:
                if (XOFF <= XHW): XWEIGHT = 1
                elif (XOFF > XHW) and (XOFF < XHW+1):
                    #Else if on the border, then weight needs to be scaled:
                    XWEIGHT = XHW + 1 - XOFF
                #If within original centroid box, set the weight to one:
                if (YOFF <= YHW): YWEIGHT = 1
                elif (YOFF > YHW) and (YOFF < YHW+1):
                    #Else if on the border, then weight needs to be scaled:
                    YWEIGHT = YHW + 1 - YOFF
                WEIGHT = XWEIGHT * YWEIGHT

                #Centroid, second moment, and third moment calculations
                #XLOC = i - int(XCEN) + XHW + 2
                #YLOC = j - int(YCEN) + YHW + 2
                XLOC = i
                YLOC = j

                #print "pix (%d, %d) weight %f" % (i, j, WEIGHT)
                SUM = SUM + image[j,i] * WEIGHT
                XSUM = XSUM + XLOC * image[j,i] * WEIGHT
                XSUM2 = XSUM2 + XLOC**2 * image[j,i] * WEIGHT
                XSUM3 = XSUM3 + XLOC**3 * image[j,i] * WEIGHT
                YSUM = YSUM + YLOC * image[j,i] * WEIGHT
                YSUM2 = YSUM2 + YLOC**2 * image[j,i] * WEIGHT
                YSUM3 = YSUM3 + YLOC**3 * image[j,i] * WEIGHT
        XCEN = XSUM / SUM
        XMOMENT2 = XSUM2 / SUM
        XMOMENT3 = XSUM3 / SUM
        YCEN = YSUM / SUM
        YMOMENT2 = YSUM2 / SUM
        YMOMENT3 = YSUM3 / SUM

        if verbose:
            print( "After iter %d , cent pos is  (%f, %f)" % (k, XCEN, YCEN))

        #Check for convergence:
        if (np.abs(XCEN - oldXCEN) <= threshold and
            np.abs(YCEN - oldYCEN) <= threshold):
            CONVERGENCEFLAG = True
            break
        else:
            CONVERGENCEFLAG = False
            oldXCEN = XCEN
            oldYCEN = YCEN

    if not CONVERGENCEFLAG:
        print("Algorithm terminated at max iterations without convergence.")

    return  YCEN, XCEN


if __name__ == '__main__':
    import pyfits as pf
    import glob as g
    import math
    import scipy.ndimage.measurements
    from scipy.ndimage import interpolation
    from support import files as fileIO
    from analysis import shape
    from support import logger as lg

    #inputs
    cut = 500
    files = g.glob('TOL*')

    log = lg.setUpLogger('centroidPSF.log')
    log.info('\n\nStarting to derive centred cutouts...')

    all = []
    for i, file in enumerate(files):
        #load data
        data = pf.getdata(file)

        #find the centroid pixel with fwcentroid
        #midy, midx = fwcentroid(data)
        #midx += 1.
        #midy += 1.

        #scipy centre-of-mass
        #midy, midx = scipy.ndimage.measurements.center_of_mass(data)

        #peak location, take a small cutout around this region
        #midy, midx = scipy.ndimage.measurements.maximum_position(data)
        #data = data[midy-1000:midy+1001, midx-1000:midx+1001]

        #data centre pixels
        ceny, cenx = data.shape
        ceny /= 2
        cenx /= 2

        #second order moments centroid finding
        settings = dict(sampling=1./12., iterations=9, sigma=0.09)
        res = shape.shapeMeasurement(data, log, **settings).measureRefinedEllipticity()
        midx = res['centreX'] - 1
        midy = res['centreY'] - 1

        #interpolate to new location
        #note however that sinc interpolation should probably be used instead of spline...
        shiftx = -midx + cenx
        shifty = -midy + ceny
        cutout = interpolation.shift(data, [shifty, shiftx], order=3)

        #take a cutout to match size
        my, mx = cutout.shape
        mx /= 2
        my /= 2
        cutout = cutout[my - cut:my + cut, mx - cut:mx + cut]

        #write output
        print 'Image %i shift' %(i+1), shiftx, shifty, np.argmax(cutout), cutout.shape
        fileIO.writeFITS(cutout, 'cutout'+file.replace('.fits.gz', '.fits'), int=False)

        all.append(cutout)

    #calculate the average PSF
    all = np.asarray(all)
    mean = np.mean(all, axis=0)
    fileIO.writeFITS(mean, 'averagePSF.fits', int=False)

    #write out residuals
    for file in g.glob('cutout*.fits'):
        data = pf.getdata(file)
        residual = data - mean
        fileIO.writeFITS(residual, file.replace('cutout', 'residual'), int=False)
        print file, np.max(residual), np.min(residual), np.std(residual)
