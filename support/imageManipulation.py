"""
Basic image manipulation algorithms such as Gaussian smoothing and free rebinning.

:requires: NumPy
:requires: SciPy

:author: Sami-Matias Niemi
:contact: sniemi@unc.edu

:version: 0.1
"""
import numpy as np
import scipy.signal as s
import scipy


def frebin(image, nsout, nlout=1, total=False):
    """
    Shrink or expand the size of an array an arbitary amount using interpolation.
    Conserves flux by ensuring that each input pixel is equally represented
    in the output array.

    .. todo:: one could do the binning faster if old and new outputs are modulo 0

    .. Note:: modelled after the IDL code frebin.pro, so this may not be the fastest solution

    :param image: input image, 1-d or 2-d ndarray
    :param nsout: number of samples in the output image, numeric scalar
    :param nlout: number of lines (ydir) in the output image, numeric scalar
    :param total: Use of the total conserves surface flux. If True, the output pixels
                  will be the sum of pixels within the appropriate box of the input image.
                  Otherwise they will be the average.

    :return: binned array
    :rtype: ndarray
    """
    shape = image.shape
    if nlout != 1:
        nl = shape[0]
        ns = shape[1]
    else:
        nl = nlout
        ns = shape[0]

    sbox = ns / float(nsout)
    lbox = nl / float(nlout)

    ns1 = ns - 1
    nl1 = nl - 1

    if nl == 1:
        #1D case
        result = np.zeros(nsout)
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            #add pixel valeus from istart to istop and subtract fraction pixel
            #from istart to rstart and fraction pixel from rstop to istop
            result[i] = np.sum(image[istart:istop + 1]) - frac1 * image[istart] - frac2 * image[istop]

        if total:
            return result
        else:
            return result / (float(sbox) * lbox)
    else:
        #2D case, first bin in second dimension
        temp = np.zeros((nlout, ns))
        result = np.zeros((nsout, nlout))

        #first lines
        for i in range(nlout):
            rstart = i * lbox
            istart = int(rstart)
            rstop = rstart + lbox

            if int(rstop) < nl1:
                istop = int(rstop)
            else:
                istop = nl1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                temp[i, :] = (1.0 - frac1 - frac2) * image[istart, :]
            else:
                temp[i, :] = np.sum(image[istart:istop + 1, :], axis=0) -\
                             frac1 * image[istart, :] - frac2 * image[istop, :]

        temp = np.transpose(temp)

        #then samples
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
            else:
                result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0) -\
                               frac1 * temp[istart, :] - frac2 * temp[istop, :]

        if total:
            return np.transpose(result)
        else:
            return np.transpose(result) / (sbox * lbox)


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    """
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    """
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print "[congrid] dimensions error. "\
              "This routine currently only support "\
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1)\
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1)\
                           * (base + ofs) - ofs)
            # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + range(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n",\
        "Currently only \'neighbour\', \'nearest\',\'linear\',",\
        "and \'spline\' are supported."
        return None


def rebinFactor(a, newshape):
    """
    """
    assert len(a.shape) == len(newshape)
    assert not sometrue(mod(a.shape, newshape))
    slices = [slice(None, None, old / new) for old, new in zip(a.shape, newshape)]
    return a[slices]


def rebin(a, *args):
    """
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args)
    evList = ['a.reshape('] +\
             ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] +\
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)] +\
             ['/factor[%d]' % i for i in range(lenShape)]
    #print ''.join(evList)
    return eval(''.join(evList))


def gaussianKernel(size, sizey=None):
    """
    Returns a normalized 2D gauss kernel array for convolutions.
    """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size + 1, -sizey:sizey + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(sizey)))
    return g / g.sum()


def blurImage(im, n, ny=None):
    """
    blurs the image by convolving with a gaussian kernel of typical
    size np. The optional keyword argument ny allows for a different
    size in the y direction.
    """
    g = gaussianKernel(n, sizey=ny)
    improc = s.convolve(im, g, mode='valid')
    return improc
