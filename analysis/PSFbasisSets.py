"""
Generating Basis Sets
=====================

This script can be used to derive a basis set for point spread functions.

:requires: Scikit-learn
:requires: PyFITS
:requires: NumPy
:requires: SciPy
:requires: matplotlib
:requires: VISsim-Python

:version: 0.2

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import glob, math, sys, os
from optparse import OptionParser
import numpy as np
import pyfits as pf
from sklearn import decomposition
from support import files as fileIO
from support import logger as lg


def deriveBasisSetsPCA(data, cut, outfolder, components=10, whiten=False):
    """
    Derives a basis set from input data using Principal component analysis (PCA).
    Saves the basis sets to a FITS file for further processing.

    Information about PCA can be found from the scikit-learn website:
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

    :param data: input data from which the basis set are derived from. The input data must be an array of arrays.
                 Each array should describe an independent data set that has been flatted to 1D.
    :type data: ndarray
    :param cut: size of the cutout region that has been used
    :type cut: int
    :param outfolder: name of the output folder e.g. 'output'
    :type outfolder: str
    :param components: the number of basis set function components to derive
    :type components: int
    :param whiten: When True (False by default) the components_ vectors are divided by n_samples times
                   singular values to ensure uncorrelated outputs with unit component-wise variances.
    :type whiten: bool

    :return: PCA components
    """
    pca = decomposition.PCA(n_components=components, whiten=whiten)
    pca.fit(data)
    image = pca.components_

    #output the variance ratio
    print 'Variance Ratio:', pca.explained_variance_ratio_*100.

    #save each component to a FITS file
    for i, img in enumerate(image):
        image = img.reshape(2*cut, 2*cut)
        #to compare IDL results
        image = -image
        fileIO.writeFITS(image, outfolder + '/PCAbasis%03d.fits' % (i+1),  int=False)
    return image


def deriveBasisSetsKernelPCA(data, components=10):
    """
    Derives a basis set from input data using Kernel Principal component analysis (KPCA).
    Saves the basis sets to a FITS file for further processing.

    Information about KPCA can be found from the scikit-learn website:
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA

    :param data: input data from which the basis set are derived from. The input data must be an array of arrays.
                 Each array should describe an independent data set that has been flatted to 1D.
    :type data: ndarray
    :param components: the number of basis set function components to derive
    :type components: int

    :return: KPCA components
    """
    pca = decomposition.KernelPCA(n_components=components, kernel='rbf')
    pca.fit(data)
    return pca


def visualiseBasisSets(files, output, outputfolder):
    """
    Generate visualisation of the basis sets.

    :param files: a list of file names that should be visualised
    :return: None
    """
    #make 3D image
    fig = plt.figure(1, figsize=(18, 28))
    fig.subplots_adjust(hspace=0.1, wspace=0.001, left=0.10, bottom=0.095, top=0.975, right=0.98)

    #number of rows needed if two columns
    rows = math.ceil(len(files) / 2.)

    for i, file in enumerate(files):
        numb = int(file.split('basis')[-1].replace('.fits', ''))
        image = pf.getdata(file)

        stopy, stopx = image.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

        #add subplot
        rect = fig.add_subplot(rows, 2, i + 1, frame_on=False, visible=False).get_position()
        ax = Axes3D(fig, rect)
        plt.title('Basis Function $B_{%i}$' % numb)
        ax.plot_surface(X, Y, image, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-0.1, 0.1)

    plt.savefig(outputfolder + '/' + output.replace('.pdf', '3D.pdf'))
    plt.close()

    #individual 3D images
    for file in files:
        numb = int(file.split('basis')[-1].replace('.fits', ''))

        image = pf.getdata(file)

        stopy, stopx = image.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

        #make plot
        fig = plt.figure(figsize=(12, 12))
        ax = Axes3D(fig)
        plt.title('Basis Function $B_{%i}$' % numb)
        ax.plot_surface(X, Y, image, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-0.1, 0.1)
        plt.savefig(file.replace('.fits', '.pdf'))
        plt.close()

    #show the mean file
    if os.path.isfile(outputfolder + '/' + 'mean.fits'):
        image = pf.getdata(outputfolder + '/' + 'mean.fits')
        stopy, stopx = image.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))

        #make plot
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        plt.title('Average PSF')
        ax.plot_surface(X, Y, image, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.savefig(outputfolder + '/' + 'meanPSF.pdf')
        plt.close()

    #make 2D image
    fig = plt.figure(1, figsize=(18, 28))
    fig.subplots_adjust(hspace=0.1, wspace=0.001, left=0.10, bottom=0.095, top=0.975, right=0.98)

    #number of rows needed if two columns
    rows = math.ceil(len(files) / 2.)

    for i, file in enumerate(files):
        image = pf.getdata(file)

        #add subplot
        ax = fig.add_subplot(rows, 2, i + 1, frame_on=False)
        plt.title(file)
        plt.imshow(image, origin='lower')
        ax.xaxis.set_major_locator(NullLocator()) # remove ticks
        ax.yaxis.set_major_locator(NullLocator())

    plt.savefig(outputfolder + '/' + output.replace('.pdf', '2D.pdf'))
    plt.close()

    #generate a movie
    fig = plt.figure(2, figsize=(8, 8))
    ax = Axes3D(fig)

    ims = [] #to store the images for the movie
    for i, file in enumerate(files):
        image = pf.getdata(file)
        stopy, stopx = image.shape
        X, Y = np.meshgrid(np.arange(0, stopx, 1), np.arange(0, stopy, 1))
        #store image for the movie
        ims.append((ax.plot_surface(X, Y, image, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False),))

    ax.set_title('PCA Basis Sets')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.set_zlabel('')
    anime = animation.ArtistAnimation(fig, ims, interval=2000, blit=True)
    anime.save(outputfolder + '/' + output.replace('.pdf', '.mp4'), fps=0.5)


def processArgs(printHelp=False):
    """
    Processes command line arguments.
    """
    parser = OptionParser()

    parser.add_option('-i', '--input', dest='input',
        help="Input files from which the basis sets will be derived from (e.g. '*.fits')", metavar='string')
    parser.add_option('-o', '--output', dest='output',
        help="Name of the output directory, [default=./]", metavar='string')
    parser.add_option('-c', '--cutout', dest='cutout',
        help='Size of the cutout region [default=50]', metavar='int')
    parser.add_option('-b', '--basis', dest='basis',
        help='Number of basis sets to derive [default=20]', metavar='int')

    if printHelp:
        parser.print_help()
    else:
        return parser.parse_args()


if __name__ == '__main__':
    opts, args = processArgs()

    if opts.input is None:
        processArgs(True)
        sys.exit(8)

    if opts.output is None:
        opts.output = '.'

    log = lg.setUpLogger(opts.output + '/BasisSet.log')
    log.info('\n\nStarting to derive basis set functions...')

    if opts.cutout is None:
        opts.cutout = 50
    else:
        opts.cutout = int(opts.cutout)
    log.info('Cutout size being used is %i' % opts.cutout)

    if opts.basis is None:
        opts.basis = 20
    else:
        opts.basis = int(opts.basis)
    log.info('%i basis sets will be derived' % opts.basis)

    files = glob.glob(opts.input)
    all = []
    for file in files:
        print file
        log.info('Processing %s' % file)

        data = pf.getdata(file)
        data /= np.max(data)

        midy, midx = data.shape
        midx = math.floor(midx / 2.)
        midy = math.floor(midy / 2.)

        #take a smaller cutout and normalize the peak pixel to unity
        cutout = data[midy-opts.cutout:midy+opts.cutout, midx-opts.cutout:midx+opts.cutout]
        cutout /= np.max(cutout)

        #flatten to a 1D array and save the info
        all.append(np.ravel(cutout))

    #convert to numpy array
    all = np.asarray(all)

    #save the mean
    log.info('Saving the mean of the input files')
    mean = np.mean(all, axis=0).reshape(2*opts.cutout, 2*opts.cutout)
    mean /= np.max(mean)
    fileIO.writeFITS(mean, opts.output+'/mean.fits', int=False)

    #derive the basis sets and save the files
    log.info('Deriving basis sets')
    print 'Deriving basis sets with PCA'
    deriveBasisSetsPCA(all, opts.cutout, opts.output, components=opts.basis)

    log.info('Visualising the derived basis sets')
    visualiseBasisSets(glob.glob(opts.output+'/PCAbasis*.fits'), 'PCABasisSets.pdf', opts.output)

    log.info('All done...')