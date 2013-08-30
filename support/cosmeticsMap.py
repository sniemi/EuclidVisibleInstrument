"""
A simple script to generate a random location cosmetics map.

:requires: NumPy

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import numpy as np


def generateCatalogue(deadpixels=280, hotpixels=5, xsize=4096, ysize=4132, output='cosmetics.dat'):
    """
    Generates a cosmetics catalogue with dead and hot pixels with random locations.
    The default numbers have been taken from MSSL/Euclid/TR/12003 Issue 2 Draft b.

    Results are written to a CSV file.

    :param deadpixels: number of dead (QE=0) pixels [280]
    :type deadpixels: int
    :param hotpixels: number of hot (value between 40k and 250ke) pixels [5]
    :type hotpixels: int
    :param xsize: x coordinate values between 0 and xsize [4096]
    :type xsize: int
    :param ysize: y coordinate values between 0 and ysize [4132]
    :type ysize: int
    :param output: name of the output file
    :type output: str

    :return: None
    """
    xcoords = np.random.random((deadpixels+hotpixels))*xsize
    ycoords = np.random.random((deadpixels+hotpixels))*ysize

    hotvalues = np.random.randint(40000, 250000, hotpixels)

    fh = open(output, 'w')
    fh.write('#Cosmetics map created with random positions\n')
    fh.write('#Number of hot and dead pixels from MSSL/Euclid/TR/12003 Issue 2 Draft b\n')

    hot = 0
    for x, y in zip(xcoords, ycoords):
        if hot < hotpixels:
            txt = '%f,%f,%i' % (x, y, hotvalues[hot])
            hot += 1
        else:
            txt = '%f,%f,0' % (x, y)

        fh.write(txt + '\n')

    fh.close()


if __name__ == '__main__':
    generateCatalogue()