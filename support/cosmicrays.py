"""

:version: 0.1

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class cosmicrays():
    """

    """
    def __init__(self, log, image, crInfo=None, information=None):
        """

        :param log:
        :param image:
        :param crInfo:
        :param information:
        :return:
        """
        #setup logger
        self.log = log

        #image and size
        self.image = image.copy()
        self.ysize, self.xsize = self.image.shape

        #set up the information dictionary
        self.information = (dict(cosmicraylengths='data/cdf_cr_length.dat', cosmicraydistance='data/cdf_cr_total.dat',
                                 exptime=565))
        if information is not None:
            self.information.update(information)

        if crInfo is not None:
            self.cr = crInfo


    def _readCosmicrayInformation(self):
        self.log.info('Reading in cosmic ray information from %s and %s' % (self.information['cosmicraylengths'],
                                                                            self.information['cosmicraydistance']))
        #read in the information from the files
        crLengths = np.loadtxt(self.information['cosmicraylengths'])
        crDists = np.loadtxt(self.information['cosmicraydistance'])

        #set up the cosmic ray information dictionary
        self.cr = dict(cr_u=crLengths[:, 0], cr_cdf=crLengths[:, 1], cr_cdfn=np.shape(crLengths)[0],
                       cr_v=crDists[:, 0], cr_cde=crDists[:, 1], cr_cden=np.shape(crDists)[0])

        return self.cr


    def _cosmicRayIntercepts(self, lum, x0, y0, l, phi):
        """
        Derive cosmic ray streak intercept points.

        :param lum: luminosities of the cosmic ray tracks
        :param x0: central positions of the cosmic ray tracks in x-direction
        :param y0: central positions of the cosmic ray tracks in y-direction
        :param l: lengths of the cosmic ray tracks
        :param phi: orientation angles of the cosmic ray tracks

        :return: map
        :rtype: nd-array
        """
        #create empty array
        crImage = np.zeros((self.ysize, self.xsize), dtype=np.float64)

        #this is very slow way to do this
        for cosmics in xrange(0, len(l)):
            #delta x and y
            dx = l[cosmics] * np.cos(phi[cosmics])
            dy = l[cosmics] * np.sin(phi[cosmics])

            #pixels in x-direction
            ilo = np.floor(x0[cosmics] - l[cosmics])

            if ilo < 1.:
                ilo = 1

            ihi = 1 + np.floor(x0[cosmics] + l[cosmics])

            if ihi > self.xsize:
                ihi = self.xsize

            #pixels in y-directions
            jlo = np.floor(y0[cosmics] - l[cosmics])

            if jlo < 1.:
                jlo = 1

            jhi = 1 + np.floor(y0[cosmics] + l[cosmics])
            if jhi > self.ysize:
                jhi = self.ysize

            u = []
            x = []
            y = []

            n = 0  # count the intercepts

            #Compute X intercepts on the pixel grid
            if dx > 0.:
                for j in xrange(int(ilo), int(ihi)):
                    ok = (j - x0[cosmics]) / dx
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(j)
                        y.append(y0[cosmics] + ok * dy)

            #Compute Y intercepts on the pixel grid
            if dy > 0.:
                for j in xrange(int(jlo), int(jhi)):
                    ok = (j - y0[cosmics]) / dy
                    if np.abs(ok) <= 0.5:
                        n += 1
                        u.append(ok)
                        x.append(x0[cosmics] + ok * dx)
                        y.append(j)

            #check if no intercepts were found
            if n < 1:
                i = np.floor(x0[cosmics])
                j = np.floor(y0[cosmics])
                crImage[j, i] += lum[cosmics]

            #Find the arguments that sort the intersections along the track.
            u = np.asarray(u)
            x = np.asarray(x)
            y = np.asarray(y)

            args = np.argsort(u)

            u = u[args]
            x = x[args]
            y = y[args]

            #Decide which cell each interval traverses, and the path length.
            for i in xrange(1, n - 1):
                w = u[i + 1] - u[i]
                cx = 1 + np.floor((x[i + 1] + x[i]) / 2.0)
                cy = 1 + np.floor((y[i + 1] + y[i]) / 2.0)

                if cx >= 0 and cx < self.xsize and cy >= 0 and cy < self.ysize:
                    crImage[cy, cx] += (w * lum[cosmics])

        return crImage


    def _drawCosmicRays(self, limit=None):
        """
        Add cosmic rays to the arrays based on a power-law intensity distribution for tracks.

        Cosmic ray properties (such as location and angle) are chosen from random Uniform distribution.
        """
        #estimate the number of cosmics
        cr_n = self.xsize * self.ysize * 0.014 / 43.263316 * 2.
        #scale with exposure time, the above numbers are for the nominal 565s exposure
        cr_n *= (self.information['exptime'] / 565.0)

        #assume a power-law intensity distribution for tracks
        fit = dict(cr_lo=1.0e3, cr_hi=1.0e5, cr_q=2.0e0)
        fit['q1'] = 1.0e0 - fit['cr_q']
        fit['en1'] = fit['cr_lo'] ** fit['q1']
        fit['en2'] = fit['cr_hi'] ** fit['q1']

        #pseudo-random numbers taken from a uniform distribution between 0 and 1
        luck = np.random.rand(int(np.floor(cr_n)))

        #draw the length of the tracks
        if self.cr['cr_cdfn'] > 1:
            ius = InterpolatedUnivariateSpline(self.cr['cr_cdf'], self.cr['cr_u'])
            self.cr['cr_l'] = ius(luck)
        else:
            self.cr['cr_l'] = np.sqrt(1.0 - luck ** 2) / luck

        #draw the energy of the tracks
        if self.cr['cr_cden'] > 1:
            ius = InterpolatedUnivariateSpline(self.cr['cr_cde'], self.cr['cr_v'])
            self.cr['cr_e'] = ius(luck)
        else:
            self.cr['cr_e'] = (fit['en1'] + (fit['en2'] - fit['en1']) *
                               np.random.rand(int(np.floor(cr_n)))) ** (1.0 / fit['q1'])

        #Choose the properties such as positions and an angle from a random Uniform dist
        cr_x = self.xsize * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.ysize * np.random.rand(int(np.floor(cr_n)))
        cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

        #find the intercepts
        if limit is None:
            self.cosmicrayMap = self._cosmicRayIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)
            print 'Number of cosmic ray events:', len(self.cr['cr_e'])
        else:
            #limit to electron levels < limit
            msk = self.cr['cr_e'] < limit
            print 'Number of cosmic ray events: %i / %i' % (len(self.cr['cr_e'][msk]), int(np.floor(cr_n)))
            self.cosmicrayMap = self._cosmicRayIntercepts(self.cr['cr_e'][msk], cr_x[msk], cr_y[msk],
                                                          self.cr['cr_l'][msk], cr_phi[msk])

        #count the covering factor
        area_cr = np.count_nonzero(self.cosmicrayMap)
        text = 'The cosmic ray covering factor is %i pixels i.e. %.3f per cent' \
               % (area_cr, 100.*area_cr / (self.xsize*self.ysize))
        self.log.info(text)
        print text


    def _drawSingleEvent(self, limit=1000, cr_n=1):
        """
        Generate a single cosmic ray event and include it to a cosmic ray map (self.cosmicrayMap).

        :param limit: limiting energy for the cosmic ray event
        :type limit: float
        :param cr_n: number of cosmic ray events to include
        :type cr_n: int

        :return: None
        """
        #pseudo-random numbers taken from a uniform distribution between 0 and 1
        luck = np.random.rand(cr_n)

        #draw the length of the tracks
        ius = InterpolatedUnivariateSpline(self.cr['cr_cdf'], self.cr['cr_u'])
        self.cr['cr_l'] = np.asarray([ius(luck),])

        #set the energy directly to the limit
        self.cr['cr_e'] = np.asarray([limit,])

        #Choose the properties such as positions and an angle from a random Uniform dist
        cr_x = self.xsize * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.ysize * np.random.rand(int(np.floor(cr_n)))
        cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

        #find the intercepts
        self.cosmicrayMap = self._cosmicRayIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)

        #count the covering factor
        area_cr = np.count_nonzero(self.cosmicrayMap)
        text = 'The cosmic ray covering factor is %i pixels i.e. %.3f per cent' \
               % (area_cr, 100.*area_cr / (self.xsize*self.ysize))
        self.log.info(text)
        print text


    def _drawEventsToCoveringFactor(self, coveringFraction=1.4, limit=1000, verbose=False):
        """
        Generate cosmic ray events up to a covering fraction and include it to a cosmic ray map (self.cosmicrayMap).

        :param coveringFraction: covering fraction of cosmic rya events in per cent of total number of pixels
        :type coveringFraction: float
        :param limit: limiting energy for the cosmic ray event
        :type limit: float

        :return: None
        """
        self.cosmicrayMap = np.zeros((self.ysize, self.xsize))
        cr_n = 1
        covering = 0.0
        while covering < coveringFraction:
            #pseudo-random numbers taken from a uniform distribution between 0 and 1
            luck = np.random.rand(cr_n)

            #draw the length of the tracks
            ius = InterpolatedUnivariateSpline(self.cr['cr_cdf'], self.cr['cr_u'])
            self.cr['cr_l'] = np.asarray([ius(luck),])

            #set the energy directly to the limit
            self.cr['cr_e'] = np.asarray([limit,])

            #Choose the properties such as positions and an angle from a random Uniform dist
            cr_x = self.xsize * np.random.rand(int(np.floor(cr_n)))
            cr_y = self.ysize * np.random.rand(int(np.floor(cr_n)))
            cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

            #find the intercepts
            self.cosmicrayMap += self._cosmicRayIntercepts(self.cr['cr_e'], cr_x, cr_y, self.cr['cr_l'], cr_phi)

            #count the covering factor
            area_cr = np.count_nonzero(self.cosmicrayMap)
            covering = 100.*area_cr / (self.xsize*self.ysize)
            text = 'The cosmic ray covering factor is %i pixels i.e. %.3f per cent' \
                   % (area_cr, covering)
            self.log.info(text)
            if verbose: print text


    def addCosmicRays(self, limit=None):
        """
        Include cosmic rays to the image given.

        :return: image with cosmic rays
        :rtype: ndarray
        """
        self._drawCosmicRays(limit=limit)

        #paste cosmic rays
        self.image += self.cosmicrayMap

        return self.image


    def addSingleEvent(self, limit=None):
        """
        Include a single cosmic ray event to the image given.

        :return: image with cosmic rays
        :rtype: ndarray
        """
        self._drawSingleEvent(limit=limit)

        #paste cosmic rays
        self.image += self.cosmicrayMap

        return self.image


    def addUpToFraction(self, coveringFraction, limit=None, verbose=False):
        """

        :param coveringFraction: covering fraction of cosmic rya events in per cent of total number of pixels
        :type coveringFraction: float
        :param limit: limiting energy for the cosmic ray event
        :type limit: float

        :return: image with cosmic rays
        :rtype: ndarray
        """
        self._drawEventsToCoveringFactor(coveringFraction, limit=limit, verbose=verbose)

        #paste cosmic rays
        self.image += self.cosmicrayMap

        return self.image
