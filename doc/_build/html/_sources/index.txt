Welcome to Euclid Visible InStrument (VIS) documentation
========================================================

:Author: Sami-Matias Niemi
:Contact: smn2@mssl.ucl.ac.uk
:version: 1.0


This Python package provides subpackages and methods related to the visible instrument (VIS) on board
the Euclid satellite. The subpackages include methods to e.g. generate object catalogues, simulate VIS images,
study radiation damage effects and fit new trap species, reduce and analyse data, and to include instrumental
characteristics such as readout noise and CTI to "pristine" images generated with e.g. GREAT10 photon
shooting code.


Creating Object Catalogs
========================

The *sources* subpackage contains a script to generate object catalogs with random x and y positions for
stars and galaxies. The magnitudes of stars and galaxies are drawn from input distributions that are
based on observations. As the number of stars depends on the galactic latitude, the script allows
the user to use three different (30, 60, 90 degrees) angles when generating the magnitude distribution for stars
(see the example plot below).

.. figure:: figs/Distributions.*
     :width: 800 px
     :align: center
     :figclass: align-center

     An example showing star and galaxy number counts in a source catalog suitable for VIS simulator.
     The solid lines show observations while the histograms show the distributions in the output catalogues.

For the Python code documentation, please see:

.. toctree::
   :maxdepth: 4

   sources


Generating Simulated Images
===========================

The *simulator* subpackage contains scripts to generate simulated VIS images. Two different methods
of generating mock images is provided. One which takes observed images (say from HST) as an input and
another in which analytical profiles are used for galaxies. The former code is custom made while the
latter relies hevily on IRAF's artdata package and mkobjects task.

The VIS reference simulator is the custom made with real observed galaxies as an input. The IRAF
based simulator can be used, for example, to train algorithms to derive elliptiticy of an object.
For more detailed documentation, please see:

.. toctree::
   :maxdepth: 4

   simulator

.. figure:: figs/simu.*
     :width: 800 px
     :align: center
     :figclass: align-center

     An example image generated with the reference simulator. The image shows a part of a single
     CCD.


Instrument Characteristics
==========================

The *postproc* subpackage contains methods related to either generating a CCD mosaics from simulated data
that is in quadrants like the VIS reference simulator produces or including instrument characteristics
to simulated images that contain only Poisson noise and background. For more detailed documentation
of the Python classes, please see:

.. toctree::
   :maxdepth: 4

   postproc


Data reduction
==============

The *reduction* subpackage contains a simple script to reduce VIS data. For more detailed documentation
of the classes, please see:

.. toctree::
   :maxdepth: 4

   reduction


Data Analysis
=============

The *analysis* subpackage contains classes and scripts related to data analysis. A simple source finder and shape
measuring classes are provided together with a wrapper to analyse reduced VIS data. For more detailed
documentation of the classes, please see:

.. toctree::
   :maxdepth: 4

   analysis

The *data* subfolder contains the supporting data, such as cosmic ray distributions, cosmetics maps,
flat fielding files, PSFs, and an example configuration file.


Charge Transfer Inefficiency
============================

The *fitting* subpackage contains a simple script that can be used to fit trap species so that the
Charge Transfer Inefficiency (CTI) trails forming behind charge injection lines agree with measured data.


Fortran code for CTI
--------------------

The *fortran* folder contains a CDM03 CTI model Fortran code. For speed the CDM03 model has been written in Fortran
because it contains several nested loops. One can use f2py to compile the code to a format that can be imported
from Python.


Supporting methods and files
============================

Objects
-------

A few postage stamps showing observed galaxies have been placed to the *objects* directory. These FITS files
can be used for, e.g., testing the shape measurement code.


Code
----

The *support* subpackage contains some support classes and methods related to generating log files and read in
data.


Photometric Accuracy
====================

The reference simulator code has been tested against photometric accuracy (without aperture correction). A
simulated image was generated with the reference simulator, sources were identified and photometry performed
using SExtractor, and finally the extracted magnitudes were compared against the input catalog. The following
figure shows that the photometric accuracy with realistic noise and the end-of-life radiation damage is
around 0.1 mag without aperture correction. Please note, however, that the derived magnitudes are based on a
single 565 second exposure. Because of this the faint galaxies have low signal-to-noise ratio and therefore
the derived magnitudes are inaccurate.

.. figure:: figs/Magnitudes15.*
     :width: 800 px
     :align: center
     :figclass: align-center

     Example showing the recovered photometry from a reference simulator image with realistic noise
     and end-of-life radiation damage, but without aperture correction. The offset is around 0.1mag.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
