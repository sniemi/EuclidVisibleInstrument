Welcome to VIS's documentation!
===============================

:Author: Sami-Matias Niemi
:Contact: smn2@mssl.ucl.ac.uk
:version: 0.1


This Python package provides subpackages and methods related to the VIS instrument on board the Euclid satellite.
The subpackages include methods to study CTI effects and fit new trap species, reduce and
analyse data, and to include instrumental characteristics such as readout noise and CTI to "pristine" images.


Generating simulated images
===========================

The *simulator* subpackage contains scripts to generate simulated VIS images, however, this package
is under heavy development and should therefore be treated with caution.

.. toctree::
   :maxdepth: 4

   simulator


Instrument characteristics
==========================

The *postproc* subpackage contains methods related to either generating a CCD mosaics from simulated data
that is in quadrants like the VIS reference simulator produces or including instrument characteristics
to simulated images that contain only Poisson noise and background. For more detailed documentation
of the classes, please see:

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
