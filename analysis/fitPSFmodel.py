"""
pyMC model to be imported.

Reads in the data that will be fitted,
assigns priors, generates a deterministic model,
and finally generates the probability/likelihood function.

:requires: NumPy
:requires: PyFITS
:requires: PyMC

:version: 0.1

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk
"""
import numpy as np
import pyfits as pf
import glob as g
import pymc

#load data
psf = np.asarray(np.ravel(pf.getdata('PSF800.fits')[400:601, 400:601]), dtype=np.float64)
mean = np.asarray(np.ravel(pf.getdata('mean.fits')[400:601, 400:601]), dtype=np.float64)
modes = sorted(g.glob('modes/PCA*.fits'))
modedata = [np.ravel(pf.getdata(file)[400:601, 400:601]) for file in modes]
data = [mean, ] + modedata
data = np.asarray(data, dtype=np.float64)

#set uniform priors
#these could actually be set based on the eigenvalues...
a0 = pymc.Uniform('a0', -1.0, 1.0, value=0.0)
a1 = pymc.Uniform('a1', -1.0, 1.0, value=0.0)
a2 = pymc.Uniform('a2', -1.0, 1.0, value=0.0)
a3 = pymc.Uniform('a3', -1.0, 1.0, value=0.0)
a4 = pymc.Uniform('a4', -1.0, 1.0, value=0.0)
a5 = pymc.Uniform('a5', -1.0, 1.0, value=0.0)
a6 = pymc.Uniform('a6', -1.0, 1.0, value=0.0)
a7 = pymc.Uniform('a7', -1.0, 1.0, value=0.0)
a8 = pymc.Uniform('a8', -1.0, 1.0, value=0.0)
a9 = pymc.Uniform('a9', -1.0, 1.0, value=0.0)
a10 = pymc.Uniform('a10', -1.0, 1.0, value=0.0)
a11 = pymc.Uniform('a11', -1.0, 1.0, value=0.0)
a12 = pymc.Uniform('a12', -1.0, 1.0, value=0.0)
a13 = pymc.Uniform('a13', -1.0, 1.0, value=0.0)
a14 = pymc.Uniform('a14', -1.0, 1.0, value=0.0)
a15 = pymc.Uniform('a15', -1.0, 1.0, value=0.0)
a16 = pymc.Uniform('a16', -1.0, 1.0, value=0.0)
a17 = pymc.Uniform('a17', -1.0, 1.0, value=0.0)
a18 = pymc.Uniform('a18', -1.0, 1.0, value=0.0)
a19 = pymc.Uniform('a19', -1.0, 1.0, value=0.0)
a20 = pymc.Uniform('a20', -1.0, 1.0, value=0.0)
sig = pymc.Uniform('sig', 0.0, 10.0, value=1.)

#model
@pymc.deterministic(plot=False)
def fit(x=psf, a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8, a9=a9, a10=a10,
        a11=a11, a12=a12, a13=a13, a14=a14, a15=a15, a16=a16, a17=a17, a18=a18, a19=a19, a20=a20):
    tmp = a0*x[0] + a1*x[1] + a2*x[2] + a3*x[3] + a4*x[4] + a5*x[5] + a6*x[6] + \
          a7*x[7] + a8*x[8] + a9*x[9] + a10*x[10] + a11*x[11] + a12*x[12] + a13*x[13] +\
          a14*x[14] + a15*x[15] + a16*x[16] + a17*x[17] + a18*x[18] + a19*x[19] + a20*x[20]
    return tmp

#likelihood
y = pymc.Normal('y', mu=fit, tau=1.0/sig**2, value=data, observed=True)
