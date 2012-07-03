from distutils.core import setup

setup(
    name='VISsim',
    version='1.0',
    author='Sami-Matias Niemi',
    author_email='smn2@mssl.ucl.ac.uk',
    packages=['CTI, analysis, fitting, plotting, postproc, reduction, simulator, sources, support'],
    license='LICENSE.txt',
    url='http://',
    long_description=open('./doc/index.rst').read(),
)