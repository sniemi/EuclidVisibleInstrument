from distutils.core import setup

setup(
    name='VISsim',
    version='1.0',
    author='Sami-Matias Niemi',
    author_email='smn2@mssl.ucl.ac.uk',
    packages=['CTI', 'analysis', 'fitting', 'fortran', 'plotting', 'postproc', 'reduction',
              'simulator', 'support', 'sources'],
    license='LICENSE.txt',
    url='http://',
    long_description=open('./doc/index.rst').read(),
)