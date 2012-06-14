from distutils.core import setup

setup(
    name='VISsim',
    version='0.1',
    author='Sami-Matias Niemi',
    author_email='smn2@mssl.ucl.ac.uk',
    packages=['analysis, fitting, postproc, reduction, simulator, support'],
    license='LICENSE.txt',
    url='http://',
    long_description=open('./doc/index.rst').read(),
)