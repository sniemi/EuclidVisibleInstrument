"""
http://astroquery.readthedocs.org/en/latest/index.html
"""
from astroquery import besancon
import astropy.io.ascii


if __name__ == '__main__':
    handler = besancon.Besancon(email='s.niemi@ucl.ac.uk')

    data = handler.query(glon=0., glat=30., extinction=0.7, area=1., absmag_limits=(-7.0, 30.0),
                         mag_limits={'V': (1., 30.), 'R': (1., 30.), 'I': (1., 30.)},
                         retrieve_file=True)

    astropy.io.ascii.write(data, 'output.data')