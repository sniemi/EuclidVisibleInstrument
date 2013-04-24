"""
HTML Pages
"""
from support.VISinstrumentModel import VISinformation
from ETC import ETC


def front():
    print "<html>"
    print "<head><center><b>VIS Exposure Time Calculator</b></center></head>"
    print "<body style='background-color:Moccasin;''>"
    print "<p> A simple Exposure Time Calculator for Euclid Visible Imager.</p> "

    # Radio buttons
    #print "<br><b> Input Parameters </b><br>"
    #print "<form method='POST' action = 'control2.py'>"
    #print "<input type='radio' name='Output' value='Model' checked/> Model 1 <br>"
    #print "<input type='radio' name='Output' value='Measure' checked /> Model 2 <br>"

    print '<br><b> Input Parameters </b><br>'
    print '''<br>Zodiacal light level: <form action="">
             <select name="zodi">
             <option value="low">Low</option>
             <option value="med">Med</option>
             <option value="high">High</option>
             </select>'''

    print '''<br>Object type: <form action="">
             <select name="object">
             <option value="galaxy">Galaxy</option>
             <option value="star">Star</option>
             </select>'''

    print "<br>"
    print "AB Magnitude: <input type='text' name='magnitude' size = '4' value = '%s'/> AB <br>" % 24.5
    print "Exposure Time: <input type='text' name='exptime' size = '4' value = '%s'/> seconds <br>" % 565
    print "Exposures: <input type='text' name='exposures' size = '4' value = '%s'/><br>" % 3

    # Submit reponses and move to results
    print "<br>"
    print "<input type='submit' value='Run Calculation' />"
    print "<br>"

    print "</form>"
    print "</body>"


def results(response):
    info = VISinformation()

    zodi = response.getvalue('zodi')
    magnitude = float(response.getvalue('magnitude'))
    exptime = float(response.getvalue('exptime'))
    obj = response.getvalue('object')
    exposures = int(response.getvalue('exposures'))
    ETC.SNRproptoPeak(info, exptime=exptime, exposures=exposures, server=True)

    if obj == 'galaxy':
        galaxy = True
    else:
        galaxy = False

    exp = ETC.exposureTime(info, magnitude, exposures=exposures, galaxy=galaxy, fudge=1.0)
    limit = ETC.limitingMagnitude(info, exp=exptime, galaxy=galaxy, exposures=exposures, fudge=1.0)
    snr = ETC.SNR(info, magnitude=magnitude, exptime=exptime, exposures=exposures, galaxy=galaxy)

    print "<html>"
    print "<head> <center> <b>Results</b> </center></head> "
    print "<body style='background-color:Moccasin;''>"

    print '<br><b> Results </b><br>'
    print 'Exposure time required to reach SNR=10.0 given a %.2f magnitude %s is %.1f seconds ' \
          'if combining %i exposures<br><br>' % (magnitude, obj, exp, exposures)
    print 'SNR=%f for %.2fmag %s if exposure time is %.2f seconds<br><br>' % (snr, magnitude, obj, exptime*exposures)
    print 'Limiting magnitude of a %s for %.2f second exposure is %.2f<br><br>' % (obj, exptime, limit)

    print "<p><img src='../delete.png'/></p>"

    print "</body>"

