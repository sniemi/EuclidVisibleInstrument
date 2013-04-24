#!/usr/bin/env python
import cgi
import cgitb; cgitb.enable()
import sys
import pages

sys.stderr = sys.stdout

print "Content-type: text/html\n\n"

#load front page
pages.front()

response = cgi.FieldStorage()

if 'magnitude' in response.keys():
    pages.results(response)