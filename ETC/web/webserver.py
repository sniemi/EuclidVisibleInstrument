"""
Basic html web server to run Euclid-VIS ETC.
"""
import BaseHTTPServer, SimpleHTTPServer, CGIHTTPServer

class myRequestHandler(CGIHTTPServer.CGIHTTPRequestHandler):
     def is_executable(self, path):
         return self.is_python(path)

if __name__ == '__main__':
     SimpleHTTPServer.test(myRequestHandler, BaseHTTPServer.HTTPServer)
