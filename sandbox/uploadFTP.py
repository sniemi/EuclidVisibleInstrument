"""
This module contains functions related to FTP file transfer protocol and file compression.

:Author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk

:version: 0.2
"""
import ftplib, os, glob, tarfile


def compress(filename):
    """
    Compress a file to a gzipped tar ball.

    :param filename: name of the file to compress
    :type filename: str

    :return: None
    """
    tar = tarfile.open(filename+'.tar.gz', 'w:gz')
    tar.add(filename)
    tar.close()


def upload(ftp, file):
    """
    Upload files to a server using FTP protocol.

    :param ftp: instance to the FTP server
    :type ftp: ftplib.FTP instance
    :param file: name of the file to be uploaded
    :type file: str

    :return: None
    """
    ext = os.path.splitext(file)[1]
    if ext in (".txt", ".htm", ".html"):
        ftp.storlines("STOR " + file, open(file))
    else:
        ftp.storbinary("STOR " + file, open(file, "rb"), 1024)


if __name__ == '__main__':
    #compress files
    files = glob.glob('CCD*.fits')
    for file in files:
        compress(file)

    #find all gzipped tar balls and sort
    uploads = glob.glob('*.fits.tar.gz')
    uploads.sort()

    #open connection to the FTP server
    ftp = ftplib.FTP('ftpix.iap.fr')
    ftp.login('login','password')

    #get existing files, note however, that this does
    #not check the file size, so files that were
    #uploaded only partially will not be uploaded
    #again.
    existing = ftp.nlst()

    #print some output
    print 'Found %i files from the server, these will not be uploaded again...' % len(existing)

    #upload not existing files
    i = 0
    for file in uploads:
        if not file in existing:
            print 'Uploading %s' % file
            upload(ftp, file)
            i += 1

    print '%i files were uploaded...' %i