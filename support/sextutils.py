"""
Utilities for parseing SExtractor files.

H. Ferguson - revised 4/23/03 to promote ints to floats if a value
with a decimal point appears somewhere in the column originally thought
to be integers

version::

    v2.1 - fails gracefully when the catalog has no sources
    v3.0 - added gettypes to return column types
          - create new column names when they are not explictly in the header
    v4.0 - added gettypes to return column types
    v4.1 - uses numarray by default
    v4.2 - delete attributed 'l' (input lines from catalog) before returning
    v4.3 - 1/11/06Added less-offensive alias se_catalog() == sextractor()
    v4.4hf-1/21/06 Fixed bug in creating extra column names when last is a vector
    v4.4vl - V. Laidler added new methods:
               __len__ returns number of objects in catalog
               __iter__ returns the index of the next row in the catalog
               line(self,i) returns a constructed string containing the ith line
               buildheader returns a constructed header from the hdict
            Added new attribute self.header: contains the header as read in
               from the catalog.
            Lines that start with '#' but are not followed by an integer are
               now assumed to be comment lines, which are added to the
               header but otherwise skipped.
    v4.5 - V. Laidler removed Numeric dependence
    v4.6 - V. Laidler converted to numpy
    v5.0 - 7/5/07 Numpy conversion
    v6.0 - V. Laidler: added rw_catalog class, reworked internals to avoid
               column name clashes
    v7.0 - S.-M. Niemi: some modifications
    v7.1 - S.-M. Niemi: now supports string columns
"""

__version__ = '7.1'
__author = 'Henry C. Ferguson, STScI'

import string
import numpy as N
import os, sys

class se_catalog(object):
    """
    Read a SExtractor-style catalog.
    Usage: c=se_catalog(catalog,readfile=True,preserve_case=False)
    Will read the catalog and return an object c, whose attributes are
    arrays containing the data. For example, c.mag_auto contains the
    mag_auto values.
    Arguments:
    catalog -- The input SExtractor catalog.
    readfile -- True means read the data. False means return the
    object without reading the data. The lines from the catalog
    are returned as a list of ascii strings c.l. Useful if you want
    to do some special parsing of some sort.
    preserve_case -- default (False) converts column names to lower case

    The input catalog MUST have a header with the SExtractor format:
    # 1 ID comment
    # 2 ALPHA_J200 another comment
    That is, first column is the comment symbol #, second column is
    the column number, third column is the column name, and the rest
    of the line is a comment. SExtractor allows "vectors" to be identified
    only by the first column...e.g.
    # 12 FLUX_APER
    # 20 FLUXERR_APER
    the missing columns are all aperture fluxes through different
    apertures. These will be read into attributes:
    c.flux_aper   # The first one
    c.flux_aper_1 # the second one, and so on

    The case of aperture radii is a bit nasty, since these only
    appear in the SExtractor configuration file. Use parseconfig()
    to read that file.
    """

    def __init__(self, cfile, readfile=True, preserve_case=False):
        (self._d, self._l, self._ncolumns, self._header) = initcat(cfile,
                                                                   preserve_case=preserve_case)
        self._fname = cfile
        if readfile:
            self._colentries = range(len(self._l))
            for i in range(len(self._l)):
                self._colentries[i] = self._l[i].split()
                #SMN: added
            if min(self._d.values()) == 0:
                for key in self._d: self._d[key] += 1
            self.gettypes()
            for k in self._d.keys():
                contents = getcolvalues(self._d[k],
                                        self._type[k],
                                        self._colentries)
                colname = self._okname(k)
                setattr(self, colname, contents)
            delattr(self, '_l')

    def __len__(self):
        return len(self._colentries)

    def __iter__(self):
        return range(len(self._colentries)).__iter__()

    def _okname(self, k):
        try:
            #Munge column name if it conflicts
            test = self.__getattribute__(k)
            newkey = 'c_' + k
            print "--Column '%s' changed to '%s' to avoid conflicts" % (k, newkey)
            self._d[newkey] = self._d[k]
            del self._d[k]
            return newkey
        except AttributeError:
            return k

    def line(self, i):
        """
        Returns an assembled line of this catalog suitable for writing.
        Except it doesn't really, if we modified the individual columns
        """
        ans = '    '.join(self._colentries[i]) + '\n'
        return ans

    def buildheader(self):
        """
        Reconstruct the header from the header dictionary.
        This might be useful if only a few columns were selected
        from the file; otherwise just use the 'header' attribute.
        """

        lines = {}
        for k in self._d:
            lines[self._d[k]] = '#   %d   %s' % (self._d[k], k.upper())
            #sort the new keys
        nkeys = lines.keys()
        nkeys.sort()
        #join them together with newlines
        ans = ''
        for k in nkeys:
            ans = ans + "%s\n" % lines[k]
        return ans

    def getcol(self, col, offset=0):
        column = self._d[col]
        return getcol(column + offset, self._l)

    def getcols(self, *args):
        ret = []
        for i in range(len(args)):
            ret = ret + [getcol(self._d[args[i]], self._l)]
        return ret

    def gettypes(self):
        self._type = {}
        for k in self._d.keys():
            #this line may require changing
            if len(self._l) > 1000000:
                every = 500
            elif len(self._l) > 10000:
                every = 20
            else:
                every = 10
            ret = getcol(self._d[k], self._l[::every])
            t = type(ret)
            if t == type(N.array([1])):
                if ret.dtype.char == 'i' or ret.dtype.char == 'l':
                    t = type(1)
                elif ret.dtype.char == 'd':
                    t = type(1.e99)
            else:
                t = type('string')
                #print k, t
            self._type[k] = t


class sextractor(se_catalog): # Just an alias for class se_catalog
    """ Read SExtractor catalog...just an alias for se_catalog """
    pass


class rw_catalog(se_catalog):
    """ Extend the se_catalog class to support adding new columns,
    and writing out the new version."""

    def __init__(self, fname):
        self._modflag = False #this flag will be set by add_column routines
        self._fname = fname
        self._colnames = []
        se_catalog.__init__(self, fname,
                            readfile=True, preserve_case=False)
        coldict = invert_dict(self._d)
        for k in coldict:
            self._colnames.append(coldict[k])

    def addcolumn(self, input_colname, coldata):
        """ coldata must be a 1d numarray of the correct length"""
        if len(coldata) != len(self):
            raise ValueError, "Column length must match catalog length"

        colname = self._okname(input_colname)

        #Most of the bookkeeping is the same as for an empty column
        self.addemptycolumn(colname, coldata.dtype)

        #and then we reset the column to contain the actual data
        setattr(self, colname, coldata)


    def addemptycolumn(self, input_colname, coltype):
        """ Defines a new column & updates all the bookkeeping, but
        does not actually fill in the data. """
        colname = self._okname(input_colname)

        setattr(self, colname, N.zeros((len(self),), coltype))
        self._modflag = True
        self._type[colname] = coltype

        #Looks strange here because we count columns from 1 but
        #Python counts them from 0
        self._ncolumns += 1
        self._d[colname] = self._ncolumns
        self._colnames.append(colname)
        self._header += '#  %d  %s\n' % (self._ncolumns, colname)

    def line(self, rownum):
        """ Construct a new line as to be printed out """
        if not self._modflag:
            return se_catalog.line(self, rownum)
        else:
            linelist = []
            for c in self._colnames:
                col = getattr(self, c)
                linelist.append(str(col[rownum]))
            line = '    '.join(linelist) + '\n'
            return line

    def writeto(self, outname, clobber=False):
        if not clobber:
            if os.path.isfile(outname):
                raise ValueError, """File already exists.
                   Use .writeto(fname, clobber=True) to overwrite. """

        out = open(outname, 'w')

        out.write(self._header)
        for k in range(len(self)):
            out.write(self.line(k))
        out.close()

    def printme(self):
        """ Like writeto, but for sys.stdout """
        sys.stdout.write(self._header)
        for k in range(len(self)):
            sys.stdout.write(self.line(k))


def invert_dict(d):
    """ Generate a new dictionary with the key/value relationship inverted """
    newd = {}
    for k in d:
        newd[d[k]] = k
    return newd


def parseconfig_se(cfile):
    """ parseconfig -- read a SExtractor .sex file and return a dictionary
      of options & values. Comments are ignored.
    """
    cdict = {}
    f = open(cfile, 'r')
    lines = f.readlines()
    for l in lines:
        a = string.split(l)
        if len(a) > 0:
            if a[0][0] != '#':
                maxi = len(a)
                for i in range(1, len(a)):
                    if a[i][0] == '#':
                        maxi = i
                        break
                        # Turn comma-separated lists into python lists
                entry = []
                for e in a[1:maxi]:
                    if string.find(e, ','):
                        entry = entry + string.split(e, ',')
                    else:
                        entry = entry + [e]
                cdict[a[0]] = entry
    return cdict


def initcat(catfile, preserve_case=False):
    """ parseheader -- reads the header of a SExtractor catalog file and
        returns a dictionary of parameter names and column numbers.
        Also returns a list of lines containing the data.
    """
    hdict = {}
    header = []
    f = open(catfile, 'r')
    lines = f.readlines()
    f.close()
    first = 1
    firstdata = 0
    i = 0
    previous_column = 0
    previous_key = ""
    for l in lines:
        if l.startswith('#'): #this is a header line
            header.append(l)
            a = (l.replace('#', '# ')).split() #Guard against "#10 colname"
            try:
                col = int(a[1])
                # If the column numbers skip, create new column names for
                # columns not named explicitly in the header
                if col != previous_column + 1:
                    for c in range(previous_column + 1, col):
                        column_name = previous_key + "_%d" % (c - previous_column)
                        hdict[column_name] = c
                        # Update this column in the dictionary
                if (preserve_case):
                    column_name = a[2]
                else:
                    column_name = a[2].lower()
                    hdict[column_name] = col
                firstdata = i + 1
                previous_column = col
                previous_key = column_name
            except (ValueError, IndexError):
                #it's a comment line with no column number,
                #or an entirely blank comment line: skip
                pass

        else:  # This is where the data start
            if previous_column == 0:
                raise ValueError("No valid header found in %s" % catfile)

            a = string.split(l)
            if len(a) > 0:
                if first:
                    firstdata = i
                    first = 0
                    # Check if there are extra columns
                if len(a) > previous_column:
                # If so, add keys for the last entry
                    for c in range(previous_column + 1, len(a)):
                        column_name = previous_key + "_%d" % (c - previous_column)
                        if (preserve_case):
                            hdict[column_name] = c
                        else:
                            hdict[column_name] = c.lower()
                ncolumns = len(a)
        i = i + 1
    return(hdict, lines[firstdata:], ncolumns, ''.join(header))


def getcol(col, lines):
    """ Get a column from a SExtractor catalog. Determine the type
(integer, float, string) and return either an array of that
type (Int32, Float64) or a list of strings """
    i = col - 1               # Columns start at 1, arrays start at 0
    nlines = len(lines)
    if len(lines) == 0:
        values = N.array([])
        return values
    a = string.split(lines[0])
    if string.find(a[i], '.') < 0:
        try:
            x = int(a[i])
        except:
            values = range(nlines)
            getstrings(col, lines, values)
        else:
            values = N.zeros((nlines), N.int32)
            if type(getints(col, lines, values)) == type(-1):
                values = N.zeros((nlines), N.float64)
                getfloats(col, lines, values)
    else:
        try:
            x = float(a[i])
        except:
            values = range(nlines)
            getstrings(col, lines, values)
        else:
            values = N.zeros((nlines), N.float64)
            getfloats(col, lines, values)
    return values


def getcolvalues(col, coltype, colentries, colzero=False):
    """ Get a column from a SExtractor catalog. Determine the type
(integer, float, string) and return either an array of that
type (Int32, Float64) or a list of strings """
    i = col - 1               # Columns start at 1, arrays start at 0
    nlines = len(colentries)
    if len(colentries) == 0:
        values = N.array([])
        return values

    if coltype == type('string'):
        values = range(nlines)
        for j in range(nlines):
            values[j] = colentries[j][i]

    if coltype == type(1.0):    # Convert floats
        values = N.zeros((nlines), N.float64)
        for j in range(nlines):
            values[j] = float(colentries[j][i])

    if coltype == type(1):    # Convert Ints
        values = N.zeros((nlines), N.int32)
        for j in range(nlines):
            values[j] = int(colentries[j][i])
    return values


def getstrings(col, lines, values):
    n = 0
    for l in lines:
        a = string.split(l)
        values[n] = a[col - 1]
        n = n + 1


def getints(col, lines, values):
    n = 0
    for l in lines:
        a = string.split(l)
        if string.find(a[col - 1], '.') > 0:
            return -1
        else:
            values[n] = int(a[col - 1])
        n = n + 1
    return values


def getfloats(col, lines, values):
    n = 0
    for l in lines:
        a = string.split(l)
        values[n] = float(a[col - 1])
        n = n + 1


def getcols(d, l, *args):
    """ Get multiple columns from SExtractor list using getcol() """
    ret = []
    for i in range(len(args)):
        ret = ret + [getcol(d[args[i]], l)]
    return ret


def writeheader(fh, colnames):
    """ Write an SExtractor-style header to an open file handle.

    :param fh: file handle
    :type fh: file

    :param colnames: list of column names
    :type colnames: list

    :todo: add space checking to colnames
    :todo: permit passing a filename?
    :todo: handle comments
    """
    for i in range(len(colnames)):
        fh.write('#   %d   %s\n' % (i + 1, colnames[i]))
