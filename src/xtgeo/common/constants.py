# -*- coding: utf-8 -*-
"""Module for basic XTGeo constants"""

# align with cxtgeo libxtg.h!
import xtgeo.cxtgeo._cxtgeo as cx

M_PI = 3.14159265358979323846
PI = M_PI
PIHALF = 1.57079632679489661923

UNDEF = 10e32
UNDEF_LIMIT = 9.9e32
UNDEF_INT = 2000000000
UNDEF_INT_LIMIT = 1999999999

VERYLARGEPOSITIVE = 10e30
VERYLARGENEGATIVE = -10e30

UNDEF_MAP_IRAPB = 1e30
UNDEF_MAP_IRAPA = 9999900.0000

MAXKEYWORDS = cx.MAXKEYWORDS  # maximum keywords for ECL and ROFF scanning
MAXDATES = cx.MAXDATES  # maximum keywords for ECL scanning
