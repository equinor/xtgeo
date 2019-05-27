# -*- coding: utf-8 -*-
"""Module for basic XTGeo constants"""

import xtgeo.cxtgeo

UNDEF = -999
UNDEF_LIMIT = -999
UNDEF_INT = -999
UNDEF_INT_LIMIT = -999
VERYLARGEPOSITIVE = -999
VERYLARGENEGATIVE = -999


try:
    UNDEF = xtgeo.cxtgeo.cxtgeo.UNDEF
    UNDEF_LIMIT = xtgeo.cxtgeo.cxtgeo.UNDEF_LIMIT
    UNDEF_INT = xtgeo.cxtgeo.cxtgeo.UNDEF_INT
    UNDEF_INT_LIMIT = xtgeo.cxtgeo.cxtgeo.UNDEF_INT_LIMIT
    VERYLARGENEGATIVE = xtgeo.cxtgeo.cxtgeo.VERYLARGENEGATIVE
    VERYLARGEPOSITIVE = xtgeo.cxtgeo.cxtgeo.VERYLARGEPOSITIVE
except AttributeError:
    print("Dummy settings")
