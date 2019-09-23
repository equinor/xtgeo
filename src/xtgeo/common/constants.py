# -*- coding: utf-8 -*-
"""Module for basic XTGeo constants"""

import xtgeo.cxtgeo.cxtgeo as _cxtgeo

UNDEF = -999
UNDEF_LIMIT = -999
UNDEF_INT = -999
UNDEF_INT_LIMIT = -999
VERYLARGEPOSITIVE = -999
VERYLARGENEGATIVE = -999


try:
    UNDEF = _cxtgeo.UNDEF
    UNDEF_LIMIT = _cxtgeo.UNDEF_LIMIT
    UNDEF_INT = _cxtgeo.UNDEF_INT
    UNDEF_INT_LIMIT = _cxtgeo.UNDEF_INT_LIMIT
    VERYLARGENEGATIVE = _cxtgeo.VERYLARGENEGATIVE
    VERYLARGEPOSITIVE = _cxtgeo.VERYLARGEPOSITIVE
except AttributeError:
    print("Dummy settings")
