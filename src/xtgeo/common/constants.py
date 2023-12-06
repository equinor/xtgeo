"""Module for basic XTGeo constants"""

# align with cxtgeo libxtg.h!
from xtgeo import _cxtgeo

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

MAXKEYWORDS = _cxtgeo.MAXKEYWORDS  # maximum keywords for ECL and ROFF scanning
MAXDATES = _cxtgeo.MAXDATES  # maximum keywords for ECL scanning

# for XYZ data, restricted to float32 and int32
UNDEF_CONT = UNDEF
UNDEF_DISC = UNDEF_INT

# INT_MIN is the lowest 32 bit int value and is applied in some RMSAPI settings where
# nan (for floats) cannot be applied
INT_MIN = -2147483648
