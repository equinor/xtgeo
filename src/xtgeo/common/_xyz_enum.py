from enum import Enum, unique


# to be able to list all values in an easy manner e.g. _AttrName.list()
class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [c.value for c in cls]


# default names of special column names
@unique
class _AttrName(ExtendedEnum):
    XNAME = "X_UTME"
    YNAME = "Y_UTMN"
    ZNAME = "Z_TVDSS"
    PNAME = "POLY_ID"
    M_MD_NAME = "M_MDEPTH"
    Q_MD_NAME = "Q_MDEPTH"
    M_AZI_NAME = "M_AZI"
    Q_AZI_NAME = "Q_AZI"
    M_INCL_NAME = "M_INCL"
    Q_INCL_NAME = "Q_INCL"
    I_INDEX = "I_INDEX"
    J_INDEX = "J_INDEX"
    K_INDEX = "K_INDEX"
    R_HLEN_NAME = "R_HLEN"
    HNAME = "H_CUMLEN"
    DHNAME = "H_DELTALEN"
    TNAME = "T_CUMLEN"
    DTNAME = "T_DELTALEN"
    WELLNAME = "WELLNAME"
    TRAJECTORY = "TRAJECTORY"


@unique
class _AttrType(ExtendedEnum):
    """Enumerate type of attribute/log"""

    CONT = "CONT"
    DISC = "DISC"


@unique
class _XYZType(ExtendedEnum):
    """Enumerate type of context"""

    POINTS = "POINTS"
    POLYGONS = "POLYGONS"
    WELL = "WELL"
