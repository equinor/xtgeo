from enum import Enum, unique


# to be able to list all values in an easy manner e.g. _AttrName.list()
class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


# default names of special column names
@unique
class _AttrName(ExtendedEnum):
    XNAME = "X_UTME"
    YNAME = "Y_UTMN"
    ZNAME = "Z_TVDSS"
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


@unique
class _AttrType(ExtendedEnum):
    """Enumerate type of attribute/log"""

    CONT = "CONT"
    DISC = "DISC"


@unique
class _XYZType(ExtendedEnum):
    """Enumerate type of context"""

    POINTS = 1
    POLYGONS = 2  # ie. same here as PolyLines
    WELL = 3
