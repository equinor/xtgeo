from enum import Enum, unique


@unique
class TypeOfGrid(Enum):
    """
    A Grid has three possible data layout formats, UNSTRUCTURED, CORNER_POINT,
    BLOCK_CENTER and COMPOSITE (meaning combination of the two former). Only
    CORNER_POINT layout is supported by XTGeo.
    """

    COMPOSITE = 0
    CORNER_POINT = 1
    UNSTRUCTURED = 2
    BLOCK_CENTER = 3

    @classmethod
    def alternate_code(cls, code):
        """Converts from alternate code to TypeOfGrid member.

        weirdly, type of grid sometimes (For instance init's INTHEAD and
        FILEHEAD) have an alternate integer code for type of grid.
        """
        if code == 0:
            type_of_grid = cls.CORNER_POINT
        elif code == 1:
            type_of_grid = cls.UNSTRUCTURED
        elif code == 2:
            type_of_grid = cls.COMPOSITE
        elif code == 3:
            type_of_grid = cls.BLOCK_CENTER
        else:
            raise ValueError(f"Unknown grid type {code}")
        return type_of_grid

    @property
    def alternate_value(self):
        """Inverse of alternate_code."""
        alternate_value = 0
        if self == TypeOfGrid.CORNER_POINT:
            alternate_value = 0
        elif self == TypeOfGrid.UNSTRUCTURED:
            alternate_value = 1
        elif self == TypeOfGrid.COMPOSITE:
            alternate_value = 2
        elif self == TypeOfGrid.BLOCK_CENTER:
            alternate_value = 3
        else:
            raise ValueError(f"Unknown grid type {self}")
        return alternate_value


@unique
class UnitSystem(Enum):
    METRIC = 1
    FIELD = 2
    LAB = 3


@unique
class Phases(Enum):
    E300_GENERIC = 0
    OIL = 1
    WATER = 2
    OIL_WATER = 3
    GAS = 4
    OIL_GAS = 5
    GAS_WATER = 6
    OIL_WATER_GAS = 7


@unique
class Simulator(Enum):
    ECLIPSE_100 = 100
    ECLIPSE_300 = 300
    ECLIPSE_300_THERMAL = 500
    INTERSECT = 700
    FRONTSIM = 800
