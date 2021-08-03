"""
Datastructure for the contents of grdecl files.

The grdecl file format is not specified in a strict manner
but in the most general sense it is a file that can be included
into the GRID section of a eclipse input file.

However, it is nearly impossible to support such a file format completely,
instead we narrow it down to the following subset of keywords:

    * COORD
    * ZCORN
    * ACTNUM
    * MAPAXES
    * GRIDUNIT
    * SPECGRID
    * GDORIENT

And ignore ECHO and NOECHO keywords. see _grid_format for the details
of how these keywords are layed out in a text file, and see GrdeclGrid
for how the grid geometry is interpreted from these keywords.
"""
from dataclasses import InitVar, astuple, dataclass, fields
from enum import Enum, auto, unique
from typing import Optional, Tuple

import numpy as np
from ecl_data_io import Format, lazy_read, write

from ._ecl_grid import EclGrid
from ._grdecl_format import match_keyword, open_grdecl


@unique
class GridRelative(Enum):
    """GridRelative is the second value given GRIDUNIT keyword.

    MAP means map relative units, while
    leaving it blank means relative to the origin given by the
    MAPAXES keyword.
    """

    MAP = auto()
    ORIGIN = auto()

    def to_grdecl(self):
        if self == GridRelative.MAP:
            return "MAP"
        else:
            return ""

    def to_bgrdecl(self):
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, unit_string):
        if match_keyword(unit_string, "MAP"):
            return cls.MAP
        else:
            return cls.ORIGIN

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@unique
class Order(Enum):
    """Either increasing or decreasing.

    Used for the grdecl keywords INC and DEC
    respectively.
    """

    INCREASING = auto()
    DECREASING = auto()

    def to_grdecl(self):
        return str(self.name)[0:3]

    def to_bgrdecl(self):
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, order_string):
        if match_keyword(order_string, "INC"):
            return cls.INCREASING
        if match_keyword(order_string, "DEC"):
            return cls.DECREASING

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@unique
class Orientation(Enum):
    """Either up or down, for the grdecl keywords UP and DOWN."""

    UP = auto()
    DOWN = auto()

    def to_grdecl(self):
        return self.name

    def to_bgrdecl(self):
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, orientation_string):
        if match_keyword(orientation_string, "UP"):
            return cls.UP
        if match_keyword(orientation_string, "DOWN"):
            return cls.DOWN
        raise ValueError(f"Unknown orientation string {orientation_string}")

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@unique
class Handedness(Enum):
    """The handedness of an orientation.

    Eiter left handed or right handed.  Used for the grdecl keywords LEFT and
    RIGHT.
    """

    LEFT = auto()
    RIGHT = auto()

    def to_grdecl(self):
        return self.name

    def to_bgrdecl(self):
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, orientation_string):
        if match_keyword(orientation_string, "LEFT"):
            return cls.LEFT
        if match_keyword(orientation_string, "RIGHT"):
            return cls.RIGHT
        raise ValueError(f"Unknown handedness string {orientation_string}")

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@unique
class CoordinateType(Enum):
    """The coordinate system type given in the SPECGRID keyword.

    This is given by either T or F in the last value of SPECGRID, meaning
    either cylindrical or cartesian coordinates respectively.
    """

    CARTESIAN = auto()
    CYLINDRICAL = auto()

    def to_grdecl(self):
        if self == CoordinateType.CARTESIAN:
            return "F"
        else:
            return "T"

    def to_bgrdecl(self):
        if self == CoordinateType.CARTESIAN:
            return 0
        else:
            return 1

    @classmethod
    def from_bgrdecl(cls, coord_value):
        if coord_value == 0:
            return cls.CARTESIAN
        else:
            return cls.CYLINDRICAL

    @classmethod
    def from_grdecl(cls, coord_string):
        if match_keyword(coord_string, "F"):
            return cls.CARTESIAN
        if match_keyword(coord_string, "T"):
            return cls.CYLINDRICAL
        raise ValueError(f"Unknown coordinate type {coord_string}")


@dataclass
class GrdeclKeyword:
    """A general grdecl keyword.

    Gives a general implementation of to/from grdecl which simply recurses on
    fields.
    """

    def to_grdecl(self):
        """Convert the keyword to list of grdecl keyword values.
        Returns:
            list of values of the given keyword. ie. The
            keyword read from "SPECGRID 1 1 1 F" should return
            [1,1,1,CoordinateType.CYLINDRICAL]
        """
        return [value.to_grdecl() for value in astuple(self)]

    def to_bgrdecl(self):
        return [value.to_bgrdecl() for value in astuple(self)]

    @classmethod
    def from_bgrdecl(cls, values):
        object_types = [f.type for f in fields(cls)]
        return cls(*[typ.from_bgrdecl(val) for val, typ in zip(values, object_types)])

    @classmethod
    def from_grdecl(cls, values):
        """Convert list of grdecl keyword values to a keyword.
        Args:
            values(list): list of values given after the keyword in
                the grdecl file.
        Returns:
            A GrdeclKeyword constructed from the given values.
        """
        object_types = [f.type for f in fields(cls)]
        return cls(*[typ.from_grdecl(val) for val, typ in zip(values, object_types)])


@dataclass
class MapAxes(GrdeclKeyword):
    """The mapaxes keyword gives the local coordinate system of the map.

    The map coordinate system is given by a point on the y line, the origin and
    a point on the x line. ie. The usual coordinate system is given by "MAPAXES
    0 1 0 0 1 0 /" where the two first values is a point on the y line, the
    middle two values is the origin, and the last two values is a point on the
    x line.
    """

    y_line: Tuple[float, float] = (0.0, 1.0)
    origin: Tuple[float, float] = (0.0, 0.0)
    x_line: Tuple[float, float] = (1.0, 0.0)

    def to_grdecl(self):
        return list(self.y_line) + list(self.origin) + list(self.x_line)

    def to_bgrdecl(self):
        return self.to_grdecl()

    @classmethod
    def from_bgrdecl(cls, values):
        return cls.from_grdecl(values)

    @classmethod
    def from_grdecl(cls, values):
        if len(values) != 6:
            raise ValueError("MAPAXES must contain 6 values")
        return cls(
            (float(values[0]), float(values[1])),
            (float(values[2]), float(values[3])),
            (float(values[4]), float(values[5])),
        )


@dataclass
class GdOrient(GrdeclKeyword):
    """The GDORIENT keyword gives the orientation of the grid.

    The three first values is either increasing or decreasing
    depending on whether the corresponding dimension has increasing
    or decreasing coordinates. Then comes the direction of the z dimension,
    and finally the handedness of the orientation. Defaults to
    "GDORIENT INC INC INC DOWN RIGHT /".
    """

    i_order: Order = Order.INCREASING
    j_order: Order = Order.INCREASING
    k_order: Order = Order.INCREASING
    z_direction: Orientation = Orientation.DOWN
    handedness: Handedness = Handedness.RIGHT


@dataclass
class SpecGrid(GrdeclKeyword):
    """The SPECGRID keyword gives the size of the grid.

    The 3 first values is the number of cells in each dimension
    of the grid. The next is the number of reservoirs in the file and
    the last is the type of coordinates, see CoordinateType.

    example:
        "SPECGRID 10 10 10 1 T /" meaning 10x10x10 grid with 1 reservoir
        and cylindrical coordinates.

    """

    ndivix: int = 1
    ndiviy: int = 1
    ndiviz: int = 1
    numres: int = 1
    coordinate_type: CoordinateType = CoordinateType.CARTESIAN

    def to_grdecl(self):
        return [
            self.ndivix,
            self.ndiviy,
            self.ndiviz,
            self.numres,
            self.coordinate_type.to_grdecl(),
        ]

    def to_bgrdecl(self):
        return [
            self.ndivix,
            self.ndiviy,
            self.ndiviz,
            self.numres,
            self.coordinate_type.to_bgrdecl(),
        ]

    @classmethod
    def from_bgrdecl(cls, values):
        ivalues = [int(v) for v in values[:4]]
        if len(values) < 5:
            return cls(*ivalues)
        if len(values) == 5:
            return cls(*ivalues, CoordinateType.from_bgrdecl(values[-1]))
        raise ValueError("SPECGRID should have at most 5 values")

    @classmethod
    def from_grdecl(cls, values):
        ivalues = [int(v) for v in values[:4]]
        if len(values) < 5:
            return cls(*ivalues)
        if len(values) == 5:
            return cls(*ivalues, CoordinateType.from_grdecl(values[-1]))
        raise ValueError("SPECGRID should have at most 5 values")


@dataclass
class GridUnit(GrdeclKeyword):
    """
    Defines the units used for grid dimensions. The first value
    is a string describing the units used, defaults to METRES, known
    accepted other units are FIELD and LAB. The last value describes
    whether the measurements are relative to the map or to the
    origin of MAPAXES.
    """

    unit: str = "METRES"
    grid_relative: GridRelative = GridRelative.ORIGIN

    def to_grdecl(self):
        return [
            self.unit,
            self.grid_relative.to_grdecl(),
        ]

    def to_bgrdecl(self):
        return [
            self.unit.ljust(8),
            self.grid_relative.to_bgrdecl(),
        ]

    @classmethod
    def from_bgrdecl(cls, values):
        if isinstance(values[0], bytes):
            return cls.from_grdecl([v.decode("ascii") for v in values])
        return cls.from_grdecl(values)

    @classmethod
    def from_grdecl(cls, values):
        if len(values) == 1:
            return cls(values[0].rstrip())
        if len(values) == 2:
            return cls(
                values[0].rstrip(),
                GridRelative.MAP
                if match_keyword(values[1], "MAP")
                else GridRelative.ORIGIN,
            )
        raise ValueError("GridUnit record must contain either 1 or 2 values")


@dataclass
class GrdeclGrid(EclGrid):
    """
    The main keywords that describe a grdecl grid is COORD, ZCORN and ACTNUM
    and are described in xtgeo.grid3d._ecl_grid.
    The remaining fields (SPECGRID, MAPAXES, MAPUNITS, GRIDUNIT, GDORIENT)
    describe units, orientation and dimensions, see corresponding dataclasses.
    The number of cells in each direction is described in the SPECGRID keyword.
    """

    coord: np.ndarray
    zcorn: np.ndarray
    specgrid: SpecGrid = None
    actnum: Optional[np.ndarray] = None
    mapaxes: Optional[MapAxes] = None
    mapunits: Optional[str] = None
    gridunit: Optional[GridUnit] = None
    gdorient: Optional[GdOrient] = None
    size: InitVar[Tuple[int, int, int]] = None

    def __post_init__(self, size):
        if not size and not self.specgrid:
            raise ValueError(
                "Either size or specgrid has to be given when constructing GrdeclGrid"
            )

        if size and not self.specgrid:
            self.specgrid = SpecGrid(*size)
        if (
            self.specgrid
            and size
            and not (
                size
                == (
                    self.specgrid.ndivix,
                    self.specgrid.ndiviy,
                    self.specgrid.ndiviz,
                )
            )
        ):
            raise ValueError(
                "GrdeclGrid given both specgrid and size with conflicting values"
            )

    def __eq__(self, other):
        if not isinstance(other, GrdeclGrid):
            return False
        return (
            self.specgrid == other.specgrid
            and self.mapaxes == other.mapaxes
            and self.mapunits == other.mapunits
            and self.gridunit == other.gridunit
            and self.gdorient == other.gdorient
            and (
                (self.actnum is None and other.actnum is None)
                or np.array_equal(self.actnum, other.actnum)
            )
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    @property
    def dimensions(self):
        return (self.specgrid.ndivix, self.specgrid.ndiviy, self.specgrid.ndiviz)

    @classmethod
    def from_file(cls, filename, fileformat="grdecl"):
        """
        write the grdeclgrid to a file.
        :param filename: path to file to write.
        :param fileformat: Either "grdecl" or "bgrdecl" to
            indicate binary or ascii format.
        """
        if fileformat == "grdecl":
            return cls._from_grdecl_file(filename)
        if fileformat == "bgrdecl":
            return cls._from_bgrdecl_file(filename)
        raise ValueError(b"Unknown grdecl file format {fileformat}")

    @classmethod
    def _from_bgrdecl_file(cls, filename, fileformat=None):
        keyword_factories = {
            "COORD": lambda x: np.array(x, dtype=np.float32),
            "ZCORN": lambda x: np.array(x, dtype=np.float32),
            "ACTNUM": lambda x: np.array(x, dtype=np.int32),
            "MAPAXES": MapAxes.from_bgrdecl,
            "MAPUNITS": lambda x: x,
            "GRIDUNIT": GridUnit.from_bgrdecl,
            "SPECGRID": SpecGrid.from_bgrdecl,
            "GDORIENT": GdOrient.from_bgrdecl,
        }
        results = {}
        for entry in lazy_read(filename, fileformat=fileformat):
            if len(results) == len(keyword_factories):
                break
            kw = entry.read_keyword().rstrip()
            if kw in results:
                raise ValueError(f"Duplicate keyword {kw} in {filename}")
            try:
                factory = keyword_factories[kw]
            except KeyError as e:
                raise ValueError(f"Unknown grdecl keyword {kw}") from e
            results[kw.lower()] = factory(entry.read_array())
        return cls(**results)

    @classmethod
    def _from_grdecl_file(cls, filename):
        keyword_factories = {
            "COORD": lambda x: np.array(x, dtype=np.float32),
            "ZCORN": lambda x: np.array(x, dtype=np.float32),
            "ACTNUM": lambda x: np.array(x, dtype=np.int32),
            "MAPAXES": MapAxes.from_grdecl,
            "MAPUNITS": lambda x: x,
            "GRIDUNIT": GridUnit.from_grdecl,
            "SPECGRID": SpecGrid.from_grdecl,
            "GDORIENT": GdOrient.from_grdecl,
        }
        results = {}
        with open_grdecl(
            filename,
            keywords=["MAPAXES", "MAPUNITS", "GRIDUNIT", "SPECGRID", "GDORIENT"],
            simple_keywords=["COORD", "ZCORN", "ACTNUM"],
            max_len=8,
            ignore=["ECHO", "NOECHO"],
        ) as keyword_generator:
            for kw, values in keyword_generator:
                if len(results) == len(keyword_factories):
                    break
                if kw in results:
                    raise ValueError(f"Duplicate keyword {kw} in {filename}")
                try:
                    factory = keyword_factories[kw]
                except KeyError as e:
                    raise ValueError(f"Unknown grdecl keyword {kw}") from e
                results[kw.lower()] = factory(values)
        return cls(**results)

    def to_file(self, filename, fileformat="grdecl"):
        """
        write the grdeclgrid to a file.
        :param filename: path to file to write.
        :param fileformat: Either "grdecl" or "bgrdecl" to
            indicate binary or ascii format.
        """
        if fileformat == "grdecl":
            return self._to_grdecl_file(filename)
        if fileformat == "bgrdecl":
            return self._to_bgrdecl_file(filename)
        raise ValueError(b"Unknown grdecl file format {fileformat}")

    def _to_grdecl_file(self, filename):
        with open(filename, "w") as filestream:
            keywords = [
                ("SPECGRID", self.specgrid.to_grdecl()),
                ("MAPAXES", self.mapaxes.to_grdecl() if self.mapaxes else None),
                ("MAPUNITS", [self.mapunits] if self.mapunits else None),
                ("GRIDUNIT", self.gridunit.to_grdecl() if self.gridunit else None),
                ("GDORIENT", self.gdorient.to_grdecl() if self.gdorient else None),
                ("COORD", self.coord),
                ("ZCORN", self.zcorn),
                ("ACTNUM", self.actnum),
            ]
            for kw, values in keywords:
                if values is None:
                    continue
                filestream.write(kw)
                if values is not None:
                    filestream.write("\n")
                    for value in values:
                        filestream.write(" ")
                        filestream.write(str(value))
                filestream.write("\n /\n")

    def _to_bgrdecl_file(self, filename, fileformat=Format.UNFORMATTED):
        contents = filter(
            lambda x: x[1] is not None,
            [
                ("SPECGRID", self.specgrid.to_bgrdecl()),
                ("MAPAXES ", self.mapaxes.to_bgrdecl() if self.mapaxes else None),
                ("MAPUNITS", [self.mapunits] if self.mapunits else None),
                ("GRIDUNIT", self.gridunit.to_bgrdecl() if self.gridunit else None),
                ("GDORIENT", self.gdorient.to_bgrdecl() if self.gdorient else None),
                ("COORD   ", self.coord),
                ("ZCORN   ", self.zcorn),
                ("ACTNUM  ", self.actnum),
            ],
        )
        write(
            filename,
            contents,
        )

    def _check_xtgeo_compatible(self):
        if self.specgrid.coordinate_type == CoordinateType.CYLINDRICAL:
            raise NotImplementedError(
                "Xtgeo does not currently support cylindrical coordinate systems"
            )
        if self.specgrid.numres != 1:
            raise NotImplementedError(
                "Xtgeo does not currently support multiple reservoirs"
            )
        if self.gridunit and self.gridunit.grid_relative == GridRelative.MAP:
            raise NotImplementedError(
                "Xtgeo does not currently support conversion of map relative grid units"
            )
