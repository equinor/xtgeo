from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, fields
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ._grdecl_format import match_keyword


@unique
class GridRelative(Enum):
    """GridRelative is the second value given GRIDUNIT keyword.

    MAP means map relative units, while
    leaving it blank means relative to the origin given by the
    MAPAXES keyword.
    """

    MAP = auto()
    ORIGIN = auto()

    def to_grdecl(self) -> str:
        if self == GridRelative.MAP:
            return "MAP"
        else:
            return ""

    def to_bgrdecl(self) -> str:
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, unit_string: str):
        if match_keyword(unit_string, "MAP"):
            return cls.MAP
        else:
            return cls.ORIGIN

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@dataclass
class GrdeclKeyword:
    """An abstract grdecl keyword.

    Gives a general implementation of to/from grdecl which recurses on
    fields. Ie. a dataclass such as

    >>> @dataclass
    >>> class MyKeyword(GrdeclKeyword):
    >>>      field1 : A
    >>>      field2 : B

    will have a to_grdec  method that returns

    >>> [self.field1.to_grdecl(), self.field2.to_grdecl]

    Similarly from_grdecl will call fields from_grdecl
    to construct the object

    >>> MyKeyword(A.from_grdecl(values[0]), B.from_grdecl(values[1]))
    """

    def to_grdecl(self) -> List[Any]:
        """Convert the keyword to list of grdecl keyword values.
        Returns:
            list of values of the given keyword. ie. The
            keyword read from "SPECGRID 1 1 1 F" should return
            [1,1,1,CoordinateType.CYLINDRICAL]
        """
        return [value.to_grdecl() for value in astuple(self)]

    def to_bgrdecl(self) -> List[Any]:
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


@unique
class Order(Enum):
    """Either increasing or decreasing.

    Used for the grdecl keywords INC and DEC
    respectively.
    """

    INCREASING = auto()
    DECREASING = auto()

    def to_grdecl(self) -> str:
        return str(self.name)[0:3]

    def to_bgrdecl(self) -> str:
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, order_string):
        if match_keyword(order_string, "INC"):
            return cls.INCREASING
        if match_keyword(order_string, "DEC"):
            return cls.DECREASING

    @classmethod
    def from_bgrdecl(cls, unit_string: Union[bytes, str]):
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

    def to_grdecl(self) -> str:
        return self.name

    def to_bgrdecl(self) -> str:
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, orientation_string: str):
        if match_keyword(orientation_string, "LEFT"):
            return cls.LEFT
        if match_keyword(orientation_string, "RIGHT"):
            return cls.RIGHT
        raise ValueError(f"Unknown handedness string {orientation_string}")

    @classmethod
    def from_bgrdecl(cls, unit_string: Union[bytes, str]):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


@unique
class Orientation(Enum):
    """Either up or down, for the grdecl keywords UP and DOWN."""

    UP = auto()
    DOWN = auto()

    def to_grdecl(self) -> str:
        return self.name

    def to_bgrdecl(self) -> str:
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, orientation_string: str):
        if match_keyword(orientation_string, "UP"):
            return cls.UP
        if match_keyword(orientation_string, "DOWN"):
            return cls.DOWN
        raise ValueError(f"Unknown orientation string {orientation_string}")

    @classmethod
    def from_bgrdecl(cls, unit_string: Union[bytes, str]):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


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
class GridUnit(GrdeclKeyword):
    """Defines the units used for grid dimensions.

    The first value is a string describing the units used, defaults to METRES,
    known accepted other units are FIELD and LAB. The last value describes
    whether the measurements are relative to the map or to the origin of
    MAPAXES.
    """

    unit: str = "METRES"
    grid_relative: GridRelative = GridRelative.ORIGIN

    def to_grdecl(self) -> List[str]:
        return [
            self.unit,
            self.grid_relative.to_grdecl(),
        ]

    def to_bgrdecl(self) -> List[str]:
        return [
            self.unit.ljust(8),
            self.grid_relative.to_bgrdecl(),
        ]

    @classmethod
    def from_bgrdecl(cls, values: Union[bytes, str]):
        if isinstance(values[0], bytes):
            return cls.from_grdecl([v.decode("ascii") for v in values])
        return cls.from_grdecl(values)

    @classmethod
    def from_grdecl(cls, values: List[str]):
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

    def to_grdecl(self) -> List[float]:
        return list(self.y_line) + list(self.origin) + list(self.x_line)

    def to_bgrdecl(self) -> List[float]:
        return self.to_grdecl()

    @classmethod
    def from_bgrdecl(cls, values: List[Union[float, str]]):
        return cls.from_grdecl(values)

    @classmethod
    def from_grdecl(cls, values: List[Union[float, str]]):
        if len(values) != 6:
            raise ValueError("MAPAXES must contain 6 values")
        return cls(
            (float(values[0]), float(values[1])),
            (float(values[2]), float(values[3])),
            (float(values[4]), float(values[5])),
        )


@unique
class CoordinateType(Enum):
    """The coordinate system type given in the SPECGRID keyword.

    This is given by either T or F in the last value of SPECGRID, meaning
    either cylindrical or cartesian coordinates respectively.
    """

    CARTESIAN = auto()
    CYLINDRICAL = auto()

    def to_grdecl(self) -> str:
        if self == CoordinateType.CARTESIAN:
            return "F"
        else:
            return "T"

    def to_bgrdecl(self) -> int:
        if self == CoordinateType.CARTESIAN:
            return 0
        else:
            return 1

    @classmethod
    def from_bgrdecl(cls, coord_value: int):
        if coord_value == 0:
            return cls.CARTESIAN
        else:
            return cls.CYLINDRICAL

    @classmethod
    def from_grdecl(cls, coord_string: str):
        if match_keyword(coord_string, "F"):
            return cls.CARTESIAN
        if match_keyword(coord_string, "T"):
            return cls.CYLINDRICAL
        raise ValueError(f"Unknown coordinate type {coord_string}")


@dataclass
class EclGrid(ABC):
    """
    The main keywords that describe a grdecl grid is COORD, ZCORN and ACTNUM.

    The grid is made up of nx*ny*nz cells in three corresponding dimensions.
    The number of cells in each direction is described in the SPECGRID keyword.

    The values in COORD, ZCORN and ACTNUM are stored flattened in F-order and
    have dimensions (nx+1,ny+1,6), (nx,2,ny,2,nz,2), and (nx,ny,nz) respectively.

    COORD and ZCORN descibe a corner point geometry for the grid. There is a
    straight line from the bottom to the top of the grid on which the corners
    of each grid lie. COORD describe the top and bottom (x,y,z) values of these
    corner lines, hence, it contains six floats for each corner line.

    ZCORN has 8 values for each grid, which describes the z-value (height) at
    which that cells corners intersect with the corresponding corner line. The
    order of corners is  "left" before "right" in the second dimension of
    ZCORN, "near"  before "far" in the fourth dimension , and "upper" before
    "bottom" in the last dimension. Note that this orientation assumes,
    increasing first dimension as to the "right", increasing second dimension
    towards "far", and increasing third dimension as towards "bottom".

    The topology is such that, assuming no gaps between cells, the (i,j,k)th
    cell and the (i+1,j+1,k+1)th cell share the upper near left corner of the
    (i+1,j+1,k+1)th cell which is the lower far right corner of the (i,j,k)th
    cell.

    ACTNUM describes the active status of each cell. 0 means inactive, 1
    means active, 2 means rock volume only, 3 means pore volume only.
    """

    coord: np.ndarray
    zcorn: np.ndarray
    actnum: Optional[np.ndarray] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, EclGrid):
            return False
        return (
            (
                (self.actnum is None and other.actnum is None)
                or np.array_equal(self.actnum, other.actnum)
            )
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    @property
    @abstractmethod
    def mapaxes(self) -> Optional[MapAxes]:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def _check_xtgeo_compatible(self):
        pass

    @staticmethod
    def valid_mapaxes(mapaxes: MapAxes) -> bool:
        y_line = mapaxes.y_line
        x_line = mapaxes.x_line
        origin = mapaxes.origin
        x_axis = np.array(x_line) - origin
        y_axis = np.array(y_line) - origin

        return np.linalg.norm(x_axis) > 1e-5 and np.linalg.norm(y_axis) > 1e-5

    def transform_xtgeo_coord_by_mapaxes(self, coord: np.ndarray):
        """Transforms xtgeo coord values by mapaxes.

        The mapaxes keyword in a grdecl file defines a new coordinate system by
        which x and y values are to be interpreted. The given xtgeo coord
        values are transformed from the local coordinate system defined by
        mapaxes to global coordinates.
        """
        x_point = self.mapaxes.x_line
        y_point = self.mapaxes.y_line
        origin = self.mapaxes.origin

        x_axis = np.array(x_point) - origin
        y_axis = np.array(y_point) - origin

        x_unit = x_axis / np.linalg.norm(x_axis)
        y_unit = y_axis / np.linalg.norm(y_axis)

        coord[:, :, (0, 1)] = (
            origin
            + coord[:, :, 0, np.newaxis] * x_unit
            + coord[:, :, 1, np.newaxis] * y_unit
        )
        coord[:, :, (3, 4)] = (
            origin
            + coord[:, :, 3, np.newaxis] * x_unit
            + coord[:, :, 4, np.newaxis] * y_unit
        )

        return coord

    def xtgeo_coord(self):
        """
        Returns:
            coord in xtgeo format.
        """
        self._check_xtgeo_compatible()
        nx, ny, _ = self.dimensions

        xtgeo_coord = np.swapaxes(self.coord.reshape((ny + 1, nx + 1, 6)), 0, 1).astype(
            np.float64
        )
        if self.mapaxes:
            self.transform_xtgeo_coord_by_mapaxes(xtgeo_coord)
        return np.ascontiguousarray(xtgeo_coord)

    def xtgeo_actnum(self):
        """
        Returns:
            actnum in xtgeo format.
        """
        self._check_xtgeo_compatible()
        nx, ny, nz = self.dimensions
        if self.actnum is None:
            return np.ones(shape=(nx, ny, nz), dtype=np.int32)
        actnum = self.actnum.reshape((nx, ny, nz), order="F")
        return np.ascontiguousarray(actnum)

    def xtgeo_zcorn(self):
        """
        Returns:
            zcorn in xtgeo format.
        """
        self._check_xtgeo_compatible()
        nx, ny, nz = self.dimensions
        zcorn = self.zcorn.reshape((2, nx, 2, ny, 2, nz), order="F")

        if not np.allclose(
            zcorn[:, :, :, :, 1, : nz - 1], zcorn[:, :, :, :, 0, 1:], atol=1e-2
        ):

            raise ValueError("xtgeo does not support grids with horizontal split.")
        result = np.zeros((nx + 1, ny + 1, nz + 1, 4), dtype=np.float32)

        # xtgeo uses 4 z values per i,j,k to mean the 4 z values of
        # adjacent cells for the cornerline at position i,j,k assuming
        # no difference in z values between upper and lower cells. In
        # the order sw,se,nw,ne.

        # In grdecl, there are 8 zvalues per i,j,k meaning the z values
        # of each corner for the cell at i,j,k. In
        # the order "left" (west) before "right" (east) , "near" (south)
        # before "far" (north) , "upper" before "bottom"

        # set the nw value of cornerline i+1,j to
        # the near right corner of cell i,j
        result[1:, :ny, 0:nz, 2] = zcorn[1, :, 0, :, 0, :]
        result[1:, :ny, nz, 2] = zcorn[1, :, 0, :, 1, nz - 1]

        # set the ne value of cornerline i,j to
        # the near left corner of cell i,j
        result[:nx, :ny, 0:nz, 3] = zcorn[0, :, 0, :, 0, :]
        result[:nx, :ny, nz, 3] = zcorn[0, :, 0, :, 1, nz - 1]

        # set the sw value of cornerline i+1,j+1 to
        # the far right corner of cell i,j to
        result[1:, 1:, 0:nz, 0] = zcorn[1, :, 1, :, 0, :]
        result[1:, 1:, nz, 0] = zcorn[1, :, 1, :, 1, nz - 1]

        # set the se value of cornerline i,j+1 to
        # the far left corner of cell i,j
        result[:nx, 1:, 0:nz, 1] = zcorn[0, :, 1, :, 0, :]
        result[:nx, 1:, nz, 1] = zcorn[0, :, 1, :, 1, nz - 1]

        self.duplicate_insignificant_xtgeo_zcorn(result)

        return np.ascontiguousarray(result)

    def duplicate_insignificant_xtgeo_zcorn(self, zcorn: np.ndarray):
        """Duplicates values on the faces and corners of the grid.

        The xtgeo format has 4 z values for all cornerlines, refering
        to the z value for the corresponding corner of the cell that is
        sw, se, nw and ne of the cornerline. However, for the cornerlines
        that are on the boundary of the grid, there might be no such cell, ie.
        north of the northernmost cornerlines there are no cells. These are
        then duplicated of corresponding cells in the opposite direction.

        """
        nx, ny, nz = self.dimensions

        # south of the sw->se face is duplicate
        # of the north values
        zcorn[1:nx, 0, :, 0] = zcorn[1:nx, 0, :, 2]
        zcorn[1:nx, 0, :, 1] = zcorn[1:nx, 0, :, 3]

        # vertical sw corner line is duplicates of
        # the ne value
        zcorn[0, 0, :, 0] = zcorn[0, 0, :, 3]
        zcorn[0, 0, :, 1] = zcorn[0, 0, :, 3]
        zcorn[0, 0, :, 2] = zcorn[0, 0, :, 3]

        # east values of the se->ne face
        # is duplicates of the corresponding
        # west values
        zcorn[nx, 1:ny, :, 1] = zcorn[nx, 1:ny, :, 0]
        zcorn[nx, 1:ny, :, 3] = zcorn[nx, 1:ny, :, 2]

        # vertical se corner line is all duplicates
        # of its nw value
        zcorn[nx, 0, :, 0] = zcorn[nx, 0, :, 2]
        zcorn[nx, 0, :, 1] = zcorn[nx, 0, :, 2]
        zcorn[nx, 0, :, 3] = zcorn[nx, 0, :, 2]

        # north values of the nw->ne face is duplicates
        # of the corresponding south values
        zcorn[1:nx, ny, :, 2] = zcorn[1:nx, ny, :, 0]
        zcorn[1:nx, ny, :, 3] = zcorn[1:nx, ny, :, 1]

        # vertical nw corner line is all duplicates
        # of the se value
        zcorn[0, ny, :, 0] = zcorn[0, ny, :, 1]
        zcorn[0, ny, :, 2] = zcorn[0, ny, :, 1]
        zcorn[0, ny, :, 3] = zcorn[0, ny, :, 1]

        # west values of the sw->nw face is duplicates
        # of corresponding east values
        zcorn[0, 1:ny, :, 0] = zcorn[0, 1:ny, :, 1]
        zcorn[0, 1:ny, :, 2] = zcorn[0, 1:ny, :, 3]

        # vertical ne corner line is all duplicates
        # of the sw value
        zcorn[nx, ny, :, 1] = zcorn[nx, ny, :, 0]
        zcorn[nx, ny, :, 2] = zcorn[nx, ny, :, 0]
        zcorn[nx, ny, :, 3] = zcorn[nx, ny, :, 0]

    @classmethod
    def from_xtgeo_grid(cls, xtgeo_grid):
        xtgeo_grid._xtgformat2()

        nx, ny, nz = xtgeo_grid.dimensions
        actnum = xtgeo_grid._actnumsv.reshape(nx, ny, nz)
        actnum = actnum.ravel(order="F")
        if np.all(actnum == 1):
            actnum = None
        coord = np.ascontiguousarray(np.swapaxes(xtgeo_grid._coordsv, 0, 1).ravel())
        zcorn = np.zeros((2, nx, 2, ny, 2, nz))
        xtgeo_zcorn = xtgeo_grid._zcornsv.reshape((nx + 1, ny + 1, nz + 1, 4))

        # This is the reverse operation of that of xtgeo_zcorn,
        # see that function for description of operations.

        # set the nw value of cornerline i+1,j to
        # the near right corner of cell i,j
        zcorn[1, :, 0, :, 1, :] = xtgeo_zcorn[1:, :ny, 1:, 2]
        zcorn[1, :, 0, :, 0, :] = xtgeo_zcorn[1:, :ny, :nz, 2]

        # set the ne value of cornerline i,j to
        # the near left corner of cell i,j
        zcorn[0, :, 0, :, 1, :] = xtgeo_zcorn[:nx, :ny, 1:, 3]
        zcorn[0, :, 0, :, 0, :] = xtgeo_zcorn[:nx, :ny, :nz, 3]

        # set the sw value of cornerline i+1,j+1 to
        # the far right corner of cell i,j to
        zcorn[1, :, 1, :, 1, :] = xtgeo_zcorn[1:, 1:, 1:, 0]
        zcorn[1, :, 1, :, 0, :] = xtgeo_zcorn[1:, 1:, :nz, 0]

        # set the se value of cornerline i,j+1 to
        # the far left corner of cell i,j
        zcorn[0, :, 1, :, 1, :] = xtgeo_zcorn[:nx, 1:, 1:, 1]
        zcorn[0, :, 1, :, 0, :] = xtgeo_zcorn[:nx, 1:, :nz, 1]

        zcorn = zcorn.ravel(order="F")

        return cls(
            coord=coord,
            zcorn=zcorn,
            actnum=actnum,
            size=(nx, ny, nz),
        )
