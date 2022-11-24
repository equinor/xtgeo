import warnings
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, fields
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.constants import foot

from ._grdecl_format import match_keyword


@unique
class Units(Enum):
    METRES = auto()
    CM = auto()
    FEET = auto()

    def conversion_factor(self, other):
        "Conversion factor from one unit to another"
        result = 1.0
        if self == other:
            return result
        if other == Units.FEET:
            result *= 1 / foot
        if other == Units.CM:
            result *= 1e2
        if self == Units.FEET:
            result *= foot
        if self == Units.CM:
            result *= 1e-2
        return result

    def to_grdecl(self):
        return self.name

    def to_bgrdecl(self):
        return self.to_grdecl().ljust(8)

    @classmethod
    def from_grdecl(cls, unit_string):
        if match_keyword(unit_string, "METRES"):
            return cls.METRES
        if match_keyword(unit_string, "FEET"):
            return cls.FEET
        if match_keyword(unit_string, "CM"):
            return cls.CM
        raise ValueError(f"Unknown unit string {unit_string}")

    @classmethod
    def from_bgrdecl(cls, unit_string):
        if isinstance(unit_string, bytes):
            return cls.from_grdecl(unit_string.decode("ascii"))
        return cls.from_grdecl(unit_string)


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
    >>> class A(GrdeclKeyword):
    ...     ...
    >>> class B(GrdeclKeyword):
    ...     ...

    >>> @dataclass
    ... class MyKeyword(GrdeclKeyword):
    ...     field1: A
    ...     field2: B

    will have a to_grdecl method that will be similar to

    >>> def to_grdecl(self):
    ...     return [self.field1.to_grdecl(), self.field2.to_grdecl]

    Similarly from_grdecl will call fields from_grdecl
    to construct the object

    >>> @classmethod
    ... def from_grdecl(cls, values):
    ...     return cls(A.from_grdecl(values[0]), B.from_grdecl(values[1]))
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

    unit: Units = Units.METRES
    grid_relative: GridRelative = GridRelative.ORIGIN


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
        return np.array(self.to_grdecl(), dtype=np.float32)

    def in_units(self, old_units, new_units):
        factor = old_units.conversion_factor(new_units)
        y_line = (self.y_line[0] * factor, self.y_line[1] * factor)
        x_line = (self.x_line[0] * factor, self.x_line[1] * factor)
        origin = (self.origin[0] * factor, self.origin[1] * factor)
        return MapAxes(y_line, origin, x_line)

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


def transform_xtgeo_coord_by_mapaxes(mapaxes: MapAxes, coord: np.ndarray):
    """Transforms xtgeo coord values by mapaxes.

    The mapaxes keyword in a grdecl file defines a new coordinate system by
    which x and y values are to be interpreted. The given xtgeo coord
    values are transformed from the local coordinate system defined by
    mapaxes to global coordinates.
    """
    x_point = mapaxes.x_line
    y_point = mapaxes.y_line
    origin = mapaxes.origin

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


def inverse_transform_xtgeo_coord_by_mapaxes(mapaxes: MapAxes, coord: np.ndarray):
    """Inversely transforms xtgeo coord values by mapaxes.

    The inverse operation of transform_xtgeo_coord_by_mapaxes.
    """
    x_point = mapaxes.x_line
    y_point = mapaxes.y_line
    origin = mapaxes.origin

    x_axis = np.array(x_point) - origin
    y_axis = np.array(y_point) - origin

    x_unit = x_axis / np.linalg.norm(x_axis)
    y_unit = y_axis / np.linalg.norm(y_axis)

    coord[:, :, (0, 1)] -= np.array(origin)
    coord[:, :, (3, 4)] -= np.array(origin)

    inv_transform = np.linalg.inv(np.transpose([x_unit, y_unit]))

    # The following index manipulation is
    # an optimized version of

    # nx, ny, _ = coord.shape
    # for i in range(nx):
    #    for j in range(ny):
    #        coord[i, j, (0, 1)] = inv_transform @ coord[i, j, (0, 1)]
    #        coord[i, j, (3, 4)] = inv_transform @ coord[i, j, (3, 4)]
    coord[:, :, (0, 1)] = (
        inv_transform[np.newaxis, np.newaxis, :, :] @ coord[:, :, (0, 1), np.newaxis]
    )[:, :, :, 0]
    coord[:, :, (3, 4)] = (
        inv_transform[np.newaxis, np.newaxis, :, :] @ coord[:, :, (3, 4), np.newaxis]
    )[:, :, :, 0]
    return coord


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

    ACTNUM describes the active status of each cell. For simulations without
    dual porosity or thermal, 0 means inactive, 1 means active and other values
    are not used. For dual porosity, 0 means inactive, 1 means matrix only,
    2 means fracture only, and 3 means both fracture and matrix. For thermal
    simulations, 0 means inactive, 1 means active, 2 means rock volume only,
    3 means pore volume only.
    """

    @property
    @abstractmethod
    def coord(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def zcorn(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def actnum(self) -> Optional[np.ndarray]:
        pass

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
    def is_map_relative(self) -> bool:
        pass

    @property
    @abstractmethod
    def mapaxes(self) -> Optional[MapAxes]:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, int, int]:
        pass

    @property
    @abstractmethod
    def map_axis_units(self) -> Units:
        pass

    @property
    @abstractmethod
    def grid_units(self) -> Units:
        pass

    @abstractmethod
    def _check_xtgeo_compatible(self):
        pass

    def convert_grid_units(self, units):
        """Converts the units of the grid
        Args:
            units: The unit to convert to.

        After convert_grid_units is called, `EclGrid.grid_units == units`.

        """
        old_grid_units = self.grid_units
        factor = old_grid_units.conversion_factor(units)
        self.coord *= factor
        self.zcorn *= factor
        self.grid_units = units

    @staticmethod
    def valid_mapaxes(mapaxes: MapAxes) -> bool:
        y_line = mapaxes.y_line
        x_line = mapaxes.x_line
        origin = mapaxes.origin
        x_axis = np.array(x_line) - origin
        y_axis = np.array(y_line) - origin

        return np.linalg.norm(x_axis) > 1e-5 and np.linalg.norm(y_axis) > 1e-5

    def _relative_to_transform(self, xtgeo_coord, relative_to=GridRelative.MAP):
        """Handle relative transform of xtgeo_coord()."""
        mapaxes = self.mapaxes
        has_mapaxes = True
        if self.mapaxes is None:
            mapaxes = MapAxes()
            has_mapaxes = False
        axis_units = self.map_axis_units

        has_axis_units = True
        if axis_units is None:
            axis_units = self.grid_units
            has_axis_units = False

        if has_mapaxes and not has_axis_units:
            warnings.warn(
                "Axis units specification is missing in input, assuming that no "
                "unit conversion is necessary"
            )

        if relative_to == GridRelative.MAP and not self.is_map_relative:
            xtgeo_coord *= self.grid_units.conversion_factor(axis_units)
            xtgeo_coord = transform_xtgeo_coord_by_mapaxes(mapaxes, xtgeo_coord)

        elif relative_to == GridRelative.ORIGIN and self.is_map_relative:
            mapaxes = mapaxes.in_units(axis_units, self.grid_units)
            xtgeo_coord = inverse_transform_xtgeo_coord_by_mapaxes(mapaxes, xtgeo_coord)

        return xtgeo_coord

    def xtgeo_coord(self, relative_to=GridRelative.MAP):
        """
        Args:
            relative_to: Specifies the axis system the coords should be
            relative to, either map or grid. Defaults to map. If relative_to is
            GridRelative.MAP then the resulting units are that of map_axis_units.
        Returns:
            coord in xtgeo format.
        """
        self._check_xtgeo_compatible()
        nx, ny, _ = self.dimensions

        xtgeo_coord = (
            np.swapaxes(self.coord.reshape((ny + 1, nx + 1, 6)), 0, 1)
            .astype(np.float64)
            .copy()
        )
        xtgeo_coord = self._relative_to_transform(xtgeo_coord, relative_to)
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
        activity_number = self.actnum.reshape((nx, ny, nz), order="F")
        return np.ascontiguousarray(activity_number)

    def xtgeo_zcorn(self, relative_to=GridRelative.MAP):
        """
            relative_to: Specifies the axis system the zcorn should be
            relative to, either map or origin. Defaults to map. For zcorn
            this only affects which units zcorn will be in, grid units for
            relative to origin, map units for relative to map.
        Returns:
            zcorn in xtgeo format.
        """
        self._check_xtgeo_compatible()
        nx, ny, nz = self.dimensions
        zcorn = self.zcorn.reshape((2, nx, 2, ny, 2, nz), order="F")

        if not np.allclose(
            zcorn[:, :, :, :, 1, : nz - 1], zcorn[:, :, :, :, 0, 1:], atol=1e-2
        ):

            warnings.warn(
                "An Eclipse style grid with vertical ZCORN splits "
                "or overlaps between vertical neighbouring cells is detected. XTGeo "
                "will import the grid as if the cell layers are connected, "
                "hence check result carefully. "
                "(Note also that this check both active and inactive cells!)",
                UserWarning,
            )

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

        axis_units = self.map_axis_units
        if axis_units is None:
            axis_units = self.grid_units
        if relative_to == GridRelative.MAP and not self.is_map_relative:
            result *= self.grid_units.conversion_factor(self.map_axis_units)

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
    @abstractmethod
    def default_settings_grid(
        cls,
        coord: np.ndarray,
        zcorn: np.ndarray,
        actnum: Optional[np.ndarray],
        size: Tuple[int, int, int],
    ):
        pass

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

        result = cls.default_settings_grid(
            coord=coord,
            zcorn=zcorn,
            actnum=actnum,
            size=(nx, ny, nz),
        )

        if xtgeo_grid.units is not None:
            result.grid_units = xtgeo_grid.units
            result.map_axis_units = xtgeo_grid.units

        return result
