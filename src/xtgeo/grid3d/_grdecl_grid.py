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
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ecl_data_io import Format, lazy_read, write

from ._ecl_grid import (
    CoordinateType,
    EclGrid,
    GdOrient,
    GrdeclKeyword,
    GridRelative,
    GridUnit,
    MapAxes,
    Units,
)
from ._grdecl_format import IGNORE_ALL, open_grdecl


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
        return np.array(
            [
                self.ndivix,
                self.ndiviy,
                self.ndiviz,
                self.numres,
                self.coordinate_type.to_bgrdecl(),
            ],
            dtype=np.int32,
        )

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


class GrdeclGrid(EclGrid):
    """
    The main keywords that describe a grdecl grid is COORD, ZCORN and ACTNUM
    and are described in xtgeo.grid3d._ecl_grid.
    The remaining fields (SPECGRID, MAPAXES, MAPUNITS, GRIDUNIT, GDORIENT)
    describe units, orientation and dimensions, see corresponding dataclasses.
    The number of cells in each direction is described in the SPECGRID keyword.
    """

    def __init__(
        self,
        coord: np.ndarray,
        zcorn: np.ndarray,
        specgrid: SpecGrid,
        actnum: Optional[np.ndarray] = None,
        mapaxes: Optional[MapAxes] = None,
        mapunits: Optional[Units] = None,
        gridunit: Optional[GridUnit] = None,
        gdorient: Optional[GdOrient] = None,
    ):
        self._coord = coord
        self._zcorn = zcorn
        self.specgrid = specgrid
        self._actnum = actnum
        self._mapaxes = mapaxes
        self.mapunits = mapunits
        self.gridunit = gridunit
        self.gdorient = gdorient

    def __str__(self):
        return (
            "GrdeclGrid("
            f"coord={self._coord}, "
            f"zcorn={self._zcorn}, "
            f"specgrid={self.specgrid}, "
            f"actnum={self._actnum}, "
            f"mapaxes={self._mapaxes}, "
            f"mapunits={self.mapunits}, "
            f"gridunit={self.gridunit}, "
            f"gdorient={self.gdorient})"
        )

    def __repr__(self):
        return str(self)

    @property
    def mapaxes(self) -> Optional[MapAxes]:
        return self._mapaxes

    @mapaxes.setter
    def mapaxes(self, value):
        self._mapaxes = value

    @property
    def coord(self) -> np.ndarray:
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = value

    @property
    def zcorn(self) -> np.ndarray:
        return self._zcorn

    @zcorn.setter
    def zcorn(self, value):
        self._zcorn = value

    @property
    def actnum(self) -> Optional[np.ndarray]:
        return self._actnum

    @classmethod
    def default_settings_grid(
        cls,
        coord: np.ndarray,
        zcorn: np.ndarray,
        actnum: Optional[np.ndarray],
        size: Tuple[int, int, int],
    ):
        return cls(coord, zcorn, SpecGrid(*size), actnum)

    def __eq__(self, other):
        if not isinstance(other, GrdeclGrid):
            return False
        return (
            self.specgrid == other.specgrid
            and self.mapaxes == other.mapaxes
            and self.mapunits == other.mapunits
            and self.gridunit == other.gridunit
            and self.gdorient == other.gdorient
            and np.array_equal(self.actnum, other.actnum)
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    @property
    def dimensions(self):
        return (self.specgrid.ndivix, self.specgrid.ndiviy, self.specgrid.ndiviz)

    @property
    def is_map_relative(self) -> bool:
        return (
            self.gridunit is not None
            and self.gridunit.grid_relative == GridRelative.MAP
        )

    @property
    def map_axis_units(self):
        if self.mapunits is None:
            return Units.METRES
        return self.mapunits

    @map_axis_units.setter
    def map_axis_units(self, value):
        self.mapunits = value

    @property
    def grid_units(self):
        if self.gridunit is None:
            return Units.METRES
        return self.gridunit.unit

    @grid_units.setter
    def grid_units(self, value):
        if self.gridunit is None and value != Units.METRES:
            self.gridunit = GridUnit(unit=value)
        elif self.gridunit is not None:
            self.gridunit.unit = value

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
            "MAPUNITS": lambda x: Units.from_bgrdecl(x[0]),
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
            "MAPUNITS": lambda x: Units.from_grdecl(x[0]),
            "GRIDUNIT": GridUnit.from_grdecl,
            "SPECGRID": SpecGrid.from_grdecl,
            "GDORIENT": GdOrient.from_grdecl,
        }
        results = {}
        with open_grdecl(
            filename,
            keywords=["MAPAXES", "MAPUNITS", "GRIDUNIT", "SPECGRID", "GDORIENT"],
            simple_keywords=["COORD", "ZCORN", "ACTNUM"],
            max_len=None,
            ignore=IGNORE_ALL,
            strict=False,
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
                ("MAPUNITS", [self.mapunits.to_grdecl()] if self.mapunits else None),
                ("GRIDUNIT", self.gridunit.to_grdecl() if self.gridunit else None),
                ("GDORIENT", self.gdorient.to_grdecl() if self.gdorient else None),
                ("COORD", self.coord),
                ("ZCORN", self.zcorn),
                ("ACTNUM", self.actnum),
            ]
            for kw, values in keywords:
                if values is None:
                    continue
                filestream.write(f"{kw}\n")
                numcolumns = 0
                for value in values:
                    numcolumns += 1
                    filestream.write(f" {value}")

                    if numcolumns >= 6:  # 6 should ensure < 128 character width total
                        filestream.write("\n")
                        numcolumns = 0

                filestream.write("\n /\n")

    def _to_bgrdecl_file(self, filename, fileformat=Format.UNFORMATTED):
        contents = filter(
            lambda x: x[1] is not None,
            [
                ("SPECGRID", self.specgrid.to_bgrdecl()),
                ("MAPAXES ", self.mapaxes.to_bgrdecl() if self.mapaxes else None),
                ("MAPUNITS", [self.mapunits.to_bgrdecl()] if self.mapunits else None),
                ("GRIDUNIT", self.gridunit.to_bgrdecl() if self.gridunit else None),
                ("GDORIENT", self.gdorient.to_bgrdecl() if self.gdorient else None),
                ("COORD   ", self.coord.astype(np.float32)),
                ("ZCORN   ", self.zcorn.astype(np.float32)),
                (
                    "ACTNUM  ",
                    self.actnum.astype(np.int32) if self.actnum is not None else None,
                ),
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
