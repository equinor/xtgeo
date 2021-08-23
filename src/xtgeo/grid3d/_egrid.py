"""
The egrid fileformat is a file outputted by reservoir simulators such as opm
flow containing the grid geometry. The layout of cell data and units is similar
to grdecl files, but there is additional metadata.

The data is layed out similarly to other ecl output files, see the ecl_data_io
module.

There is an alternate data layout (in addition to that of grdecl files), called
unstructured, which is not widely supported. XTGeo does not currently support
that format.

egrid files like other ecl files contain tuples of keywords and list of data values
of one type (An array with a name). The enums in this file generally describe
a range of values for a position in one of these lists, the dataclasses describe
the values of one keyword or a collection of those, named a file section.

The following egrid file contents (as keyword/array pairs)::

  ("FILEHEAD", [2001,3,0,3,0,0,0])
  ("GRIDUNIT", "METRES   ")

is represented by::

    EGridHead(
        Filehead(2001,3,3,TypeOfGrid.CORNER_POINT,RockModel(0),GridFormat(0)),
        GridUnit("METRES   ")
    )

Where ``EGridHead`` is a section of the file, ``Filehead`` and ``GridUnit`` are
keywords.

keywords implement the `to_egrid` and `from_egrid` functions
which should satisfy::

    GridHead.from_egrid(x).to_egrid() == x

These convert to and from the object representation and the keyword/array
pairs, ie.

>>> grid_head_contents = [0]*100
>>> head = GridHead.from_egrid(grid_head_contents)
>>> head
GridHead(type_of_grid=<TypeOfGrid.COMPOSITE...
>>> head.to_egrid().tolist() == grid_head_contents
True
"""
import warnings
from dataclasses import InitVar, dataclass
from enum import Enum, unique
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from ecl_data_io import Format, lazy_read, write

from ._ecl_grid import CoordinateType, EclGrid, GdOrient, GridUnit, MapAxes


class EGridFileFormatError(ValueError):
    """
    Exception raised when an file unexpectedly does not conform to the egrid
    format.
    """

    pass


@unique
class TypeOfGrid(Enum):
    """
    A Grid has three possible data layout formats, UNSTRUCTURED, CORNER_POINT
    and COMPOSITE (meaning combination of the two former). Only CORNER_POINT
    layout is supported by XTGeo.
    """

    COMPOSITE = 0
    CORNER_POINT = 1
    UNSTRUCTURED = 2


@unique
class RockModel(Enum):
    """
    Type of rock model.
    """

    SINGLE_PERMEABILITY_POROSITY = 0
    DUAL_POROSITY = 1
    DUAL_PERMEABILITY = 2


@unique
class GridFormat(Enum):
    """
    The format of the "original grid", ie., what
    method was used to construct the values in the file.
    """

    UNKNOWN = 0
    IRREGULAR_CORNER_POINT = 1
    REGULAR_CARTESIAN = 2


@dataclass
class Filehead:
    """
    The first keyword in an egrid file is the FILEHEAD
    keyword, containing metadata about the file and its
    content.
    """

    version_number: int
    year: int
    version_bound: int
    type_of_grid: TypeOfGrid
    rock_model: RockModel
    grid_format: GridFormat

    @classmethod
    def from_egrid(cls, values: List[int]):
        """
        Construct a Filehead given the list of values following
        the FILEHEAD keyword.
        Args:
            values(List[int]): list of values following the FILEHEAD keyword,
                expected to contain at least 7 values (normally 100).
        Returns:
            A Filhead constructed from the given values.
        """
        if len(values) < 7:
            raise ValueError(f"Filehead given too few values, {len(values)} < 7")
        # weirdly, filehead has a different
        # code for grid type
        corner_type_code = values[4]
        type_of_grid = None
        if corner_type_code == 0:
            type_of_grid = TypeOfGrid.CORNER_POINT
        elif corner_type_code == 1:
            type_of_grid = TypeOfGrid.UNSTRUCTURED
        elif corner_type_code == 2:
            type_of_grid = TypeOfGrid.COMPOSITE
        else:
            raise ValueError(f"Unknown grid type {corner_type_code} in FILEHEAD")

        return cls(
            version_number=values[0],
            year=values[1],
            version_bound=values[3],
            type_of_grid=type_of_grid,
            rock_model=RockModel(values[5]),
            grid_format=GridFormat(values[6]),
        )

    def to_egrid(self) -> np.ndarray:
        """
        Returns:
            List of values, as layed out after the FILEHEAD keyword for
            the given filehead.
        """
        type_of_grid_code = None
        if self.type_of_grid == TypeOfGrid.CORNER_POINT:
            type_of_grid_code = 0
        elif self.type_of_grid == TypeOfGrid.UNSTRUCTURED:
            type_of_grid_code = 1
        elif self.type_of_grid == TypeOfGrid.COMPOSITE:
            type_of_grid_code = 2
        else:
            raise ValueError(f"Unknown grid type {self.type_of_grid}")
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.version_number
        result[1] = self.year
        result[3] = self.version_bound
        result[4] = type_of_grid_code
        result[5] = self.rock_model.value
        result[6] = self.grid_format.value
        return result


@dataclass
class GridHead:
    """
    Both for lgr (see LGRSection) and the global grid (see GlobalGrid)
    the GRIDHEAD keyword indicates the start of the grid layout for that
    section.
    """

    type_of_grid: TypeOfGrid
    num_x: int
    num_y: int
    num_z: int
    grid_reference_number: int
    numres: int
    nseg: int
    coordinate_type: CoordinateType
    lgr_start: Tuple[int, int, int]
    lgr_end: Tuple[int, int, int]

    @classmethod
    def from_egrid(cls, values: Sequence[int]):
        if len(values) < 33:
            raise ValueError(
                f"Too few arguments to GridHead.from_egrid {len(values)} < 33"
            )
        return cls(
            type_of_grid=TypeOfGrid(values[0]),
            num_x=values[1],
            num_y=values[2],
            num_z=values[3],
            grid_reference_number=values[4],
            numres=values[24],
            nseg=values[25],
            coordinate_type=CoordinateType.from_bgrdecl(values[26]),
            lgr_start=(values[27], values[28], values[29]),
            lgr_end=(values[30], values[31], values[32]),
        )

    def to_egrid(self) -> np.ndarray:
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.type_of_grid.value
        result[1] = self.num_x
        result[2] = self.num_y
        result[3] = self.num_z
        result[4] = self.grid_reference_number
        result[24] = self.numres
        result[25] = self.nseg
        result[26] = self.coordinate_type.to_bgrdecl()
        result[[27, 28, 29]] = np.array(list(self.lgr_start))
        result[[30, 31, 32]] = np.array(list(self.lgr_end))
        return result


@dataclass
class EGridSubGrid(EclGrid):
    """
    Both the LGR sections and the global grid contain a grid which is in the
    general format of a eclipse grid. EGridSubGrid contain the common implementation.
    """

    grid_head: Optional[GridHead] = None
    size: InitVar[Tuple[int, int, int]] = None
    mapaxes: Optional[MapAxes] = None

    def __eq__(self, other):
        return super().__eq__(other) and self.grid_head == other.grid_head

    def _check_xtgeo_compatible(self):
        if self.grid_head.coordinate_type == CoordinateType.CYLINDRICAL:
            raise NotImplementedError(
                "Xtgeo does not currently support cylindrical coordinate systems"
            )
        if self.grid_head.numres < 1:
            raise ValueError("EGrid file given with numres < 1")
        if self.grid_head.numres != 1:
            raise NotImplementedError(
                "Xtgeo does not currently support multiple reservoirs"
            )

    def __post_init__(self, size: Tuple[int, int, int]):
        if not size and not self.grid_head:
            raise ValueError(
                "Either size or grid_head has to be given when"
                " constructing an ecl grid"
            )

        if size and not self.grid_head:
            self.grid_head = GridHead(
                TypeOfGrid.CORNER_POINT,
                *size,
                0,
                1,
                1,
                CoordinateType.CARTESIAN,
                (0, 0, 0),
                (0, 0, 0),
            )
        if self.grid_head and not size:
            size = (
                self.grid_head.num_x,
                self.grid_head.num_y,
                self.grid_head.num_z,
            )
        if (
            self.grid_head
            and size
            and size
            != (
                self.grid_head.num_x,
                self.grid_head.num_y,
                self.grid_head.num_z,
            )
        ):
            raise ValueError(
                "EGridSubGrid given both grid_head and size with conflicting values"
            )

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return (
            int(self.grid_head.num_x),
            int(self.grid_head.num_y),
            int(self.grid_head.num_z),
        )

    def to_egrid(self) -> List[Tuple[str, Any]]:
        result = [
            ("GRIDHEAD", self.grid_head.to_egrid()),
            ("COORD   ", self.coord),
            ("ZCORN   ", self.zcorn),
        ]
        if self.actnum is not None:
            result.append(("ACTNUM  ", self.actnum))
        return result


def maybe_array_equal(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    Returns:
     True if the two given Optional[np.ndarray]'s are equal (either both None
     or np.array_equal).
    """
    return (arr1 is None and arr2 is None) or np.array_equal(arr1, arr2)


@dataclass
class LGRSection(EGridSubGrid):
    """
    An Egrid file can contain multiple LGR (Local Grid Refinement) sections
    which define a subgrid with finer layout.
    """

    name: Optional[str] = None
    parent: Optional[str] = None
    grid_parent: Optional[str] = None
    hostnum: Optional[np.ndarray] = None
    boxorig: Optional[Tuple[int, int, int]] = None
    coord_sys: Optional[MapAxes] = None

    def __eq__(self, other):
        if not isinstance(other, LGRSection):
            return False
        return (
            super().__eq__(other)
            and self.name == other.name
            and self.parent == other.parent
            and self.grid_parent == other.grid_parent
            and maybe_array_equal(self.hostnum, other.hostnum)
            and self.boxorig == other.boxorig
            and self.coord_sys == other.coord_sys
        )

    def __post_init__(self, size: Tuple[int, int, int]):
        super().__post_init__(size)
        if self.name is None:
            raise TypeError("Missing parameter to LGRSection: name")

    def to_egrid(self) -> List[Tuple[str, Any]]:
        result_dict = dict(super().to_egrid())
        result_dict["LGR     "] = [self.name]
        if self.parent is not None:
            result_dict["LGRPARNT"] = [self.parent]
        if self.grid_parent is not None:
            result_dict["LGRSGRID"] = [self.grid_parent]
        if self.hostnum is not None:
            result_dict["HOSTNUM "] = self.hostnum
        if self.boxorig is not None:
            result_dict["BOXORIG "] = list(self.boxorig)
        if self.coord_sys is not None:
            result_dict["COORDSYS"] = self.coord_sys.to_bgrdecl()
        result_dict["ENDGRID "] = np.array([], dtype=np.int32)
        result_dict["ENDLGR  "] = np.array([], dtype=np.int32)
        result = []
        order = [
            "LGR     ",
            "LGRPARNT",
            "LGRSGRID",
            "GRIDHEAD",
            "BOXORIG ",
            "COORD   ",
            "COORDSYS",
            "ZCORN   ",
            "ACTNUM  ",
            "HOSTNUM ",
            "ENDGRID ",
            "ENDLGR  ",
        ]
        for kw in order:
            if kw in result_dict:
                result.append((kw, result_dict[kw]))
        return result


@dataclass
class GlobalGrid(EGridSubGrid):
    """
    The global grid contains the layout of the grid before
    refinements, and the sectioning into grid coarsening
    through the optional corsnum keyword.
    """

    coord_sys: Optional[MapAxes] = None
    boxorig: Optional[Tuple[int, int, int]] = None
    corsnum: Optional[np.ndarray] = None

    def _check_xtgeo_compatible(self):
        super()._check_xtgeo_compatible()
        if self.corsnum is not None:
            warnings.warn(
                "egrid file given with coarsening, this is not directly supported"
                " by xtgeo. Instead grid is imported without coarsening."
            )

        if self.coord_sys is not None:
            warnings.warn(
                "egrid file given with coordinate definition for global"
                "grid, this is not directly supported by xtgeo. Instead"
                "grid is imported without converting by local coordsys."
            )

    def __eq__(self, other):
        if not isinstance(other, GlobalGrid):
            return False
        return (
            super().__eq__(other)
            and self.coord_sys == other.coord_sys
            and self.boxorig == other.boxorig
            and maybe_array_equal(self.corsnum, other.corsnum)
        )

    def to_egrid(self) -> List[Tuple[str, Any]]:
        result_dict = dict(super().to_egrid())
        if self.coord_sys is not None:
            result_dict["COORDSYS"] = self.coord_sys.to_bgrdecl()
        if self.boxorig is not None:
            result_dict["BOXORIG "] = list(self.boxorig)
        if self.corsnum is not None:
            result_dict["CORSNUM "] = self.corsnum
        result_dict["ENDGRID "] = np.array([], dtype=np.int32)
        result = []
        order = [
            "GRIDHEAD",
            "BOXORIG ",
            "COORD   ",
            "COORDSYS",
            "ZCORN   ",
            "ACTNUM  ",
            "CORSNUM ",
            "ENDGRID ",
        ]
        for kw in order:
            if kw in result_dict:
                result.append((kw, result_dict[kw]))
        return result


@dataclass
class NNCHead:
    """
    The NNCHead keyword denotes the start of a
    NNCSection and contains the number of nncs and
    the grid number of the grid where the NNCs applies.
    """

    num_nnc: int
    grid_identifier: int

    @classmethod
    def from_egrid(cls, values: List[int]):
        return cls(*values[0:2])

    def to_egrid(self) -> np.ndarray:
        result = np.zeros((10,), dtype=np.int32)
        result[0] = self.num_nnc
        result[1] = self.grid_identifier
        return result


@dataclass
class NNCSection:
    """
    The NNCSection's describe non-neighboor connections
    in the grid.
    """

    nnchead: NNCHead
    upstream_nnc: np.ndarray
    downstream_nnc: np.ndarray
    nncl: Optional[np.ndarray] = None
    nncg: Optional[np.ndarray] = None
    amalgamation_idxs: Optional[Tuple[int, int]] = None
    nna1: Optional[np.ndarray] = None
    nna2: Optional[np.ndarray] = None

    def __eq__(self, other):
        if not isinstance(other, NNCSection):
            return False
        return (
            self.nnchead == other.nnchead
            and np.array_equal(self.upstream_nnc, other.upstream_nnc)
            and np.array_equal(self.downstream_nnc, other.downstream_nnc)
            and maybe_array_equal(self.nncl, other.nncl)
            and maybe_array_equal(self.nncg, other.nncg)
            and self.amalgamation_idxs == other.amalgamation_idxs
            and maybe_array_equal(self.nna1, other.nna1)
            and maybe_array_equal(self.nna2, other.nna2)
        )

    def to_egrid(self) -> List[Tuple[str, Any]]:
        result = [
            ("NNCHEAD ", self.nnchead.to_egrid()),
            ("NNC1    ", self.upstream_nnc),
            ("NNC2    ", self.downstream_nnc),
        ]
        if self.nncl is not None:
            result.append(("NNCL    ", self.nncl))
        if self.nncg is not None:
            result.append(("NNCG    ", self.nncg))
        if self.amalgamation_idxs is not None:
            result.append(("NNCHEADA", list(self.amalgamation_idxs)))
        if self.nna1 is not None:
            result.append(("NNA1    ", self.nna1))
        if self.nna2 is not None:
            result.append(("NNA2    ", self.nna2))
        return result


@dataclass
class EGridHead:
    """
    The EGridHead section occurs once at the start of an EGrid file.
    """

    file_head: Filehead
    mapunits: Optional[str] = None
    mapaxes: Optional[MapAxes] = None
    gridunit: Optional[GridUnit] = None
    gdorient: Optional[GdOrient] = None

    def to_egrid(self) -> List[Tuple[str, Any]]:
        result = [
            ("FILEHEAD", self.file_head.to_egrid()),
        ]
        if self.mapunits is not None:
            result.append(("MAPUNITS", [self.mapunits]))
        if self.mapaxes is not None:
            result.append(("MAPAXES ", self.mapaxes.to_bgrdecl()))
        if self.gridunit is not None:
            result.append(("GRIDUNIT", self.gridunit.to_bgrdecl()))
        if self.gdorient is not None:
            result.append(("GDORIENT", self.gdorient.to_bgrdecl()))
        return result


@dataclass
class EGrid:
    """
    Contains the complete contents of an EGridFile.
    """

    egrid_head: EGridHead
    global_grid: GlobalGrid
    lgr_sections: List[LGRSection]
    nnc_sections: List[NNCSection]

    @classmethod
    def from_file(self, filelike, file_format: Format = None):
        """
        Read an egrid file
        Args:
            filelike (str,Path,stream): The egrid file to be read.
            file_format (None or ecl_data_io.Format): The format of the file,
                None means guess.
        Returns:
            EGrid with the contents of the file.
        """
        return EGridReader(filelike, file_format=file_format).read()

    def to_file(self, filelike, file_format: Format = Format.UNFORMATTED):
        """
        write the EGrid to file.
        Args:
            filelike (str,Path,stream): The egrid file to write to.
            file_format (ecl_data_io.Format): The format of the file.
        """
        contents = []
        contents += self.egrid_head.to_egrid()
        contents += self.global_grid.to_egrid()
        for lgr in self.lgr_sections:
            contents += lgr.to_egrid()
        for nnc in self.nnc_sections:
            contents += nnc.to_egrid()
        write(filelike, contents, file_format)

    def _check_xtgeo_compatible(self):
        if self.lgr_sections:
            warnings.warn(
                "egrid file given with local grid refinements."
                "LGR's are not directly supported,"
                "Instead unrefined grid is imported."
            )

    @property
    def dimensions(self):
        return self.global_grid.dimensions

    @classmethod
    def from_xtgeo_grid(cls, xtgeo_grid):
        xtgeo_grid._xtgformat2()
        global_grid = GlobalGrid.from_xtgeo_grid(xtgeo_grid)
        global_grid.coord = global_grid.coord.astype(np.float32)
        global_grid.zcorn = global_grid.zcorn.astype(np.float32)
        rock_model = RockModel.SINGLE_PERMEABILITY_POROSITY
        if xtgeo_grid._dualporo:
            rock_model = RockModel.DUAL_POROSITY
        if xtgeo_grid._dualperm:
            rock_model = RockModel.DUAL_PERMEABILITY
        return EGrid(
            EGridHead(
                Filehead(
                    3,
                    2007,
                    3,
                    TypeOfGrid.CORNER_POINT,
                    rock_model,
                    GridFormat.IRREGULAR_CORNER_POINT,
                )
            ),
            global_grid,
            [],
            [],
        )

    def xtgeo_coord(self) -> np.ndarray:
        self._check_xtgeo_compatible()
        previous_mapaxes = self.global_grid.mapaxes
        self.global_grid.mapaxes = self.egrid_head.mapaxes
        result = self.global_grid.xtgeo_coord()
        self.global_grid.mapaxes = previous_mapaxes
        return result

    def xtgeo_actnum(self) -> np.ndarray:
        self._check_xtgeo_compatible()
        previous_mapaxes = self.global_grid.mapaxes
        self.global_grid.mapaxes = self.egrid_head.mapaxes
        result = self.global_grid.xtgeo_actnum()
        self.global_grid.mapaxes = previous_mapaxes
        return result

    def xtgeo_zcorn(self) -> np.ndarray:
        self._check_xtgeo_compatible()
        previous_mapaxes = self.global_grid.mapaxes
        self.global_grid.mapaxes = self.egrid_head.mapaxes
        result = self.global_grid.xtgeo_zcorn()
        self.global_grid.mapaxes = previous_mapaxes
        return result


keyword_translation = {
    "FILEHEAD": "file_head",
    "MAPUNITS": "mapunits",
    "MAPAXES ": "mapaxes",
    "GRIDUNIT": "gridunit",
    "GDORIENT": "gdorient",
    "LGR     ": "name",
    "GRIDHEAD": "grid_head",
    "HOSTNUM ": "hostnum",
    "BOXORIG ": "boxorig",
    "COORDSYS": "coord_sys",
    "LGRPARNT": "parent",
    "LGRSGRID": "grid_parent",
    "COORD   ": "coord",
    "ZCORN   ": "zcorn",
    "ACTNUM  ": "actnum",
    "NNCHEAD ": "nnchead",
    "NNC1    ": "upstream_nnc",
    "NNC2    ": "downstream_nnc",
    "NNCL    ": "nncl",
    "NNCG    ": "nncg",
    "NNCHEADA": "amalgamation_idxs",
    "NNA1    ": "nna1",
    "NNA2    ": "nna2",
    "CORSNUM ": "corsnum",
}


class EGridReader:
    """
    The EGridReader reads an egrid file through the `read` method.

    Args:
        filelike (str, Path, stream): The egrid file to read from.
        file_format (None or ecl_data_io.Format): The format of the file,
            None means guess.

    """

    def __init__(self, filelike, file_format: Format = None):
        self.filelike = filelike
        self.keyword_generator = lazy_read(filelike, file_format)

    def read_section(
        self,
        keyword_factories: Dict[str, Callable],
        required_keywords: Set[str],
        stop_keywords: Iterable[str],
        skip_keywords: Iterable[str] = [],
        keyword_visitors: Iterable[Callable] = [],
    ):
        """
        Read a general egrid file section.
        Args:
            keyword_factories (dict[str, func]): The function used
                to construct a section member.
            required_keywords (List[str]): List of keywords that are required
                for the given section.
            stop_keywords (List[str]): List of keywords which when read ends
                the section. The keyword generator will be at the first keyword
                in stop_keywords after read_section is called.
            skip_keywords (List[str]): List of keywords that does not
                have a factory, which should just be skipped.
            keyword_visitors (List[func]): List of functions that
                "visit" each keyword. Each of these functions are called
                for each keyword, value pair and can be used to
                preprocess the data.

        Returns:
            dictionary of parameters for the constructor of the given section.
        """
        results = {}
        i = 0
        while True:
            try:
                entry = next(self.keyword_generator)
            except StopIteration:
                break
            kw = entry.read_keyword()
            if kw in skip_keywords:
                continue
            if kw in stop_keywords and i > 0:
                # Optional keywords were possibly omitted and
                # we have reached the global grid section
                # push back the grid head of the global grid
                # and proceed
                self.keyword_generator = chain([entry], self.keyword_generator)
                break
            if kw in results:
                raise EGridFileFormatError(f"Duplicate keyword {kw} in {self.filelike}")
            try:
                factory = keyword_factories[kw]
            except KeyError as err:
                raise EGridFileFormatError(f"Unknown egrid keyword {kw}") from err
            try:
                value = factory(entry.read_array())
                results[kw] = value
            except (ValueError, IndexError, TypeError) as err:
                raise EGridFileFormatError(f"Incorrect values in keyword {kw}") from err
            for visit in keyword_visitors:
                visit(kw, value)
            i += 1

        missing_keywords = required_keywords.difference(results.keys())
        params = {keyword_translation[kw]: v for kw, v in results.items()}
        if missing_keywords:
            raise EGridFileFormatError(f"Missing required keywords {missing_keywords}")
        return params

    def read_header(self) -> EGridHead:
        """
        Reads the EGrid header from the start of the stream. Ensures
        that the keyword_generator is at the first GRIDHEAD keyword
        after the header.
        """
        params = self.read_section(
            keyword_factories={
                "FILEHEAD": Filehead.from_egrid,
                "MAPUNITS": lambda x: x[0].decode("ascii"),
                "MAPAXES ": MapAxes.from_bgrdecl,
                "GRIDUNIT": GridUnit.from_bgrdecl,
                "GDORIENT": GdOrient.from_bgrdecl,
            },
            required_keywords={"FILEHEAD"},
            stop_keywords=["GRIDHEAD"],
        )
        return EGridHead(**params)

    def read_global_grid(self) -> GlobalGrid:
        """
        Reads the global grid section from the start of the keyword_generator,
        ensures the keyword_generator is at the keyword after the first ENDGRID
        keyword encountered.
        """

        def check_gridhead(kw: str, value):
            if kw == "GRIDHEAD" and value.type_of_grid != TypeOfGrid.CORNER_POINT:
                raise NotImplementedError(
                    "XTGeo does not support unstructured or mixed grids."
                )

        params = self.read_section(
            keyword_factories={
                "GRIDHEAD": GridHead.from_egrid,
                "BOXORIG ": tuple,
                "COORDSYS": MapAxes.from_bgrdecl,
                "COORD   ": lambda x: np.array(x, dtype=np.float32),
                "ZCORN   ": lambda x: np.array(x, dtype=np.float32),
                "ACTNUM  ": lambda x: np.array(x, dtype=np.int32),
                "CORSNUM ": lambda x: np.array(x, dtype=np.int32),
            },
            required_keywords={"GRIDHEAD", "COORD   ", "ZCORN   "},
            stop_keywords=["ENDGRID "],
            keyword_visitors=[check_gridhead],
        )
        try:
            entry = next(self.keyword_generator)
        except StopIteration as err:
            raise EGridFileFormatError(
                "Did not read ENDGRID after global grid"
            ) from err
        if entry.read_keyword() != "ENDGRID ":
            raise EGridFileFormatError("Did not read ENDGRID after global grid")
        return GlobalGrid(**params)

    def read_subsections(self) -> Tuple[List[LGRSection], List[NNCSection]]:
        """
        Reads lgr and nnc subsections from the start of the keyword_generator.
        """
        lgr_sections = []
        nnc_sections = []
        while True:
            try:
                entry = next(self.keyword_generator)
            except StopIteration:
                break
            self.keyword_generator = chain([entry], self.keyword_generator)
            keyword = entry.read_keyword().rstrip()
            if keyword == "LGR":
                lgr_sections.append(self.read_lgr_subsection())
            elif keyword == "NNCHEAD":
                nnc_sections.append(self.read_nnc_subsection())
            else:
                raise EGridFileFormatError(
                    f"egrid subsection started with unexpected keyword {keyword}"
                )
        return lgr_sections, nnc_sections

    def read_lgr_subsection(self) -> LGRSection:
        """
        Reads one lgr subsection from the start of the keyword generator.
        After read_lgr_subsection is called, The keyword_generator is at the
        keyword after the first ENDLGR keyword encountered, or end of stream.
        """
        params = self.read_section(
            keyword_factories={
                "LGR     ": lambda x: x[0].decode("ascii"),
                "LGRPARNT": lambda x: x[0].decode("ascii"),
                "LGRSGRID": lambda x: x[0].decode("ascii"),
                "GRIDHEAD": GridHead.from_egrid,
                "BOXORIG ": tuple,
                "COORDSYS": MapAxes.from_bgrdecl,
                "COORD   ": lambda x: np.array(x, dtype=np.float32),
                "ZCORN   ": lambda x: np.array(x, dtype=np.float32),
                "ACTNUM  ": lambda x: np.array(x, dtype=np.int32),
                "HOSTNUM ": lambda x: np.array(x, dtype=np.int32),
            },
            required_keywords={
                "LGR     ",
                "GRIDHEAD",
                "COORD   ",
                "ZCORN   ",
                "HOSTNUM ",
            },
            skip_keywords=["ENDGRID "],
            stop_keywords=["ENDLGR  "],
        )
        try:
            entry = next(self.keyword_generator)
        except StopIteration as err:
            raise EGridFileFormatError("Did not read ENDLGR after lgr section") from err
        if entry.read_keyword() != "ENDLGR  ":
            raise EGridFileFormatError("Did not read ENDLGR after lgr section")
        return LGRSection(**params)

    def read_nnc_subsection(self) -> NNCSection:
        """
        Reads one lgr subsection from the start of the keyword generator.
        After read_lgr_subsection is called, The keyword_generator is
        at the next NNCHEAD or LGR keyword, or end of stream.
        """
        params = self.read_section(
            keyword_factories={
                "NNCHEAD ": NNCHead.from_egrid,
                "NNC1    ": lambda x: np.array(x, dtype=np.int32),
                "NNC2    ": lambda x: np.array(x, dtype=np.int32),
                "NNCL    ": lambda x: np.array(x, dtype=np.int32),
                "NNCG    ": lambda x: np.array(x, dtype=np.int32),
                "NNCHEADA": lambda x: tuple(x[0:2]),
                "NNA1    ": lambda x: np.array(x, dtype=np.int32),
                "NNA2    ": lambda x: np.array(x, dtype=np.int32),
            },
            required_keywords={"NNCHEAD ", "NNC1    ", "NNC2    "},
            stop_keywords=["NNCHEAD ", "LGR     "],
        )
        return NNCSection(**params)

    def read(self) -> EGrid:
        header = self.read_header()
        if header.file_head.type_of_grid != TypeOfGrid.CORNER_POINT:
            raise NotImplementedError(
                "XTGeo does not support unstructured or mixed grids."
            )
        global_grid = self.read_global_grid()
        lgr_sections, nnc_sections = self.read_subsections()
        return EGrid(header, global_grid, lgr_sections, nnc_sections)
