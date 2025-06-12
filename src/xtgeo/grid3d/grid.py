"""Module/class for 3D grids (corner point geometry) with XTGeo."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo import _cxtgeo
from xtgeo.common import XTGDescription, null_logger
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.sys import generic_hash
from xtgeo.common.types import Dimensions
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.surface.regular_surface import surface_from_grid3d

from . import (
    _grid3d_fence,
    _grid_boundary,
    _grid_etc1,
    _grid_export,
    _grid_hybrid,
    _grid_import,
    _grid_import_ecl,
    _grid_refine,
    _grid_roxapi,
    _grid_wellzone,
    _gridprop_lowlevel,
)
from ._grid3d import _Grid3D
from .grid_properties import GridProperties

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Hashable

    import pandas as pd

    from xtgeo import Polygons, Well
    from xtgeo._internal.grid3d import Grid as GridCPP  # type: ignore
    from xtgeo._internal.regsurf import (  # type: ignore
        RegularSurface as RegularSurfaceCPP,
    )
    from xtgeo.common.types import FileLike
    from xtgeo.xyz.points import Points

    from ._ecl_grid import Units
    from .grid_property import GridProperty
    from .types import METRIC

xtg = xtgeo.common.XTGeoDialog()
logger = null_logger(__name__)

# --------------------------------------------------------------------------------------
# Comment on "asmasked" vs "activeonly:
#
# "asmasked"=True will return a np.ma array, while "asmasked" = False will
# return a np.ndarray
#
# The "activeonly" will filter out masked entries, or use None or np.nan
# if "activeonly" is False.
#
# Use word "zerobased" for a bool regarding if startcell basis is 1 or 0
#
# For functions with mask=... ,they should be replaced with asmasked=...
# --------------------------------------------------------------------------------------


def _estimate_grid_ijk_handedness(coordsv: np.array) -> Literal["left", "right"] | None:
    """Helper: estimate the ijk handedness from the coordinates.

    Args:
        coordsv: The coordinates (coordsv) of the grid.

    Returns:
        "left" or "right" depending on the handedness, or None if cannot determine

    Note:
        Z array is always assumed positive downwards
    """
    p0 = coordsv[0, 0, 0:3]  # origin
    pz = coordsv[0, 0, 3:6]  # layer direction
    px = coordsv[-1, 0, 0:3]  # columns direction, aka X
    py = coordsv[0, -1, 0:3]  # rows direction, aka Y

    # sometimes the pillars are collapsed; assume that the pillars are postive down
    if pz[2] <= p0[2]:
        pz[2] = p0[2] + 0.001

    # multiply z by -1 since system is Z positive down
    p0[2] *= -1
    pz[2] *= -1
    px[2] *= -1
    py[2] *= -1

    vx = np.array(px) - np.array(p0)
    vy = np.array(py) - np.array(p0)
    vz = np.array(pz) - np.array(p0)
    det = np.dot(np.cross(vx, vy), vz)

    if det > 0:
        return "right"

    if det < 0:
        return "left"

    return None


# METHODS as wrappers to class init + import
def _handle_import(
    grid_constructor: Callable[..., Grid],
    gfile: FileLike | FileWrapper,
    fformat: str | None = None,
    **kwargs: dict,
) -> Grid:
    """Handles the import given a constructor.

    For backwards compatability we need to call different constructors
    with grid __init__ parameters (As returned by _grid_import.from_file).
    These are generally either the _reset method of an instance or Grid().

    This function takes such a constructor, remaining arguments are interpreted
    as they are in _grid_import.from_file and calls the constructor with the
    resulting arguments.

    """
    gfile = FileWrapper(gfile, mode="rb")
    if fformat == "eclipserun":
        ecl_grid = grid_constructor(
            **_grid_import.from_file(
                FileWrapper(gfile.name + ".EGRID", mode="rb"), fformat=FileFormat.EGRID
            )
        )
        _grid_import_ecl.import_ecl_run(gfile.name, ecl_grid=ecl_grid, **kwargs)
        return ecl_grid

    fmt = gfile.fileformat(fformat)
    return grid_constructor(**_grid_import.from_file(gfile, fmt, **kwargs))


def grid_from_file(
    gfile: FileLike | FileWrapper,
    fformat: str | None = None,
    **kwargs: dict[str, Any],
) -> Grid:
    """Read a grid (cornerpoint) from filelike and an returns a Grid() instance.

    Args:
        gfile: File name to be imported. If fformat="eclipse_run"
            then a fileroot name shall be input here, see example below.
        fformat: File format egrid/roff/grdecl/bgrdecl/eclipserun/xtgcpgeom
            (None is default and means "guess")
        initprops (str list): Optional, and only applicable for file format
            "eclipserun". Provide a list the names of the properties here. A
            special value "all" can be get all properties found in the INIT file
        restartprops (str list): Optional, see initprops
        restartdates (int list): Optional, required if restartprops
        ijkrange (list-like): Optional, only applicable for hdf files, see
            :meth:`Grid.from_hdf`.
        zerobased (bool): Optional, only applicable for hdf files, see
            :meth:`Grid.from_hdf`.
        mmap (bool): Optional, only applicable for xtgf files, see
            :meth:`Grid.from_xtgf`.

    Example::

        >>> import xtgeo
        >>> mygrid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")

    Example using "eclipserun"::

        >>> mycase = "REEK"  # meaning REEK.EGRID, REEK.INIT, REEK.UNRST
        >>> xg = xtgeo.grid_from_file(
        ...     reek_dir + "/" + mycase,
        ...     fformat="eclipserun",
        ...     initprops="all",
        ... )
        Grid ... filesrc='.../REEK.EGRID'

    Raises:
        OSError: if file is not found etc

    """
    return _handle_import(Grid, gfile, fformat, **kwargs)


def grid_from_roxar(
    project: str,
    gname: str,
    realisation: int = 0,
    info: bool = False,
) -> Grid:
    """Read a 3D grid inside a RMS project and return a Grid() instance.

    Args:
        project (str or special): The RMS project or the project variable
            from inside RMS.
        gname (str): Name of Grid Model in RMS.
        realisation (int): Realisation number.
        dimensions_only (bool): If True, only the ncol, nrow, nlay will
            read. The actual grid geometry will remain empty (None). This will
            be much faster of only grid size info is needed, e.g.
            for initalising a grid property.
        info (bool): If true, only grid info

    Example::

        # inside RMS
        import xtgeo
        mygrid = xtgeo.grid_from_roxar(project, "REEK_SIM")

    """
    return Grid(**_grid_roxapi.load_grid_from_rms(project, gname, realisation, info))


def create_box_grid(
    dimension: Dimensions,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    oricenter: bool = False,
    increment: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation: float = 0.0,
    flip: Literal[1, -1] = 1,
) -> Grid:
    """Create a rectangular 'shoebox' grid from spec.

    Args:
        dimension (NamedTuple of int): A tuple of (NCOL, NROW, NLAY)
        origin (tuple of float): Startpoint of grid (x, y, z)
        oricenter (bool): If False, startpoint is node, if True, use cell center
        increment (tuple of float): Grid increments (xinc, yinc, zinc)
        rotation (float): Rotation in degrees, anticlock from X axis.
        flip (int): If +1, grid origin is lower left and left-handed;
                    if -1, origin is upper left and right-handed (row flip).

    Returns:
        Instance is updated (previous instance content will be erased)

    .. versionadded:: 2.1
    """
    kwargs = _grid_etc1.create_box(
        dimension=dimension,
        origin=origin,
        oricenter=oricenter,
        increment=increment,
        rotation=rotation,
        flip=flip,
    )

    return Grid(**kwargs)


def grid_from_cube(
    cube: xtgeo.Cube,
    propname: str | None = "seismics",
    oricenter: bool = True,
) -> Grid:
    """Create a rectangular 'shoebox' grid from an existing cube.

    The cube values itself will then be stored with name given by ``propname`` key.

    Since the cube actually is node centered, while grids are cell oriented,
    the geometries here are shifted half an increment as default. To avoid this, use
    oricenter=False.

    Args:
        cube: The xtgeo Cube instance
        propname: Name of seismic property, if None then only the grid geometry
            will be made
        oricenter: Default is True, to treat seismic nodes as cell center values in
            a grid.

    .. versionadded:: 3.4
    """

    grd = create_box_grid(
        cube.dimensions,
        (cube.xori, cube.yori, cube.zori),
        oricenter=oricenter,
        increment=(cube.xinc, cube.yinc, cube.zinc),
        rotation=cube.rotation,
        flip=cube.yflip,
    )
    if propname is not None:
        grd.props = [
            xtgeo.GridProperty(
                ncol=cube.ncol,
                nrow=cube.nrow,
                nlay=cube.nlay,
                values=cube.values.copy(),
                name=propname,
            )
        ]
    return grd


def grid_from_surfaces(
    surfaces: xtgeo.Surfaces,
    ij_dimension: tuple[int, int] | None = None,
    ij_origin: tuple[float, float] | None = None,
    ij_increment: tuple[float, float] | None = None,
    rotation: float | None = None,
    tolerance: float | None = None,
) -> Grid:
    """Create a simple grid (non-faulted) from a stack of surfaces.

    The surfaces shall be sorted from top to base, and they should not cross in depth.
    In addition, it is required that they have the same settings (origin, rotation,
    etc).

    Args:
        surfaces: The Surfaces instance
        ij_dimension: The dimensions of the grid (ncol, nrow), default is to use the
            dimension from the surfaces
        ij_origin: The origin of the grid (x, y)
        ij_increment: The increment of the grid (xinc, yinc), default is to use the
            increment from the surfaces
        rotation: The rotation of the grid in degrees, default is to use the rotation
            from the surfaces
        tolerance: The tolerance for sampling the surfaces. In particualar if the grid
            origin, resolution and rotation are exactly the same as the surfaces,
            the grid may not be able to sample the surfaces exactly at edges.
            Lowering the tolerance may be used to avoid this. Default is 1e-6.

    Example::

        import xtgeo
        surf1 = xtgeo.surface_from_file("top.surf")
        surf2 = xtgeo.surface_from_file("base.surf")
        surfaces = xtgeo.Surfaces([surf1, surf2])

        grid = xtgeo.grid_from_surfaces(surfaces, ij_dimension=(20,30),
                                        ij_increment=(50, 50), rotation=10)

    """

    tolerance = tolerance if tolerance else 1e-6

    return _grid_etc1.create_grid_from_surfaces(
        surfaces, ij_dimension, ij_origin, ij_increment, rotation, tolerance
    )


class _GridCache:
    """Internal class for caching data related to grid.

    For example special grid, maps that are applied to speed up certain operations.

    The "onegrid" is a 3D grid with only one layer, utilizing that a corner point grid
    has straight pillars. The "top" and "base" surfaces are the top and base of the
    onelayer grid, and extracting indices may be useful for some operations.

    """

    def __init__(self, grid: Grid) -> None:
        logger.debug("Initialize cache for grid %s", grid.name)
        self.onegrid: Grid | None = None
        self.bbox: tuple[float, float, float, float, float, float] | None = None
        self.top_depth: xtgeo.RegularSurface | None = None
        self.base_depth: xtgeo.RegularSurface | None = None
        self.top_i_index: xtgeo.RegularSurface | None = None
        self.top_j_index: xtgeo.RegularSurface | None = None
        self.base_i_index: xtgeo.RegularSurface | None = None
        self.base_j_index: xtgeo.RegularSurface | None = None
        logger.debug("Initialize cache for grid .. %s", grid.name)

        self.threshold_magic_1: float = 0.1  # used in a C++ algorithm

        # cpp (pybind11) objects
        self.onegrid_cpp: GridCPP | None = None
        self.top_depth_cpp: RegularSurfaceCPP | None = None
        self.base_depth_cpp: RegularSurfaceCPP | None = None
        self.top_i_index_cpp: RegularSurfaceCPP | None = None
        self.top_j_index_cpp: RegularSurfaceCPP | None = None
        self.base_i_index_cpp: RegularSurfaceCPP | None = None
        self.base_j_index_cpp: RegularSurfaceCPP | None = None
        logger.debug("Initialize cache for grid .... %s", grid.name)

        # these are special SWIG arrays kept until necessary SWIG methods are replaced
        # with pypind 11 / C++
        self.top_i_index_carr: Any | None = None
        self.top_j_index_carr: Any | None = None
        self.base_i_index_carr: Any | None = None
        self.base_j_index_carr: Any | None = None

        self.name: str = grid.name
        self.hash = hash(grid)  # to remember the grid
        logger.debug("Initialize cache for grid ...... %s", grid.name)
        # initialize the cache with a one layer grid and surfaces
        self._initialize(grid)
        logger.debug("Initialized cache for grid %s, DONE", grid.name)

    @staticmethod
    def _get_swig_carr_double(surface: xtgeo.RegularSurface) -> Any:
        carr = _cxtgeo.new_doublearray(surface.ncol * surface.nrow)
        _cxtgeo.swig_numpy_to_carr_1d(surface.get_values1d(), carr)
        return carr

    def _initialize(self, grid: Grid) -> None:
        """Initialize the cache with a one layer grid and surfaces."""
        grid._set_xtgformat2()  # ensure xtgformat 2
        grid_cpp = grid._get_grid_cpp()

        dz = grid.get_dz()
        self.threshold_magic_1 = dz.values.mean()  # cf. get_indices_from_pointset

        logger.debug("Extracting one layer grid from %s", grid.name)
        coordsv1, zcornsv1, actnumsv1 = grid_cpp.extract_onelayer_grid()

        one = self.onegrid = Grid(
            coordsv=coordsv1,
            zcornsv=zcornsv1,
            actnumsv=actnumsv1,
        )

        logger.debug("Created one layer grid in python from %s", grid.name)

        minv, maxv = one._get_grid_cpp().get_bounding_box()
        self.bbox = (minv.x, minv.y, minv.z, maxv.x, maxv.y, maxv.z)

        self.top_depth = surface_from_grid3d(
            one, where="top", property="depth", rfactor=4, index_position="top"
        )
        self.base_depth = surface_from_grid3d(
            one, where="base", property="depth", rfactor=4, index_position="base"
        )
        self.top_i_index = surface_from_grid3d(
            one, where="top", property="i", rfactor=4, index_position="top"
        )
        self.top_j_index = surface_from_grid3d(
            one, where="top", property="j", rfactor=4, index_position="top"
        )
        self.base_i_index = surface_from_grid3d(
            one, where="base", property="i", rfactor=4, index_position="base"
        )
        self.base_j_index = surface_from_grid3d(
            one, where="base", property="j", rfactor=4, index_position="base"
        )

        # need to fill (interpolate) eventual holes in maps
        self.top_depth.fill()
        self.base_depth.fill()
        self.top_i_index.fill()
        self.top_j_index.fill()
        self.base_i_index.fill()
        self.base_j_index.fill()

        self.onegrid_cpp = one._get_grid_cpp()
        self.top_depth_cpp = _internal.regsurf.RegularSurface(self.top_depth)
        self.base_depth_cpp = _internal.regsurf.RegularSurface(self.base_depth)
        self.top_i_index_cpp = _internal.regsurf.RegularSurface(self.top_i_index)
        self.top_j_index_cpp = _internal.regsurf.RegularSurface(self.top_j_index)
        self.base_i_index_cpp = _internal.regsurf.RegularSurface(self.base_i_index)
        self.base_j_index_cpp = _internal.regsurf.RegularSurface(self.base_j_index)

        # these are special SWIG array objects kept until necessary SWIG
        # methods are replaced
        self.top_i_index_carr = self._get_swig_carr_double(self.top_i_index)
        self.top_j_index_carr = self._get_swig_carr_double(self.top_j_index)
        self.base_i_index_carr = self._get_swig_carr_double(self.base_i_index)
        self.base_j_index_carr = self._get_swig_carr_double(self.base_j_index)

        logger.info("Initialized cache for grid %s", grid.name)

    def clear(self) -> None:
        """Clear the cache."""
        self.threshold_magic_1 = 0.1
        self.onegrid = None
        self.bbox = None
        self.top_depth = None
        self.base_depth = None
        self.top_i_index = None
        self.top_j_index = None
        self.base_i_index = None
        self.base_j_index = None

        self.onegrid_cpp = None
        self.top_depth_cpp = None
        self.base_depth_cpp = None
        self.top_i_index_cpp = None
        self.top_j_index_cpp = None
        self.base_i_index_cpp = None
        self.base_j_index_cpp = None

        self.top_i_index_carr = None
        self.top_j_index_carr = None
        self.base_i_index_carr = None
        self.base_j_index_carr = None

        logger.info("Clear cache for grid %s", self.name)


# --------------------------------------------------------------------------------------
# Comment on dual porosity grids:
#
# Simulation grids may hold a "dual poro" or/and a "dual perm" system. This is
# supported here for EGRID format (only, so far), which:
# * Index 5 in FILEHEAD will be 1 if dual poro is True
# * Index 5 in FILEHEAD will be 2 if dual poro AND dual perm is True
# * ACTNUM values will be: 0, 1, 2 (inactive) or 3 (active) instead of normal
#   0 / 1 in the file:
# * 0 both Fracture and Matrix are inactive
# * 1 Matrix is active, Fracture is inactive (set to zero)
# * 2 Matrix is inactive (set to zero), Fracture is active
# * 3 Both Fracture and Matrix are active
#
#   However, XTGeo will convert this 0..3 scheme back to 0..1 scheme for ACTNUM!
#   In case of dualporo/perm, a special property holding the initial actnum
#   will be made, which is self._dualactnum
#
# The property self._dualporo is True in case of Dual Porosity
# BOTH self._dualperm AND self._dualporo are True in case of Dual Permeability
#
# All properties in a dual p* system will be given a postfix "M" of "F", e.g.
# PORO -->  POROM and POROF
# --------------------------------------------------------------------------------------

IJKRange: tuple[int, int, int, int, int, int]


class Grid(_Grid3D):
    """Class for a 3D grid corner point geometry in XTGeo.

    I.e. the geometric grid cells and the active cell indicator.

    The grid geometry class instances are normally created when
    importing a grid from file, as it is normally too complex to create from
    scratch.

    Args:
        coordsv: numpy array of dtype float64 and dimensions (nx + 1, ny + 1, 6)
            Giving the x,y,z values of the upper and lower corners in the grid.
        zcornsv: numpy array of dtype float32 and dimensions (nx + 1, ny + 1, nz + 1, 4)
            giving the sw, se, nw, ne corners along the i,jth corner line for
            the kth layer.
        actnumsv: numpy array of dtype int32 and dimensions (nx, ny, nz) giving
            the activity number for each cell. 0 means inactive, 1 means
            active. For dualporo=True/dualperm=True grids, value can also be 2
            or 3 meaning rock volume only and pore volume only respectively.
        dualporo (bool): True if dual porosity grid.
        dualperm (bool): True if dual permeability grid.
        subgrids: dictionary giving names to subset of layers. Has name as key and
            list of layer indices as values. Defaults to no names given.
        units: The length units the coordinates are in,
            (either Units.CM, Units.METRES, Units.FEET for cm, metres and
            feet respectively).  Default (None) is unitless.
        filesrc: Optional filename of grid.
        props: GridProperties instance containing the properties of the grid, defaults
            to empty instance.
        name: Optional name of the grid.
        roxgrid: Roxar Grid the Grid originates from if any, defaults to no such grid.
        roxindexer: Roxar grid indexer for the roxgrid. Defaults to no such indexer.

    See Also:
        The :class:`.GridProperty` and the :class:`.GridProperties` classes.

    """

    def __init__(
        self,
        coordsv: np.ndarray,
        zcornsv: np.ndarray,
        actnumsv: np.ndarray,
        dualporo: bool = False,
        dualperm: bool = False,
        subgrids: dict | None = None,
        units: Units | None = None,
        filesrc: pathlib.Path | str | None = None,
        props: GridProperties | None = None,
        name: str | None = None,
        roxgrid: Any | None = None,
        roxindexer: Any | None = None,
    ):
        logger.debug("Initialize Grid...")
        coordsv = np.asarray(coordsv)
        zcornsv = np.asarray(zcornsv)
        actnumsv = np.asarray(actnumsv)
        if coordsv.dtype != np.float64:
            raise TypeError(
                f"The dtype of the coordsv array must be float64, got {coordsv.dtype}"
            )
        if zcornsv.dtype != np.float32:
            raise TypeError(
                f"The dtype of the zcornsv array must be float32, got {zcornsv.dtype}"
            )
        if actnumsv.dtype != np.int32:
            raise TypeError(
                f"The dtype of the actnumsv array must be int32, got {actnumsv.dtype}"
            )
        if len(coordsv.shape) != 3 or coordsv.shape[2] != 6:
            raise ValueError(
                f"shape of coordsv should be (nx+1,ny+1,6), got {coordsv.shape}"
            )
        if len(zcornsv.shape) != 4 or zcornsv.shape[3] != 4:
            raise ValueError(
                f"shape of zcornsv should be (nx+1,ny+1,nz+1, 4), got {zcornsv.shape}"
            )
        if zcornsv.shape[0:2] != coordsv.shape[0:2]:
            raise ValueError(
                f"Mismatch between zcornsv and coordsv shape: {zcornsv.shape}"
                f" vs {coordsv.shape}"
            )
        if np.any(np.asarray(zcornsv.shape[0:3]) != np.asarray(actnumsv.shape) + 1):
            raise ValueError(
                f"Mismatch between zcornsv and actnumsv shape: {zcornsv.shape}"
                f" vs {actnumsv.shape}"
            )

        super().__init__(*actnumsv.shape)

        self._xtgformat = 2
        self._ncol = actnumsv.shape[0]
        self._nrow = actnumsv.shape[1]
        self._nlay = actnumsv.shape[2]

        self._coordsv = coordsv
        self._zcornsv = zcornsv
        self._actnumsv = actnumsv
        self._dualporo = dualporo
        self._dualperm = dualperm

        # this is a reference to the Grid object in C++. Never access this directly,
        # but use self._get_grid_cpp() function which validates against hash
        self._grid_cpp = _internal.grid3d.Grid(self)
        logger.debug("Initialize Grid... grid_cpp initialized")

        self._filesrc = filesrc

        self._props: GridProperties | None = (
            GridProperties(props=[]) if props is None else props
        )

        self._name = name
        self._subgrids = subgrids
        self._ijk_handedness: Literal["left", "right"] | None = None
        logger.debug("Initialize Grid... subgrids initialized")

        self._dualactnum = None
        if dualporo:
            self._dualactnum = self.get_actnum(name="DUALACTNUM")
            acttmp = self._dualactnum.copy()
            acttmp.values[acttmp.values >= 1] = 1
            self.set_actnum(acttmp)

        self._metadata = xtgeo.MetaDataCPGeometry()
        self._metadata.required = self
        logger.debug("Initialize Grid... metadata initialized")

        # Roxar api spesific:
        self._roxgrid = roxgrid
        self._roxindexer = roxindexer
        logger.debug("Initialize Grid... roxar api initialized (if required)")

        self.units = units

        self._ijk_handedness = _estimate_grid_ijk_handedness(coordsv.copy())
        logger.debug(
            "Initialize Grid... ijk_handedness set to %s", self._ijk_handedness
        )

        self._hash = hash(self)

        self._tmp = {}  # TMP!
        self._cache: _GridCache | None = None
        logger.debug("Initialize Grid... DONE!")

    def __repr__(self) -> str:
        """The __repr__ method."""
        logger.info("Invoke __repr__ for grid")
        return (
            f"{self.__class__.__name__} (id={id(self)}) ncol={self._ncol!r}, "
            f"nrow={self._nrow!r}, nlay={self._nlay!r}, filesrc={self._filesrc!r}"
        )

    def __str__(self) -> str:
        """The __str__ method for user friendly print."""
        logger.debug("Invoke __str__ for grid", stack_info=True)

        return self.describe(flush=False) or ""

    def __hash__(self):
        """The __hash__ method, i.e hash(self)."""

        return hash(self.generate_hash())

    # ==================================================================================
    # Public Properties:
    # ==================================================================================

    @property
    def metadata(self) -> xtgeo.MetaDataCPGeometry:
        """obj: Return or set metadata instance of type MetaDataCPGeometry."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj: xtgeo.MetaDataCPGeometry) -> None:
        # The current metadata object can be replaced. A bit dangerous so further
        # check must be done to validate. TODO.
        if not isinstance(obj, xtgeo.MetaDataCPGeometry):
            raise ValueError("Input obj not an instance of MetaDataCPGeometry")

        self._metadata = obj  # checking is currently missing! TODO

    @property
    def filesrc(self) -> str | pathlib.Path | None:
        """str: Source for grid (filepath or name in RMS)."""
        return self._filesrc

    @property
    def name(self) -> str | None:
        """str: Name attribute of grid."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise ValueError("Input name is not a text string")
        self._name = name

    @property
    def dimensions(self) -> Dimensions:
        """Dimensions NamedTuple: The grid dimensions (read only)."""
        return Dimensions(self.ncol, self.nrow, self.nlay)

    @property
    def vectordimensions(self) -> tuple[int, int, int]:
        """3-tuple: The storage grid array dimensions tuple of 3 integers (read only).

        The tuple is (ncoord, nzcorn, nactnum).
        """
        ncoord = (self.ncol + 1) * (self.nrow + 1) * 2 * 3
        nzcorn = self.ncol * self.nrow * (self.nlay + 1) * 4
        ntot = self.ncol * self.nrow * self.nlay

        return (ncoord, nzcorn, ntot)

    @property
    def ijk_handedness(self) -> Literal["left", "right"] | None:
        """str: IJK handedness for grids, "right" or "left".

        For a non-rotated grid with K increasing with depth, 'left' is corner in
        lower-left, while 'right' is origin in upper-left corner.

        """
        return self._ijk_handedness

    @ijk_handedness.setter
    def ijk_handedness(self, value: Literal["left", "right"]) -> None:
        if value not in ("right", "left"):
            raise ValueError("The value must be 'right' or 'left'")
        self.reverse_row_axis(ijk_handedness=value)

    @property
    def subgrids(self) -> dict[str, range | list[int]] | None:
        """:obj:`list` of :obj:`int`: A dict with subgrid name and an array as value.

        I.e. a dict on the form ``{"name1": [1, 2, 3, 4], "name2": [5, 6, 7],
        "name3": [8, 9, 10]}``, here meaning 3 subgrids where upper is 4
        cells vertically, then 3, then 3. The numbers must sum to NLAY.

        The numbering in the arrays are 1 based; meaning uppermost layer is 1
        (not 0).

        None will be returned if no subgrid indexing is present.

        See also :meth:`set_subgrids()` and :meth:`get_subgrids()` which
        have a similar function, but differs a bit.

        Note that this design is a bit different from the Roxar API, where
        repeated sections are allowed, and where indices start from 0,
        not one.
        """
        return None if self._subgrids is None else self._subgrids

    @subgrids.setter
    def subgrids(
        self,
        sgrids: dict[str, range | list[int]] | None,
    ) -> None:
        if sgrids is None:
            self._subgrids = None
            return

        if not isinstance(sgrids, dict):
            raise ValueError("Input to subgrids must be an ordered dictionary")

        lengths = 0
        zarr: list[Hashable] = []
        keys: list[Hashable] = []

        for key, val in sgrids.items():
            lengths += len(val)
            keys.append(key)
            zarr.extend(val)

        if lengths != self._nlay:
            raise ValueError(
                f"Subgrids lengths <{lengths}> not equal NLAY <{self.nlay}>"
            )

        if set(zarr) != set(range(1, self._nlay + 1)):
            raise ValueError(
                f"Arrays are not valid as the do not sum to vertical range, {zarr}"
            )

        if len(keys) != len(set(keys)):
            raise ValueError(f"Subgrid keys are not unique: {keys}")

        self._subgrids = sgrids

    @property
    def nactive(self) -> int:
        """int: Returns the number of active cells (read only)."""
        return len(self.actnum_indices)

    @property
    def actnum_array(self) -> np.ndarray:
        """Returns the 3D ndarray which for active cells.

        Values are 1 for active, 0 for inactive, in C order (read only).

        """
        actnumv = self.get_actnum().values
        return ma.filled(actnumv, fill_value=0)

    @property
    def actnum_indices(self) -> np.ndarray:
        """:obj:np.ndrarray: Indices (1D array) for active cells (read only).

        In dual poro/perm systems, this will be the active indices for the
        matrix cells and/or fracture cells (i.e. actnum >= 1).
        """
        actnumv = self.get_actnum()
        actnumv = np.ravel(actnumv.values)
        return np.flatnonzero(actnumv)

    @property
    def ntotal(self) -> int:
        """Returns the total number of cells (read only)."""
        return self.ncol * self.nrow * self.nlay

    @property
    def dualporo(self) -> bool:
        """Boolean flag for dual porosity scheme (read only)."""
        return self._dualporo

    @property
    def dualperm(self) -> bool:
        """Boolean flag for dual porosity scheme (read only)."""
        return self._dualperm

    @property
    def gridprops(self) -> GridProperties:
        """Return or set a XTGeo GridProperties objects attached to the Grid."""
        # Note, internally, the _props is a GridProperties instance, which is
        # a class that holds a list of properties.
        # Note that the `props` methods below will deal with properties in a
        # list context

        return self._props

    @gridprops.setter
    def gridprops(self, gprops: GridProperties) -> None:
        if not isinstance(gprops, GridProperties):
            raise ValueError("Input must be a GridProperties instance")

        self._props = gprops  # self._props is a GridProperties instance

    @property
    def props(self) -> list[GridProperty] | None:
        """Return or set a list of XTGeo GridProperty objects.

        When setting, the dimension of the property object is checked,
        and will raise an IndexError if it does not match the grid.

        When setting props, the current property list is replaced.

        See also :meth:`append_prop()` method to add a property to the current list.

        """
        # Note, internally, the _props is a GridProperties instance, which is
        # a class that holds a list of properties.

        if isinstance(self._props, GridProperties):
            return self._props.props
        if isinstance(self._props, list):
            raise RuntimeError("self._props is a list, not a GridProperties instance")
        return None

    @props.setter
    def props(self, plist: list[GridProperty]) -> None:
        if not isinstance(plist, list):
            raise ValueError("Input to props must be a list")

        for gridprop in plist:
            if gridprop.dimensions != self.dimensions:
                raise IndexError(
                    f"Property NX NY NZ <{gridprop.name}> does not match grid!"
                )

        self._props.props = plist  # self._props is a GridProperties instance

    @property
    def propnames(self) -> list[str] | None:
        """Returns a list of property names that are hooked to a grid."""
        return None if self._props is None else self._props.names

    @property
    def roxgrid(self) -> Any | None:
        """Get the Roxar native proj.grid_models[gname].get_grid() object."""
        return self._roxgrid

    @property
    def roxindexer(self) -> Any | None:
        """The Roxar native proj.grid_models[gname].get_grid().grid_indexer object."""
        return self._roxindexer

    # ==================================================================================
    # Private, special
    # TODO: do a threading lock of the cache
    # ==================================================================================

    def _get_grid_cpp(self) -> GridCPP:
        """Get the C++ grid object, creating or updating if needed."""
        grid_cpp = getattr(self, "_grid_cpp", None)
        current_hash = hash(self)
        if grid_cpp is not None and self._hash == current_hash:
            logger.info("Cache for python Grid is valid, returning current")
            return grid_cpp

        # update reference to C++ Grid object if hash has changed
        self._grid_cpp = _internal.grid3d.Grid(self)
        self._hash = current_hash
        return self._grid_cpp

    def _get_cache(self) -> _GridCache:
        """Get the grid cache object, creating or updating if needed."""
        cache = getattr(self, "_cache", None)
        current_hash = hash(self)
        if cache is not None:
            if cache.hash == current_hash:
                logger.info("Cache for python Grid is valid, returning current")
                return cache
            logger.info(
                "Python Grid has changed, current cache is invalid, creating new"
            )
            cache.clear()
        self._cache = _GridCache(self)
        return self._cache

    def _clear_cache(self) -> None:
        """Clear the grid cache object."""
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()
            self._cache = None
            logger.debug("Cache cleared")
        else:
            logger.debug("No cache to clear")

    # ==================================================================================
    # Other
    # ==================================================================================
    def generate_hash(
        self,
        hashmethod: Literal["md5", "sha256", "blake2b"] = "md5",
    ) -> str:
        """Return a unique hash ID for current instance (for persistance).

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        .. versionadded:: 2.14
        """
        self._set_xtgformat2()  # ensure xtgformat 2!

        required = (
            "_ncol",
            "_nrow",
            "_nlay",
            "_coordsv",
            "_zcornsv",
            "_actnumsv",
        )

        gid = "".join(str(getattr(self, att)) for att in required)

        return generic_hash(gid, hashmethod=hashmethod)

    # ==================================================================================
    # Create/import/export
    # ==================================================================================

    def to_file(self, gfile: FileLike, fformat: str = "roff") -> None:
        """Export grid geometry to file, various vendor formats.

        Args:
            gfile (str): Name of output file
            fformat (str): File format; roff/roff_binary/roff_ascii/
                grdecl/bgrdecl/egrid.

        Raises:
            OSError: Directory does not exist

        Example::
            >>> grid = create_box_grid((2,2,2))
            >>> grid.to_file(outdir + "/myfile.roff")
        """
        _gfile = FileWrapper(gfile, mode="wb")

        _gfile.check_folder(raiseerror=OSError)

        if fformat in FileFormat.ROFF_BINARY.value:
            _grid_export.export_roff(self, _gfile.name, "binary")
        elif fformat in FileFormat.ROFF_ASCII.value:
            _grid_export.export_roff(self, _gfile.name, "ascii")
        elif fformat in FileFormat.GRDECL.value:
            _grid_export.export_grdecl(self, _gfile.name, 1)
        elif fformat in FileFormat.BGRDECL.value:
            _grid_export.export_grdecl(self, _gfile.name, 0)
        elif fformat in FileFormat.EGRID.value:
            _grid_export.export_egrid(self, _gfile.name)
        elif fformat in FileFormat.FEGRID.value:
            _grid_export.export_fegrid(self, _gfile.name)
        elif fformat in FileFormat.HDF.value:
            self.to_hdf(gfile)
        elif fformat in FileFormat.XTG.value:
            self.to_xtgf(gfile)
        else:
            extensions = FileFormat.extensions_string(
                [
                    FileFormat.ROFF_BINARY,
                    FileFormat.ROFF_ASCII,
                    FileFormat.EGRID,
                    FileFormat.FEGRID,
                    FileFormat.GRDECL,
                    FileFormat.BGRDECL,
                    FileFormat.XTG,
                    FileFormat.HDF,
                ]
            )
            raise InvalidFileFormatError(
                f"File format {fformat} is invalid for type Grid. "
                f"Supported formats are {extensions}."
            )

    def to_hdf(
        self,
        gfile: str | pathlib.Path,
        compression: str | None = None,
        chunks: bool | None = False,
        subformat: int | None = 844,
    ) -> FileLike:
        """Export grid geometry to HDF5 storage format (experimental!).

        Args:
            gfile: Name of output file
            compression: Compression method, such as "blosc" or "lzf"
            chunks: chunks settings
            subformat: Format of output arrays in terms of bytes. E.g. 844 means
                8 byte for COORD, 4 byte for ZCORNS, 4 byte for ACTNUM.

        Raises:
            OSError: Directory does not exist

        Returns:
            Used file object, or None if memory stream

        Example:

            >>> grid = create_box_grid((2,2,2))
            >>> filename = grid.to_hdf(outdir + "/myfile_grid.h5")
        """
        _gfile = FileWrapper(gfile, mode="wb", obj=self)
        _gfile.check_folder(raiseerror=OSError)

        _grid_export.export_hdf5_cpgeom(
            self, _gfile, compression=compression, chunks=chunks, subformat=subformat
        )

        return _gfile.file

    def to_xtgf(
        self,
        gfile: str | pathlib.Path,
        subformat: int | None = 844,
    ) -> pathlib.Path:
        """Export grid geometry to xtgeo native binary file format (experimental!).

        Args:
            gfile: Name of output file
            subformat: Format of output arryas in terms of bytes. E.g. 844 means
                8 byte for COORD, 4 byte for ZCORNS, 4 byte for ACTNUM.

        Raises:
            OSError: Directory does not exist

        Returns:
            gfile (pathlib.Path): Used pathlib.Path file object, or None if
                memory stream

        Example::
            >>> grid = create_box_grid((2,2,2))
            >>> filename = grid.to_xtgf(outdir + "/myfile.xtg")
        """
        _gfile = FileWrapper(gfile, mode="wb", obj=self)
        _gfile.check_folder(raiseerror=OSError)

        _grid_export.export_xtgcpgeom(self, _gfile, subformat=subformat)

        return _gfile.file

    def to_roxar(
        self,
        project: str,
        gname: str,
        realisation: int = 0,
        info: bool = False,
        method: Literal["cpg", "roff"] = "cpg",
    ) -> None:
        """Export (upload) a grid from XTGeo to RMS via Roxar API.

        Note:
            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Args:
            project (str or roxar._project): Inside RMS use the magic 'project',
                else use path to RMS project, or a project reference
            gname (str): Name of grid in RMS
            realisation (int): Realisation umber, default 0
            info (bool): TBD
            method (str): Save approach, the default is 'cpg' which applied the internal
                RMS API, while 'roff' will do a save to a temporary area, and then load
                into RMS. For strange reasons, the 'roff' method is per RMS version
                14.2 a faster method (strange since file i/o is way more costly than
                direct API access, in theory).

        Note:
            When storing grids that needs manipulation of inactive cells, .e.g.
            ``activate_all()`` method, using method='roff' is recommended. The reason
            is that saving cells using 'cpg' method will force zero depth values on
            inactive cells.

        """
        _grid_roxapi.save_grid_to_rms(
            self, project, gname, realisation, info=info, method=method
        )

    def convert_units(self, units: Units) -> None:
        """
        Convert the units of the grid.
        Args:
            units: The unit to convert to.
        Raises:
            ValueError: When the grid is unitless (no initial
                unit information available).
        """
        if self.units is None:
            raise ValueError("convert_units called on unitless grid.")
        if self.units == units:
            return
        factor = self.units.conversion_factor(units)
        self._coordsv *= factor
        self._zcornsv *= factor
        self.units = units

    # ==================================================================================
    # Various public methods
    # ==================================================================================

    def copy(self) -> Grid:
        """Copy from one existing Grid instance to a new unique instance.

        Note that associated properties will also be copied.

        Example::

            >>> grd = create_box_grid((5,5,5))
            >>> newgrd = grd.copy()
        """
        logger.info("Copy a Grid instance")
        return _grid_etc1.copy(self)

    def describe(
        self,
        details: bool = False,
        flush: bool = True,
    ) -> str | None:
        """Describe an instance by printing to stdout."""
        logger.info("Print a description...")

        dsc = XTGDescription()
        dsc.title("Description of Grid instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("File source", self._filesrc)
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        dsc.txt("Number of active cells", self.nactive)

        if details:
            geom = self.get_geometrics(cellcenter=True, return_dict=True)

            assert isinstance(geom, dict)

            prp1: list[str] = []
            for prp in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                prp1.append(f"{geom[prp]:10.3f}")

            prp2: list[str] = []
            for prp in ("avg_dx", "avg_dy", "avg_dz", "avg_rotation"):
                prp2.append(f"{geom[prp]:7.4f}")

            geox = self.get_geometrics(
                cellcenter=False, allcells=True, return_dict=True
            )
            assert isinstance(geox, dict)
            prp3: list[str] = []
            for prp in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                prp3.append(f"{geox[prp]:10.3f}")

            prp4 = []
            for prp in ("avg_dx", "avg_dy", "avg_dz", "avg_rotation"):
                prp4.append(f"{geox[prp]:7.4f}")

            dsc.txt("For active cells, using cell centers:")
            dsc.txt("Xmin, Xmax, Ymin, Ymax, Zmin, Zmax:", *prp1)
            dsc.txt("Avg DX, Avg DY, Avg DZ, Avg rotation:", *prp2)
            dsc.txt("For all cells, using cell corners:")
            dsc.txt("Xmin, Xmax, Ymin, Ymax, Zmin, Zmax:", *prp3)
            dsc.txt("Avg DX, Avg DY, Avg DZ, Avg rotation:", *prp4)

        dsc.txt("Attached grid props objects (names)", self.propnames)

        if details:
            dsc.txt("Attached grid props objects (id)", self.props)
        if self.subgrids:
            dsc.txt("Number of subgrids", len(list(self.subgrids.keys())))
        else:
            dsc.txt("Number of subgrids", "No subgrids")
        if details:
            dsc.txt("Subgrids details", json.dumps(self.get_subgrids()))
            dsc.txt("Subgrids with values array", self.subgrids)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def get_bounding_box(self) -> tuple[float, float, float, float, float, float]:
        """Get the bounding box of the grid.

        Returns:
            A tuple with the bounding box coordinates (xmin, ymin, zmin, xmax, ymax,
            zmax).

        Example::

            >>> import xtgeo
            >>> grd = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID", fformat="egrid")
            >>> bb = grd.get_bounding_box()

        _versionadded:: 4.9
        """
        min_pt, max_pt = self._get_grid_cpp().get_bounding_box()
        return (min_pt.x, min_pt.y, min_pt.z, max_pt.x, max_pt.y, max_pt.z)

    def get_dataframe(
        self,
        activeonly: bool = True,
        ijk: bool = True,
        xyz: bool = True,
        doubleformat: bool = False,
    ) -> pd.DataFrame:
        """Returns a Pandas dataframe for the grid and any attached grid properties.

        Note that this dataframe method is rather similar to GridProperties
        dataframe function, but have other defaults.

        Args:
            activeonly (bool): If True (default), return only active cells.
            ijk (bool): If True (default), show cell indices, IX JY KZ columns
            xyz (bool): If True (default), show cell center coordinates.
            doubleformat (bool): If True, floats are 64 bit, otherwise 32 bit.
                Note that coordinates (if xyz=True) is always 64 bit floats.

        Returns:
            A Pandas dataframe object

        Example::

            >>> import xtgeo
            >>> grd = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID", fformat="egrid")
            >>> names = ["SOIL", "SWAT", "PRESSURE"]
            >>> dates = [19991201]
            >>> xpr = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.UNRST",
            ...     fformat="unrst",
            ...     names=names,
            ...     dates=dates,
            ...     grid=grd,
            ... )
            >>> grd.gridprops = xpr  # attach properties to grid

            >>> df = grd.get_dataframe()

            >>> # save as CSV file
            >>> df.to_csv(outdir + "/mygrid.csv")
        """
        return self.gridprops.get_dataframe(
            grid=self,
            activeonly=activeonly,
            ijk=ijk,
            xyz=xyz,
            doubleformat=doubleformat,
        )

    def get_vtk_esg_geometry_data(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Get grid geometry data suitable for use with VTK's vtkExplicitStructuredGrid.

        Builds and returns grid geometry data in a format tailored for use with VTK's
        explicit structured grid (ESG). Essentially this entails building an
        unstructured grid representation where all the grid cells are represented as
        hexahedrons with explicit connectivities. The cell connectivity array refers
        into the accompanying vertex array.

        In VTK, cell order increases in I fastest, then J, then K.

        The returned tuple contains:
            - numpy array with dimensions in terms of points (not cells)
            - vertex array, numpy array with vertex coordinates
            - connectivity array for all the cells, numpy array with integer indices
            - inactive cell indices, numpy array with integer indices

        This function also tries to remove/weld duplicate vertices, but this is still
        a work in progress.

        Example usage with VTK::

            dims, vert_arr, conn_arr, inact_arr = xtg_grid.get_vtk_esg_geometry_data()

            vert_arr = vert_arr.reshape(-1, 3)
            vtk_points = vtkPoints()
            vtk_points.SetData(numpy_to_vtk(vert_arr, deep=1))

            vtk_cell_array = vtkCellArray()
            vtk_cell_array.SetData(8, numpy_to_vtkIdTypeArray(conn_arr, deep=1))

            vtk_esgrid = vtkExplicitStructuredGrid()
            vtk_esgrid.SetDimensions(dims)
            vtk_esgrid.SetPoints(vtk_points)
            vtk_esgrid.SetCells(vtk_cell_array)

            vtk_esgrid.ComputeFacesConnectivityFlagsArray()

            ghost_arr_vtk = vtk_esgrid.AllocateCellGhostArray()
            ghost_arr_np = vtk_to_numpy(ghost_arr_vtk)
            ghost_arr_np[inact_arr] = vtkDataSetAttributes.HIDDENCELL

        .. versionadded:: 2.20
        """
        return _grid_etc1.get_vtk_esg_geometry_data(self)

    def get_vtk_geometries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get necessary arrays on correct layout for VTK ExplicitStructuredGrid usage.

        Example::

            import pyvista as pv
            dim, crn, inactind = grd.get_vtk_geometries()
            grid = pv.ExplicitStructuredGrid(dim, crn)
            grid.flip_z(inplace=True)
            grid.hide_cells(inactind, inplace=True)
            grid.plot(show_edges=True)

        Returns:
            dims, corners, inactive_indices

        .. versionadded:: 2.18
        """

        return _grid_etc1.get_vtk_geometries(self)

    def append_prop(self, prop: GridProperty) -> None:
        """Append a single property to the grid."""
        if prop.dimensions != self.dimensions:
            raise ValueError(
                f"Dimensions does not match, got: {prop.dimensions} "
                f"expected: {self.dimensions}."
            )

        self._props.append_props([prop])

    def set_subgrids(self, sdict: dict[str, int] | None) -> None:
        """Set the subgrid from a simplified ordered dictionary.

        The simplified dictionary is on the form
        {"name1": 3, "name2": 5}

        Note that the input must be an dict!

        """
        if sdict is None:
            return

        if not isinstance(sdict, dict):
            raise ValueError("Input sdict is not an dict")

        newsub: dict[str, range | list[int]] = {}

        inn1 = 1
        for name, nsub in sdict.items():
            inn2 = inn1 + nsub
            newsub[name] = range(inn1, inn2)
            inn1 = inn2

        self.subgrids = newsub

    def get_subgrids(self) -> dict[str, int] | None:
        """Get the subgrids on a simplified ordered dictionary.

        The simplified dictionary is on the form {"name1": 3, "name2": 5}
        """
        if not self.subgrids:
            return None

        return {name: len(subb) for name, subb in self.subgrids.items()}

    def rename_subgrids(self, names: list[str] | tuple[str, ...]) -> None:
        """Rename the names in the subgrids with the new names.

        Args:
            names (list): List of new names, length of list must be same as length of
                subgrids


        Example::

            >>> grd = create_box_grid((3, 3, 3))
            >>> grd.subgrids = dict(
            ...     [("1", range(1,2)), ("2", range(2,3)), ("3", range(3,4))]
            ... )
            >>> grd.rename_subgrids(["Inky", "Tinky", "Pinky"])

        Raises:
            ValueError: Input names not a list or a tuple
            ValueError: Lenght of names list not same as number of subgrids

        .. versionadded:: 2.12
        """
        if not isinstance(names, (list, tuple)):
            raise ValueError("Input names not a list or a tuple")

        assert self.subgrids is not None
        if len(names) != len(list(self.subgrids.keys())):
            raise ValueError("Lenght of names list not same as number of subgrids")

        subs = self.get_subgrids()
        assert subs is not None
        subs_copy = subs.copy()
        for num, oldname in enumerate(self.subgrids.keys()):
            subs_copy[str(names[num])] = subs_copy.pop(oldname)

        self.set_subgrids(subs_copy)

    def estimate_design(
        self,
        nsub: str | int | None = None,
    ) -> dict[str, str | float] | None:
        """Estimate design and simbox thickness of the grid or a subgrid.

        If the grid consists of several subgrids, and nsub is not specified, then
        a failure should be raised.

        Args:
            nsub (int or str): Subgrid index to check, either as a number (starting
                with 1) or as subgrid name. If set to None, the whole grid will
                examined.

        Returns:
            result (dict): where key "design" gives one letter in(P, T, B, X, M)
                P=proportional, T=topconform, B=baseconform,
                X=underdetermined, M=Mixed conform. Key "dzsimbox" is simbox thickness
                estimate per cell. None if nsub is given, but subgrids are missing, or
                nsub (name or number) is out of range.

        Example::

            >>> import xtgeo
            >>> grd = xtgeo.grid_from_file(emerald_dir + "/emerald_hetero_grid.roff")
            >>> print(grd.subgrids)
            dict([('subgrid_0', range(1, 17)), ('subgrid_1', range(17, 47))])
            >>> res = grd.estimate_design(nsub="subgrid_0")
            >>> print("Subgrid design is", res["design"])
            Subgrid design is P
            >>> print("Subgrid simbox thickness is", res["dzsimbox"])
            Subgrid simbox thickness is 2.548...



        """
        nsubname = None

        if nsub is None and self.subgrids:
            raise ValueError("Subgrids exists, nsub cannot be None")

        if nsub is not None:
            if not self.subgrids:
                return None

            if isinstance(nsub, int):
                try:
                    nsubname = list(self.subgrids.keys())[nsub - 1]
                except IndexError:
                    return None

            elif isinstance(nsub, str):
                nsubname = nsub
            else:
                raise ValueError("Key nsub of wrong type, must be a number or a name")

            if nsubname not in self.subgrids:
                return None

        return _grid_etc1.estimate_design(self, nsubname)

    def estimate_flip(self) -> Literal[1, -1]:
        """Estimate flip (handedness) of grid returns as 1 or -1.

        The flip numbers are 1 for left-handed and -1 for right-handed.

        .. seealso:: :py:attr:`~ijk_handedness`
        """
        self._set_xtgformat2()  # ensure xtgformat 2!
        handedness = _estimate_grid_ijk_handedness(self._coordsv.copy())
        return -1 if handedness == "right" else 1

    def subgrids_from_zoneprop(self, zoneprop: GridProperty) -> dict[str, int] | None:
        """Estimate subgrid index from a zone property.

        The new will estimate which will replace the current if any.

        Args:
            zoneprop(GridProperty): a XTGeo GridProperty instance.

        Returns:
            Will also return simplified dictionary is on the form
                {"name1": 3, "name2": 5}
        """
        _, _, k_index = self.get_ijk()
        kval = k_index.values
        zprval = zoneprop.values
        minzone = int(zprval.min())
        maxzone = int(zprval.max())

        newd: dict[str, range] = {}
        for izone in range(minzone, maxzone + 1):
            mininzn = int(kval[zprval == izone].min())  # 1 base
            maxinzn = int(kval[zprval == izone].max())  # 1 base

            newd[zoneprop.codes.get(izone, "zone" + str(izone))] = range(
                mininzn, maxinzn + 1
            )

        self.subgrids = newd  # type: ignore

        return self.get_subgrids()

    def get_zoneprop_from_subgrids(self) -> NoReturn:
        """Make a XTGeo GridProperty instance for a Zone property subgrid index."""
        raise NotImplementedError("Not yet; todo")

    def get_boundary_polygons(
        self: Grid,
        alpha_factor: float = 1.0,
        convex: bool = False,
        simplify: bool | dict[str, Any] = True,
        filter_array: np.ndarray | None = None,
    ) -> Polygons:
        """Extract boundary polygons from the grid cell centers.

        A ``filter_array`` can be applied to extract boundaries around specific
        parts of the grid e.g. a region or a zone.

        The concavity and precision of the boundaries are controlled by the
        ``alpha_factor``. A low ``alpha_factor`` makes more precise boundaries,
        while a larger value makes more rough polygons.

        Note that the ``alpha_factor`` is a multiplier (default value 1) on top
        of an auto estimated value, derived from the maximum xinc and yinc from
        the grid cells. Dependent on the regularity of the grid, tuning of the
        alpha_factor (up/down) is sometimes necessary to get satisfactory results.

        Args:
            alpha_factor: An alpha multiplier, which controls the precision of the
                boundaries. A higher number will produce smoother and less accurate
                polygons. Not applied if convex is set to True.
            convex: The default is False, which means that a "concave hull" algorithm
                is used. If convex is True, the alpha factor is overridden to a large
                number, producing a 'convex' shape boundary instead.
            simplify: If True, a simplification is done in order to reduce the number
                of points in the polygons, where tolerance is 0.1. Another
                alternative to True is to input a Dict on the form
                ``{"tolerance": 2.0, "preserve_topology": True}``, cf. the
                :func:`Polygons.simplify()` method. For details on e.g. tolerance, see
                Shapely's simplify() method.
            filter_array: An numpy boolean array with equal shape as the grid dimension,
                used to filter the grid cells and define where to extract boundaries.

        Returns:
            A XTGeo Polygons instance

        Example::

            grid = xtgeo.grid_from_roxar(project, "Simgrid")
            # extract polygon for a specific region, here region 3
            region = xtgeo.gridproperty_from_roxar(project, "Simgrid", "Regions")
            filter_array = (region.values==3)
            boundary = grid.get_boundary_polygons(filter_array=filter_array)

        See also:
            The :func:`Polygons.boundary_from_points()` class method.

        """
        return _grid_boundary.create_boundary(
            self, alpha_factor, convex, simplify, filter_array
        )

    def get_actnum_indices(
        self,
        order: Literal["C", "F", "A", "K"] = "C",
        inverse: bool = False,
    ) -> np.ndarray:
        """Returns the 1D ndarray which holds the indices for active cells.

        Args:
            order (str): "Either 'C' (default) or 'F' order).
            inverse (bool): Default is False, returns indices for inactive cells
                if True.

        .. versionchanged:: 2.18 Added inverse option
        """
        actnumv = self.get_actnum().values.copy(order=order)
        actnumv = np.ravel(actnumv, order=order)
        if inverse:
            actnumv -= 1
            return np.flatnonzero(actnumv)
        return np.flatnonzero(actnumv)

    def get_dualactnum_indices(
        self,
        order: Literal["C", "F", "A", "K"] = "C",
        fracture: bool = False,
    ) -> np.ndarray | None:
        """Returns the 1D ndarray which holds the indices for matrix/fracture cases.

        Args:
            order (str): "Either 'C' (default) or 'F' order).
            fracture (bool): If True use Fracture properties.
        """
        if not self._dualporo:
            return None

        assert self._dualactnum is not None
        actnumv = self._dualactnum.values.copy(order=order)
        actnumv = np.ravel(actnumv, order=order)

        if fracture:
            actnumvf = actnumv.copy()
            actnumvf[(actnumv == 3) | (actnumv == 2)] = 1
            actnumvf[(actnumv == 1) | (actnumv == 0)] = 0
            return np.flatnonzero(actnumvf)

        actnumvm = actnumv.copy()
        actnumvm[(actnumv == 3) | (actnumv == 1)] = 1
        actnumvm[(actnumv == 2) | (actnumv == 0)] = 0
        return np.flatnonzero(actnumvm)

    def get_prop_by_name(self, name: str) -> GridProperty | None:
        """Gets a property object by name lookup, return None if not present."""

        if self.props is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no gird "
                "property objects (self.props is None)"
            )

        for obj in self.props:
            if obj.name == name:
                return obj

        return None

    def get_actnum(
        self,
        name: str = "ACTNUM",
        asmasked: bool = False,
        dual: bool = False,
    ) -> GridProperty:
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): Actnum is returned with all cells shown
                as default. Use asmasked=True to make 0 entries masked.
            dual (bool): If True, and the grid is a dualporo/perm grid, an
                extended ACTNUM is applied (numbers 0..3)

        Example::

            >>> import xtgeo
            >>> mygrid = xtgeo.create_box_grid((2,2,2))
            >>> act = mygrid.get_actnum()
            >>> print("{}% of cells are active".format(act.values.mean() * 100))
            100.0% of cells are active

        .. versionchanged:: 2.6 Added ``dual`` keyword
        """
        if dual and self._dualactnum:
            act = self._dualactnum.copy()
        else:
            act = xtgeo.grid3d.GridProperty(
                ncol=self._ncol,
                nrow=self._nrow,
                nlay=self._nlay,
                values=np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.int32),
                name=name,
                discrete=True,
            )

            if self._xtgformat == 1:
                values = _gridprop_lowlevel.f2c_order(self, self._actnumsv)
            else:
                values = self._actnumsv

            act.values = values
            act.mask_undef()

        if asmasked:
            act.values = ma.masked_equal(act.values, 0)

        act.codes = {0: "0", 1: "1"}
        if dual and self._dualactnum:
            act.codes = {0: "0", 1: "1", 2: "2", 3: "3"}

        return act

    def set_actnum(self, actnum: GridProperty) -> None:
        """Modify the existing active cell index, ACTNUM.

        Args:
            actnum (GridProperty): a gridproperty instance with 1 for active
                cells, 0 for inactive cells

        Example::
            >>> mygrid = create_box_grid((5,5,5))
            >>> act = mygrid.get_actnum()
            >>> act.values[:, :, :] = 1
            >>> act.values[:, :, 4] = 0
            >>> mygrid.set_actnum(act)
        """
        val1d = actnum.values.ravel()

        if self._xtgformat == 1:
            self._actnumsv = _gridprop_lowlevel.c2f_order(self, val1d)
        else:
            self._actnumsv = np.ma.filled(actnum.values, fill_value=0).astype(np.int32)

    def get_dz(
        self,
        name: str = "dZ",
        flip: bool = True,
        asmasked: bool = True,
        metric: METRIC = "z projection",
    ) -> GridProperty:
        """Return the dZ as GridProperty object.

        Returns the average length of z direction edges for each
        cell as a GridProperty. The length is by default the
        z delta, ie. projected onto the z dimension (see the metric parameter).

        Args:
            name (str): name of property
            flip (bool): Use False for Petrel grids were Z is negative down
                (experimental)
            asmasked (bool): True if only for active cells, False for all cells
            metric (str): One of the following metrics:
                * "euclid": sqrt(dx^2 + dy^2 + dz^2)
                * "horizontal": sqrt(dx^2 + dy^2)
                * "east west vertical": sqrt(dy^2 + dz^2)
                * "north south vertical": sqrt(dx^2 + dz^2)
                * "x projection": dx
                * "y projection": dy
                * "z projection": dz

        Returns:
            A XTGeo GridProperty object dZ
        """
        return _grid_etc1.get_dz(
            self,
            name=name,
            flip=flip,
            asmasked=asmasked,
            metric=metric,
        )

    def get_dx(
        self,
        name: str = "dX",
        asmasked: bool = True,
        metric: METRIC = "horizontal",
    ) -> GridProperty:
        """Return the dX as GridProperty object.

        Returns the average length of x direction edges for each
        cell as a GridProperty. The length is by default horizontal
        vector length (see the metric parameter).

        Args:
            name (str): names of properties
            asmasked (bool). If True, make a np.ma array where inactive cells
                are masked.
            metric (str): One of the following metrics:
                * "euclid": sqrt(dx^2 + dy^2 + dz^2)
                * "horizontal": sqrt(dx^2 + dy^2)
                * "east west vertical": sqrt(dy^2 + dz^2)
                * "north south vertical": sqrt(dx^2 + dz^2)
                * "x projection": dx
                * "y projection": dy
                * "z projection": dz

        Returns:
            XTGeo GridProperty objects containing dx.
        """
        return _grid_etc1.get_dx(self, name=name, asmasked=asmasked, metric=metric)

    def get_dy(
        self,
        name: str = "dY",
        asmasked: bool = True,
        metric: METRIC = "horizontal",
    ) -> GridProperty:
        """Return the dY as GridProperty object.

        Returns the average length of y direction edges for each
        cell as a GridProperty. The length is by default horizontal
        vector length (see the metric parameter).

        Args:
            name (str): names of properties
            asmasked (bool). If True, make a np.ma array where inactive cells
                are masked.
            metric (str): One of the following metrics:
                * "euclid": sqrt(dx^2 + dy^2 + dz^2)
                * "horizontal": sqrt(dx^2 + dy^2)
                * "east west vertical": sqrt(dy^2 + dz^2)
                * "north south vertical": sqrt(dx^2 + dz^2)
                * "x projection": dx
                * "y projection": dy
                * "z projection": dz

        Returns:
            Two XTGeo GridProperty objects (dx, dy).
        """
        return _grid_etc1.get_dy(self, name=name, asmasked=asmasked, metric=metric)

    def get_cell_volume(
        self,
        ijk: tuple[int, int, int] = (1, 1, 1),
        activeonly: bool = True,
        zerobased: bool = False,
        precision: Literal[1, 2, 4] = 2,
    ) -> float:
        """Return the bulk volume for a given cell.

        This method is currently *experimental*.

        A bulk volume of a cornerpoint cell is actually a non-trivial and a non-unique
        entity. The volume is approximated by dividing the cell (hexahedron) into
        6 tetrehedrons; there is however a large number of ways to do this division.

        As default (precision=2) an average of two different ways to divide the cell
        into tetrahedrons is averaged.

        Args:
            ijk (tuple): A tuple of I J K (NB! cell counting starts from 1
                unless zerobased is True).
            activeonly (bool): Skip undef cells if True; return None for inactive.
            precision (int): An even number indication precision level,where
                a higher number means increased precision but also increased computing
                time. Currently 1, 2, 4 are supported.

        Returns:
            Cell total bulk volume

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> print(grid.get_cell_volume(ijk=(10,13,2)))
            107056...

        .. versionadded:: 2.13 (as experimental)
        """
        return _grid_etc1.get_cell_volume(
            self,
            ijk=ijk,
            activeonly=activeonly,
            zerobased=zerobased,
            precision=precision,
        )

    def get_bulk_volume(
        self,
        name: str = "bulkvol",
        asmasked: bool = True,
        precision: Literal[1, 2, 4] = 2,
    ) -> GridProperty:
        """Return the geometric cell volume for all cells as a GridProperty object.

        This method is currently *experimental*.

        A bulk volume of a cornerpoint cell is actually a non-trivial and a non-unique
        entity. The volume is approximated by dividing the cell (hexahedron) into
        6 tetrehedrons; there is however a large number of ways to do this division.

        As default (precision=2) an average of two different ways to divide the cell
        into tetrahedrons is averaged.

        Args:
            name (str): name of property, default to "bulkvol"
            asmasked (bool). If True, make a np.ma array where inactive cells
                are masked. Otherwise a numpy array will all bulk for all cells is
                returned
            precision (int): Not applied!

        Returns:
            XTGeo GridProperty object

        .. versionadded:: 2.13 (as experimental)

        """
        return _grid_etc1.get_bulk_volume(
            self, name=name, precision=precision, asmasked=asmasked
        )

    def get_heights_above_ffl(
        self,
        ffl: GridProperty,
        option: str = "cell_center_above_ffl",
    ) -> tuple[GridProperty, GridProperty, GridProperty]:
        """Returns 3 properties: htop, hbot and hmid, primarely for use in Sw models."

        Args:
            ffl: Free fluid level e.g. FWL (or level whatever is required; a level from
                which cells above will be shown as delta heights (positive), while
                cells below will have 0.0 values.
            option: How to compute values, as either "cell_center_above_ffl" or
                "cell_corners_above_ffl". The first one looks at cell Z centerlines, and
                compute the top, the bottom and the midpoint. The second will look at
                cell corners, using the uppermost corner for top, and the lowermost
                corner for bottom. In both cases, values are modified if cell is
                intersected with the provided ffl.

        Returns:
            (htop, hbot, hmid) delta heights, as xtgeo GridProperty objects

        .. versionadded:: 3.9

        """
        return _grid_etc1.get_heights_above_ffl(self, ffl=ffl, option=option)

    def get_property_between_surfaces(
        self,
        top: xtgeo.RegularSurface,
        base: xtgeo.RegularSurface,
        value: int = 1,
        name: str = "between_surfaces",
    ) -> GridProperty:
        """Returns a 3D GridProperty object with <value> between two surfaces."

        Args:
            top: The bounding top surface (RegularSurface object)
            base: The bounding base surface (RegularSurface object)
            value: An integer > 0 to assign to cells between surfaces, 1 as default
            name: Name of the property, default is "between_surfaces"

        Returns:
            xtgeo GridProperty object with <value> if cell center is between surfaces,
            otherwise 0. Note that the property wil be discrete if input value is an
            integer, otherwise it will be continuous.

        .. versionadded:: 4.5

        """
        return _grid_etc1.get_property_between_surfaces(self, top, base, value, name)

    def get_ijk(
        self,
        names: tuple[str, str, str] = ("IX", "JY", "KZ"),
        asmasked: bool = True,
        zerobased: bool = False,
    ) -> tuple[GridProperty, GridProperty, GridProperty]:
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter, J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            asmasked: If True, UNDEF cells are masked, default is True
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """
        ixc, jyc, kzc = _grid_etc1.get_ijk(
            self, names=names, asmasked=asmasked, zerobased=zerobased
        )

        # return the objects
        return ixc, jyc, kzc

    def get_ijk_from_points(
        self,
        points: Points,
        activeonly: bool = True,
        zerobased: bool = False,
        dataframe: bool = True,
        includepoints: bool = True,
        columnnames: tuple[str, str, str] = ("IX", "JY", "KZ"),
        fmt: Literal["int", "float"] = "int",
        undef: int = -1,
    ) -> pd.DataFrame | list:
        """Returns a list/dataframe of cell indices based on a Points() instance.

        If a point is outside the grid, -1 values are returned

        Args:
            points (Points): A XTGeo Points instance
            activeonly (bool): If True, UNDEF cells are not included
            zerobased (bool): If True, counter start from 0, otherwise 1 (default=1).
            dataframe (bool): If True result is Pandas dataframe, otherwise a list
                of tuples
            includepoints (bool): If True, include the input points in result
            columnnames (tuple): Name of columns if dataframe is returned
            fmt (str): Format of IJK arrays (int/float). Default is "int"
            undef (int or float): Value to assign to undefined (outside) entries.

        .. versionadded:: 2.6
        .. versionchanged:: 2.8 Added keywords `columnnames`, `fmt`, `undef`
        """
        return _grid_etc1.get_ijk_from_points(
            self,
            points,
            activeonly=activeonly,
            zerobased=zerobased,
            dataframe=dataframe,
            includepoints=includepoints,
            columnnames=columnnames,
            fmt=fmt,
            undef=undef,
        )

    def get_xyz(
        self,
        names: tuple[str, str, str] = ("X_UTME", "Y_UTMN", "Z_TVDSS"),
        asmasked: bool = True,
    ) -> tuple[
        GridProperty,
        GridProperty,
        GridProperty,
    ]:
        """Returns 3 xtgeo.grid3d.GridProperty objects for x, y, z coordinates.

        The values are mid cell values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates). However, the option asmasked=True will mask
        the numpies for undef cells.

        Args:
            names: a 3 x tuple of names per property (default is X_UTME,
            Y_UTMN, Z_TVDSS).
            asmasked: If True, then inactive cells is masked (numpy.ma).
        """
        return _grid_etc1.get_xyz(self, names=tuple(names), asmasked=asmasked)

    def get_xyz_cell_corners(
        self,
        ijk: tuple[int, int, int] = (1, 1, 1),
        activeonly: bool = True,
        zerobased: bool = False,
    ) -> tuple[int, ...]:
        """Return a 8 * 3 tuple x, y, z for each corner.

        .. code-block:: none

           2       3
           !~~~~~~~!
           !  top  !
           !~~~~~~~!    Listing corners with Python index (0 base)
           0       1

           6       7
           !~~~~~~~!
           !  base !
           !~~~~~~~!
           4       5

        Args:
            ijk (tuple): A tuple of I J K (NB! cell counting starts from 1
                unless zerobased is True)
            activeonly (bool): Skip undef cells if set to True.

        Returns:
            A tuple with 24 elements (x1, y1, z1, ... x8, y8, z8)
                for 8 corners. None if cell is inactive and activeonly=True.

        Example::

            >>> grid = grid_from_file(reek_dir + "REEK.EGRID")
            >>> print(grid.get_xyz_cell_corners(ijk=(10,13,2)))
            (458704.10..., 1716.969970703125)

        Raises:
            RuntimeWarning if spesification is invalid.
        """

        return _grid_etc1.get_xyz_cell_corners_internal(
            self, ijk=ijk, activeonly=activeonly, zerobased=zerobased
        )

    def get_xyz_corners(
        self, names: tuple[str, str, str] = ("X_UTME", "Y_UTMN", "Z_TVDSS")
    ) -> tuple[GridProperty, ...]:
        """Returns 8*3 (24) xtgeo.grid3d.GridProperty objects, x, y, z for each corner.

        The values are cell corner values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates).

        .. code-block:: none

           2       3
           !~~~~~~~!
           !  top  !
           !~~~~~~~!    Listing corners with Python index (0 base)
           0       1

           6       7
           !~~~~~~~!
           !  base !
           !~~~~~~~!
           4       5

        Args:
            names (list): Generic name of the properties, will have a
                number added, e.g. X0, X1, etc.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.create_box_grid((2,2,2))
            >>> gps = grid.get_xyz_corners() # list of 24 grid properties
            >>> len(gps)
            24
            >>> gps[0].values.tolist()
            [[[0.0, 0.0], ... [[1.0, 1.0], [1.0, 1.0]]]


        Raises:
            RunetimeError if corners has wrong spesification
        """
        # return the 24 objects in a long tuple (x1, y1, z1, ... x8, y8, z8)
        return _grid_etc1.get_xyz_corners(self, names=names)

    def get_layer_slice(
        self, layer: int, top: bool = True, activeonly: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays for cell coordinates e.g. for plotting.

        In each cell there are 5 XY pairs, making a closed polygon as illustrated here:

        XY3  <  XY2
        !~~~~~~~!
        !       ! ^
        !~~~~~~~!
        XY0 ->  XY1
        XY4

        Note that cell ordering is C ordering (row is fastest)

        Args:
            layer (int): K layer, starting with 1 as topmost
            tip (bool): If True use top of cell, otherwise use base
            activeonly (bool): If True, only return active cells

        Returns:
            layerarray (np): [[[X0, Y0], [X1, Y1]...[X4, Y4]], [[..][..]]...]
            icarray (np): On the form [ic1, ic2, ...] where ic is cell count (C order)

        Example:

            Return two arrays forr cell corner for bottom layer::

                grd = xtgeo.grid_from_file(REEKFILE)

                parr, ibarr = grd.get_layer_slice(grd.nlay, top=False)

        .. versionadded:: 2.3
        """
        return _grid_etc1.get_layer_slice(self, layer, top=top, activeonly=activeonly)

    def get_geometrics(
        self,
        allcells: bool = False,
        cellcenter: bool = True,
        return_dict: bool = False,
        _ver: Literal[1, 2] = 1,
    ) -> dict | tuple:
        """Get a list of grid geometrics such as origin, min, max, etc.

        This returns a tuple: (xori, yori, zori, xmin, xmax, ymin, ymax, zmin,
        zmax, avg_rotation, avg_dx, avg_dy, avg_dz, grid_regularity_flag)

        If a dictionary is returned, the keys are as in the list above.

        Args:
            allcells (bool): If True, return also for inactive cells
            cellcenter (bool): If True, use cell center, otherwise corner
                coords
            return_dict (bool): If True, return a dictionary instead of a
                list, which is usually more convinient.
            _ver (int): Private option; only for developer!

        Raises: Nothing

        Example::

            >>> mygrid = grid_from_file(reek_dir + "REEK.EGRID")
            >>> gstuff = mygrid.get_geometrics(return_dict=True)
            >>> print(f"X min/max is {gstuff['xmin']:.2f} {gstuff['xmax']:.2f}")
            X min/max is 456620.79 467106.33

        """
        # TODO(JB): _grid_etc1.get_geometrics(False, False, True): Looks like it will
        # fail due to glist and gkeys will be out of sync (lengths will be differnt)
        return _grid_etc1.get_geometrics(
            self,
            allcells=allcells,
            cellcenter=cellcenter,
            return_dict=return_dict,
            _ver=_ver,
        )

    def get_adjacent_cells(
        self,
        prop: GridProperty,
        val1: int,
        val2: int,
        activeonly: bool = True,
    ) -> GridProperty:
        """Get a discrete property which reports val1 properties vs neighbouring val2.

        The result will be a new gridproperty, which in general has value 0
        but 1 if criteria is met, and 2 if criteria is met but cells are
        faulted.

        Args:
            prop (xtgeo.GridProperty): A discrete grid property, e.g region
            val1 (int): Primary value to evaluate
            val2 (int): Neighbourung value
            activeonly (bool): If True, do not look at inactive cells

        Raises: Nothing

        """
        return _grid_etc1.get_adjacent_cells(
            self, prop, val1, val2, activeonly=activeonly
        )

    def get_gridquality_properties(self) -> GridProperties:
        """Return a GridProperties() instance with grid quality measures.

        These measures are currently:

        * minangle_topbase (degrees) - minimum angle of top and base
        * maxangle_topbase (degrees) - maximum angle of top and base
        * minangle_topbase_proj (degrees) min angle projected (bird view)
        * maxangle_topbase_proj (degrees) max angle projected (bird view)
        * minangle_sides (degress) minimum angle, all side surfaces
        * maxangle_sides (degress) maximum angle, all side surfaces
        * collapsed (int) Integer, 1 of one or more corners are collpased in Z
        * faulted (int) Integer, 1 if cell is faulted (one or more neighbours offset)
        * negative_thickness (int) Integer, 1 if cell has negative thickness
        * concave_proj (int) 1 if cell is concave seen from projected bird view

        Example::

            # store grid quality measures in RMS
            gprops = grd.gridquality()
            for gprop in gprops:
                gprop.to_roxar(project, "MyGrid", gprop.name)


        """
        return _grid_etc1.get_gridquality_properties(self)

    # =========================================================================
    # Some more special operations that changes the grid or actnum
    # =========================================================================
    def activate_all(self) -> None:
        """Activate all cells in the grid, by manipulating ACTNUM."""
        self._actnumsv = np.ones(self.dimensions, dtype=np.int32)

        if self._xtgformat == 1:
            self._actnumsv = self._actnumsv.flatten()

    def inactivate_by_dz(self, threshold: float) -> None:
        """Inactivate cells thinner than a given threshold."""
        _grid_etc1.inactivate_by_dz(self, threshold)

    def inactivate_inside(
        self,
        poly: Polygons,
        layer_range: tuple[int, int] | None = None,
        inside: bool = True,
        force_close: bool = False,
    ) -> None:
        """Inacativate grid inside a polygon.

        The Polygons instance may consist of several polygons. If a polygon
        is open, then the flag force_close will close any that are not open
        when doing the operations in the grid.

        Args:
            poly(Polygons): A polygons object
            layer_range (tuple): A tuple of two ints, upper layer = 1, e.g.
                (1, 14). Note that base layer count is 1 (not zero)
            inside (bool): True if remove inside polygon
            force_close (bool): If True then force polygons to be closed.

        Raises:
            RuntimeError: If a problems with one or more polygons.
            ValueError: If Polygon is not a XTGeo object
        """
        _grid_etc1.inactivate_inside(
            self, poly, layer_range=layer_range, inside=inside, force_close=force_close
        )

    def inactivate_outside(
        self,
        poly: Polygons,
        layer_range: tuple[int, int] | None = None,
        force_close: bool = False,
    ) -> None:
        """Inacativate grid outside a polygon."""
        self.inactivate_inside(
            poly, layer_range=layer_range, inside=False, force_close=force_close
        )

    def collapse_inactive_cells(self) -> None:
        """Collapse inactive layers where, for I J with other active cells."""
        _grid_etc1.collapse_inactive_cells(self)

    def crop(
        self,
        colcrop: tuple[int, int],
        rowcrop: tuple[int, int],
        laycrop: tuple[int, int],
        props: Literal["all"] | list[GridProperty] | None = None,
    ) -> None:
        """Reduce the grid size by cropping.

        The new grid will get new dimensions.

        If props is "all" then all properties assosiated (linked) to then
        grid are also cropped, and the instances are updated.

        Args:
            colcrop (tuple): A tuple on the form (i1, i2)
                where 1 represents start number, and 2 represent end. The range
                is inclusive for both ends, and the number start index is 1 based.
            rowcrop (tuple): A tuple on the form (j1, j2)
            laycrop (tuple): A tuple on the form (k1, k2)
            props (list or str): None is default, while properties can be listed.
                If "all", then all GridProperty objects which are linked to the
                Grid instance are updated.

        Returns:
            The instance is updated (cropped)

        Example::

            >>> import xtgeo
            >>> mygrid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> mygrid.crop((3, 6), (4, 20), (1, 10))
            >>> mygrid.to_file(outdir + "/gf_reduced.roff")

        """
        _grid_etc1.crop(self, (colcrop, rowcrop, laycrop), props=props)

    def reduce_to_one_layer(self) -> None:
        """Reduce the grid to one single layer.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> grid.nlay
            14
            >>> grid.reduce_to_one_layer()
            >>> grid.nlay
            1

        """
        _grid_etc1.reduce_to_one_layer(self)

    def get_onelayer_grid(self) -> Grid:
        """Return a copy of the grid with only one layer."""

        new_grid_cpp = self._get_grid_cpp().extract_onelayer_grid()
        return Grid(
            coordsv=new_grid_cpp.coordsv,
            zcornsv=new_grid_cpp.zcornsv,
            actnumsv=new_grid_cpp.actnumsv,
        )

    def translate_coordinates(
        self,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        flip: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Translate (move) and/or flip grid coordinates in 3D.

        By 'flip' here, it means that the full coordinate array are multiplied
        with -1.

        Args:
            translate (tuple): Translation distance in X, Y, Z coordinates
            flip (tuple): Flip array. The flip values must be 1 or -1.

        Raises:
            RuntimeError: If translation goes wrong for unknown reasons
        """
        _grid_etc1.translate_coordinates(self, translate=translate, flip=flip)

    def reverse_row_axis(
        self, ijk_handedness: Literal["left", "right"] | None = None
    ) -> None:
        """Reverse the row axis (J indices).

        This means that IJK system will switched between a left vs right handed system.
        It is here (by using ijk_handedness key), possible to set a wanted stated.

        Note that properties that are assosiated with the grid (through the
        :py:attr:`~gridprops` or :py:attr:`~props` attribute) will also be
        reversed (which is desirable).

        Args:
            ijk_handedness (str): If set to "right" or "left", do only reverse rows if
                handedness is not already achieved.

        Example::

            grd = xtgeo.grid_from_file("somefile.roff")
            prop1 = xtgeo.gridproperty_from_file("somepropfile1.roff")
            prop2 = xtgeo.gridproperty_from_file("somepropfile2.roff")

            grd.props = [prop1, prop2]

            # secure that the grid geometry is IJK right-handed
            grd.reverse_row_axis(ijk_handedness="right")

        .. versionadded:: 2.5

        """
        _grid_etc1.reverse_row_axis(self, ijk_handedness=ijk_handedness)

    def make_zconsistent(self, zsep: float | int = 1e-5) -> None:
        """Make the 3D grid consistent in Z, by a minimal gap (zsep).

        Args:
            zsep (float): Minimum gap
        """
        _grid_etc1.make_zconsistent(self, zsep)

    def convert_to_hybrid(
        self,
        nhdiv: int = 10,
        toplevel: float = 1000.0,
        bottomlevel: float = 1100.0,
        region: GridProperty | None = None,
        region_number: int | None = None,
    ) -> None:
        """Convert to hybrid grid, either globally or in a selected region.

        This function will convert the internal structure in the corner point grid,
        so that the cells between two levels ``toplevel`` and ``bottomlevel`` become
        horizontal, which can be useful in flow simulators when e.g. liquid
        contact movements are dominating. See example on `usage in the Troll field`_.

        Note that the resulting hybrid will have an increased number of layers.
        If the initial grid has N layers, and the number of horizontal layers
        is NHDIV, then the result grid will have N * 2 + NHDIV layers.

        .. image:: images/hybridgrid2.jpg
           :width: 600
           :align: center

        Args:
            nhdiv (int): Number of hybrid layers.
            toplevel (float): Top of hybrid grid.
            bottomlevel (float): Base of hybrid grid.
            region (GridProperty, optional): Region property (if needed).
            region_number (int): Which region to apply hybrid grid in if region.

        Example:
            Create a hybridgrid from file, based on a GRDECL file (no region)::

               import xtgeo
               grd = xtgeo.grid_from_file("simgrid.grdecl", fformat="grdecl")
               grd.convert_to_hybrid(nhdiv=12, toplevel=2200, bottomlevel=2250)
               # save in binary GRDECL fmt:
               grd.to_file("simgrid_hybrid.bgrdecl", fformat="bgrdecl")

        See Also:
               :ref:`hybrid` example.

        .. _usage in the Troll field: https://doi.org/10.2118/148023-MS

        """
        _grid_hybrid.make_hybridgrid(
            self,
            nhdiv=nhdiv,
            toplevel=toplevel,
            bottomlevel=bottomlevel,
            region=region,
            region_number=region_number,
        )

    def refine(
        self,
        refine_col: int | dict[int, int],
        refine_row: int | dict[int, int],
        refine_layer: int | dict,
        zoneprop: GridProperty | None = None,
    ) -> None:
        """Refine grid in all direction, proportionally.

        The refine_layer can be a scalar or a dictionary.

        If refine_layer is a dict and zoneprop is None, then the current
        subgrids array is used. If zoneprop is defined, the
        current subgrid index will be redefined for the case. A warning will
        be issued if subgrids are defined, but the give zone
        property is inconsistent with this.

        Also, if a zoneprop is defined but no current subgrids in the grid,
        then subgrids will be added to the grid, if more than 1 subgrid.

        Args:
            self (object): A grid XTGeo object
            refine_col (scalar or dict): Refinement factor for each column.
            refine_row (scalar or dict): Refinement factor for each row.
            refine_layer (scalar or dict): Refinement factor for layer, if dict, then
                the dictionary must be consistent with self.subgrids if this is
                present.
            zoneprop (GridProperty): Zone property; must be defined if refine_layer
                is a dict

        Returns:
            ValueError: if..
            RuntimeError: if mismatch in dimensions for refine_layer and zoneprop

        """
        _grid_refine.refine(
            self, refine_col, refine_row, refine_layer, zoneprop=zoneprop
        )
        self._tmp = {}

    def refine_vertically(
        self,
        rfactor: int | dict,
        zoneprop: GridProperty | None = None,
    ) -> None:
        """Refine vertically, proportionally.

        The rfactor can be a scalar or a dictionary.

        If rfactor is a dict and zoneprop is None, then the current
        subgrids array is used. If zoneprop is defined, the
        current subgrid index will be redefined for the case. A warning will
        be issued if subgrids are defined, but the give zone
        property is inconsistent with this.

        Also, if a zoneprop is defined but no current subgrids in the grid,
        then subgrids will be added to the grid, if more than 1 subgrid.

        Args:
            self (object): A grid XTGeo object
            rfactor (scalar or dict): Refinement factor, if dict, then the
                dictionary must be consistent with self.subgrids if this is
                present.
            zoneprop (GridProperty): Zone property; must be defined if rfactor
                is a dict

        Returns:
            ValueError: if..
            RuntimeError: if mismatch in dimensions for rfactor and zoneprop


        Examples::

            # refine vertically all by factor 3

            grd.refine_vertically(3)

            # refine by using a dictionary; note that subgrids must exist!
            # and that subgrids that are not mentioned will have value 1
            # in refinement (1 is meaning no refinement)

            grd.refine_vertically({1: 3, 2: 4, 4: 1})

            # refine by using a a dictionary and a zonelog. If subgrids exists
            # but are inconsistent with the zonelog; the current subgrids will
            # be redefined, and a warning will be issued! Note also that ranges
            # in the dictionary rfactor and the zone property must be aligned.

            grd.refine_vertically({1: 3, 2: 4, 4: 0}, zoneprop=myzone)

        """
        _grid_refine.refine_vertically(self, rfactor, zoneprop=zoneprop)

    def report_zone_mismatch(
        self,
        well: Well | None = None,
        zonelogname: str = "ZONELOG",
        zoneprop: GridProperty | None = None,
        zonelogrange: tuple[int, int] = (0, 9999),
        zonelogshift: int = 0,
        depthrange: tuple | None = None,
        perflogname: str | None = None,
        perflogrange: tuple[int, int] = (1, 9999),
        filterlogname: str | None = None,
        filterlogrange: tuple[float, float] = (1e-32, 9999.0),
        resultformat: Literal[1, 2] = 1,
    ) -> tuple | dict | None:
        """Reports mismatch between wells and a zone.

        Approaches on matching:
            1. Use the well zonelog as basis, and compare sampled zone with that
               interval. This means that zone cells outside well range will not be
               counted
            2. Compare intervals with wellzonation in range or grid zonations in
               range. This gives a wider comparison, and will capture cases
               where grid zonations is outside well zonation

        .. image:: images/zone-well-mismatch-plain.svg
           :width: 200
           :align: center

        Note if `zonelogname` and/or `filterlogname` and/or `perflogname` is given,
        and such log(s) are not present, then this function will return ``None``.

        Args:
            well (Well): a XTGeo well object
            zonelogname (str): Name of the zone logger
            zoneprop (GridProperty): Grid property instance to use for
                zonation
            zonelogrange (tuple): zone log range, from - to (inclusive)
            zonelogshift (int): Deviation (numerical shift) between grid and zonelog,
                e.g. if Zone property starts with 1 and this corresponds to a zonelog
                index of 3 in the well, the shift shall be -2.
            depthrange (tuple): Interval for search in TVD depth, to speed up
            perflogname (str): Name of perforation log to filter on (> 0 default).
            perflogrange (tuple): Range of values where perforations are present.
            filterlogname (str): General filter, work as perflog, filter on values > 0
            filterlogrange (tuple): Range of values where filter shall be present.
            resultformat (int): If 1, consider the zonelogrange in the well as
                basis for match ratio, return (percent, match count, total count).
                If 2 then a dictionary is returned with various result members

        Returns:
            res (tuple or dict): report dependent on `resultformat`
                * A tuple with 3 members:
                    (match_as_percent, number of matches, total count) approach 1
                * A dictionary with keys:
                    * MATCH1 - match as percent, approach 1
                    * MCOUNT1 - number of match samples approach 1
                    * TCOUNT1 - total number of samples approach 1
                    * MATCH2 - match as percent, approach 2
                    * MCOUNT2 - a.a for option 2
                    * TCOUNT2 - a.a. for option 2
                    * WELLINTV - a Well() instance for the actual interval
                * None, if perflogname or zonelogname of filtername is given, but
                  the log does not exists for the well

        Example::

            g1 = xtgeo.grid_from_file("gullfaks2.roff")

            z = xtgeo.gridproperty_from_file(gullfaks2_zone.roff", name="Zone")

            w2 = xtgeo.well_from_file("34_10-1.w", zonelogname="Zonelog")

            w3 = xtgeo.well_from_file("34_10-B-21_B.w", zonelogname="Zonelog"))

            wells = [w2, w3]

            for w in wells:
                response = g1.report_zone_mismatch(
                    well=w, zonelogname="ZONELOG", zoneprop=z,
                    zonelogrange=(0, 19), depthrange=(1700, 9999))

                print(response)

        .. versionchanged:: 2.8 Added several new keys and better precision in result
        .. versionchanged:: 2.11 Added ``perflogrange`` and ``filterlogrange``
        """
        return _grid_wellzone.report_zone_mismatch(
            self,
            well=well,
            zonelogname=zonelogname,
            zoneprop=zoneprop,
            zonelogrange=zonelogrange,
            zonelogshift=zonelogshift,
            depthrange=depthrange,
            perflogname=perflogname,
            perflogrange=perflogrange,
            filterlogname=filterlogname,
            filterlogrange=filterlogrange,
            resultformat=resultformat,
        )

    # ==================================================================================
    # Extract a fence/randomline by sampling, ready for plotting with e.g. matplotlib
    # ==================================================================================
    def get_randomline(
        self,
        fencespec: np.ndarray | Polygons,
        prop: str | GridProperty,
        zmin: float | None = None,
        zmax: float | None = None,
        zincrement: float = 1.0,
        hincrement: float | None = None,
        atleast: int = 5,
        nextend: int = 2,
    ) -> tuple[float, float, float, float, np.ndarray]:
        """Get a sampled randomline from a fence spesification.

        This randomline will be a 2D numpy with depth on the vertical
        axis, and length along as horizontal axis. Undefined values will have
        the np.nan value.

        The input fencespec is either a 2D numpy where each row is X, Y, Z, HLEN,
        where X, Y are UTM coordinates, Z is depth/time, and HLEN is a
        length along the fence, or a Polygons instance.

        If input fencspec is a numpy 2D, it is important that the HLEN array
        has a constant increment and ideally a sampling that is less than the
        Grid resolution. If a Polygons() instance, this will be automated if
        hincrement is None.

        Args:
            fencespec (:obj:`~numpy.ndarray` or :class:`~xtgeo.xyz.polygons.Polygons`):
                2D numpy with X, Y, Z, HLEN as rows or a xtgeo Polygons() object.
            prop (GridProperty or str): The grid property object, or name, which shall
                be plotted.
            zmin (float): Minimum Z (default is Grid Z minima/origin)
            zmax (float): Maximum Z (default is Grid Z maximum)
            zincrement (float): Sampling vertically, default is 1.0
            hincrement (float): Resampling horizontally. This applies only
                if the fencespec is a Polygons() instance. If None (default),
                the distance will be deduced automatically.
            atleast (int): Minimum number of horizontal samples This applies
                only if the fencespec is a Polygons() instance.
            nextend (int): Extend with nextend * hincrement in both ends.
                This applies only if the fencespec is a Polygons() instance.

        Returns:
            A tuple: (hmin, hmax, vmin, vmax, ndarray2d)

        Raises:
            ValueError: Input fence is not according to spec.

        Example::

            mygrid = xtgeo.grid_from_file("somegrid.roff")
            poro = xtgeo.gridproperty_from_file("someporo.roff")
            mywell = xtgeo.well_from_file("somewell.rmswell")
            fence = mywell.get_fence_polyline(sampling=5, tvdmin=1750, asnumpy=True)
            (hmin, hmax, vmin, vmax, arr) = mygrid.get_randomline(
                 fence, poro, zmin=1750, zmax=1850, zincrement=0.5,
            )
            # matplotlib ...
            plt.imshow(arr, cmap="rainbow", extent=(hmin1, hmax1, vmax1, vmin1))

        .. versionadded:: 2.1

        .. seealso::
           Class :class:`~xtgeo.xyz.polygons.Polygons`
              The method :meth:`~xtgeo.xyz.polygons.Polygons.get_fence()` which can be
              used to pregenerate `fencespec`

        """
        if not isinstance(fencespec, (np.ndarray, xtgeo.Polygons)):
            raise ValueError("fencespec must be a numpy or a Polygons() object")
        logger.info("Getting randomline...")

        res = _grid3d_fence.get_randomline(
            self,
            fencespec,
            prop,
            zmin=zmin,
            zmax=zmax,
            zincrement=zincrement,
            hincrement=hincrement,
            atleast=atleast,
            nextend=nextend,
        )
        logger.info("Getting randomline... DONE")
        return res

    # ----------------------------------------------------------------------------------
    # Special private functions; these may only live for while
    # ----------------------------------------------------------------------------------

    def _convert_xtgformat2to1(self) -> None:
        """Convert arrays from new structure xtgformat=2 to legacy xtgformat=1."""
        _grid_etc1._convert_xtgformat2to1(self)

    def _convert_xtgformat1to2(self) -> None:
        """Convert arrays from old structure xtgformat=1 to new xtgformat=2."""
        _grid_etc1._convert_xtgformat1to2(self)

    def _set_xtgformat1(self) -> None:
        """Shortform... arrays from new structure xtgformat=2 to legacy xtgformat=1."""
        self._convert_xtgformat2to1()

    def _set_xtgformat2(self) -> None:
        """Shortform... arrays from old structure xtgformat=1 to new xtgformat=2."""
        self._convert_xtgformat1to2()
