"""Module/class for 3D grids (corner point geometry) with XTGeo."""
from __future__ import annotations

import functools
import json
import pathlib
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import deprecation
import numpy as np
import numpy.ma as ma

import xtgeo
from xtgeo.common import XTGDescription, _XTGeoFile, null_logger
from xtgeo.common.sys import generic_hash
from xtgeo.common.version import __version__

from . import (
    _grid3d_fence,
    _grid_etc1,
    _grid_export,
    _grid_hybrid,
    _grid_import,
    _grid_import_ecl,
    _grid_import_xtgcpgeom,
    _grid_refine,
    _grid_roxapi,
    _grid_wellzone,
    _gridprop_lowlevel,
)
from ._ecl_grid import Units
from ._grid3d import _Grid3D
from .grid_properties import GridProperties, GridProperty

xtg = xtgeo.common.XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    import pandas as pd

    from xtgeo import Polygons, Well
    from xtgeo.common.types import FileLike
    from xtgeo.xyz.points import Points

METRIC = Literal[
    "euclid",
    "horizontal",
    "east west vertical",
    "north south vertical",
    "x projection",
    "y projection",
    "z projection",
]

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


# METHODS as wrappers to class init + import
def _handle_import(
    grid_constructor: Callable[..., Grid],
    gfile: FileLike | _XTGeoFile,
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
    gfile = xtgeo._XTGeoFile(gfile, mode="rb")
    if fformat == "eclipserun":
        ecl_grid = grid_constructor(
            **_grid_import.from_file(
                xtgeo._XTGeoFile(gfile.name + ".EGRID", mode="rb"), fformat="egrid"
            )
        )
        _grid_import_ecl.import_ecl_run(gfile.name, ecl_grid=ecl_grid, **kwargs)
        return ecl_grid
    return grid_constructor(**_grid_import.from_file(gfile, fformat, **kwargs))


def grid_from_file(
    gfile: str | pathlib.Path,
    fformat: str | None = None,
    **kwargs: dict[str, Any],
) -> Grid:
    """Read a grid (cornerpoint) from filelike and an returns a Grid() instance.

    Args:
        gfile (str or Path): File name to be imported. If fformat="eclipse_run"
            then a fileroot name shall be input here, see example below.
        fformat (str): File format egrid/roff/grdecl/bgrdecl/eclipserun/xtgcpgeom
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
    dimensions_only: bool = False,
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
    return Grid(
        **_grid_roxapi.import_grid_roxapi(
            project, gname, realisation, dimensions_only, info
        )
    )


def create_box_grid(
    dimension: tuple[int, int, int],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    oricenter: bool = False,
    increment: tuple[int, int, int] = (1, 1, 1),
    rotation: float = 0.0,
    flip: Literal[1, -1] = 1,
) -> Grid:
    """Create a rectangular 'shoebox' grid from spec.

    Args:
        dimension (tuple of int): A tuple of (NCOL, NROW, NLAY)
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


def allow_deprecated_init(func: Callable) -> Callable:
    # This decorator is here to maintain backwards compatibility in the construction
    # of RegularSurface and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):  # type: ignore
        # Checking if we are doing an initialization
        # from file and raise a deprecation warning if
        # we are.
        if "gfile" in kwargs or (
            len(args) >= 1 and isinstance(args[0], (str, pathlib.Path, _XTGeoFile))
        ):
            warnings.warn(
                "Initializing directly from file name is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                "mygrid = xtgeo.grid_from_file('some_name.roff') instead",
                DeprecationWarning,
            )

            def constructor(**kwargs):  # type: ignore
                func(self, **kwargs)
                return self

            _handle_import(constructor, *args, **kwargs)
            return None

        # Check if we are doing default value init
        if len(args) == 0 and len(kwargs) == 0:
            warnings.warn(
                "Initializing default box grid is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                "mygrid = xtgeo.create_box_grid() or, alternatively,"
                "directly from file with mygrid = xtgeo.grid_from_file().",
                DeprecationWarning,
            )
            kwargs = _grid_etc1.create_box(
                dimension=(4, 3, 5),
                origin=(10.0, 20.0, 1000.0),
                oricenter=False,
                increment=(100, 150, 5),
                rotation=30.0,
                flip=1,
            )
            return func(self, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


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

    # pylint: disable=too-many-public-methods
    @allow_deprecated_init
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

        self._reset(
            coordsv=coordsv,
            zcornsv=zcornsv,
            actnumsv=actnumsv,
            dualporo=dualporo,
            dualperm=dualperm,
            subgrids=subgrids,
            units=units,
            filesrc=filesrc,
            props=props,
            name=name,
            roxgrid=roxgrid,
            roxindexer=roxindexer,
        )

    def _reset(
        self,
        coordsv: np.ndarray,
        zcornsv: np.ndarray,
        actnumsv: np.ndarray,
        dualporo: bool = False,
        dualperm: bool = False,
        subgrids: dict[str, range | list[int]] | None = None,
        units: Units | None = None,
        filesrc: pathlib.Path | str | None = None,
        props: GridProperties | None = None,
        name: str | None = None,
        roxgrid: Any | None = None,
        roxindexer: Any | None = None,
    ) -> None:
        """This function only serves to allow deprecated initialization."""
        # TODO: Remove once implicit initialization such as Grid().from_file()
        # is removed
        self._xtgformat = 2
        self._ncol = actnumsv.shape[0]
        self._nrow = actnumsv.shape[1]
        self._nlay = actnumsv.shape[2]

        self._coordsv = coordsv
        self._zcornsv = zcornsv
        self._actnumsv = actnumsv
        self._dualporo = dualporo
        self._dualperm = dualperm

        self._filesrc = filesrc

        if props is None:
            self._props = GridProperties(props=[])
        else:
            self._props = props
        self._name = name
        self._subgrids = subgrids
        self._ijk_handedness: Literal["left", "right"] | None = None

        self._dualactnum = None
        if dualporo:
            self._dualactnum = self.get_actnum(name="DUALACTNUM")
            acttmp = self._dualactnum.copy()
            acttmp.values[acttmp.values >= 1] = 1
            self.set_actnum(acttmp)

        self._metadata = xtgeo.MetaDataCPGeometry()
        self._metadata.required = self

        # Roxar api spesific:
        self._roxgrid = roxgrid
        self._roxindexer = roxindexer

        self.units = units

        self._tmp: dict = {}

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
    def dimensions(self) -> tuple[int, int, int]:
        """3-tuple: The grid dimensions as a tuple of 3 integers (read only)."""
        return (self.ncol, self.nrow, self.nlay)

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
        nflip = _grid_etc1.estimate_flip(self)
        if nflip == 1:
            self._ijk_handedness = "left"
        elif nflip == -1:
            self._ijk_handedness = "right"
        else:
            self._ijk_handedness = None  # cannot determine

        return self._ijk_handedness

    @ijk_handedness.setter
    def ijk_handedness(self, value: Literal["left", "right"]) -> None:
        if value not in ("right", "left"):
            raise ValueError("The value must be 'right' or 'left'")
        self.reverse_row_axis(ijk_handedness=value)
        self._ijk_handedness = value

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
        actnumv = ma.filled(actnumv, fill_value=0)

        return actnumv

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
        elif isinstance(self._props, list):
            raise RuntimeError(
                "self._props is a list, not a GridProperties " "instance"
            )
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

    def generate_hash(
        self,
        hashmethod: Literal["md5", "sha256", "blake2b"] = "md5",
    ) -> str:
        """Return a unique hash ID for current instance.

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        .. versionadded:: 2.14
        """
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

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.create_box_grid() instead",
    )
    def create_box(
        self,
        dimension: tuple[int, int, int] = (10, 12, 6),
        origin: tuple[float, float, float] = (10.0, 20.0, 1000.0),
        oricenter: bool = False,
        increment: tuple[int, int, int] = (100, 150, 5),
        rotation: float = 30.0,
        flip: Literal[1, -1] = 1,
    ) -> None:
        """Create a rectangular 'shoebox' grid from spec.

        Args:
            dimension (tuple of int): A tuple of (NCOL, NROW, NLAY)
            origin (tuple of float): Startpoint of grid (x, y, z)
            oricenter (bool): If False, startpoint is node, if True, use cell center
            increment (tuple of float): Grid increments (xinc, yinc, zinc)
            rotation (float): Roations in degrees, anticlock from X axis.
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

        self._reset(**kwargs)

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
        _gfile = xtgeo._XTGeoFile(gfile, mode="wb")

        if not _gfile.memstream:
            _gfile.check_folder(raiseerror=OSError)

        valid_formats = {
            "roff": ["roff", "roff_binary", "roff_bin", "roffbin"],
            "roff_ascii": ["roff_ascii", "roff_asc", "roffasc"],
            "grdecl": ["grdecl"],
            "bgrdecl": ["bgrdecl"],
            "egrid": ["egrid"],
            "fegrid": ["fegrid"],
        }

        if fformat in valid_formats["roff"]:
            _grid_export.export_roff(self, _gfile.name, "binary")
        elif fformat in valid_formats["roff_ascii"]:
            _grid_export.export_roff(self, _gfile.name, "ascii")
        elif fformat in valid_formats["grdecl"]:
            _grid_export.export_grdecl(self, _gfile.name, 1)
        elif fformat in valid_formats["bgrdecl"]:
            _grid_export.export_grdecl(self, _gfile.name, 0)
        elif fformat in valid_formats["egrid"]:
            _grid_export.export_egrid(self, _gfile.name)
        elif fformat in valid_formats["fegrid"]:
            _grid_export.export_fegrid(self, _gfile.name)
        else:
            raise ValueError(
                f"Invalid file format: {fformat}, valid options are: "
                f"{', '.join(v for vv in valid_formats.values() for v in vv)}"
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
        _gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)
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
        _gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)
        _gfile.check_folder(raiseerror=OSError)

        _grid_export.export_xtgcpgeom(self, _gfile, subformat=subformat)

        return _gfile.file

    def to_roxar(
        self,
        project: str,
        gname: str,
        realisation: int = 0,
        info: bool = False,
        method: str = "cpg",
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
            method (str): Save approach

        """
        _grid_roxapi.export_grid_roxapi(
            self, project, gname, realisation, info=info, method=method
        )

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.grid_from_file() instead",
    )
    def from_file(
        self,
        gfile: FileLike,
        fformat: str | None = None,
        **kwargs: Any,
    ) -> Grid:
        """Import grid geometry from file, and makes an instance of this class.

        If file extension is missing, then the extension will guess the fformat
        key, e.g. fformat egrid will be guessed if ".EGRID". The "eclipserun"
        will try to input INIT and UNRST file in addition the grid in "one go".

        Arguments:
            gfile (str or Path): File name to be imported. If fformat="eclipse_run"
                then a fileroot name shall be input here, see example below.
            fformat (str): File format egrid/roff/grdecl/bgrdecl/eclipserun/xtgcpgeom
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

            >>> xg = Grid()
            >>> xg.from_file(reek_dir + "/REEK.EGRID", fformat="egrid")
            Grid ... filesrc='.../REEK.EGRID'

        Example using "eclipserun"::

            >>> mycase = "REEK"  # meaning REEK.EGRID, REEK.INIT, REEK.UNRST
            >>> xg = Grid()
            >>> xg.from_file(
            ...     reek_dir + "/" + mycase,
            ...     fformat="eclipserun",
            ...     initprops="all",
            ... )
            Grid ... filesrc='.../REEK.EGRID'

        Raises:
            OSError: if file is not found etc
        """

        def constructor(*args, **kwargs):  # type: ignore
            self._reset(*args, **kwargs)
            return self

        _handle_import(constructor, gfile, fformat, **kwargs)
        return self

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.grid_from_file() instead",
    )
    def from_hdf(
        self,
        gfile: FileLike,
        ijkrange: Sequence[int] | None = None,
        zerobased: bool = False,
    ) -> None:
        """Import grid geometry from HDF5 file (experimental!).

        Args:
            gfile (str): Name of output file
            ijkrange (list-like): Partial read, e.g. (1, 20, 1, 30, 1, 3) as
                (i1, i2, j1, j2, k1, k2). Numbering scheme depends on `zerobased`,
                where default is `eclipse-like` i.e. first cell is 1. Numbering
                is inclusive for both ends. If ijkrange exceeds original range,
                an Exception is raised. Using existing boundaries can be defaulted
                by "min" and "max", e.g. (1, 20, 5, 10, "min", "max")
            zerobased (bool): If True index in ijkrange is zero based.

        Raises:
            ValueError: The ijkrange spesification exceeds boundaries.
            ValueError: The ijkrange list must have 6 elements

        Example::

            >>> xg = create_box_grid((20,20,5))
            >>> filename = xg.to_hdf(outdir + "/myfile_grid.h5")
            >>> xg.from_hdf(filename, ijkrange=(1, 10, 10, 15, 1, 4))
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)

        kwargs = _grid_import_xtgcpgeom.import_hdf5_cpgeom(
            gfile, ijkrange=ijkrange, zerobased=zerobased
        )
        self._reset(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.grid_from_file() instead",
    )
    def from_xtgf(self, gfile: FileLike, mmap: bool = False) -> None:
        """Import grid geometry from native xtgeo file format (experimental!).

        Args:
            gfile (str): Name of output file
            mmap (bool): If true, reading with memory mapping is active

        Example::

            >>> xg = create_box_grid((5,5,5))
            >>> filename = xg.to_xtgf(outdir + "/myfile_grid.xtg")
            >>> xg.from_xtgf(filename)
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)

        kwargs = _grid_import_xtgcpgeom.import_xtgcpgeom(gfile, mmap)
        self._reset(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.grid_from_roxar() instead",
    )
    def from_roxar(
        self,
        projectname: str,
        gname: str,
        realisation: int = 0,
        dimensions_only: bool = False,
        info: bool = False,
    ) -> None:
        """Import grid model geometry from RMS project, and makes an instance.

        Args:
            projectname (str): Name of RMS project
            gname (str): Name of grid model
            realisation (int): Realisation number.
            dimensions_only (bool): If True, only the ncol, nrow, nlay will
                read. The actual grid geometry will remain empty (None). This
                will be much faster of only grid size info is needed, e.g.
                for initalising a grid property.
            info (bool): If True, various info will printed to screen. This
                info will depend on version of ROXAPI, and is mainly a
                developer/debugger feature. Default is False.


        """
        kwargs = _grid_roxapi.import_grid_roxapi(
            projectname, gname, realisation, dimensions_only, info
        )
        self._reset(**kwargs)

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
        other = _grid_etc1.copy(self)
        return other

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

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Method dataframe is deprecated, use get_dataframe instead.",
    )
    def dataframe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Returns:
            A Pandas dataframe
        """
        return self.get_dataframe(*args, **kwargs)

    def get_vtk_esg_geometry_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
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
            return None

        if not isinstance(sdict, dict):
            raise ValueError("Input sdict is not an dict")

        newsub: dict[str, range | list[int]] = dict()

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

        return dict((name, len(subb)) for name, subb in self.subgrids.items())

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

            if nsubname not in self.subgrids.keys():
                return None

        res = _grid_etc1.estimate_design(self, nsubname)

        return res

    def estimate_flip(self) -> Literal[1, -1]:
        """Estimate flip (handedness) of grid returns as 1 or -1.

        The flip numbers are 1 for left-handed and -1 for right-handed.

        .. seealso:: :py:attr:`~ijk_handedness`
        """
        return _grid_etc1.estimate_flip(self)

    def subgrids_from_zoneprop(self, zoneprop: GridProperty) -> dict[str, int] | None:
        """Estimate subgrid index from a zone property.

        The new will esimate which will replace the current if any.

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

        newd: dict[str, range] = dict()
        for izone in range(minzone, maxzone + 1):
            mininzn = int(kval[zprval == izone].min())  # 1 base
            maxinzn = int(kval[zprval == izone].max())  # 1 base
            newd["zone" + str(izone)] = range(mininzn, maxinzn + 1)

        self.subgrids = newd  # type: ignore

        return self.get_subgrids()

    def get_zoneprop_from_subgrids(self) -> NoReturn:
        """Make a XTGeo GridProperty instance for a Zone property subgrid index."""
        raise NotImplementedError("Not yet; todo")

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
        else:
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

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.Grid().gridprops instead",
    )
    def get_gridproperties(self) -> GridProperties:
        """Return the :obj:`GridProperties` instance attached to the grid.

        See also the :meth:`gridprops` property
        """
        return self._props

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
        mask: bool | None = None,
        dual: bool = False,
    ) -> GridProperty:
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): Actnum is returned with all cells shown
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead!
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
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = self._evaluate_mask(mask)

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
        mask: bool | None = None,
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
            mask (bool): Deprecated, use asmasked instead!
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
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = self._evaluate_mask(mask)

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

    @deprecation.deprecated(
        deprecated_in="3.0",
        removed_in="4.0",
        current_version=__version__,
        details="Use xtgeo.Grid.get_dx() and/or xtgeo.Grid.get_dy() instead.",
    )
    def get_dxdy(
        self,
        names: tuple[str, str] = ("dX", "dY"),
        asmasked: bool = False,
    ) -> tuple[GridProperty, GridProperty]:
        """Return the dX and dY as GridProperty object.

        The values lengths are projected to a constant Z.

        Args:
            name (tuple): names of properties
            asmasked (bool). If True, make a np.ma array where inactive cells
                are masked.

        Returns:
            Two XTGeo GridProperty objects (dx, dy).
            XTGeo GridProperty objects containing dy.
        """
        return (
            self.get_dx(name=names[0], asmasked=asmasked),
            self.get_dy(name=names[1], asmasked=asmasked),
        )

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
            precision (int): An number indication precision level, where
                a higher number means increased precision but also increased computing
                time. Currently 1, 2 (default), 4 are supported.

        Returns:
            XTGeo GridProperty object

        .. versionadded:: 2.13 (as experimental)

        """
        return _grid_etc1.get_bulk_volume(
            self, name=name, asmasked=asmasked, precision=precision
        )

    def get_ijk(
        self,
        names: tuple[str, str, str] = ("IX", "JY", "KZ"),
        asmasked: bool = True,
        mask: bool | None = None,
        zerobased: bool = False,
    ) -> tuple[GridProperty, GridProperty, GridProperty]:
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter, J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            asmasked: If True, UNDEF cells are masked, default is True
            mask (bool): Deprecated, use asmasked instead!
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = self._evaluate_mask(mask)

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
        mask: bool | None = None,
    ) -> tuple[GridProperty, GridProperty, GridProperty,]:
        """Returns 3 xtgeo.grid3d.GridProperty objects for x, y, z coordinates.

        The values are mid cell values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates). However, the option asmasked=True will mask
        the numpies for undef cells.

        Args:
            names: a 3 x tuple of names per property (default is X_UTME,
            Y_UTMN, Z_TVDSS).
            asmasked: If True, then inactive cells is masked (numpy.ma).
            mask (bool): Deprecated, use asmasked instead!
        """
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = self._evaluate_mask(mask)

        return _grid_etc1.get_xyz(self, names=names, asmasked=asmasked)

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
        return _grid_etc1.get_xyz_cell_corners(
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

        self._tmp = {}

    def inactivate_by_dz(self, threshold: float) -> None:
        """Inactivate cells thinner than a given threshold."""
        _grid_etc1.inactivate_by_dz(self, threshold)
        self._tmp = {}

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
        self._tmp = {}

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
        self._tmp = {}

    def collapse_inactive_cells(self) -> None:
        """Collapse inactive layers where, for I J with other active cells."""
        _grid_etc1.collapse_inactive_cells(self)
        self._tmp = {}

    def crop(
        self,
        colcrop: tuple[int, int],
        rowcrop: tuple[int, int],
        laycrop: tuple[int, int],
        props: str | list[GridProperty] | None = None,
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
        self._tmp = {}

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
        self._tmp = {}

    def translate_coordinates(
        self,
        translate: tuple[int, int, int] = (0, 0, 0),
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
        self._tmp = {}

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
        self._tmp = {}

    def make_zconsistent(self, zsep: float = 1e-5) -> None:
        """Make the 3D grid consistent in Z, by a minimal gap (zsep).

        Args:
            zsep (float): Minimum gap
        """
        _grid_etc1.make_zconsistent(self, zsep)
        self._tmp = {}

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
        self._tmp = {}

    def refine_vertically(
        self,
        rfactor: int | dict | None,
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
        self._tmp = {}

    def report_zone_mismatch(
        self,
        well: Well | None = None,
        zonelogname: str = "ZONELOG",
        zoneprop: GridProperty | None = None,
        onelayergrid: tuple | None = None,
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
            onelayergrid (Grid): Redundant from version 2.8, please skip!
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
            onelayergrid=onelayergrid,
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

    def _xtgformat1(self) -> None:
        """Shortform... arrays from new structure xtgformat=2 to legacy xtgformat=1."""
        self._convert_xtgformat2to1()

    def _xtgformat2(self) -> None:
        """Shortform... arrays from old structure xtgformat=1 to new xtgformat=2."""
        self._convert_xtgformat1to2()
