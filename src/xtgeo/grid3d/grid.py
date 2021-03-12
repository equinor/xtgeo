# -*- coding: utf-8 -*-
"""Module/class for 3D grids (corner point geometry) with XTGeo."""

import pathlib
import json
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Union, Optional, List, Tuple

import numpy as np
import numpy.ma as ma

import xtgeo

from xtgeo.common import XTGDescription
from ._grid3d import _Grid3D

from . import _grid_hybrid
from . import _grid_import
from . import _grid_import_xtgcpgeom
from . import _grid_export
from . import _grid_refine
from . import _grid_etc1
from . import _grid_wellzone
from . import _grid3d_fence
from . import _grid_roxapi
from . import _gridprop_lowlevel

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

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


def grid_from_file(
    gfile, fformat=None, initprops=None, restartprops=None, restartdates=None
):
    """Read a grid (cornerpoint) from file and an returns a Grid() instance.

    See :meth:`Grid.from_file` method for details on keywords.

    Example::

        import xtgeo
        mygrid = xtgeo.grid_from_file("reek.roff")

    """
    obj = Grid()

    obj.from_file(
        gfile,
        initprops=initprops,
        restartprops=restartprops,
        restartdates=restartdates,
        fformat=fformat,
    )

    return obj


def grid_from_roxar(
    project, gname, realisation=0, dimensions_only=False, info=False
):  # pragma: no cover
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
    obj = Grid()

    obj.from_roxar(
        project,
        gname,
        realisation=realisation,
        dimensions_only=dimensions_only,
        info=info,
    )

    return obj


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

IJKRange = Tuple[int, int, int, int, int, int]


class Grid(_Grid3D):
    """Class for a 3D grid corner point geometry in XTGeo.

    I.e. the geometric grid cells and the active cell indicator.

    The grid geometry class instances are normally created when
    importing a grid from file, as it is normally too complex to create from
    scratch.

    See Also:
        The :class:`.GridProperty` and the :class:`.GridProperties` classes.

    """

    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        gfile: Optional[Union[str, Path]] = None,
        fformat: Optional[str] = "guess",
        initprops: Optional[List[str]] = None,
        restartprops: Optional[List[str]] = None,
        restartdates: Optional[List[Union[int, str]]] = None,
        ijkrange: Optional[IJKRange] = None,
        zerobased: Optional[bool] = False,
    ):
        """Instantating.

        Args:
            gfile: Input file, or leave blank.
            fformat: File format input, default is ``guess`` based on file extension.
                Other options are ...
            initprops: List of initial properties (Eclipse based ``eclrun`` import).
            restartprops: List of restart properties (Eclipse based ``eclrun`` import).
            restartdates: List of restart dates as YYYYMMDD (Eclipse based ``eclrun``
                import).
            ijkrange: Tuple of 6 integers defining (imin, imax, jmin, jmax, kmin, kmax)
                when import from ``hdf`` files. Ranges are implicit at both ends.
            zerobased: Whether `ijkrange` uses 1 (default) or 0 as base.

        Example::

            import xtgeo

            geo = xtgeo.Grid()
            geo.from_file("myfile.roff")

            # alternative (make instance directly from file):
            geo = xtgeo.Grid("myfile.roff")

            # or use
            geo = xtgeo.grid_from_file("myfile.roff")
        """
        super().__init__()

        self._coordsv = None  # numpy array to coords vector
        self._zcornsv = None  # numpy array to zcorns vector
        self._actnumsv = None  # numpy array to actnum vector

        # _xtgformat: internal flag for storage structure. 1 is the "current" while 2
        # will be the new one. This will affect how _coordsv _zcornsv and _actnumsv
        # are organized! The corresponding C routines for 1: grd3d_*, while 2: grdcp3d_*
        self._xtgformat = 2

        # the following block is currently not in use, but required for metadata
        self._xshift = 0.0
        self._yshift = 0.0
        self._zshift = 0.0
        self._xscale = 1.0
        self._yscale = 1.0
        self._zscale = 1.0

        self._actnum_indices = None  # Index numpy array for active cells
        self._filesrc = None

        self._props = None  # None or a GridProperties instance
        self._name = "noname"
        self._subgrids = None  # A python dict if subgrids are given
        self._ijk_handedness = None

        # Simulators like Eclipse may have a dual poro/perm model
        self._dualporo = False
        self._dualperm = False
        self._dualactnum = None  # will be a GridProperty()

        self._metadata = xtgeo.MetaDataCPGeometry()

        # Roxar api spesific:
        self._roxgrid = None
        self._roxindexer = None

        # For storage of more private stuff in order to speed up certain functions
        # See _grid3d_fence for instance; note! reset this if any kind of grid change!
        self._tmp = {}

        if gfile is not None:
            gfile = pathlib.Path(gfile)
            if gfile.suffix == "hdf":
                self.from_hdf(gfile, ijkrange, zerobased)
            self.from_file(
                gfile,
                fformat=fformat,
                initprops=initprops,
                restartprops=restartprops,
                restartdates=restartdates,
            )
        else:
            # make a simple empty box grid (from version 2.13)
            self.create_box((self._ncol, self._nrow, self._nlay))

        self._metadata.required = self
        logger.info("Ran __init__ for %s", repr(self))

    def __repr__(self):
        """The __repr__ method."""
        logger.info("Invoke __repr__ for grid")
        myrp = (
            "{0.__class__.__name__} (id={1}) ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, nlay={0._nlay!r}, "
            "filesrc={0._filesrc!r}".format(self, id(self))
        )
        return myrp

    def __str__(self):
        """The __str__ method for user friendly print."""
        logger.debug("Invoke __str__ for grid", stack_info=True)

        return self.describe(flush=False)

    # ==================================================================================
    # Public Properties:
    # ==================================================================================

    @property
    def metadata(self):
        """obj: Return or set metadata instance of type MetaDataCPGeometry."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. A bit dangerous so further
        # check must be done to validate. TODO.
        if not isinstance(obj, xtgeo.MetaDataCPGeometry):
            raise ValueError("Input obj not an instance of MetaDataCPGeometry")

        self._metadata = obj  # checking is currently missing! TODO

    @property
    def filesrc(self):
        """str: Source for grid (filepath or name in RMS)."""
        return self._filesrc

    @property
    def name(self):
        """str: Name attribute of grid."""
        return self._name

    @name.setter
    def name(self, name):
        if isinstance(name, str):
            self._name = name
        else:
            raise ValueError("Input name is not a text string")

    @property
    def ncol(self):
        """int: Number of columns (read only)."""
        return super().ncol

    @property
    def nrow(self):
        """int: Number of rows (read only)."""
        return super().nrow

    @property
    def nlay(self):
        """int: Number of layers (read only)."""
        return super().nlay

    @property
    def dimensions(self):
        """3-tuple: The grid dimensions as a tuple of 3 integers (read only)."""
        return (self._ncol, self._nrow, self._nlay)

    @property
    def vectordimensions(self):
        """3-tuple: The storage grid array dimensions tuple of 3 integers (read only).

        The tuple is (ncoord, nzcorn, nactnum).
        """
        ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
        nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4
        ntot = self._ncol * self._nrow * self._nlay

        return (ncoord, nzcorn, ntot)

    @property
    def ijk_handedness(self):
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
    def ijk_handedness(self, value):
        if value in ("right", "left"):
            self.reverse_row_axis(ijk_handedness=value)
        else:
            raise ValueError("The value must be 'right' or 'left'")
        self._ijk_handedness = value

    @property
    def subgrids(self):
        """:obj:`list` of :obj:`int`: A dict with subgrid name and an array as value.

        I.e. a dict on the form ``{"name1": [1, 2, 3, 4], "name2:" [5, 6, 7],
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
        if self._subgrids is None:
            return None

        return self._subgrids

    @subgrids.setter
    def subgrids(self, sgrids):

        if sgrids is None:
            self._subgrids = None

        if not isinstance(sgrids, OrderedDict):
            raise ValueError("Input to subgrids must be an ordered dictionary")

        lengths = 0
        zarr = []
        keys = []
        for key, val in sgrids.items():
            lengths += len(val)
            keys.append(key)
            zarr.extend(val)

        if lengths != self._nlay:
            raise ValueError("Subgrids lengths not equal NLAY")

        if set(zarr) != set(range(1, self._nlay + 1)):
            raise ValueError(
                "Arrays are not valid as the do not sum to "
                "vertical range, {}".format(zarr)
            )

        if len(keys) != len(set(keys)):
            raise ValueError("Subgrid keys are not unique: {}".format(keys))

        self._subgrids = sgrids

    @property
    def nactive(self):
        """int: Returns the number of active cells (read only)."""
        return len(self.actnum_indices)

    @property
    def actnum_array(self):
        """Returns the 3D ndarray which for active cells.

        Values are 1 for active, 0 for inactive, in C order (read only).

        """
        actnumv = self.get_actnum().values
        actnumv = ma.filled(actnumv, fill_value=0)

        return actnumv

    @property
    def actnum_indices(self):
        """:obj:np.ndrarray: Indices (1D array) for active cells (read only).

        In dual poro/perm systems, this will be the active indices for the
        matrix cells and/or fracture cells (i.e. actnum >= 1).
        """
        actnumv = self.get_actnum()
        actnumv = np.ravel(actnumv.values)
        self._actnum_indices = np.flatnonzero(actnumv)

        return self._actnum_indices

    @property
    def ntotal(self):
        """Returns the total number of cells (read only)."""
        return self._ncol * self._nrow * self._nlay

    @property
    def dualporo(self):
        """Boolean flag for dual porosity scheme (read only)."""
        return self._dualporo

    @property
    def dualperm(self):
        """Boolean flag for dual porosity scheme (read only)."""
        return self._dualperm

    @property
    def gridprops(self):
        """Return or set a XTGeo GridProperties objects attached to the Grid."""
        # Note, internally, the _props is a GridProperties instance, which is
        # a class that holds a list of properties.
        # Note that the `props` methods below will deal with properties in a
        # list context

        return self._props

    @gridprops.setter
    def gridprops(self, gprops):

        if not isinstance(gprops, xtgeo.grid3d.GridProperties):
            raise ValueError("Input must be a GridProperties instance")

        self._props = gprops  # self._props is a GridProperties instance

    @property
    def props(self):
        """Return or set a list of XTGeo GridProperty objects.

        When setting, the dimension of the property object is checked,
        and will raise an IndexError if it does not match the grid.

        When setting props, the current property list is replaced.

        See also :meth:`append_prop()` method to add a property to the current list.

        """
        # Note, internally, the _props is a GridProperties instance, which is
        # a class that holds a list of properties.

        prplist = None
        if isinstance(self._props, xtgeo.grid3d.GridProperties):
            prplist = self._props.props
        elif isinstance(self._props, list):
            raise RuntimeError(
                "self._props is a list, not a GridProperties " "instance"
            )
        return prplist

    @props.setter
    def props(self, plist):

        if not isinstance(plist, list):
            raise ValueError("Input to props must be a list")

        if self._props is None:
            self._props = xtgeo.grid3d.GridProperties()

        for litem in plist:
            if litem.dimensions != self.dimensions:
                raise IndexError(
                    "Property NX NY NZ <{}> does not match grid!".format(litem.name)
                )

        self._props.props = plist  # self._props is a GridProperties instance

    @property
    def propnames(self):
        """Returns a list of property names that are hooked to a grid."""
        plist = None
        if self._props is not None:
            plist = self._props.names

        return plist

    @property
    def roxgrid(self):
        """Get the Roxar native proj.grid_models[gname].get_grid() object."""
        return self._roxgrid

    @property
    def roxindexer(self):
        """The Roxar native proj.grid_models[gname].get_grid().grid_indexer object."""
        return self._roxindexer

    def generate_hash(self, hashmethod="md5"):
        """Return a unique hash ID for current instance.

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        .. versionadded:: 2.14
        """
        required = (
            "ncol",
            "nrow",
            "nlay",
            "coordsv",
            "zcornsv",
            "actnumsv",
        )

        gid = ""
        for req in required:
            gid += f"{getattr(self, '_' + req)}"

        return xtgeo.common.sys.generic_hash(gid, hashmethod=hashmethod)

    # ==================================================================================
    # Create/import/export
    # ==================================================================================

    def create_box(
        self,
        dimension=(10, 12, 6),
        origin=(10.0, 20.0, 1000.0),
        oricenter=False,
        increment=(100, 150, 5),
        rotation=30.0,
        flip=1,
    ):
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
        _grid_etc1.create_box(
            self,
            dimension=dimension,
            origin=origin,
            oricenter=oricenter,
            increment=increment,
            rotation=rotation,
            flip=flip,
        )
        self._tmp = {}

    def to_file(self, gfile, fformat="roff"):
        """Export grid geometry to file, various vendor formats.

        Args:
            gfile (str): Name of output file
            fformat (str): File format; roff/roff_binary/roff_ascii/
                grdecl/bgrdecl/egrid.

        Raises:
            OSError: Directory does not exist

        Example::

            xg.to_file("myfile.roff")
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb")

        gfile.check_folder(raiseerror=OSError)

        if fformat in ("roff", "roff_binary", "roff_bin", "roffbin"):
            _grid_export.export_roff(self, gfile.name, 0)
        elif fformat in ("roff_ascii", "roff_asc", "roffasc"):
            _grid_export.export_roff(self, gfile.name, 1)
        elif fformat == "grdecl":
            _grid_export.export_grdecl(self, gfile.name, 1)
        elif fformat == "bgrdecl":
            _grid_export.export_grdecl(self, gfile.name, 0)
        elif fformat == "egrid":
            _grid_export.export_egrid(self, gfile.name)
        else:
            raise SystemExit("Invalid file format")

    def to_hdf(
        self,
        gfile: Union[str, Path],
        compression: Optional[str] = None,
        chunks: Optional[bool] = False,
        subformat: Optional[int] = 844,
    ) -> Path:
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

            >>> xg.to_hdf("myfile_grid.h5")
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)
        gfile.check_folder(raiseerror=OSError)

        _grid_export.export_hdf5_cpgeom(
            self, gfile, compression=compression, chunks=chunks, subformat=subformat
        )

        return gfile.file

    def to_xtgf(
        self,
        gfile: Union[str, Path],
        subformat: Optional[int] = 844,
    ) -> Path:
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

            xg.to_xtg("myfile_grid.xtgf")
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)
        gfile.check_folder(raiseerror=OSError)

        _grid_export.export_xtgcpgeom(self, gfile, subformat=subformat)

        return gfile.file

    def to_roxar(
        self, project, gname, realisation=0, info=False, method="cpg"
    ):  # pragma: no cover
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

    def from_file(
        self, gfile, fformat=None, initprops=None, restartprops=None, restartdates=None
    ):
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

        Example::

            >>> myfile = "../../testdata/Zone/gullfaks.roff"
            >>> xg = Grid()
            >>> xg.from_file(myfile, fformat="roff")
            >>> # or shorter:
            >>> xg = Grid(myfile)  # will guess the file format

        Example using "eclipserun"::

            >>> mycase = "ECL"  # meaning ECL.EGRID, ECL.INIT, ECL.UNRST
            >>> xg = Grid()
            >>> xg.from_file(mycase, fformat="eclipserun", initprops="all")

        Raises:
            OSError: if file is not found etc
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="rb")

        obj = _grid_import.from_file(
            self,
            gfile,
            fformat=fformat,
            initprops=initprops,
            restartprops=restartprops,
            restartdates=restartdates,
        )
        self._tmp = {}
        self._metadata.required = self
        return obj

    def from_hdf(self, gfile, ijkrange=None, zerobased=False):
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

            xg.from_hdf("myfile_grid.h5", ijkrange=(1, 10, 10, 15, 1, 4))
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)

        _grid_import_xtgcpgeom.import_hdf5_cpgeom(
            self, gfile, ijkrange=ijkrange, zerobased=zerobased
        )

    def from_xtgf(self, gfile, mmap=False):
        """Import grid geometry from native xtgeo file format (experimental!).

        Args:
            gfile (str): Name of output file
            mmap (bool): If true, reading with memory mapping is active

        Example::

            xg.from_xtgf("myfile_grid.xtg")
        """
        gfile = xtgeo._XTGeoFile(gfile, mode="wb", obj=self)

        _grid_import_xtgcpgeom.import_xtgcpgeom(self, gfile, mmap)

    def from_roxar(
        self, projectname, gname, realisation=0, dimensions_only=False, info=False
    ):  # pragma: no cover
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
        _grid_roxapi.import_grid_roxapi(
            self, projectname, gname, realisation, dimensions_only, info
        )
        self._tmp = {}
        self._metadata.required = self

    # ==================================================================================
    # Various public methods
    # ==================================================================================

    def numpify_carrays(self):
        """Numpify pointers from C (SWIG) arrays so instance is easier to pickle."""
        warnings.warn(
            "Method numpify_carrays is deprecated and can be removed ({})".format(self),
            DeprecationWarning,
            stacklevel=2,
        )

    def copy(self):
        """Copy from one existing Grid instance to a new unique instance.

        Note that associated properties will also be copied.

        Example::

            newgrd = grd.copy()
        """
        logger.info("Copy a Grid instance")
        other = _grid_etc1.copy(self)
        return other

    def describe(self, details=False, flush=True):
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

            prp1 = []
            for prp in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                prp1.append("{:10.3f}".format(geom[prp]))

            prp2 = []
            for prp in ("avg_dx", "avg_dy", "avg_dz", "avg_rotation"):
                prp2.append("{:7.4f}".format(geom[prp]))

            geox = self.get_geometrics(
                cellcenter=False, allcells=True, return_dict=True
            )
            prp3 = []
            for prp in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                prp3.append("{:10.3f}".format(geox[prp]))

            prp4 = []
            for prp in ("avg_dx", "avg_dy", "avg_dz", "avg_rotation"):
                prp4.append("{:7.4f}".format(geox[prp]))

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

    def get_dataframe(self, activeonly=True, ijk=True, xyz=True, doubleformat=False):
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

            grd = Grid(gfile1, fformat="egrid")
            xpr = GridProperties()

            names = ["SOIL", "SWAT", "PRESSURE"]
            dates = [19991201]
            xpr.from_file(rfile1, fformat="unrst", names=names, dates=dates,
                        grid=grd)
            grd.gridprops = xpr  # attach properties to grid

            df = grd.dataframe()

            # save as CSV file
            df.to_csv("mygrid.csv")
        """
        if self.gridprops is None:
            self.gridprops = xtgeo.grid3d.GridProperties(
                ncol=self.ncol, nrow=self.nrow, nlay=self.nlay
            )

        return self.gridprops.dataframe(
            activeonly=activeonly,
            ijk=ijk,
            xyz=xyz,
            doubleformat=doubleformat,
            grid=self,
        )

    dataframe = get_dataframe  # backward compatibility...

    def append_prop(self, prop):
        """Append a single property to the grid."""
        if prop.dimensions == self.dimensions:
            if self._props is None:
                self._props = xtgeo.grid3d.GridProperties()

            self._props.append_props([prop])
        else:
            raise ValueError("Dimensions does not match")

    def set_subgrids(self, sdict):
        """Set the subgrid from a simplified ordered dictionary.

        The simplified dictionary is on the form
        {"name1": 3, "name2": 5}

        Note that the input must be an OrderedDict!

        """
        if sdict is None:
            return

        if not isinstance(sdict, OrderedDict):
            raise ValueError("Input sdict is not an OrderedDict")

        newsub = OrderedDict()

        inn1 = 1
        for name, nsub in sdict.items():
            inn2 = inn1 + nsub
            newsub[name] = range(inn1, inn2)
            inn1 = inn2

        self.subgrids = newsub

    def get_subgrids(self):
        """Get the subgrids on a simplified ordered dictionary.

        The simplified dictionary is on the form {"name1": 3, "name2": 5}
        """
        if not self.subgrids:
            return None

        newd = OrderedDict()
        for name, subarr in self.subgrids.items():
            newd[name] = len(subarr)

        return newd

    def rename_subgrids(self, names):
        """Rename the names in the subgrids with the new names.

        Args:
            names (list): List of new names, length of list must be same as length of
                subgrids


        Example::

            grd = xtgeo.Grid("somefile.roff")

            if len(grd.subgrids) == 3:
                grd.rename_subgrids(["Inky", "Tinky", "Pinky"])

        Raises:
            ValueError: Input names not a list or a tuple
            ValueError: Lenght of names list not same as number of subgrids

        .. versionadded:: 2.12
        """
        if not isinstance(names, (list, tuple)):
            raise ValueError("Input names not a list or a tuple")

        if len(names) != len(list(self.subgrids.keys())):
            raise ValueError("Lenght of names list not same as number of subgrids")

        subs = self.get_subgrids().copy()
        for num, oldname in enumerate(self.subgrids.keys()):
            subs[str(names[num])] = subs.pop(oldname)

        self.set_subgrids(subs)

    def estimate_design(self, nsub=None):
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

            grd = xtgeo.Grid("emerald.roff")
            res = grd.estimate_design(nsub="Etive")
            print("Subgrid design is ", res["design"])
            print("Subgrid simbox thickness is ", res["dzsimbox"])

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

    def estimate_flip(self):
        """Estimate flip (handedness) of grid returns as 1 or -1.

        The flip numbers are 1 for left-handed and -1 for right-handed.

        .. seealso:: :py:attr:`~ijk_handedness`
        """
        return _grid_etc1.estimate_flip(self)

    def subgrids_from_zoneprop(self, zoneprop):
        """Estimate subgrid index from a zone property.

        The new will esimate which will replace the current if any.

        Args:
            zoneprop(GridProperty): a XTGeo GridProperty instance.

        Returns:
            Will also return simplified dictionary is on the form
                {"name1": 3, "name2": 5}
        """
        newd = OrderedDict()
        _, _, k_index = self.get_ijk()
        kval = k_index.values
        zprval = zoneprop.values
        minzone = int(zprval.min())
        maxzone = int(zprval.max())

        for izone in range(minzone, maxzone + 1):
            mininzn = int(kval[zprval == izone].min())  # 1 base
            maxinzn = int(kval[zprval == izone].max())  # 1 base
            newd["zone" + str(izone)] = range(mininzn, maxinzn + 1)

        self.subgrids = newd

        return self.get_subgrids()

    def get_zoneprop_from_subgrids(self):
        """Make a XTGeo GridProperty instance for a Zone property subgrid index."""
        raise NotImplementedError("Not yet; todo")

    def get_actnum_indices(self, order="C"):
        """Returns the 1D ndarray which holds the indices for active cells.

        Args:
            order (str): "Either 'C' (default) or 'F' order).
        """
        actnumv = self.get_actnum().values.copy(order=order)
        actnumv = np.ravel(actnumv, order="K")
        return np.flatnonzero(actnumv)

    def get_dualactnum_indices(self, order="C", fracture=False):
        """Returns the 1D ndarray which holds the indices for matrix/fracture cases.

        Args:
            order (str): "Either 'C' (default) or 'F' order).
            fracture (bool): If True use Fracture properties.
        """
        if not self._dualporo:
            return None

        actnumv = self._dualactnum.values.copy(order=order)
        actnumv = np.ravel(actnumv, order="K")

        if not fracture:
            actnumvm = actnumv.copy()
            actnumvm[(actnumv == 3) | (actnumv == 1)] = 1
            actnumvm[(actnumv == 2) | (actnumv == 0)] = 0
            ind = np.flatnonzero(actnumvm)
        else:
            actnumvf = actnumv.copy()
            actnumvf[(actnumv == 3) | (actnumv == 2)] = 1
            actnumvf[(actnumv == 1) | (actnumv == 0)] = 0
            ind = np.flatnonzero(actnumvf)

        return ind

    def get_gridproperties(self):
        """Return the :obj:`GridProperties` instance attached to the grid.

        See also the :meth:`gridprops` property
        """
        return self._props

    def get_prop_by_name(self, name):
        """Gets a property object by name lookup, return None if not present."""
        for obj in self.props:
            if obj.name == name:
                return obj

        return None

    def get_cactnum(self):
        """Returns the C pointer object reference to the ACTNUM array (deprecated)."""
        warnings.warn(
            "Method get_cactnum is deprecated and will be removed. ({})".format(self),
            DeprecationWarning,
            stacklevel=2,
        )

    def get_actnum(self, name="ACTNUM", asmasked=False, mask=None, dual=False):
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): Actnum is returned with all cells shown
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead!
            dual (bool): If True, and the grid is a dualporo/perm grid, an
                extended ACTNUM is applied (numbers 0..3)

        Example::

            act = mygrid.get_actnum()
            print("{}% cells are active".format(act.values.mean() * 100))

        .. versionchanged:: 2.6 Added ``dual`` keyword
        """
        if mask is not None:
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

        # return the object
        return act

    def set_actnum(self, actnum):
        """Modify the existing active cell index, ACTNUM.

        Args:
            actnum (GridProperty): a gridproperty instance with 1 for active
                cells, 0 for inactive cells

        Example::

            act = mygrid.get_actnum()
            act.values[:, :, :] = 1
            act.values[:, :, 4] = 0
            grid.set_actnum(act)
        """
        val1d = actnum.values.ravel(order="K")

        if self._xtgformat == 1:
            self._actnumsv = _gridprop_lowlevel.c2f_order(self, val1d)
        else:
            self._actnumsv = np.ma.filled(actnum.values, fill_value=0).astype(np.int32)

    def get_dz(self, name="dZ", flip=True, asmasked=True, mask=None):
        """Return the dZ as GridProperty object.

        The dZ is computed as an average height of the vertical pillars in
        each cell, projected to vertical dimension.

        Args:
            name (str): name of property
            flip (bool): Use False for Petrel grids were Z is negative down
                (experimental)
            asmasked (bool): True if only for active cells, False for all cells
            mask (bool): Deprecated, use asmasked instead!

        Returns:
            A XTGeo GridProperty object dZ
        """
        if mask is not None:
            asmasked = self._evaluate_mask(mask)

        deltaz = _grid_etc1.get_dz(self, name=name, flip=flip, asmasked=asmasked)

        return deltaz

    def get_dxdy(self, names=("dX", "dY"), asmasked=False):
        """Return the dX and dY as GridProperty object.

        The values lengths are projected to a constant Z.

        Args:
            name (tuple): names of properties
            asmasked (bool). If True, make a np.ma array where inactive cells
                are masked.

        Returns:
            Two XTGeo GridProperty objects (dx, dy).
        """
        deltax, deltay = _grid_etc1.get_dxdy(self, names=names, asmasked=asmasked)

        # return the property objects
        return deltax, deltay

    def get_cell_volume(
        self, ijk=(1, 1, 1), activeonly=True, zerobased=False, precision=2
    ):
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

            >>> grid = Grid()
            >>> grid.from_file("gullfaks2.roff")
            >>> vol = grid.get_cell_volume(ijk=(45,13,2))

        .. versionadded:: 2.13 (as experimental)
        """
        vol = _grid_etc1.get_cell_volume(
            self,
            ijk=ijk,
            activeonly=activeonly,
            zerobased=zerobased,
            precision=precision,
        )

        return vol

    def get_bulk_volume(self, name="bulkvol", asmasked=True, precision=2):
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

    def get_indices(self, names=("I", "J", "K")):
        """Return 3 GridProperty objects for column, row, and layer index.

        Note that the indexes starts with 1, not zero (i.e. upper
        cell layer is K=1)

        This method is deprecated, use :meth:`get_ijk()` instead.

        Args:
            names (tuple): Names of the columns (as property names)

        Examples::

            i_index, j_index, k_index = grd.get_indices()

        """
        warnings.warn("Use method get_ijk() instead", DeprecationWarning, stacklevel=2)
        return self.get_ijk(names=names, asmasked=False)

    def get_ijk(
        self, names=("IX", "JY", "KZ"), asmasked=True, mask=None, zerobased=False
    ):
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter, J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            asmasked: If True, UNDEF cells are masked, default is True
            mask (bool): Deprecated, use asmasked instead!
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """
        if mask is not None:
            asmasked = self._evaluate_mask(mask)

        ixc, jyc, kzc = _grid_etc1.get_ijk(
            self, names=names, asmasked=asmasked, zerobased=zerobased
        )

        # return the objects
        return ixc, jyc, kzc

    def get_ijk_from_points(
        self,
        points,
        activeonly=True,
        zerobased=False,
        dataframe=True,
        includepoints=True,
        columnnames=("IX", "JY", "KZ"),
        fmt="int",
        undef=-1,
    ):
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
        ijklist = _grid_etc1.get_ijk_from_points(
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

        # return the dataframe or list of tuples
        return ijklist

    def get_xyz(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS"), asmasked=True, mask=None):
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
            asmasked = self._evaluate_mask(mask)

        xcoord, ycoord, zcoord = _grid_etc1.get_xyz(
            self, names=names, asmasked=asmasked
        )

        # return the objects
        return xcoord, ycoord, zcoord

    def get_xyz_cell_corners(self, ijk=(1, 1, 1), activeonly=True, zerobased=False):
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

            >>> grid = Grid()
            >>> grid.from_file("gullfaks2.roff")
            >>> xyzlist = grid.get_xyz_corners_cell(ijk=(45,13,2))

        Raises:
            RuntimeWarning if spesification is invalid.
        """
        clist = _grid_etc1.get_xyz_cell_corners(
            self, ijk=ijk, activeonly=activeonly, zerobased=zerobased
        )

        return clist

    def get_xyz_corners(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS")):
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

            >>> grid = Grid()
            >>> grid.from_file("gullfaks2.roff")
            >>> clist = grid.get_xyz_corners()


        Raises:
            RunetimeError if corners has wrong spesification
        """
        grid_props = _grid_etc1.get_xyz_corners(self, names=names)

        # return the 24 objects in a long tuple (x1, y1, z1, ... x8, y8, z8)
        return grid_props

    def get_layer_slice(self, layer, top=True, activeonly=True):
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

                grd = Grid()
                grd.from_file(REEKFILE)

                parr, ibarr = grd.get_layer_slice(grd.nlay, top=False)

        .. versionadded:: 2.3
        """
        return _grid_etc1.get_layer_slice(self, layer, top=top, activeonly=activeonly)

    def get_geometrics(
        self, allcells=False, cellcenter=True, return_dict=False, _ver=1
    ):
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

            mygrid = Grid("gullfaks.roff")
            gstuff = mygrid.get_geometrics(return_dict=True)
            print("X min/max is {} {}".format(gstuff["xmin", gstuff["xmax"]))

        """
        gresult = _grid_etc1.get_geometrics(
            self,
            allcells=allcells,
            cellcenter=cellcenter,
            return_dict=return_dict,
            _ver=_ver,
        )

        return gresult

    def get_adjacent_cells(self, prop, val1, val2, activeonly=True):
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
        presult = _grid_etc1.get_adjacent_cells(
            self, prop, val1, val2, activeonly=activeonly
        )

        return presult

    def get_gridquality_properties(self):
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
        gprops = _grid_etc1.get_gridquality_properties(self)

        return gprops

    # =========================================================================
    # Some more special operations that changes the grid or actnum
    # =========================================================================
    def activate_all(self):
        """Activate all cells in the grid, by manipulating ACTNUM."""
        self._actnumsv = np.ones(self.dimensions, dtype=np.int32)

        if self._xtgformat == 1:
            self._actnumsv = self._actnumsv.flatten()

        self._tmp = {}

    def inactivate_by_dz(self, threshold):
        """Inactivate cells thinner than a given threshold."""
        _grid_etc1.inactivate_by_dz(self, threshold)
        self._tmp = {}

    def inactivate_inside(self, poly, layer_range=None, inside=True, force_close=False):
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

    def inactivate_outside(self, poly, layer_range=None, force_close=False):
        """Inacativate grid outside a polygon."""
        self.inactivate_inside(
            poly, layer_range=layer_range, inside=False, force_close=force_close
        )
        self._tmp = {}

    def collapse_inactive_cells(self):
        """Collapse inactive layers where, for I J with other active cells."""
        _grid_etc1.collapse_inactive_cells(self)
        self._tmp = {}

    def crop(self, colcrop, rowcrop, laycrop, props=None):
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

                >>> from xtgeo.grid3d import Grid
                >>> gf = Grid("gullfaks2.roff")
                >>> gf.crop((3, 6), (4, 20), (1, 10))
                >>> gf.to_file("gf_reduced.roff")

        """
        _grid_etc1.crop(self, (colcrop, rowcrop, laycrop), props=props)
        self._tmp = {}

    def reduce_to_one_layer(self):
        """Reduce the grid to one single layer.

        Example::

            >>> from xtgeo.grid3d import Grid
            >>> gf = Grid("gullfaks2.roff")
            >>> gf.nlay
            47
            >>> gf.reduce_to_one_layer()
            >>> gf.nlay
            1

        """
        _grid_etc1.reduce_to_one_layer(self)
        self._tmp = {}

    def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):
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

    def reverse_row_axis(self, ijk_handedness=None):
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

            grd = xtgeo.Grid("somefile.roff")
            prop1 = xtgeo.GridProperty("somepropfile1.roff")
            prop2 = xtgeo.GridProperty("somepropfile2.roff")

            grd.props = [prop1, prop2]

            # secure that the grid geometry is IJK right-handed
            grd.reverse_row_axis(ijk_handedness="right")

        .. versionadded:: 2.5

        """
        _grid_etc1.reverse_row_axis(self, ijk_handedness=ijk_handedness)
        self._tmp = {}

    def make_zconsistent(self, zsep=1e-5):
        """Make the 3D grid consistent in Z, by a minimal gap (zsep).

        Args:
            zsep (float): Minimum gap
        """
        _grid_etc1.make_zconsistent(self, zsep)
        self._tmp = {}

    def convert_to_hybrid(
        self,
        nhdiv=10,
        toplevel=1000.0,
        bottomlevel=1100.0,
        region=None,
        region_number=None,
    ):
        """Convert to hybrid grid, either globally or in a selected region.

        Note that the resulting hybrid will have a increased number of layers.
        In the initial grid has N layers, and the number of horizontal layers
        is NHDIV, then the result grid will have N * 2 + NHDIV layers.

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

    def refine_vertically(self, rfactor, zoneprop=None):
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
        well=None,
        zonelogname="ZONELOG",
        zoneprop=None,
        onelayergrid=None,
        zonelogrange=(0, 9999),
        zonelogshift=0,
        depthrange=None,
        perflogname=None,
        perflogrange=(1, 9999),
        filterlogname=None,
        filterlogrange=(1e-32, 9999.0),
        resultformat=1,
    ):
        """Reports mismatch between wells and a zone.

        Approaches on matching:
            1. Use the well zonelog as basis, and compare sampled zone with that
               interval. This means that zone cells outside well range will not be
               counted
            2. Compare intervals with wellzonation in range or grid zonations in
               range. This gives a wider comparison, and will capture cases
               where grid zonations is outside well zonation

        .. image:: ../../docs/images/zone-well-mismatch-plain.svg
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

            g1 = xtgeo.Grid("gullfaks2.roff")

            z = xtgeo.GridProperty(gullfaks2_zone.roff", name="Zone")

            w2 = xtgeo.Well("34_10-1.w", zonelogname="Zonelog")

            w3 = xtgeo.Well("34_10-B-21_B.w", zonelogname="Zonelog"))

            wells = [w2, w3]

            for w in wells:
                response = g1.report_zone_mismatch(
                    well=w, zonelogname="ZONELOG", zoneprop=z,
                    zonelogrange=(0, 19), depthrange=(1700, 9999))

                print(response)

        .. versionchanged:: 2.8 Added several new keys and better precision in result
        .. versionchanged:: 2.11 Added ``perflogrange`` and ``filterlogrange``
        """
        reports = _grid_wellzone.report_zone_mismatch(
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

        return reports

    # ==================================================================================
    # Extract a fence/randomline by sampling, ready for plotting with e.g. matplotlib
    # ==================================================================================
    def get_randomline(
        self,
        fencespec,
        prop,
        zmin=None,
        zmax=None,
        zincrement=1.0,
        hincrement=None,
        atleast=5,
        nextend=2,
    ):
        """Get a sampled randomline from a fence spesification.

        This randomline will be a 2D numpy with depth on the vertical
        axis, and length along as horizontal axis. Undefined values will have
        the np.nan value.

        The input fencespec is either a 2D numpy where each row is X, Y, Z, HLEN,
        where X, Y are UTM coordinates, Z is depth/time, and HLEN is a
        length along the fence, or a Polygons instance.

        If input fencspec is a numpy 2D, it is important that the HLEN array
        has a constant increment and ideally a sampling that is less than the
        Grid resolution. If a Polygons() instance, this is automated if hincrement is
        None, and ignored if hincrement is False.

        Args:
            fencespec (:obj:`~numpy.ndarray` or :class:`~xtgeo.xyz.polygons.Polygons`):
                2D numpy with X, Y, Z, HLEN as rows or a xtgeo Polygons() object.
            prop (GridProperty or str): The grid property object, or name, which shall
                be plotted.
            zmin (float): Minimum Z (default is Grid Z minima/origin)
            zmax (float): Maximum Z (default is Grid Z maximum)
            zincrement (float): Sampling vertically, default is 1.0
            hincrement (float or bool): Resampling horizontally. This applies only
                if the fencespec is a Polygons() instance. If None (default),
                the distance will be deduced automatically. If False, then it assumes
                the Polygons can be used as-is.
            atleast (int): Minimum number of horizontal samples (only if
                fencespec is a Polygons instance and hincrement != False)
            nextend (int): Extend with nextend * hincrement in both ends (only if
                fencespec is a Polygons instance and hincrement != False)

        Returns:
            A tuple: (hmin, hmax, vmin, vmax, ndarray2d)

        Raises:
            ValueError: Input fence is not according to spec.

        Example::

            mygrid = xtgeo.Grid("somegrid.roff")
            poro = xtgeo.GridProperty("someporo.roff")
            mywell = xtgeo.Well("somewell.rmswell")
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

    def _convert_xtgformat2to1(self):
        """Convert arrays from new structure xtgformat=2 to legacy xtgformat=1."""
        _grid_etc1._convert_xtgformat2to1(self)

    def _convert_xtgformat1to2(self):
        """Convert arrays from old structure xtgformat=1 to new xtgformat=2."""
        _grid_etc1._convert_xtgformat1to2(self)

    def _xtgformat1(self):
        """Shortform... arrays from new structure xtgformat=2 to legacy xtgformat=1."""
        self._convert_xtgformat2to1()

    def _xtgformat2(self):
        """Shortform... arrays from old structure xtgformat=1 to new xtgformat=2."""
        self._convert_xtgformat1to2()
