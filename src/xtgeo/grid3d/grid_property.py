from __future__ import annotations

import copy
import functools
import hashlib
from types import FunctionType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.types import Dimensions
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.metadata.metadata import MetaDataCPProperty

from . import (
    _gridprop_export,
    _gridprop_lowlevel,
    _gridprop_op1,
    _gridprop_roxapi,
    _gridprop_value_init,
)
from ._grid3d import _Grid3D
from ._gridprop_import_eclrun import (
    import_gridprop_from_init,
    import_gridprop_from_restart,
)
from ._gridprop_import_grdecl import import_bgrdecl_prop, import_grdecl_prop
from ._gridprop_import_roff import import_roff
from ._gridprop_import_xtgcpprop import import_xtgcpprop

xtg = XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Union

    import numpy.typing as npt

    from xtgeo.common.types import FileLike
    from xtgeo.xyz.polygons import Polygons

    from ._gridprop_op1 import XYValueLists
    from .grid import Grid

    GridProperty_DType = Union[
        type[np.uint8],
        type[np.uint16],
        type[np.int16],
        type[np.int32],
        type[np.int64],
        type[np.float16],
        type[np.float32],
        type[np.float64],
    ]
    Roxar_DType = Union[type[np.uint8], type[np.uint16], type[np.float32]]

# --------------------------------------------------------------------------------------
# Comment on 'asmasked' vs 'activeonly:
#
# 'asmasked'=True will return a np.ma array, while 'asmasked' = False will
# return a np.ndarray
#
# The 'activeonly' will filter out masked entries, or use None or np.nan
# if 'activeonly' is False.
#
# Use word 'zerobased' for a bool regrading startcell basis is 1 or 0
#
# For functions with mask=... ,they should be replaced with asmasked=...
# --------------------------------------------------------------------------------------

# ======================================================================================
# Functions outside the class, for rapid access. Will be exposed as
# xxx = xtgeo.gridproperty_from_file.
# ======================================================================================


def _data_reader_factory(fformat: FileFormat) -> Callable:
    if fformat in (FileFormat.ROFF_BINARY, FileFormat.ROFF_ASCII):
        return import_roff
    if fformat in (FileFormat.FINIT, FileFormat.INIT):
        return import_gridprop_from_init
    if fformat in (FileFormat.FUNRST, FileFormat.UNRST):
        return functools.partial(import_gridprop_from_restart, fformat=fformat)
    if fformat == FileFormat.GRDECL:
        return import_grdecl_prop
    if fformat == FileFormat.BGRDECL:
        return import_bgrdecl_prop
    if fformat == FileFormat.XTG:
        return import_xtgcpprop

    extensions = FileFormat.extensions_string(
        [
            FileFormat.ROFF_BINARY,
            FileFormat.ROFF_ASCII,
            FileFormat.INIT,
            FileFormat.FINIT,
            FileFormat.UNRST,
            FileFormat.FUNRST,
            FileFormat.GRDECL,
            FileFormat.BGRDECL,
            FileFormat.XTG,
        ]
    )
    raise InvalidFileFormatError(
        f"File format {fformat} is invalid for type GridProperty. "
        f"Supported formats are {extensions}."
    )


def gridproperty_from_file(
    pfile: FileLike,
    fformat: str | None = None,
    **kwargs: dict[str, Any],
) -> GridProperty:
    """
    Make a GridProperty instance directly from a file import.

    Note that the the property may be linked to its geometrical grid
    through the ``grid=`` option. Sometimes this is required, for instance
    for most Eclipse input.

    Args:
        pfile: Name of file to be imported.
        fformat: File format to be used (roff/init/unrst/grdecl).
            Defaults to None and tries to infer from file extension.
        name (str): Name of property to import
        date (int or str): For restart files, date in YYYYMMDD format. Also
            the YYYY-MM-DD form is allowed (string), and for Eclipse,
            mnemonics like 'first', 'last' is also allowed.
        grid (Grid, optional): Grid object for checks. Optional for
            ROFF, required for Eclipse).
        gridlink (bool): If True, and grid is not None, a link from the grid
            instance to the property is made. If False, no such link is made.
            Avoiding gridlink is recommended when running statistics of multiple
            realisations of a property.
        fracture (bool): Only applicable for DUAL POROSITY systems. If True
            then the fracture property is read. If False then the matrix
            property is read. Names will be appended with "M" or "F"
        ijrange (list-like): A list of 4 numbers (i1, i2, j1, j2) for a subrange
            of cells to read. Only applicable for xtgcpprop format.
        zerobased (bool): Input if cells counts are zero- or one-based in
            ijrange. Only applicable for xtgcpprop format.

    Returns:
        A GridProperty instance.

    Examples::

        import xtgeo
        gprop = xtgeo.gridproperty_from_file("somefile.roff", fformat="roff")

        # or

        mygrid = xtgeo.grid_from_file("ECL.EGRID")
        pressure_1 = xtgeo.gridproperty_from_file("ECL.UNRST", name="PRESSURE",
                                                 date="first", grid=mygrid)

    """
    return GridProperty._read_file(pfile, fformat, **kwargs)


def gridproperty_from_roxar(
    project: Any,  # project can be a path but also a magic variable in RMS
    gname: str,
    pname: str,
    realisation: int = 0,
    faciescodes: bool = False,
) -> GridProperty:
    """
    Make a GridProperty instance directly inside RMS.

    Args:
        project: The Roxar project path or magical pre-defined variable in RMS
        gname: Name of the grid model
        pname: Name of the grid property
        realisation: Realisation number (default 0; first)
        faciescodes: If a Roxar property is of the special body_facies type
            (e.g. result from a channel facies object modelling), the default
            is to get the body code values. If faciescodes is True, the facies
            code values will be read instead. For other roxar properties this
            key is not relevant.

    Returns:
        A GridProperty instance.

    Example::

        import xtgeo
        myporo = xtgeo.gridproperty_from_roxar(project, 'Geogrid', 'Poro')

    """
    return GridProperty._read_roxar(
        project,
        gname,
        pname,
        realisation=realisation,
        faciescodes=faciescodes,
    )


class GridProperty(_Grid3D):
    """
    Class for a single 3D grid property, e.g porosity or facies.

    An GridProperty instance may or may not 'belong' to a grid (geometry) object.
    E.g. for ROFF input, ncol, nrow, nlay are given in the import file and the grid
    geometry file is not needed. For many Eclipse files, the grid geometry is needed
    as this holds the active number indices (ACTNUM).

    Normally the instance is created when importing a grid
    property from file, but it can also be created directly, as e.g.::

        poro = GridProperty(ncol=233, nrow=122, nlay=32)

    The grid property values ``someinstance.values`` by themselves is a 3D masked
    numpy usually as either float64 (double) or int32 (if discrete), and undefined
    cells are displayed as masked. The internal array order is now C_CONTIGUOUS.
    (i.e. not in Eclipse manner). A 1D view (C order) is achieved by the
    values1d property, e.g.::

       poronumpy = poro.values1d

    .. versionchanged:: 2.6 Possible to make GridProperty instance directly from Grid
    .. versionchanged:: 2.8 Possible to base it on existing GridProperty instance

    """

    def __init__(
        self,
        gridlike: Grid | GridProperty | None = None,
        ncol: int | None = None,
        nrow: int | None = None,
        nlay: int | None = None,
        name: str = "unknown",
        discrete: bool = False,
        date: str | None = None,
        grid: Grid | None = None,
        linkgeometry: bool = True,
        fracture: bool = False,
        codes: dict[int, str] | None = None,
        dualporo: bool = False,
        dualperm: bool = False,
        roxar_dtype: Roxar_DType | None = None,
        values: np.ndarray | float | int | None = None,
        roxorigin: bool = False,
        filesrc: str | None = None,
    ) -> None:
        """
        Instantiating.

        Args:
            gridlike: Grid or GridProperty instance, or leave blank.
            ncol: Number of columns (nx). Defaults to 4.
            nrow: Number of rows (ny). Defaults to 3.
            nlay: Number of layers (nz). Defaults to 5.
            name: Name of property. Defaults to "unknown".
            discrete: True or False. Defaults to False.
            date: Date on YYYYMMDD form.
            grid: Attached Grid object.
            linkgeometry: If True, establish a link between GridProperty
                and Grid. Defaults to True.
            fracture: True if fracture option (relevant for flow simulator data).
                Defaults to False.
            codes: Codes in case a discrete property e.g. {1: "Sand", 4: "Shale"}.
            dualporo: True if dual porosity system. Defaults to False.
            dualperm: True if dual porosity and dual permeability system.
                Defaults to False.
            roxar_dtype: Specify Roxar datatype e.g. np.uint8.
            values: Values to apply.
            roxorigin: True if the object comes from Roxar API. Defaults to False.
            filesrc: Where the file came from.

        Raises:
            RuntimeError: If something goes wrong (e.g. file not found).

        Examples::

            import xtgeo
            myprop = xtgeo.gridproperty_from_file("emerald.roff", name="PORO")

            # or

            values = np.ma.ones((12, 17, 10), dtype=np.float64),
            myprop = GridProperty(ncol=12, nrow=17, nlay=10,
                                  values=values, discrete=False,
                                  name="MyValue")

            # or create properties from a Grid() instance

            mygrid = xtgeo.grid_from_file("grid.roff")
            myprop1 = xtgeo.GridProperty(mygrid, name="PORO")
            myprop2 = xtgeo.GridProperty(mygrid, name="FACIES", discrete=True, values=1,
                                   linkgeometry=True)  # alternative 1
            myprop2.geometry = mygrid  # alternative 2 to link grid geometry to property

            # from Grid instance:
            grd = xtgeo.grid_from_file("somefile_grid_file")
            myprop = GridProperty(grd, values=99, discrete=True)  # based on grd

            # or from existing GridProperty instance:
            myprop2 = GridProperty(myprop, values=99, discrete=False)  # based on myprop

        """
        super().__init__(ncol or 4, nrow or 3, nlay or 5)

        # Instance attributes defaults:
        self._name = name
        self._date = date
        self._isdiscrete = discrete
        self._geometry = grid
        self._fracture = fracture
        self._codes = {} if codes is None else codes

        # Not primary input:
        self._dualporo = dualporo
        self._dualperm = dualperm

        self._filesrc = filesrc
        self._roxorigin = roxorigin

        if roxar_dtype is None:
            self._roxar_dtype: Roxar_DType = np.uint8 if discrete else np.float32
        else:
            self.roxar_dtype = roxar_dtype

        self._undef = UNDEF_INT if discrete else UNDEF

        self._set_initial_dimensions(gridlike, (ncol, nrow, nlay))

        self._values = _gridprop_value_init.gridproperty_non_dummy_values(
            gridlike, self.dimensions, values, discrete
        )

        if isinstance(gridlike, xtgeo.grid3d.Grid):
            if linkgeometry:
                # Associate this grid property with a Grid instance. This is not default
                # since sunch links may affect garbage collection
                self.geometry = gridlike
            gridlike.append_prop(self)

        self._metadata: MetaDataCPProperty = MetaDataCPProperty()

    def _set_initial_dimensions(
        self,
        gridlike: Grid | GridProperty | None,
        input_dimensions: tuple[int | None, int | None, int | None],
    ) -> None:
        """
        Sets the initial dimensions either from input, grid or default.

        Args:
            gridlike: Grid/GridProperty instance or leave blank.
            input_dimensions: The (ncol, nrow, nlay) tuple describing the
            dimensions.

        If a gridlike is given, we use its dimensions, but make sure it matches
        the input dimensions if given (not None). Otherwise, dimensions are either
        set to the input dimensions or defaulted.

        """
        if gridlike is not None:
            self._ncol = gridlike.ncol
            self._nrow = gridlike.nrow
            self._nlay = gridlike.nlay
            self._check_dimensions_match(*input_dimensions)
        else:
            ncol, nrow, nlay = input_dimensions
            if ncol is None:
                self._ncol = 4
            else:
                self._ncol = ncol
            if nrow is None:
                self._nrow = 3
            else:
                self._nrow = nrow
            if nlay is None:
                self._nlay = 5
            else:
                self._nlay = nlay

    def _check_dimensions_match(
        self, ncol: int | None, nrow: int | None, nlay: int | None
    ) -> None:
        """
        Checks that Grid/GridProperty dimensions match provided input dimensions.

        Args:
            input_dimensions: The (ncol, nrow, nlay) tuple describing the
            dimensions.

        Raises:
            ValueError: If given dimensions are not None and do not
                match dimensions of the GridProperty

        """
        if ncol is not None and self._ncol != ncol:
            raise ValueError(
                f"Mismatching column dimension given: {ncol} vs {self._ncol}"
            )
        if nrow is not None and self._nrow != nrow:
            raise ValueError(f"Mismatching row dimension given: {nrow} vs {self._nrow}")
        if nlay is not None and self._nlay != nlay:
            raise ValueError(
                f"Mismatching layer dimension given: {nlay} vs {self._nlay}"
            )

    def __del__(self) -> None:
        logger.debug("DELETING property instance %s", self.name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} (id={id(self)}) ncol={self._ncol!r}, "
            f"nrow={self._nrow!r}, nlay={self._nlay!r}, filesrc={self._filesrc!r}"
        )

    def __str__(self) -> str:
        return self.describe(flush=False)

    # ==================================================================================
    # Properties
    # Some properties such as ncol, nrow, nlay are from _Grid3d
    # ==================================================================================

    @property
    def metadata(self) -> MetaDataCPProperty:
        """Get or set metadata object instance of type MetaDataCPProperty."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: MetaDataCPProperty) -> None:
        if not isinstance(metadata, MetaDataCPProperty):
            raise ValueError("Input metadata not an instance of MetaDataCPProperty")
        # TODO: validate this?
        self._metadata = metadata

    @property
    def name(self) -> str | None:
        """Get or set the property name."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def dimensions(self) -> Dimensions:
        """Get the grid dimensions as a NamedTuple of 3 integers."""
        return Dimensions(self.ncol, self.nrow, self.nlay)

    @property
    def nactive(self) -> int:
        """Get the number of active cells."""
        return len(self.actnum_indices)

    @property
    def geometry(self) -> Grid | None:
        """Get or set the linked geometry, i.e. the Grid instance."""
        return self._geometry

    @geometry.setter
    def geometry(self, grid: Grid | None) -> None:
        if grid is None:
            self._geometry = None
        elif isinstance(grid, xtgeo.grid3d.Grid) and grid.dimensions == self.dimensions:
            self._geometry = grid
        else:
            raise ValueError("Could not set geometry; wrong type or size")

    @property
    def actnum_indices(self) -> np.ndarray:
        """
        Get the 1D ndarray which holds the indices for active cells
        given in 1D, C order.

        """
        gridprop = self.get_actnum()
        actnumv = np.ravel(gridprop.values)
        return np.flatnonzero(actnumv)

    @property
    def isdiscrete(self) -> bool:
        """
        Get or set whether this property is discrete.

        This can also be used to convert from continuous to discrete
        or from discrete to continuous::

            myprop.isdiscrete = False

        """
        return self._isdiscrete

    @isdiscrete.setter
    def isdiscrete(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise ValueError("Input to {__name__} must be a bool")

        if flag is self._isdiscrete:
            return

        if flag is True and self._isdiscrete is False:
            self.continuous_to_discrete()
        else:
            self.discrete_to_continuous()

    @property
    def dtype(self) -> GridProperty_DType:
        """
        Get or set the ``values`` numpy dtype.

        When setting, note that the the dtype must correspond to the
        `isdiscrete` property. Hence dtype cannot alter isdiscrete status

        Example::

            if myprop.isdiscrete:
                myprop.dtype = np.uint16

        """
        return self._values.dtype

    @dtype.setter
    def dtype(self, dtype: GridProperty_DType) -> None:
        allowed: list[GridProperty_DType] = (
            [np.uint8, np.uint16, np.int16, np.int32, np.int64]
            if self.isdiscrete
            else [np.float16, np.float32, np.float64]
        )
        if dtype not in allowed:
            raise ValueError(
                f"{__name__}: Wrong input for dtype. Use one of {allowed}!"
            )
        # https://github.com/numpy/numpy/issues/24392
        self.values = self.values.astype(dtype)  # type: ignore

    @property
    def filesrc(self) -> str | None:
        """Get or set the GridProperty file src (if any)."""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, src: str) -> None:
        self._filesrc = src

    @property
    def roxar_dtype(self) -> Roxar_DType | None:
        """Get or set the roxar dtype (if any)."""
        return self._roxar_dtype

    @roxar_dtype.setter
    def roxar_dtype(self, dtype: Roxar_DType) -> None:
        allowed = [np.uint8, np.uint16, np.float32]
        if dtype not in allowed:
            raise ValueError(
                f"{__name__}: Wrong input for roxar_dtype. Use one of {allowed}!"
            )
        self._roxar_dtype = dtype

    @property
    def date(self) -> str | None:
        """Get or set the property date as string in YYYYMMDD format."""
        return self._date

    @date.setter
    def date(self, date: str | None) -> None:
        self._date = date

    @property
    def codes(self) -> dict[int, str]:
        """Get or set the property codes as a dictionary."""
        return self._codes

    @codes.setter
    def codes(self, codes: dict[int, str]) -> None:
        if not isinstance(codes, dict):
            raise ValueError(
                "The codes must be a python dictionary, current input "
                f"is type: {type(codes)}"
            )
        self._codes = copy.deepcopy(codes)

    @property
    def ncodes(self) -> int:
        """Get number of codes if discrete grid property."""
        return len(self._codes)

    @property
    def values(self) -> np.ma.MaskedArray:
        """Get or set the grid property as a masked 3D numpy array."""
        return self._values

    @values.setter
    def values(self, values: np.ndarray) -> None:
        values = self.ensure_correct_values(self.ncol, self.nrow, self.nlay, values)
        self._values = values

    @property
    def ntotal(self) -> int:
        """Get total number of cells (ncol * nrow * nlay)."""
        return self.ncol * self.nrow * self.nlay

    @property
    def roxorigin(self) -> bool:
        """Get boolean value of True if the property comes from ROXAPI."""
        return self._roxorigin

    @roxorigin.setter
    def roxorigin(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise ValueError("Input to roxorigin must be True or False")
        self._roxorigin = val

    @property
    def values1d(self) -> np.ma.MaskedArray:
        """Get a masked 1D array view of values."""
        return self._values.reshape(-1)

    @property
    def undef(self) -> float | int:
        """Get the actual undef value for floats or ints in numpy arrays."""
        if self._isdiscrete:
            return UNDEF_INT
        return UNDEF

    @property
    def undef_limit(self) -> float | int:
        """
        Get the undef limit number, which is slightly less than the
        undef value.

        Hence for numerical precision, one can force undef values
        to a given number, e.g.::

           x[x<x.undef_limit] = 999

        Undef limit values cannot be changed (read only).

        """
        if self._isdiscrete:
            return UNDEF_INT_LIMIT
        return UNDEF_LIMIT

    # ==================================================================================
    # Class and special methods
    # ==================================================================================

    def generate_hash(self) -> str:
        """
        Generates a sha256 hash id representing a GridProperty.

        Returns:
            A unique hash id string.

        .. versionadded:: 2.10

        """
        mhash = hashlib.sha256()
        gid = (
            f"{self._filesrc}{self._ncol}{self._nrow}{self._nlay}"
            f"{self._values.mean()}{self._values.min()}{self._values.max()}"
        )
        mhash.update(gid.encode())
        return mhash.hexdigest()

    @classmethod
    def methods(cls) -> str:
        """
        A list of methods in the class as a string.

        Returns:
            The names of the methods in the class.

        Example::
            >>> print(GridProperty.methods())
            METHODS for GridProperty():
            ======================
            __init__
            _reset
            _set_initial_dimensions
            _check_dimensions_match
            ...

        """
        mets = [x for x, y in cls.__dict__.items() if isinstance(y, FunctionType)]

        txt = "METHODS for GridProperty():\n======================\n"
        for met in mets:
            txt += str(met) + "\n"

        return txt

    def ensure_correct_values(
        self,
        ncol: int,
        nrow: int,
        nlay: int,
        invalues: npt.ArrayLike,
    ) -> np.ma.MaskedArray:
        """
        Ensures that values is a 3D masked numpy (ncol, nrol, nlay).

        Args:
            ncol: Number of columns.
            nrow: Number of rows.
            nlay: Number of layers.
            invalues: Values to process.

        Returns:
            The values as a masked numpy array.

        """
        currentmask = (
            np.ma.getmaskarray(self._values)
            if self._values is not None and isinstance(self._values, np.ma.MaskedArray)
            else None
        )

        if isinstance(invalues, (int, float)):
            vals = np.ma.zeros((ncol, nrow, nlay), order="C", dtype=self.dtype)
            vals = np.ma.array(vals, mask=currentmask)
            values = vals + invalues
            invalues = values

        if not isinstance(invalues, np.ma.MaskedArray):
            values = np.ma.array(invalues, mask=currentmask, order="C")
        else:
            values = invalues  # new mask is possible

        if values.shape != (ncol, nrow, nlay):
            try:
                values = np.ma.reshape(values, (ncol, nrow, nlay), order="C")
            except ValueError as emsg:
                xtg.error(f"Cannot reshape array: {emsg}")
                raise

        # replace any undef or nan with mask
        values = np.ma.masked_greater(values, self.undef_limit)
        values = np.ma.masked_invalid(values)

        if not values.flags.c_contiguous:
            mask = np.ma.getmaskarray(values)
            mask = np.asanyarray(mask, order="C")
            values = np.asanyarray(values, order="C")
            values = np.ma.array(values, mask=mask, order="C")

        # the self._isdiscrete property shall win over numpy dtype
        if "int" in str(values.dtype) and not self._isdiscrete:
            values = values.astype(np.float64)

        if "float" in str(values.dtype) and self._isdiscrete:
            values = values.astype(np.int32)

        return values

    # ==================================================================================
    # Import and export
    # ==================================================================================

    @classmethod
    def _read_file(
        cls,
        filelike: FileLike,
        fformat: str | None = None,
        **kwargs: Any,
    ) -> GridProperty:
        pfile = FileWrapper(filelike)
        fmt = pfile.fileformat(fformat)
        kwargs = _data_reader_factory(fmt)(pfile, **kwargs)
        kwargs["filesrc"] = pfile.file
        return cls(**kwargs)

    def to_file(
        self,
        pfile: FileLike,
        fformat: Literal["roff", "roffasc", "grdecl", "bgrdecl", "xtgcpprop"] = "roff",
        name: str | None = None,
        append: bool = False,
        dtype: type[np.float32] | type[np.float64] | type[np.int32] | None = None,
        fmt: str | None = None,
    ) -> None:
        """
        Export the grid property to file.

        Args:
            pfile: File name or pathlib.Path to export to.
            fformat: The file format to be used. Default is
                roff binary, else roff_ascii/grdecl/bgrdecl.
            name: If provided, will explicitly give property name;
                else the existing name of the instance will used.
            append: Append to existing file, only for (b)grdecl formats.
            dtype: The values data type. This is valid only for grdecl or bgrdecl
                formats, where the default is None which means 'float32' for
                floating point numbers and 'int32' for discrete properties.
                Other choices are 'float64' which are 'DOUB' entries in
                Eclipse formats.
            fmt: Format for ascii grdecl format. Default is None. If specified,
                the user is responsible for a valid format specifier, e.g. "%8.4f".

        Example::

            # This example demonstrates that file formats can be mixed
            import xtgeo
            rgrid = xtgeo.grid_from_file("reek.roff")
            poro = GridProperty("reek_poro.grdecl", grid=rgrid, name='PORO')

            poro.values += 0.05

            poro.to_file("reek_export_poro.bgrdecl", format="bgrdecl")

        .. versionadded:: 2.13  Key `fmt` was added and default format for float output
            to grdecl is now "%e" if `fmt=None`

        """
        _gridprop_export.to_file(
            self,
            pfile,
            fformat=fformat,
            name=name,
            append=append,
            dtype=dtype,
            fmt=fmt,
        )

    @classmethod
    def _read_roxar(
        cls,
        projectname: str,
        gridname: str,
        propertyname: str,
        realisation: int = 0,
        faciescodes: bool = False,
    ) -> GridProperty:
        return cls(
            **_gridprop_roxapi.import_prop_roxapi(
                projectname, gridname, propertyname, realisation, faciescodes
            )
        )

    def to_roxar(
        self,
        projectname: str,
        gridname: str,
        propertyname: str,
        realisation: int = 0,
        casting: (
            Literal["no", "equiv", "safe", "same_kind", "unsafe"] | None
        ) = "unsafe",
    ) -> None:
        """
        Store a grid model property into a RMS project.

        Note:
            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Note:
            Beware values casting, see ``casting`` key.
            Default is "unsafe" which may create issues if your property has
            values that is outside the valid range. I.e. for float values XTGeo
            normally use `float64` (8 byte) while roxar use `float32` (4 byte).
            With extreme values, e.g. 10e40, such values will be truncated if
            "unsafe" casting. More common is casting issues with discrete as
            Roxar (RMS) often use `uint8` which only allow values in range 1..256.

        Args:
            projectname: Inside RMS use the magic 'project' string. Otherwise
                use a path to an RMS project, or a project reference.
            gridname: Name of grid model.
            propertyname: Name of grid property.
            realisation: Realisation number. Default is 0 (the first).
            casting: This refers to numpy `astype(... casting=...)` settings.

        .. versionchanged:: 2.10 Key `saveproject` has been removed and will
            have no effect
        .. versionadded:: 2.12 Key `casting` was added

        """
        _gridprop_roxapi.export_prop_roxapi(
            self,
            projectname,
            gridname,
            propertyname,
            realisation=realisation,
            casting=casting,
        )

    # ==================================================================================
    # Various public methods
    # ==================================================================================

    def describe(self, flush: bool = True) -> str:
        """
        Describe a GridProperty instance by printing its properties
        to stdout

        Args:
            flush: Print to stdout. True by default.

        Returns:
            A string description of the grid property instance.

        """
        from xtgeo.common import XTGDescription

        dsc = XTGDescription()
        dsc.title("Description of GridProperty instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("Name", self.name)
        dsc.txt("Date", self.date)
        dsc.txt("File source", self._filesrc)
        dsc.txt("Discrete status", self._isdiscrete)
        dsc.txt("Codes", self._codes)
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        np.set_printoptions(threshold=16)
        dsc.txt("Values", self._values.reshape(-1), self._values.dtype)
        np.set_printoptions(threshold=1000)
        dsc.txt(
            "Values, mean, stdev, minimum, maximum",
            self.values.mean(),
            self.values.std(),
            self.values.min(),
            self.values.max(),
        )
        itemsize = self.values.itemsize
        msize = float(self.values.size * itemsize) / (1024 * 1024 * 1024)
        dsc.txt("Roxar datatype", self.roxar_dtype)
        dsc.txt("Minimum memory usage of array (GB)", msize)

        if flush:
            dsc.flush()
            return ""

        return dsc.astext()

    def get_npvalues3d(self, fill_value: npt.ArrayLike | None = None) -> np.ndarray:
        """
        Get a pure numpy copy (not masked) of the values in 3D shape.

        Note that Numpy dtype will be reset; int32 if discrete or float64 if
        continuous. The reason for this is to avoid inconsistensies regarding
        UNDEF values.

        If fill_value is not None, than the returning dtype is always `np.float64`.

        Args:
            fill_value: Value of masked entries. Default is None which
                means the XTGeo UNDEF value (a high number). This UNDEF
                value is different for a continuous or discrete property.

        Returns:
            Non-masked array copy of 3D-shaped values

        """
        if fill_value is None:
            if self._isdiscrete:
                fvalue: npt.ArrayLike = UNDEF_INT
                dtype: type[np.int32] | type[np.float64] = np.int32
            else:
                fvalue = UNDEF
                dtype = np.float64
        else:
            fvalue = fill_value
            dtype = np.float64

        val = self.values.copy().astype(dtype)
        npv3d = np.ma.filled(val, fill_value=fvalue)
        del val

        return npv3d

    def get_actnum(
        self,
        name: str = "ACTNUM",
        asmasked: bool = False,
    ) -> GridProperty:
        """
        Return an ACTNUM GridProperty object.

        Note that this method is similar to, but not identical to,
        the job with same name in Grid(). Here, the maskedarray of the values
        is applied to deduce the ACTNUM array.

        Args:
            name: Name of property in the XTGeo GridProperty object.
                Default is "ACTNUM".
            asmasked: Default is False, so that actnum is returned with all cells
                shown. Use asmasked=True to make 0 entries masked.

        Returns:
            The ACTNUM GridProperty object.

        Example::

            act = mygrid.get_actnum()
            print('{}% cells are active'.format(act.values.mean() * 100))

        """
        act = GridProperty(
            ncol=self._ncol, nrow=self._nrow, nlay=self._nlay, name=name, discrete=True
        )

        orig = self.values
        vact = np.ma.ones(self.values.shape)
        vact[orig.mask] = 0

        if asmasked:
            vact = np.ma.masked_equal(vact, 0)

        act.values = vact.astype(np.int32)
        act.isdiscrete = True
        act.codes = {0: "0", 1: "1"}

        return act

    def get_active_npvalues1d(self) -> np.ma.MaskedArray:
        """
        Get the active cells as a 1D numpy masked array.

        Returns:
            The grid property as a 1D numpy masked array, active cells only.

        """
        return self.get_npvalues1d(activeonly=True)

    def get_npvalues1d(
        self,
        activeonly: bool = False,
        fill_value: npt.ArrayLike = np.nan,
        order: Literal["C", "F"] = "C",
    ) -> np.ma.MaskedArray:
        """
        Return the grid property as a 1D numpy array (copy) for active or all
        cells, but inactive have a fill value.

        Args:
            activeonly: If True, then only return active cells.
                Default is False.
            fill_value: Fill value for inactive cells. Default is `np.nan`.
            order: Array internal order. Default is "C", alternative is "F".

        Returns:
            The grid property as a 1D numpy masked array.

        .. versionadded:: 2.3
        .. versionchanged:: 2.8 Added `fill_value` and `order`

        """
        vact = self.values1d.copy()

        if order == "F":
            vact = _gridprop_lowlevel.c2f_order(self, vact)

        if activeonly:
            return vact.compressed()  # safer than vact[~vact.mask] if no masked

        return vact.filled(fill_value)

    def copy(self, newname: str | None = None) -> GridProperty:
        """
        Copy a GridProperty object to another instance.

        Args:
            newname: Give the copied instance a new name.

        Returns:
            A copy of the GridProperty instance.

        ::

            >>> import xtgeo
            >>> myporo = xtgeo.gridproperty_from_file(
            ...    reek_dir + '/reek_sim_poro.roff',
            ...    name="PORO"
            ... )
            >>> mycopy = myporo.copy(newname='XPROP')
            >>> print(mycopy.name)
            XPROP

        """
        if newname is None:
            newname = self.name
        assert newname is not None

        xprop = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=self._values.copy(),
            name=newname,
        )

        xprop.geometry = self._geometry
        xprop.isdiscrete = self._isdiscrete
        xprop.codes = self._codes
        xprop.date = self._date
        xprop.roxorigin = self._roxorigin
        xprop.roxar_dtype = self.roxar_dtype

        xprop.filesrc = self._filesrc

        return xprop

    def mask_undef(self) -> None:
        """Make UNDEF values masked."""
        if self._isdiscrete:
            self._values = np.ma.masked_greater(self._values, UNDEF_INT_LIMIT)
        else:
            self._values = np.ma.masked_greater(self._values, UNDEF_LIMIT)

    def crop(
        self, spec: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> None:
        """
        Crop a property between grid coordinates.

        Args:
            spec: Provide a tuple of i, j, k lower and upper bounds
                to crop between, e.g. ((1, 3), (2, 4), (1, 5)) would
                crop a grid property such that only values from 1:3 in
                the i plane, 2:4 in the j plane, and 1:5 in the k plane
                would be present.

        """
        (ic1, ic2), (jc1, jc2), (kc1, kc2) = spec

        # Compute size of new cropped grid
        self._ncol = ic2 - ic1 + 1
        self._nrow = jc2 - jc1 + 1
        self._nlay = kc2 - kc1 + 1

        newvalues = self.values.copy()

        self.values = newvalues[ic1 - 1 : ic2, jc1 - 1 : jc2, kc1 - 1 : kc2]

    def get_xy_value_lists(
        self, grid: Grid | None = None, activeonly: bool = True
    ) -> XYValueLists:
        """
        Get lists of xy coords and values for Webportal format.

        The coordinates are on the form (two cells)::

            [[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
            [(x5,y5), (x6,y6), (x7,y7), (x8,y8)]]]

        Args:
            grid: The XTGeo Grid object for the property.  Defaults to None.
            activeonly: If True (default), active cells only,
                otherwise cell geometries will be listed and property will
                have value -999 in undefined cells.

        Returns:
            A tuple of two lists, one being the xr coords, the other
            the values at those coords.


        Example::

            import xtgeo
            grid = xtgeo.grid_from_file("../xtgeo-testdata/3dgrids/bri/b_grid.roff")
            prop = xtgeogridproperty_from_file(
                "../xtgeo-testdata/3dgrids/bri/b_poro.roff", grid=grid, name="PORO"
            )

            clist, valuelist = prop.get_xy_value_lists(
                grid=grid, activeonly=False
            )

        """
        clist, vlist = _gridprop_op1.get_xy_value_lists(
            self, grid=grid, mask=activeonly
        )
        return clist, vlist

    def get_values_by_ijk(
        self, iarr: np.ndarray, jarr: np.ndarray, karr: np.ndarray, base: int = 1
    ) -> np.ma.MaskedArray | None:
        """
        Get a 1D ndarray of values by I J K arrays.

        This could for instance be a well path where I J K
        exists as well logs.

        Note that the input arrays have 1 as base as default

        Args:
            iarr: Numpy array of I
            jarr: Numpy array of J
            karr: Numpy array of K
            base: Should be 1 or 0, dependent on what
                number base the input arrays has.

        Returns:
            A 1D numpy array of property values,
            with NaN if undefined. Returns None
            on IndexErrors.

        """
        res = np.zeros(iarr.shape, dtype="float64")
        res = np.ma.masked_equal(res, 0)  # mask all

        # get indices where defined (note the , after valids)
        (valids,) = np.where(~np.isnan(iarr))

        iarr = iarr[~np.isnan(iarr)]
        jarr = jarr[~np.isnan(jarr)]
        karr = karr[~np.isnan(karr)]

        try:
            res[valids] = self.values[
                iarr.astype("int") - base,
                jarr.astype("int") - base,
                karr.astype("int") - base,
            ]
            return np.ma.filled(res, fill_value=np.nan)
        except IndexError as ier:
            xtg.warn(f"Error {ier}, return None")
            return None
        except:  # noqa
            xtg.warn("Unexpected error")
            raise

    def discrete_to_continuous(self) -> None:
        """Convert from discrete to continuous values."""
        if not self.isdiscrete:
            logger.debug("No need to convert, already continuous")
            return

        logger.debug("Converting to continuous ...")
        val = self._values.copy()
        val = val.astype("float64")
        self._values = val
        self._isdiscrete = False
        self._codes = {}
        self.roxar_dtype = np.float32

    def continuous_to_discrete(self) -> None:
        """Convert from continuous to discrete values."""
        if self.isdiscrete:
            logger.debug("No need to convert, already discrete")
            return

        logger.debug("Converting to discrete ...")
        val = self._values.copy()
        val = val.astype(np.int32)
        self._values = val
        self._isdiscrete = True

        # make the code list
        uniq = np.unique(val).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {k: str(v) for k, v in codes.items()}  # val as strings
        self._codes = codes
        self.roxar_dtype = np.uint16

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # ==================================================================================

    def operation_polygons(
        self,
        poly: Polygons,
        value: float | int,
        opname: Literal["add", "sub", "mul", "div", "set"] = "add",
        inside: bool = True,
    ) -> None:
        """
        A generic function for doing 3D grid property operations
        restricted to inside or outside polygon(s).

        This method requires that the property geometry is known
        (prop.geometry is set to a grid instance).

        Args:
            poly: A XTGeo Polygons instance.
            value: Value to add, subtract etc.
            opname: Name of operation... "add", "sub", etc.
                Defaults to "add".
            inside: If True do operation inside polygons; else outside.
                Defaults to True.

        """
        if self.geometry is None:
            msg = """
            You need to link the property to a grid geometry:"

                myprop.geometry = mygrid

            """
            xtg.warnuser(msg)
            raise ValueError("The geometry attribute is not set")

        _gridprop_op1.operation_polygons(
            self, poly, value, opname=opname, inside=inside
        )

    def add_inside(self, poly: Polygons, value: float | int) -> None:
        """Add a value (scalar) inside polygons."""
        self.operation_polygons(poly, value, opname="add", inside=True)

    def add_outside(self, poly: Polygons, value: float | int) -> None:
        """Add a value (scalar) outside polygons."""
        self.operation_polygons(poly, value, opname="add", inside=False)

    def sub_inside(self, poly: Polygons, value: float | int) -> None:
        """Subtract a value (scalar) inside polygons."""
        self.operation_polygons(poly, value, opname="sub", inside=True)

    def sub_outside(self, poly: Polygons, value: float | int) -> None:
        """Subtract a value (scalar) outside polygons."""
        self.operation_polygons(poly, value, opname="sub", inside=False)

    def mul_inside(self, poly: Polygons, value: float | int) -> None:
        """Multiply a value (scalar) inside polygons."""
        self.operation_polygons(poly, value, opname="mul", inside=True)

    def mul_outside(self, poly: Polygons, value: float | int) -> None:
        """Multiply a value (scalar) outside polygons."""
        self.operation_polygons(poly, value, opname="mul", inside=False)

    def div_inside(self, poly: Polygons, value: float | int) -> None:
        """Divide a value (scalar) inside polygons."""
        self.operation_polygons(poly, value, opname="div", inside=True)

    def div_outside(self, poly: Polygons, value: float | int) -> None:
        """Divide a value (scalar) outside polygons."""
        self.operation_polygons(poly, value, opname="div", inside=False)

    def set_inside(self, poly: Polygons, value: float | int) -> None:
        """Set a value (scalar) inside polygons."""
        self.operation_polygons(poly, value, opname="set", inside=True)

    def set_outside(self, poly: Polygons, value: float | int) -> None:
        """Set a value (scalar) outside polygons."""
        self.operation_polygons(poly, value, opname="set", inside=False)
