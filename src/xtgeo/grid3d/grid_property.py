# -*- coding: utf-8 -*-
"""Module for a 3D grid property."""


import copy
import numbers
import hashlib
import pathlib
from types import FunctionType
from typing import Optional, Union, Any
import numpy as np

import xtgeo

from ._grid3d import _Grid3D
from . import _gridprop_etc
from . import _gridprop_op1
from . import _gridprop_import
from . import _gridprop_roxapi
from . import _gridprop_export
from . import _gridprop_lowlevel

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

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

# pylint: disable=logging-format-interpolation, too-many-public-methods

# ======================================================================================
# Functions outside the class, for rapid access. Will be exposed as
# xxx = xtgeo.gridproperty_from_file. pylint: disable=fixme
# ======================================================================================


def gridproperty_from_file(
    pfile,
    fformat="guess",
    name="unknown",
    grid=None,
    gridlink=True,
    date=None,
    fracture=False,
):
    """Make a GridProperty instance directly from file import.

    For arguments, see :func:`GridProperty.from_file()`

    Args:
        pfile (str): Property file
        kwargs: See :func:`GridProperty.from_file()`.

    Example::

        import xtgeo
        myporo = xtgeo.gridproperty_from_file('myporofile.roff')
    """

    obj = GridProperty()
    obj.from_file(
        pfile,
        fformat=fformat,
        name=name,
        grid=grid,
        gridlink=gridlink,
        date=date,
        fracture=fracture,
    )

    return obj


def gridproperty_from_roxar(
    project, gname, pname, realisation=0, faciescodes=False
):  # pragma: no cover
    """Make a GridProperty instance directly inside RMS.

    For arguments, see :func:`GridProperty.from_roxar()`

    Example::

        import xtgeo
        myporo = xtgeo.gridproperty_from_roxar(project, 'Geogrid', 'Poro')

    """
    obj = GridProperty()
    obj.from_roxar(
        project, gname, pname, realisation=realisation, faciescodes=faciescodes
    )

    return obj


# ======================================================================================
# GridProperty class
# ======================================================================================


class GridProperty(_Grid3D):
    """Class for a single 3D grid property, e.g porosity or facies.

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

    .. versionchanged:: 2.6 Possible to make GridProperty instance directly from Grid()
    .. versionchanged:: 2.8 Possible to base it on existing GridProperty() instance

    """

    def __init__(
        self,
        pfile: Optional[Union[str, pathlib.Path, Any]] = None,
        fformat: Optional[str] = "guess",
        ncol: Optional[int] = 4,
        nrow: Optional[int] = 3,
        nlay: Optional[int] = 5,
        name: Optional[str] = "unknown",
        discrete: Optional[bool] = False,
        date: Optional[str] = None,
        grid: Optional[Any] = None,
        linkgeometry: Optional[bool] = True,
        fracture: Optional[bool] = False,
        codes: Optional[dict] = None,
        dualporo: Optional[bool] = False,
        dualperm: Optional[bool] = False,
        roxar_dtype: Optional[Any] = np.float32,
        values: Optional[Union[np.ndarray, float, int]] = None,
    ):
        """Instantating.

            Args:
                pfile: Input file-like or a Grid/GridProperty instance or leave blank.
                fformat: File format input, default is ``guess``.
                ncol: Number of columns (nx).
                nrow: Number of rows (ny).
                ncol: Number of layers (nz).
                name: Name of property.
                discrete: True or False.
                date: Date on YYYYMMDD form.
                grid: Attached grid object
                linkgeometry: If True, establish a link between GridProperty and Grid
                fracture: True if fracture option (relevant for flow simulator data)
                codes: Codes in case a discrete property e.g. {1: "Sand", 4: "Shale"}
                dualporo: True if dual porosity system.
                dualperm: True if dual porosity and dual permeability system.
                roxar_dtype: Spesify roxar datatype e.g. np.uint8
                values: Values to apply (will not be applied if a file-like is input)

        Returns:
            A GridProperty object instance.

        Raises:
            RuntimeError: if something goes wrong (e.g. file not found)

        Examples::

            from xtgeo.grid3d import GridProperty
            myprop = GridProperty()
            myprop.from_file('emerald.roff', name='PORO')

            # or

            values = np.ma.ones((12, 17, 10), dtype=np.float64),
            myprop = GridProperty(ncol=12, nrow=17, nlay=10,
                                  values=values, discrete=False,
                                  name='MyValue')

            # or

            myprop = GridProperty('emerald.roff', name='PORO')

            # or create properties from a Grid() instance

            mygrid = Grid("grid.roff")
            myprop1 = GridProperty(mygrid, name='PORO')
            myprop2 = GridProperty(mygrid, name='FACIES', discrete=True, values=1,
                                   linkgeometry=True)  # alternative 1
            myprop2.geometry = mygrid  # alternative 2 to link grid geometry to property

            # from Grid instance:
            grd = Grid("somefile_grid_file")
            myprop = GridProperty(grd, values=99, discrete=True)  # based on grd

            # or from existing GridProperty instance:
            myprop2 = GridProperty(myprop, values=99, discrete=False)  # based on myprop

        """

        super().__init__()

        self._ncol = ncol
        self._nrow = nrow
        self._nlay = nlay

        # instance attributes defaults:
        self._name = name
        self._date = date
        self._isdiscrete = discrete
        self._geometry = grid
        self._fracture = fracture
        self._codes = dict() if codes is None else codes

        # not primary input:
        self._dualporo = dualporo
        self._dualperm = dualperm

        self._filesrc = None
        self._actnum_indices = None
        self._roxorigin = False  # true if the object comes from the ROXAPI
        self._roxar_dtype = roxar_dtype
        self._values = values

        self._undef = xtgeo.UNDEF_INT if discrete else xtgeo.UNDEF

        if pfile is not None:
            # make instance through grid/gridprop instance or file import
            if isinstance(pfile, (xtgeo.grid3d.Grid, xtgeo.grid3d.GridProperty)):
                _gridprop_etc.gridproperty_fromgrid(
                    self,
                    pfile,
                    linkgeometry=linkgeometry,
                    values=values,
                )

            elif isinstance(pfile, (str, pathlib.Path)):
                self.from_file(
                    pfile,
                    fformat=fformat,
                    name=name,
                    grid=grid,
                    gridlink=linkgeometry,
                    date=date,
                    fracture=fracture,
                )

        else:
            # make values
            _gridprop_etc.gridvalues_fromspec(self, values)

        self._metadata = xtgeo.MetaDataCPProperty()

    def __del__(self):
        logger.debug("DELETING property instance %s", self.name)

    def __repr__(self):
        myrp = (
            "{0.__class__.__name__} (id={1}) ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, nlay={0._nlay!r}, "
            "filesrc={0._filesrc!r}".format(self, id(self))
        )
        return myrp

    def __str__(self):
        # user friendly print
        return self.describe(flush=False)

    # ==================================================================================
    # Properties
    # Some proprerties such as ncol, nrow, nlay are from the Super class
    # ==================================================================================

    @property
    def metadata(self):
        """Return metadata object instance of type MetaDataRegularSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. A bit dangerous so further
        # check must be done to validate. TODO.
        if not isinstance(obj, xtgeo.MetaDataCPProperty):
            raise ValueError("Input obj not an instance of MetaDataCPProperty")

        self._metadata = obj  # checking is currently missing! TODO

    @property
    def name(self):
        """Returns or rename the property name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def dimensions(self):
        """3-tuple: The grid dimensions as a tuple of 3 integers (read only)"""
        return (self._ncol, self._nrow, self._nlay)

    @property
    def nactive(self):
        """int: Returns the number of active cells (read only)."""
        return len(self.actnum_indices)

    @property
    def geometry(self):
        """Returns or set the linked geometry, i.e. the Grid instance)"""
        return self._geometry

    @geometry.setter
    def geometry(self, geom):

        if geom is None:
            self._geometry = None
        elif isinstance(geom, xtgeo.grid3d.Grid) and geom.dimensions == self.dimensions:
            self._geometry = geom
        else:
            raise ValueError("Could not set geometry; wrong type or size")

    @property
    def actnum_indices(self):
        """Returns the 1D ndarray which holds the indices for active cells
        given in 1D, C order (read only).

        """
        actnumv = self.get_actnum()
        actnumv = np.ravel(actnumv.values)
        self._actnum_indices = np.flatnonzero(actnumv)

        return self._actnum_indices

    @property
    def isdiscrete(self):
        """Return True if property is discrete.

        This can also be used to convert from continuous to discrete
        or from discrete to continuous::

            myprop.isdiscrete = False
        """

        return self._isdiscrete

    @isdiscrete.setter
    def isdiscrete(self, flag):

        if not isinstance(flag, bool):
            raise ValueError("Input to {__name__} must be a bool")

        if flag is self._isdiscrete:
            pass
        else:
            if flag is True and self._isdiscrete is False:
                self.continuous_to_discrete()
            else:
                self.discrete_to_continuous()

    @property
    def dtype(self):
        """Return or set the values numpy dtype.

        When setting, note that the the dtype must correspond to the
        `isdiscrete` property. Hence dtype cannot alter isdiscrete status

        Example::

            if myprop.isdiscrete:
                myprop.dtype = np.uint16


        """
        return self._values.dtype

    @dtype.setter
    def dtype(self, dtype):
        allowedfloat = [np.float16, np.float32, np.float64]
        allowedint = [np.uint8, np.uint16, np.int16, np.int32, np.int64]

        okv = True
        msg = ""
        if self.isdiscrete:
            if dtype in allowedint:
                self.values = self.values.astype(dtype)
            else:
                okv = False
                msg = "{}: Wrong input for dtype. Use one of {}!".format(
                    __name__, allowedint
                )
        else:
            if dtype in allowedfloat:
                self.values = self.values.astype(dtype)
            else:
                okv = False
                msg = "{}: Wrong input for dtype. Use one of {}!".format(
                    __name__, allowedfloat
                )

        if not okv:
            raise ValueError(msg)

    @property
    def filesrc(self):
        """Return or set file src (if any)"""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, src):
        self._filesrc = src

    @property
    def roxar_dtype(self):
        """Return or set the roxar dtype (if any)"""
        return self._roxar_dtype

    @roxar_dtype.setter
    def roxar_dtype(self, dtype):
        allowed = [np.uint16, np.uint8, np.float32]
        if dtype in allowed:
            self._roxar_dtype = dtype
        else:
            raise ValueError(
                "{}: Wrong input for roxar_dtype. Use one of {}!".format(
                    __name__, allowed
                )
            )

    @property
    def date(self):
        """Returns or rename the property date on YYYYMMDD numerical format."""
        return self._date

    @date.setter
    def date(self, date):
        self._date = date

    @property
    def codes(self):
        """The property codes as a dictionary."""
        return self._codes

    @codes.setter
    def codes(self, cdict):
        self._codes = cdict.copy()

    @property
    def ncodes(self):
        """Number of codes if discrete grid property (read only)."""
        return len(self._codes)

    @property
    def values(self):
        """ Return or set the grid property as a masked 3D numpy array"""
        return self._values

    @values.setter
    def values(self, values):

        values = self.ensure_correct_values(self.ncol, self.nrow, self.nlay, values)

        self._values = values

    @property
    def ntotal(self):
        """Returns total number of cells ncol*nrow*nlay (read only)"""
        return self._ncol * self._nrow * self._nlay

    @property
    def roxorigin(self):
        """Returns True if the property comes from ROXAPI"""
        return self._roxorigin

    @roxorigin.setter
    def roxorigin(self, val):
        if isinstance(val, bool):
            self._roxorigin = val
        else:
            raise ValueError("Input to roxorigin must be True or False")

    @property
    def values3d(self):
        """For backward compatibility (use values instead)"""
        return self._values

    @values3d.setter
    def values3d(self, values):
        # kept for backwards compatibility
        self.values = values

    @property
    def values1d(self):
        """Returns a 1D view of values (masked numpy) (read only)."""
        return self._values.reshape(-1)

    @property
    def undef(self):
        """Get the actual undef value for floats or ints
        numpy arrays (read only).
        """
        if self._isdiscrete:
            return xtgeo.UNDEF_INT

        return xtgeo.UNDEF

    @property
    def undef_limit(self):
        """Returns the undef limit number, which is slightly less than the
        undef value.

        Hence for numerical precision, one can force undef values
        to a given number, e.g.::

           x[x<x.undef_limit]=999

        Undef limit values cannot be changed (read only).

        """
        if self._isdiscrete:
            return xtgeo.UNDEF_INT_LIMIT

        return xtgeo.UNDEF_LIMIT

    # ==================================================================================
    # Class and special methods
    # ==================================================================================

    def generate_hash(self):
        """str: Return a unique hash ID for current grid; can e.g. be used to compare
        two gridproperty instances with same source.

        .. versionadded:: 2.10
        """

        mhash = hashlib.sha256()
        gid = "{}{}{}{}{}{}{}".format(
            self._filesrc,
            self._ncol,
            self._nrow,
            self._nlay,
            self._values.mean(),
            self._values.min(),
            self._values.max(),
        )

        mhash.update(gid.encode())
        return mhash.hexdigest()

    @classmethod
    def methods(cls):
        """Returns the names of the methods in the class.

        >>> print(RegularSurface.methods())
        """
        mets = [x for x, y in cls.__dict__.items() if isinstance(y, FunctionType)]

        txt = "\nMETHODS for GridProperty():\n======================\n"
        for met in mets:
            txt += str(met) + "\n"

        return txt

    def ensure_correct_values(self, ncol, nrow, nlay, invalues):
        """Ensures that values is a 3D masked numpy (ncol, nrol, nlay).

        Args:
            ncol (int): Number of columns.
            nrow (int): Number of rows.
            nlay (int): Number of layers.
            invalues (array or scalar): Values to process.

        Return:
            values (MaskedArray): Numpy masked array on correct format.

        """

        currentmask = None
        if self._values is not None:
            if isinstance(self._values, np.ma.MaskedArray):
                currentmask = np.ma.getmaskarray(self._values)

        if isinstance(invalues, numbers.Number):
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
                xtg.error("Cannot reshape array: {}".format(emsg))
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

    def from_file(
        self,
        pfile,
        fformat=None,
        name="unknown",
        grid=None,
        gridlink=True,
        date=None,
        fracture=False,
        _roffapiv=1,
        ijrange=None,
        zerobased=False,
    ):  # _roffapiv for devel.
        """
        Import grid property from file, and makes an instance of this class.

        Note that the the property may be linked to its geometrical grid,
        through the ``grid=`` option. Sometimes this is required, for instance
        for most Eclipse input.

        Args:
            pfile (str): name of file to be imported
            fformat (str): file format to be used roff/init/unrst/grdecl
                (None is default, which means "guess" from file extension).
            name (str): name of property to import
            date (int or str): For restart files, date on YYYYMMDD format. Also
                the YYYY-MM-DD form is allowed (string), and for Eclipse,
                mnemonics like 'first', 'last' is also allowed.
            grid (Grid object): Grid Object for checks (optional for ROFF,
                required for Eclipse).
            gridlink (bool): If True, and grid is not None, a link from the grid
                instance to the property is made. If False, no such link is made.
                Avoiding gridlink is recommended when running statistics of multiple
                realisations of a property.
            fracture (bool): Only applicable for DUAL POROSITY systems, if True
                then the fracture property is read; if False then the matrix
                property is read. Names will be appended with "M" or "F"
            ijrange (list-like): A list of 4 number: (i1, i2, j1, j2) for subrange
                of cells to read. Only applicable for xtgcpprop format.
            zerobased (bool): Input if cells counts are zero- or one-based in
                ijrange. Only applicable for xtgcpprop format.

        Examples::

           x = GridProperty()
           x.from_file('somefile.roff', fformat='roff')
           #
           mygrid = Grid('ECL.EGRID')
           pressure_1 = GridProperty()
           pressure_1.from_file('ECL.UNRST', name='PRESSURE', date='first',
                                grid=mygrid)

        Returns:
           True if success, otherwise False

        .. versionchanged:: 2.8 Added gridlink option, default is True
        """
        pfile = xtgeo._XTGeoFile(pfile, mode="rb")

        obj = _gridprop_import.from_file(
            self,
            pfile,
            fformat=fformat,
            name=name,
            grid=grid,
            date=date,
            fracture=fracture,
            _roffapiv=_roffapiv,
            ijrange=ijrange,
            zerobased=zerobased,
        )

        if grid and gridlink:
            grid.append_prop(self)

        return obj

    def to_file(
        self, pfile, fformat="roff", name=None, append=False, dtype=None, fmt=None
    ):
        """Export the grid property to file.

        Args:
            pfile (str or Path): File name or pathlib.Path to export to
            fformat (str): The file format to be used. Default is
                roff binary , else roff_ascii/grdecl/bgrdecl
            name (str): If provided, will explicitly give property name;
                else the existing name of the instance will used.
            append (bool): Append to existing file, only for (b)grdecl formats.
            dtype (str): Data type; this is valid only for grdecl or bgrdecl
                formats, where default is None which means 'float32' for
                floating point number and 'int32' for discrete properties.
                Other choices are 'float64' which are 'DOUB' entries in
                Eclipse formats.
            fmt (str): Format for ascii grdecl format, default is None. If spesified,
                the user is responsible for a valid format specifier, e.g. "%8.4f"

        Example::

            # This example demonstrates that file formats can be mixed
            rgrid = Grid('reek.roff')
            poro = GridProperty('reek_poro.grdecl', grid=rgrid, name='PORO')

            poro.values += 0.05

            poro.to_file('reek_export_poro.bgrdecl', format='bgrdecl')

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

    def from_roxar(
        self, projectname, gname, pname, realisation=0, faciescodes=False
    ):  # pragma: no cover
        """Import grid model property from RMS project, and makes an instance.

        Arguments:
            projectname (str): Name of RMS project; use pure 'project'
                if inside RMS
            gfile (str): Name of grid model
            pfile (str): Name of grid property
            realisation (int): Realisation number (default 0; first)
            faciescodes (bool): If a Roxar property is of the special ``body_facies``
                type (e.g. result from a channel facies object modelling), the
                default is to get the body code values. If faciescodes is True,
                the facies code values will be read instead. For other roxar properties
                this key is not relevant.

        .. versionadded:: 2.12  Key `faciescodes` was added

        """

        self._filesrc = None

        _gridprop_roxapi.import_prop_roxapi(
            self, projectname, gname, pname, realisation, faciescodes
        )

    def to_roxar(
        self, project, gname, pname, realisation=0, casting="unsafe"
    ):  # pragma: no cover
        """Store a grid model property into a RMS project.

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
            project (str or roxar._project): Inside RMS use the magic 'project',
                else use path to RMS project, or a project reference
            gfile (str): Name of grid model
            pfile (str): Name of grid property
            projectname (str): Name of RMS project (None if inside a project)
            realisation (int): Realisation number (default 0 first)
            casting (str): This refers to numpy `astype(... casting=...)` settings.


        .. versionchanged: 2.10
            Key `saveproject` has been removed and will have no effect

        .. versionadded:: 2.12 Key `casting` was added

        """
        _gridprop_roxapi.export_prop_roxapi(
            self, project, gname, pname, realisation=realisation, casting=casting
        )

    # ==================================================================================
    # Various public methods
    # ==================================================================================

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = xtgeo.common.XTGDescription()
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
            return None

        return dsc.astext()

    def get_npvalues3d(self, fill_value=None):
        """Get a pure numpy copy (not masked) copy of the values, 3D shape.

        Note that Numpy dtype will be reset; int32 if discrete or float64 if
        continuous. The reason for this is to avoid inconsistensies regarding
        UNDEF values.

        If fill_value is not None, than the returning dtype is always `np.float64`.

        Args:
            fill_value: Value of masked entries. Default is None which
                means the XTGeo UNDEF value (a high number), different
                for a continuous or discrete property
        """
        # this is a function, not a property by design

        if fill_value is None:
            if self._isdiscrete:
                fvalue = xtgeo.UNDEF_INT
                dtype = np.int32
            else:
                fvalue = xtgeo.UNDEF
                dtype = np.float64
        else:
            fvalue = fill_value
            dtype = np.float64

        val = self.values.copy().astype(dtype)
        npv3d = np.ma.filled(val, fill_value=fvalue)
        del val

        return npv3d

    def get_actnum(self, name="ACTNUM", asmasked=False, mask=None):
        """Return an ACTNUM GridProperty object.

        Note that this method is similar to, but not identical to,
        the job with same name in Grid(). Here, the maskedarray of the values
        is applied to deduce the ACTNUM array.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): Actnum is returned with all cells shown
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead!

        Example::

            act = mygrid.get_actnum()
            print('{}% cells are active'.format(act.values.mean() * 100))
        """

        if mask is not None:
            asmasked = super()._evaluate_mask(mask)

        act = GridProperty(
            ncol=self._ncol, nrow=self._nrow, nlay=self._nlay, name=name, discrete=True
        )

        orig = self.values
        vact = np.ones(self.values.shape)
        vact[orig.mask] = 0

        if asmasked:
            vact = np.ma.masked_equal(vact, 0)

        act.values = vact.astype(np.int32)
        act.isdiscrete = True
        act.codes = {0: "0", 1: "1"}

        # return the object
        return act

    def get_active_npvalues1d(self):
        """Return the grid property as a 1D numpy array (copy), active
        cells only.
        """

        return self.get_npvalues1d(activeonly=True)

    def get_npvalues1d(self, activeonly=False, fill_value=np.nan, order="C"):
        """Return the grid property as a 1D numpy array (copy) for active or all
        cells, but inactive have a fill value.

        Args:
            activeonly (bool): If True, then only return active cells
            fill_value (float): Fill value for inactive cells
            order (str): Array internal order; default is "C", alternative is "F"

        .. versionadded:: 2.3
        .. versionchanged:: 2.8 Added `fill_value` and `order`
        """
        vact = self.values1d.copy()

        if order == "F":
            vact = _gridprop_lowlevel.c2f_order(self, vact)

        if activeonly:
            return vact.compressed()  # safer than vact[~vact.mask] if no masked

        return vact.filled(fill_value)

    def copy(self, newname=None):
        """Copy a xtgeo.grid3d.GridProperty() object to another instance.

        ::

            >>> mycopy = xx.copy(newname='XPROP')
        """

        if newname is None:
            newname = self.name

        xprop = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=self._values.copy(),
            name=newname,
        )

        xprop.geometry = self._geometry
        xprop.codes = copy.deepcopy(self._codes)
        xprop.isdiscrete = self._isdiscrete
        xprop.date = self._date
        xprop.roxorigin = self._roxorigin
        xprop.roxar_dtype = self._roxar_dtype

        xprop.filesrc = self._filesrc

        return xprop

    def mask_undef(self):
        """Make UNDEF values masked."""
        if self._isdiscrete:
            self._values = np.ma.masked_greater(self._values, xtgeo.UNDEF_INT_LIMIT)
        else:
            self._values = np.ma.masked_greater(self._values, xtgeo.UNDEF_LIMIT)

    def crop(self, spec):
        """Crop a property, see method under grid"""

        (ic1, ic2), (jc1, jc2), (kc1, kc2) = spec

        # compute size of new cropped grid
        self._ncol = ic2 - ic1 + 1
        self._nrow = jc2 - jc1 + 1
        self._nlay = kc2 - kc1 + 1

        newvalues = self.values.copy()

        self.values = newvalues[ic1 - 1 : ic2, jc1 - 1 : jc2, kc1 - 1 : kc2]

    def get_xy_value_lists(self, grid=None, activeonly=True):
        """Get lists of xy coords and values for Webportal format.

        The coordinates are on the form (two cells)::

            [[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
            [(x5,y5), (x6,y6), (x7,y7), (x8,y8)]]]

        Args:
            grid (object): The XTGeo Grid object for the property
            activeonly (bool): If true (default), active cells only,
                otherwise cell geometries will be listed and property will
                have value -999 in undefined cells.

        Example::

            grid = Grid()
            grid.from_file('../xtgeo-testdata/3dgrids/bri/b_grid.roff')
            prop = GridProperty()
            prop.from_file('../xtgeo-testdata/3dgrids/bri/b_poro.roff',
                           grid=grid, name='PORO')

            clist, valuelist = prop.get_xy_value_lists(grid=grid,
                                                       activeonly=False)


        """

        clist, vlist = _gridprop_op1.get_xy_value_lists(
            self, grid=grid, mask=activeonly
        )
        return clist, vlist

    def get_values_by_ijk(self, iarr, jarr, karr, base=1):
        """Get a 1D ndarray of values by I J K arrays.

        This could for instance be a well path where I J K
        exists as well logs.

        Note that the input arrays have 1 as base as default

        Args:
            iarr (ndarray): Numpy array of I
            jarr (ndarray): Numpy array of J
            karr (ndarray): Numpy array of K
            base (int): Should be 1 or 0, dependent on what
                number base the input arrays has.

        Returns:
            pvalues (ndarray): A 1D numpy array of property values,
                with NaN if undefined

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
            xtg.warn("Error {}, return None".format(ier))
            return None
        except:  # noqa
            xtg.warn("Unexpected error")
            raise

    def discrete_to_continuous(self):
        """Convert from discrete to continuous values"""

        if self.isdiscrete:
            logger.info("Converting to continuous ...")
            val = self._values.copy()
            val = val.astype("float64")
            self._values = val
            self._isdiscrete = False
            self._codes = {}
            self._roxar_dtype = np.float32
        else:
            logger.info("No need to convert, already continuous")

    def continuous_to_discrete(self):
        """Convert from continuous to discrete values"""

        if not self.isdiscrete:
            logger.info("Converting to discrete ...")
            val = self._values.copy()
            val = val.astype(np.int32)
            self._values = val
            self._isdiscrete = True

            # make the code list
            uniq = np.unique(val).tolist()
            codes = dict(zip(uniq, uniq))
            codes = {k: str(v) for k, v in codes.items()}  # val as strings
            self._codes = codes
            self._roxar_dtype = np.uint16
        else:
            logger.info("No need to convert, already discrete")

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # ==================================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True):
        """A generic function for doing 3D grid property operations
        restricted to inside or outside polygon(s).

        This method requires that the property geometry is known
        (prop.geometry is set to a grid instance)

        Args:
            poly (Polygons): A XTGeo Polygons instance
            value (float): Value to add, subtract etc
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside polygons; else outside.
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

    # shortforms
    def add_inside(self, poly, value):
        """Add a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=True)

    def add_outside(self, poly, value):
        """Add a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=False)

    def sub_inside(self, poly, value):
        """Subtract a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=True)

    def sub_outside(self, poly, value):
        """Subtract a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=False)

    def mul_inside(self, poly, value):
        """Multiply a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=True)

    def mul_outside(self, poly, value):
        """Multiply a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=False)

    def div_inside(self, poly, value):
        """Divide a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="div", inside=True)

    def div_outside(self, poly, value):
        """Divide a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="div", inside=False)

    def set_inside(self, poly, value):
        """Set a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=True)

    def set_outside(self, poly, value):
        """Set a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=False)
