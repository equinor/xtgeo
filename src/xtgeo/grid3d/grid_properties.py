"""Module for Grid Properties."""

from __future__ import annotations

import hashlib
import warnings
from typing import TYPE_CHECKING, List, Literal, Tuple, Union

import numpy as np
import pandas as pd

from xtgeo.common import XTGDescription, XTGeoDialog, null_logger
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.io._file import FileFormat, FileWrapper

from . import _grid3d_utils as utils, _grid_etc1
from ._grid3d import _Grid3D
from ._gridprops_import_eclrun import (
    import_ecl_init_gridproperties,
    import_ecl_restart_gridproperties,
    read_eclrun_properties,
)
from ._gridprops_import_roff import import_roff_gridproperties, read_roff_properties

xtg = XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from xtgeo import Grid
    from xtgeo.common.types import FileLike

    from .grid_property import GridProperty

KeywordTuple = Tuple[str, str, int, int]
KeywordDateTuple = Tuple[str, str, int, int, Union[str, int]]
GridPropertiesKeywords = Union[
    List[Union[KeywordTuple, KeywordDateTuple]], pd.DataFrame
]


def list_gridproperties(
    property_file: FileLike,
    fformat: Literal["roffasc", "roff", "finit", "init", "funrst", "unrst"] = None,
) -> list[str]:
    """List the properties in a ROFF or Eclipse INIT/UNRST file.

    Args:
        property_file: The `FileLike` containing the file to inspect.
        fformat: The optional format of the provided file. If not provided will
            attempt to detect it. None by default.

    Returns:
        A list of property names within the file.

    Raises:
        ValueError: Unknown or invalid file format.

    Example::
        >>> static_props = xtgeo.list_gridproperties(reek_dir + "/REEK.INIT")
        >>> roff_props = xtgeo.list_gridproperties(
        ...     reek_dir + "/reek_grd_w_props.roff",
        ...     fformat="roff",
        ... )
    """
    pfile = FileWrapper(property_file, mode="rb")
    pfile.check_file(raiseerror=ValueError)

    fmt = pfile.fileformat(fformat)
    if fmt in (FileFormat.ROFF_ASCII, FileFormat.ROFF_BINARY):
        return list(read_roff_properties(pfile))
    if fmt in (
        FileFormat.FINIT,
        FileFormat.INIT,
        FileFormat.FUNRST,
        FileFormat.UNRST,
    ):
        return list(read_eclrun_properties(pfile))

    extensions = FileFormat.extensions_string(
        [
            FileFormat.ROFF_BINARY,
            FileFormat.ROFF_ASCII,
            FileFormat.INIT,
            FileFormat.FINIT,
            FileFormat.UNRST,
            FileFormat.FUNRST,
        ]
    )
    raise InvalidFileFormatError(
        f"File format {fformat} is invalid for type GridProperties. "
        f"Supported formats are {extensions}."
    )


def gridproperties_from_file(
    property_file: FileLike,
    fformat: str | None = None,
    names: list[str] | None = None,
    dates: list[str] | None = None,
    grid: Grid | None = None,
    namestyle: int = 0,
    strict: tuple[bool, bool] = (True, False),
) -> GridProperties:
    """Import grid properties from file.

    In case of names='all' then all vectors which have a valid length
    (number of total or active cells in the grid) will be read

    Args:
        property_file (str or Path): Name of file with properties
        fformat (str): roff/init/unrst
        names: list of property names, e.g. ['PORO', 'PERMX'] or 'all'
        dates: list of dates on YYYYMMDD format, for restart files, or 'all'
        grid (obj): The grid geometry object (optional if ROFF)
        namestyle (int): 0 (default) for style SWAT_20110223,
            1 for SWAT--2011_02_23 (applies to restart only)
        strict (tuple of (bool, bool)): If (True, False) (default) then an
            Error is raised if keyword name is not found, or a key-date combination
            is not found. However, the dates will processed so that non-valid dates
            are skipped (still, invalid key-date combinations may occur!).
            If (True, True) all keywords and dates are tried, while (False, False)
            means that that only valid entries are imported, more or less silently.
            Saturations keywords SWAT/SOIL/SGAS are not evaluated as they may be
            derived.
    Example::
        >>> grd = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID", fformat='egrid')
        >>> gps = xtgeo.gridproperties_from_file(
        ...     reek_dir + "/REEK.INIT",
        ...     fformat='init',
        ...     names=["PORO", "PERMX"],
        ...     grid=grd,
        ... )
    """
    pfile = FileWrapper(property_file, mode="rb")
    pfile.check_file(raiseerror=ValueError)

    fmt = pfile.fileformat(fformat)
    if fmt in (FileFormat.ROFF_ASCII, FileFormat.ROFF_BINARY):
        return GridProperties(
            props=import_roff_gridproperties(pfile, names, strict=strict)
        )
    if fmt in (FileFormat.FINIT, FileFormat.INIT):
        return GridProperties(
            props=import_ecl_init_gridproperties(
                pfile,
                grid=grid,
                names=names,
                strict=strict[0],
                maxkeys=MAXKEYWORDS,
            )
        )
    if fmt in (FileFormat.FUNRST, FileFormat.UNRST):
        return GridProperties(
            props=import_ecl_restart_gridproperties(
                pfile,
                dates=dates,
                grid=grid,
                names=names,
                namestyle=namestyle,
                strict=strict,
                maxkeys=MAXKEYWORDS,
            )
        )

    extensions = FileFormat.extensions_string(
        [
            FileFormat.ROFF_BINARY,
            FileFormat.ROFF_ASCII,
            FileFormat.INIT,
            FileFormat.FINIT,
            FileFormat.UNRST,
            FileFormat.FUNRST,
        ]
    )
    raise InvalidFileFormatError(
        f"File format {fformat} is invalid for type GridProperties. "
        f"Supported formats are {extensions}."
    )


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


def gridproperties_dataframe(
    gridproperties: Iterable[GridProperties],
    grid: Grid | None = None,
    activeonly: bool = True,
    ijk: bool = False,
    xyz: bool = False,
    doubleformat: bool = False,
) -> pd.DataFrame:
    """Returns a Pandas dataframe table for the properties.

    Similar to :meth:`GridProperties.get_dataframe()` but takes any list of
    grid properties as its first argument.

    Args:
        gridproperties: List (also GridProperties or iterable) of GridProperty
            to create dataframe of.
        activeonly (bool): If True, return only active cells, NB!
            If True, will require a grid instance (see grid key)
        ijk (bool): If True, show cell indices, IX JY KZ columns
        xyz (bool): If True, show cell center coordinates (needs grid).
        doubleformat (bool): If True, floats are 64 bit, otherwise 32 bit.
            Note that coordinates (if xyz=True) is always 64 bit floats.
        grid (Grid): The grid geometry object. This is required for the
            xyz option.
    Returns:
        Pandas dataframe object
    Examples::
        >>> grd = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID", fformat='egrid')
        >>> names = ['SOIL', 'SWAT', 'PRESSURE']
        >>> dates = [19991201]
        >>> gps = xtgeo.gridproperties_from_file(
        ...     reek_dir + "/REEK.UNRST",
        ...     fformat='unrst',
        ...     names=names,
        ...     dates=dates,
        ...     grid=grd
        ... )
        >>> df = xtgeo.gridproperties_dataframe(gps, grid=grd)
    """

    proplist = list(gridproperties)

    dataframe_dict = {}
    if ijk:
        if activeonly:
            if grid:
                ix, jy, kz = _grid_etc1.get_ijk(grid)
                dataframe_dict["IX"] = ix.get_active_npvalues1d()
                dataframe_dict["JY"] = jy.get_active_npvalues1d()
                dataframe_dict["KZ"] = kz.get_active_npvalues1d()
            elif proplist:
                ix, jy, kz = _grid_etc1.get_ijk(proplist[0])
                dataframe_dict["IX"] = ix.get_active_npvalues1d()
                dataframe_dict["JY"] = jy.get_active_npvalues1d()
                dataframe_dict["KZ"] = kz.get_active_npvalues1d()
        else:
            if not grid:
                raise ValueError(
                    "You ask for active_only but no Grid is present. Use grid=..."
                )
            act = grid.get_actnum(dual=True)
            ix, jy, kz = grid.get_ijk(asmasked=False)
            dataframe_dict["ACTNUM"] = act.values1d
            dataframe_dict["IX"] = ix.values1d
            dataframe_dict["JY"] = jy.values1d
            dataframe_dict["KZ"] = kz.values1d

    if xyz:
        if not grid:
            raise ValueError("You ask for xyz but no Grid is present. Use grid=...")

        xc, yc, zc = grid.get_xyz(asmasked=activeonly)
        if activeonly:
            dataframe_dict["X_UTME"] = xc.get_active_npvalues1d()
            dataframe_dict["Y_UTMN"] = yc.get_active_npvalues1d()
            dataframe_dict["Z_TVDSS"] = zc.get_active_npvalues1d()
        else:
            dataframe_dict["X_UTME"] = xc.values1d
            dataframe_dict["Y_UTMN"] = yc.values1d
            dataframe_dict["Z_TVDSS"] = zc.values1d

    for prop in gridproperties:
        if activeonly:
            vector = prop.get_active_npvalues1d()
        else:
            vector = prop.values1d.copy()
            # mask values not supported in Pandas:
            if prop.isdiscrete:
                vector = vector.filled(fill_value=0)
            else:
                vector = vector.filled(fill_value=np.nan)

        if doubleformat:
            vector = vector.astype(np.float64)
        else:
            vector = vector.astype(np.float32)

        dataframe_dict[prop.name] = vector

    return pd.DataFrame.from_dict(dataframe_dict)


class GridProperties(_Grid3D):
    """Class for a collection of 3D grid props, belonging to the same grid topology.

     It is a thin wrapper on a list that 1) checks that the GridProperties
     belong to the same Grid (loosely). 2) Contains operations that can be
     called on lists of GridProperty objects for easy discoverability.

     Examples::
         >>> import xtgeo
         >>> # Create an
         >>> grid_properties = xtgeo.GridProperties(props=[])
         >>> # Get the dataframe via the gridproperties object
         >>> grid_properties.get_dataframe()
         Empty DataFrame...
         >>> # Convert the gridproperties to a list
         >>> grid_properties_list = list(grid_properties)
         >>> # Get the dataframe of the list:
         >>> gridproperties_dataframe(grid_properties_list)
         Empty DataFrame...

    Args:
        props: The list of GridProperty objects.

    See Also:
        The :class:`GridProperty` class.
    """

    def __init__(
        self,
        props: list[GridProperty] | None = None,
    ):
        ncol = 4
        nrow = 3
        nlay = 5

        if props:
            ncol, nrow, nlay = props[0].dimensions

        super().__init__(ncol, nrow, nlay)

        # This triggers the setter for 'props', ensuring proper
        # setup of related attributes like '_ncol', '_nrow',
        # and '_nlay', and performs a consistency check.
        self.props = props or []

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__} (id={id(self)}) ncol={self._ncol!r}, "
            f"nrow={self._nrow!r}, nlay={self._nlay!r}, filesrc={self.names!r}"
        )

    def __str__(self) -> str:
        """str: User friendly print."""
        return self.describe(flush=False) or ""

    def __contains__(self, name: str) -> bool:
        """bool: Emulate 'if "PORO" in props'."""
        return bool(self.get_prop_by_name(name, raiseserror=False))

    def __getitem__(self, name: str) -> GridProperty:  # noqa: D105
        prop = self.get_prop_by_name(name, raiseserror=False)
        if prop is None:
            raise KeyError(f"Key {name} does not exist")

        return prop

    def __iter__(self) -> Iterator[GridProperty]:  # noqa: D105
        return iter(self._props)

    @property
    def names(self) -> list[str]:
        """Returns a list of property names.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> props = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.INIT",
            ...     fformat="init",
            ...     names=["PERMX"],
            ...     grid=grid,
            ... )

            >>> namelist = props.names
            >>> for name in namelist:
            ...     print ('Property name is {}'.format(name))
            Property name is PERMX

        """
        return [prop.name for prop in self._props]

    @property
    def props(self) -> list[GridProperty] | None:
        """Returns a list of XTGeo GridProperty objects, None if empty.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> myprops = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.INIT",
            ...     fformat="init",
            ...     names=["PERMX"],
            ...     grid=grid,
            ... )

            >>> proplist = myprops.props
            >>> for prop in proplist:
            ...     print ('Property object name is {}'.format(prop.name))
            Property object name is PERMX

            >>> # adding a property, e.g. get ACTNUM as a property from the grid
            >>> actn = grid.get_actnum()  # this will get actn as a GridProperty
            >>> myprops.append_props([actn])
        """
        if not self._props:
            return None

        return self._props

    @props.setter
    def props(self, propslist: list[GridProperty]) -> None:
        self._props = propslist
        if propslist:
            self._ncol = propslist[0].ncol
            self._nrow = propslist[0].nrow
            self._nlay = propslist[0].nlay
        self._consistency_check()

    @property
    def dates(self) -> list[str | None] | None:
        """Returns a list of valid (found) dates after import.

        Returns None if no dates present

        Note:
            See also :meth:`GridProperties.scan_dates` for scanning available dates
            in advance

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> props = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.INIT",
            ...     fformat="init",
            ...     names=["PERMX"],
            ...     grid=grid,
            ... )

            >>> datelist = props.dates
            >>> for date in datelist:
            ...     print ('Date applied is {}'.format(date))
            Date applied is 19991201

        .. versionchanged:: 2.16 dates is no longer an alias (undocumented behavior),
            and simply return the dates of the underlying list of GridProperty.
        """
        if not self.props:
            return None

        return [p.date for p in self.props]

    # Copy, and etc aka setters and getters

    def copy(self) -> GridProperties:
        """Copy a GridProperties instance to a new unique instance.

        Note that the GridProperty instances will also be unique.
        """

        return GridProperties(
            props=[p.copy() for p in self.props] if self.props else []
        )

    def describe(self, flush: bool = True) -> str | None:
        """Describe an instance by printing to stdout."""
        dsc = XTGDescription()

        dsc.title("Description of GridProperties instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        dsc.txt("Attached grid props objects (names)", self.names)

        if flush:
            dsc.flush()
            return None
        return dsc.astext()

    def generate_hash(self) -> str:
        """str: Return a unique hash ID for current gridproperties instance.

        .. versionadded:: 2.10
        """
        mhash = hashlib.sha256()

        hashinput = ""
        for prop in self._props:
            gid = (
                f"{prop.ncol}{prop.nrow}{prop.nlay}{prop.values.mean()}"
                f"{prop.values.min()}{prop.values.max()}"
            )
            hashinput += gid

        mhash.update(hashinput.encode())
        return mhash.hexdigest()

    def get_prop_by_name(
        self, name: str, raiseserror: bool = True
    ) -> GridProperty | None:
        """Find and return a property object (GridProperty) by name.

        Args:
            name (str): Name of property to look for
            raiseserror (bool): If True, raises a ValueError if not found, otherwise
                return None

        """
        for prop in self._props:
            logger.debug("Look for %s, actual is %s", name, prop.name)
            if prop.name == name:
                logger.debug(repr(prop))
                return prop

        if raiseserror:
            raise ValueError(f"Cannot find property with name <{name}>")

        return None

    def append_props(self, proplist: list[GridProperty]) -> None:
        """Add a list of GridProperty objects to current GridProperties instance."""
        if not self._props and proplist:
            self._ncol = proplist[0].ncol
            self._nrow = proplist[0].nrow
            self._nlay = proplist[0].nlay
        self._props += proplist
        self._consistency_check()

    def get_ijk(
        self,
        names: tuple[str, str, str] = ("IX", "JY", "KZ"),
        zerobased: bool = False,
        asmasked: bool = False,
    ) -> tuple[GridProperty, GridProperty, GridProperty]:
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter, J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            asmasked: If True, then active cells only.
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """
        return _grid_etc1.get_ijk(
            self, names=names, zerobased=zerobased, asmasked=asmasked
        )

    def get_actnum(
        self,
        name: str = "ACTNUM",
        asmasked: bool = False,
    ) -> GridProperty | None:
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): ACTNUM is returned with all cells
                as default. Use asmasked=True to make 0 entries masked.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> myprops = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.INIT",
            ...     fformat="init",
            ...     names=["PERMX"],
            ...     grid=grid,
            ... )
            >>> act = myprops.get_actnum()
            >>> print('{}% of cells are active'.format(act.values.mean() * 100))
            99.99...% of cells are active

        Returns:
            A GridProperty instance of ACTNUM, or None if no props present.
        """
        # borrow function from GridProperty class:
        if self._props:
            return self._props[0].get_actnum(name=name, asmasked=asmasked)

        warnings.warn("No gridproperty in list", UserWarning)
        return None

    def get_dataframe(
        self,
        activeonly: bool = False,
        ijk: bool = False,
        xyz: bool = False,
        doubleformat: bool = False,
        grid: Grid | None = None,
    ) -> pd.DataFrame:
        """Returns a Pandas dataframe table for the properties.

        See also :func:`xtgeo.gridproperties_dataframe()`

        Args:
            activeonly (bool): If True, return only active cells, NB!
                If True, will require a grid instance (see grid key)
            ijk (bool): If True, show cell indices, IX JY KZ columns
            xyz (bool): If True, show cell center coordinates (needs grid).
            doubleformat (bool): If True, floats are 64 bit, otherwise 32 bit.
                Note that coordinates (if xyz=True) is always 64 bit floats.
            grid (Grid): The grid geometry object. This is required for the
                xyz option.

        Returns:
            Pandas dataframe object

        Examples::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> pps = xtgeo.gridproperties_from_file(
            ...     reek_dir + "/REEK.UNRST",
            ...     fformat="unrst",
            ...     names=['SOIL', 'SWAT', 'PRESSURE'],
            ...     dates=[19991201],
            ...     grid=grid,
            ... )
            >>> df = pps.get_dataframe(activeonly=False, ijk=True, xyz=True, grid=grid)
            >>> print(df)
                   ACTNUM  IX  JY  ...  SOIL_19991201  SWAT_19991201  PRESSURE_19991201
            0           1   1   1  ...            0.0            1.0         341.694183
            1           1   1   1  ...            0.0            1.0         342.097107
            2           1   1   1  ...            0.0            1.0         342.500061
            3           1   1   1  ...            0.0            1.0         342.902954
            4           1   1   1  ...            0.0            1.0         343.305908
            ...

        """
        return gridproperties_dataframe(
            self,
            activeonly=activeonly,
            ijk=ijk,
            xyz=xyz,
            doubleformat=doubleformat,
            grid=grid,
        )

    def _consistency_check(self) -> None:
        for p in self._props:
            if (p.ncol, p.nrow, p.nlay) != (self.ncol, self.nrow, self.nlay):
                raise ValueError("Mismatching dimensions in GridProperties members.")

    @staticmethod
    def scan_dates(
        pfile: FileLike,
        fformat: Literal["unrst"] = "unrst",
        maxdates: int = MAXDATES,
        dataframe: bool = False,
        datesonly: bool = False,
    ) -> list | pd.DataFrame:
        """Quick scan dates in a simulation restart file.

        Args:
            pfile (str): Name of file or file handle with properties
            fformat (str): unrst (so far)
            maxdates (int): Maximum number of dates to collect
            dataframe (bool): If True, return a Pandas dataframe instead
            datesonly (bool): If True, SEQNUM is skipped,

        Return:
            A list of tuples or a dataframe with (seqno, date),
            date is on YYYYMMDD form. If datesonly is True and dataframe is False,
            the returning list will be a simple list of dates.

        Example::
            >>> dlist = GridProperties.scan_dates(reek_dir + "/REEK.UNRST")
            >>> #or getting all dates a simple list:
            >>> dlist = GridProperties.scan_dates(
            ... reek_dir + "/REEK.UNRST",
            ... datesonly=True)

        .. versionchanged:: 2.13 Added datesonly keyword
        """
        logger.info("Format supported as default is %s", fformat)

        _pfile = FileWrapper(pfile)
        _pfile.check_file(raiseerror=ValueError)

        dlist = utils.scan_dates(_pfile, maxdates=maxdates, dataframe=dataframe)

        if datesonly and dataframe:
            assert isinstance(dlist, pd.DataFrame)
            dlist.drop("SEQNUM", axis=1, inplace=True)

        if datesonly and not dataframe:
            dlist = [date for (_, date) in dlist]

        return dlist
