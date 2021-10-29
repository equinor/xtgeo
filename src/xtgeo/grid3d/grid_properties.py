# -*- coding: utf-8 -*-

"""Module for Grid Properties."""


import hashlib
import warnings
from collections import OrderedDict
from typing import Optional

import deprecation
import numpy as np
import pandas as pd

import xtgeo
from xtgeo.common import XTGDescription, XTGeoDialog

from . import _grid3d_utils as utils
from . import _grid_etc1, _gridprops_import_eclrun
from ._grid3d import _Grid3D
from .grid_property import GridProperty

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def gridproperties_from_file(
    pfile,
    fformat=None,
    names=None,
    dates=None,
    grid=None,
    namestyle=0,
    strict=(True, False),
):
    """Import grid properties from file in one go.

    In case of names='all' then all vectors which have a valid length
    (number of total or active cells in the grid) will be read

    Args:
        pfile (str or Path): Name of file with properties
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
        >>> xtgeo.gridproperties_from_file(
        ...     "ECL.UNRST",
        ...     fformat="unrst",
        ...     dates=[20110101, 20141212],
        ...     names=["PORO", "DZ"]
        ... )

    Raises:
        FileNotFoundError: if input file is not found
        DateNotFoundError: The date is not found
        KeywordNotFoundError: The keyword is not found
        KeywordFoundDateNotFoundError: The keyword but not date found

    .. versionadded:: 2.13 Added strict key
    """
    pfile = xtgeo._XTGeoFile(pfile, mode="rb")

    pfile.check_file(raiseerror=OSError)

    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat()
    else:
        fformat = pfile.generic_format_by_proposal(fformat)  # default

    if fformat.lower() in ["roff_ascii", "roff_binary"]:
        return [GridProperty(pfile, fformat="roff", name=name) for name in names]

    elif fformat.lower() == "init":
        return _gridprops_import_eclrun.import_ecl_init_gridproperties(
            pfile,
            grid=grid,
            names=names,
            strict=strict[0],
        )
    elif fformat.lower() == "unrst":
        return _gridprops_import_eclrun.import_ecl_restart_gridproperties(
            pfile,
            dates=dates,
            grid=grid,
            names=names,
            namestyle=namestyle,
            strict=strict,
        )
    else:
        raise OSError("Invalid file format {fformat}")


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
    gridproperties, grid=None, activeonly=True, ijk=False, xyz=False, doubleformat=False
):  # pylint: disable=too-many-branches, too-many-statements
    """Returns a Pandas dataframe table for the properties.

    Args:
        gridproperties: List (or iterable) of GridProperty to
            create dataframe of.
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

        >>> grd = xtgeo.grid_from_file(gfile1, fformat='egrid')
        >>> names = ['SOIL', 'SWAT', 'PRESSURE']
        >>> dates = [19991201]
        >>> gps = xtgeo.gridproperties_from_file(
        ...     rfile1,
        ...     fformat='unrst',
        ...     names=names,
        ...     dates=dates,
        ...     grid=grd
        ... )
        >>> df = xtgeo.gridproperties_dataframe(gps, grid=grd)

    """

    proplist = OrderedDict()

    if ijk:
        logger.info("IJK is active")
        if activeonly:
            logger.info("Active cells only")
            ix, jy, kz = grid.get_ijk(asmasked=True)
            proplist["IX"] = ix.get_active_npvalues1d()
            proplist["JY"] = jy.get_active_npvalues1d()
            proplist["KZ"] = kz.get_active_npvalues1d()
        else:
            logger.info("All cells (1)")
            act = grid.get_actnum(dual=True)
            ix, jy, kz = grid.get_ijk(asmasked=False)
            proplist["ACTNUM"] = act.values1d
            proplist["IX"] = ix.values1d
            proplist["JY"] = jy.values1d
            proplist["KZ"] = kz.values1d

    if xyz:
        if isinstance(grid, GridProperties):
            raise ValueError("You ask for xyz but no Grid is present. Use " "grid=...")

        logger.info("XYZ is active")
        option = False
        if activeonly:
            logger.info("Active cells only")
            option = True

        xc, yc, zc = grid.get_xyz(asmasked=option)
        if activeonly:
            proplist["X_UTME"] = xc.get_active_npvalues1d()
            proplist["Y_UTMN"] = yc.get_active_npvalues1d()
            proplist["Z_TVDSS"] = zc.get_active_npvalues1d()
        else:
            logger.info("All cells (2)")
            proplist["X_UTME"] = xc.values1d
            proplist["Y_UTMN"] = yc.values1d
            proplist["Z_TVDSS"] = zc.values1d

    logger.info("Proplist: %s", proplist)

    for prop in gridproperties:
        logger.info("Getting property %s", prop.name)
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

        proplist[prop.name] = vector

    for key, prop in proplist.items():
        logger.info("Property %s has length %s", key, prop.shape)
    mydataframe = pd.DataFrame.from_dict(proplist)
    logger.debug("Dataframe: \n%s", mydataframe)

    return mydataframe


def scan_restart_dates(
    pfile, fformat="unrst", maxdates=1000, dataframe=False, datesonly=False
):
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
        >>> from xtgeo import scan_restart_dates
        >>> dlist = scan_restart_dates("ECL.UNRST", datesonly=True)

    .. versionchanged:: 2.13 Added datesonly keyword
    """
    pfile = xtgeo._XTGeoFile(pfile)

    logger.info("Format supported as default is %s", fformat)

    dlist = utils.scan_dates(pfile, maxdates=maxdates, dataframe=dataframe)

    if datesonly and dataframe:
        dlist.drop("SEQNUM", axis=1, inplace=True)

    if datesonly and not dataframe:
        dlist = [date for (_, date) in dlist]

    return dlist


def scan_ecl_keywords(
    pfile, fformat="xecl", maxkeys=100000, dataframe=False, dates=False
):
    """Quick scan of keywords in Eclipse binary files, or ROFF binary files.

    For Eclipse files:
    Returns a list of tuples (or dataframe), e.g. ('PRESSURE',
    'REAL', 355299, 3582700), where (keyword, type, no_of_values,
    byteposition_in_file)

    For ROFF files
    Returns a list of tuples (or dataframe), e.g.
    ('translate!xoffset', 'float', 1, 3582700),
    where (keyword, type, no_of_values, byteposition_in_file).

    For Eclipse, the byteposition is to the KEYWORD, while for ROFF
    the byte position is to the beginning of the actual data.

    Args:
        pfile (str): Name or a filehandle to file with properties
        fformat (str): xecl (Eclipse INIT, RESTART, ...) or roff for
            ROFF binary,
        maxkeys (int): Maximum number of keys
        dataframe (bool): If True, return a Pandas dataframe instead
        dates (bool): if True, the date is the last column (only
            menaingful for restart files). Default is False.

    Return:
        A list of tuples or dataframe with keyword info

    Example::
        >>> props = GridProperties()
        >>> dlist = props.scan_keywords('ECL.UNRST')

    """
    pfile = xtgeo._XTGeoFile(pfile)

    dlist = utils.scan_keywords(
        pfile, fformat=fformat, maxkeys=maxkeys, dataframe=dataframe, dates=dates
    )

    return dlist


def gridproperties_hash(props):
    """Return a unique hash ID for a list of gridproperties.

    .. versionadded:: 2.16
    """
    mhash = hashlib.sha256()

    hashinput = ""
    for prop in props:
        gid = "{}{}{}{}{}{}".format(
            prop.ncol,
            prop.nrow,
            prop.nlay,
            prop.values.mean(),
            prop.values.min(),
            prop.values.max(),
        )
        hashinput += gid

    mhash.update(hashinput.encode())
    return mhash.hexdigest()


class GridProperties(_Grid3D):
    """Deprecated: Collection for properties belonging to the same grid.

    See Also:
        The :class:`GridProperty` class.
    .. deprecated:: 2.16
    """

    def __init__(
        self,
        ncol: Optional[int] = 4,
        nrow: Optional[int] = 3,
        nlay: Optional[int] = 5,
    ):
        """Instantiating.

        Args:
            ncol: Number of columns
            nrow: Number of rows
            nlay: Number of layers

        """
        super().__init__()

        warnings.warn(
            "The GridProperties class is deprecated,"
            " use a list of GridProperty instead",
            DeprecationWarning,
        )

        self._ncol = ncol
        self._nrow = nrow
        self._nlay = nlay

        self._props = []  # list of GridProperty objects
        self._names = []  # list of GridProperty names
        self._dates = []  # list of dates (_after_ import) YYYYDDMM

    @classmethod
    def from_list(cls, proplist):
        gps = cls()
        if len(proplist) == 0:
            return gps
        dimensions = proplist[0].dimensions

        for litem in proplist:
            if litem.dimensions != dimensions:
                raise ValueError(
                    f"Non-uniform dimensions of gridproperties in {proplist}"
                )
        gps._ncol, gps._nrow, gps._nlay = dimensions
        gps._props = proplist
        gps._names = [p.name for p in proplist]
        gps._dates = [p.date for p in proplist]
        return gps

    def __repr__(self):  # noqa: D105
        myrp = (
            "{0.__class__.__name__} (id={1}) ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, nlay={0._nlay!r}, "
            "filesrc={0._names!r}".format(self, id(self))
        )
        return myrp

    def __str__(self):
        """str: User friendly print."""
        return self.describe(flush=False)

    def __contains__(self, name):
        """bool: Emulate 'if "PORO" in props'."""
        prop = self.get_prop_by_name(name, raiseserror=False)
        if prop:
            return True

        return False

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated"
        " so instead of gridproperties['PORO_1991231'] use eg. a list comprehension"
        " to get properties matching a name, like this: "
        " [prop for prop in gridproperties if prop.name == 'PORO_1991231']",
    )
    def __getitem__(self, name):  # noqa: D105
        prop = self.get_prop_by_name(name, raiseserror=False)
        if prop is None:
            raise KeyError(f"Key {name} does not exist")

        return prop

    def __iter__(self):  # noqa: D105
        return iter(self._props)

    # ----------------------------------------------------------------------------------
    # Properties:

    @property
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated"
        "use [p.name for p in gridproperties] instead",
    )
    def names(self):
        """Returns or sets a list of property names.

        Example::

            namelist = props.names
            for prop in namelist:
                print ('Property name is {}'.format(name))

            # when setting names, an empty string 'trick' will make the namelist to be
            # defaulted as each individual property name

            props.names = ""

        """
        return self._names

    @names.setter
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated",
    )
    def names(self, nameslist):
        if not nameslist and self._props:
            # try the get the names from each individual property automatic
            for prop in self._props:
                self._names.append(prop.name)
                return

        if len(nameslist) != len(self._props):
            raise ValueError("Number of names does not match number of properties")

        # look for duplicates
        if len(nameslist) > len(set(nameslist)):
            raise ValueError("List of names contains duplicates; names must be unique")

        self._names = nameslist

    @property
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated,"
        "use list(gridproperties) instead",
    )
    def props(self):
        """Returns a list of XTGeo GridProperty objects, None if empty.

        Example::

            gg = xtgeo.grid3d.Grid('TEST1.EGRID')
            myprops = XTGeo.grid3d.GridProperties()
            from_file('TEST1.INIT', fformat='init', names=['PORO']         )
            proplist = myprops.props
            for prop in proplist:
                print ('Property object ID is {}'.format(prop))

            # adding a property, e.g. get ACTNUM as a property from the grid
            actn = gg.get_actnum()  # this will get actn as a GridProperty
            myprops.add_props([actn])
        """
        if not self._props:
            return None

        return self._props

    @props.setter
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated",
    )
    def props(self, propslist):
        self._props = propslist
        self._dates = [p.date for p in propslist]
        self._names = [p.name for p in propslist]

    @property
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated,"
        "use [p.date for p in gridproperties] instead",
    )
    def dates(self):
        """Returns a list of valid (found) dates after import.

        Returns None if no dates present

        Note:
            See also :meth:`GridProperties.scan_dates` for scanning available dates
            in advance

        Example::

            datelist = props.dates
            for date in datelist:
                print ('Date applied is {}'.format(date))

        """
        if not self._dates:
            return None

        return self._dates

    # Copy, and etc aka setters and getters

    def copy(self):
        """Copy a GridProperties instance to a new unique instance.

        Note that the GridProperty instances will also be unique.
        """
        new = GridProperties()

        new._ncol = self._ncol
        new._nrow = self._nrow
        new._nlay = self._nlay

        for prp in self._props:
            newprp = prp.copy()
            new._props.append(newprp)

        for name in self._names:
            new._names.append(name)

        for date in self._dates:
            new._dates.append(date)

        return new

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated.",
    )
    def describe(self, flush=True):
        """Describe an instance by printing to stdout."""
        dsc = XTGDescription()

        dsc.title("Description of GridProperties instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        dsc.txt("Attached grid props objects (names)", self._names)

        if flush:
            dsc.flush()
            return None
        return dsc.astext()

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="generate_hash is deprecated, see :meth:`gridproperties_hash`",
    )
    def generate_hash(self):
        """Deprecated, see gridproperties_hash
        .. versionadded:: 2.10
        .. deprecated:: 2.16
        """
        return gridproperties_hash(self)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Only using gridproperties as an iterable is non-deprecated"
        " so instead of calling get_prop_by_name, use eg. a list comprehension to get "
        " properties matching a name, like this: "
        " [prop for prop in gridproperties if prop.name == 'PORO_1991231']",
    )
    def get_prop_by_name(self, name, raiseserror=True):
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
            raise ValueError("Cannot find property with name <{}>".format(name))

        return None

    def __add__(self, other):
        """Concatenate two gridproperties instances."""
        copy = self.copy()
        copy += other
        return copy

    def __iadd__(self, other):
        """Add a list of GridProperty objects to current GridProperties instance."""
        for prop in other:
            if isinstance(prop, GridProperty):
                # an prop instance can only occur once
                if prop not in self._props:

                    self._props.append(prop)
                    self._names.append(prop.name)
                    self._dates.append(prop.date)
                    self._ncol = prop.ncol
                    self._nrow = prop.nrow
                    self._nlay = prop.nlay
                else:
                    logger.info("Not added. GridProperty instance already present")
            else:
                raise ValueError("Input property is not a valid GridProperty " "object")

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Method append_props is deprecated,"
        " use grid_properties += proplist instead",
    )
    def append_props(self, proplist):
        """Deprecated, use the + operation instead."""
        self += proplist

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Method get_actnum is deprecated, use xtgeo.Grid.get_ijk instead",
    )
    def get_ijk(
        self, names=("IX", "JY", "KZ"), zerobased=False, asmasked=False, mask=None
    ):
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter, J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            asmasked: If True, then active cells only.
            mask: If True, then active cells only (deprecated).
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = super()._evaluate_mask(mask)

        # resuse method from grid
        ixc, jyc, kzc = _grid_etc1.get_ijk(
            self, names=names, zerobased=zerobased, asmasked=asmasked
        )

        # return the objects
        return ixc, jyc, kzc

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Method get_actnum is deprecated, use xtgeo.Grid.get_actnum instead",
    )
    def get_actnum(self, name="ACTNUM", asmasked=False, mask=None):
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): ACTNUM is returned with all cells
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead.

        Example::

            act = myprops.get_actnum()
            print('{}% cells are active'.format(act.values.mean() * 100))

        Returns:
            A GridProperty instance of ACTNUM, or None if no props present.
        """
        if mask is not None:
            xtg.warndeprecated(
                "The mask option is deprecated,"
                "and will be removed in version 4.0. Use asmasked instead."
            )
            asmasked = super()._evaluate_mask(mask)

        # borrow function from GridProperty class:
        if self._props:
            return self._props[0].get_actnum(name=name, asmasked=asmasked)

        warnings.warn("No gridproperty in list", UserWarning)
        return None

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="The GridProperties class is deprecated. To load"
        "a list of properties, use xtgeo.gridproperties_from_file() instead.",
    )
    def from_file(self, *args, **kwargs):
        """Deprecated, see :meth:`gridproperties_from_file`.

        .. deprecated:: 2.16
        """
        self.props = gridproperties_from_file(*args, **kwargs)
        if self.props:
            self._ncol = self.props[0].ncol
            self._nrow = self.props[0].nrow
            self._nlay = self.props[0].nlay
        return self

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Method get_dataframe is deprecated, use get_dataframe instead.",
    )
    def get_dataframe(
        self, activeonly=False, ijk=False, xyz=False, doubleformat=False, grid=None
    ):
        """
        Deprecated, see :meth:`gridproperties_dataframe`.

        .. deprecated:: 2.16
        """
        gridlike = self
        if grid is not None:
            gridlike = grid
        return gridproperties_dataframe(
            self,
            grid=gridlike,
            activeonly=activeonly,
            ijk=ijk,
            xyz=xyz,
            doubleformat=doubleformat,
        )

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Method dataframe is deprecated, use"
        " xtgeo.gridproperties_dataframe() instead.",
    )
    def dataframe(self, *args, **kwargs):
        """
        Deprecated, see :meth:`gridproperties_dataframe`.
        .. versiondeprecated:: 2.16
        """
        return self.get_dataframe(*args, **kwargs)

    @staticmethod
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.scan_ecl_keywords() instead.",
    )
    def scan_keywords(*args, **kwargs):
        """Deprecated, see :meth:`scan_elc_keywords`.
        .. versiondeprecated:: 2.16
        """
        return scan_ecl_keywords(*args, **kwargs)

    @staticmethod
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.scan_restart_dates() instead.",
    )
    def scan_dates(*args, **kwargs):
        """Deprecated, see :meth:`scan_restart_dates`.
        .. versiondeprecated:: 2.16
        """
        return scan_restart_dates(*args, **kwargs)
