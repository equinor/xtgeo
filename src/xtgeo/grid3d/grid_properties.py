# -*- coding: utf-8 -*-

"""Module for Grid Properties."""
import hashlib
import warnings
from typing import List, Optional

import deprecation
import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGDescription, XTGeoDialog
from xtgeo.common.constants import MAXDATES, MAXKEYWORDS

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
    """Import grid properties from file.

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
        >>> grd = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID", fformat='egrid')
        >>> gps = xtgeo.gridproperties_from_file(
        ...     reek_dir + "/REEK.INIT",
        ...     fformat='init',
        ...     names=["PORO", "PERMX"],
        ...     grid=grd,
        ... )
    """
    pfile = xtgeo._XTGeoFile(pfile, mode="rb")

    pfile.check_file(raiseerror=ValueError)

    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat()
    else:
        fformat = pfile.generic_format_by_proposal(fformat)  # default

    if fformat.lower() in ["roff_ascii", "roff_binary"]:
        return GridProperties(
            props=[
                xtgeo.gridproperty_from_file(
                    pfile.file, fformat="roff", name=name, grid=grid
                )
                for name in names
            ]
        )

    elif fformat.lower() == "init":
        return GridProperties(
            props=_gridprops_import_eclrun.import_ecl_init_gridproperties(
                pfile,
                grid=grid,
                names=names,
                strict=strict[0],
                maxkeys=MAXKEYWORDS,
            )
        )
    elif fformat.lower() == "unrst":
        return GridProperties(
            props=_gridprops_import_eclrun.import_ecl_restart_gridproperties(
                pfile,
                dates=dates,
                grid=grid,
                names=names,
                namestyle=namestyle,
                strict=strict,
                maxkeys=MAXKEYWORDS,
            )
        )
    else:
        raise ValueError("Invalid file format {fformat}")


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

    dataframe_dict = dict()
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
        ncol: Deprecated argument.
        nrow: Deprecated argument.
        nlay: Deprecated argument.
        props: The list of GridProperty objects.

    See Also:
        The :class:`GridProperty` class.
    """

    def __init__(
        self,
        ncol: Optional[int] = None,
        nrow: Optional[int] = None,
        nlay: Optional[int] = None,
        props: List[GridProperty] = None,
    ):
        dims_given = False
        if ncol is not None:
            warnings.warn(
                "Initializing GridProperties with ncol is deprecated.",
                DeprecationWarning,
            )
            dims_given = True
        else:
            ncol = 4
        if nrow is not None:
            warnings.warn(
                "Initializing GridProperties with nrow is deprecated.",
                DeprecationWarning,
            )
            dims_given = True
        else:
            nrow = 3
        if nlay is not None:
            warnings.warn(
                "Initializing GridProperties with nlay is deprecated.",
                DeprecationWarning,
            )
            dims_given = True
        else:
            nlay = 5

        if props:
            if dims_given:
                raise ValueError(
                    "Giving both ncol/nrow/nlay and props list is not supported. "
                    "Please give just props as ncol/nrow/nlay is deprecated."
                )
            ncol, nrow, nlay = props[0].dimensions

        super().__init__(ncol, nrow, nlay)

        # The _names field is just kept for backwards
        # compatability until the names setter has been
        # deprecated
        self._names = []

        self.props = props or []

    def __repr__(self):  # noqa: D105
        myrp = (
            "{0.__class__.__name__} (id={1}) ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, nlay={0._nlay!r}, "
            "filesrc={0.names!r}".format(self, id(self))
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

    def __getitem__(self, name):  # noqa: D105
        prop = self.get_prop_by_name(name, raiseserror=False)
        if prop is None:
            raise KeyError(f"Key {name} does not exist")

        return prop

    def __iter__(self):  # noqa: D105
        return iter(self._props)

    @property
    def names(self):
        """Returns a list of property names.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> props = GridProperties()
            >>> props.from_file(
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
        return self._names

    @names.setter
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="The behavior of the setting names would before create an alias name "
        "the behavior of which was not always consistent. "
        "Note that setting names on the GridProperties has "
        "_no effect_ on the behavior of its methods except the names getter."
        "This name aliasing is now going away. "
        "In order to change the name of properties, "
        "use\nfor p in gridprops:\n    p.name = newname",
    )
    def names(self, nameslist):
        if len(nameslist) != len(self._props):
            raise ValueError("Number of names does not match number of properties")

        # look for duplicates
        if len(nameslist) > len(set(nameslist)):
            raise ValueError("List of names contains duplicates; names must be unique")

        self._names = nameslist

    @property
    def props(self):
        """Returns a list of XTGeo GridProperty objects, None if empty.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> myprops = GridProperties()
            >>> myprops.from_file(
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
    def props(self, propslist):
        self._props = propslist
        if propslist:
            self._ncol = propslist[0].ncol
            self._nrow = propslist[0].nrow
            self._nlay = propslist[0].nlay
        self._names = [p.name for p in self._props]
        self._consistency_check()

    @property
    def dates(self):
        """Returns a list of valid (found) dates after import.

        Returns None if no dates present

        Note:
            See also :meth:`GridProperties.scan_dates` for scanning available dates
            in advance

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> props = GridProperties()
            >>> props.from_file(
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

    def copy(self):
        """Copy a GridProperties instance to a new unique instance.

        Note that the GridProperty instances will also be unique.
        """

        gps = GridProperties(props=[p.copy() for p in self.props] if self.props else [])
        gps._names = self._names.copy()
        return gps

    def describe(self, flush=True):
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

    def generate_hash(self):
        """str: Return a unique hash ID for current gridproperties instance.

        .. versionadded:: 2.10
        """
        mhash = hashlib.sha256()

        hashinput = ""
        for prop in self._props:
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

    def append_props(self, proplist):
        """Add a list of GridProperty objects to current GridProperties instance."""
        if not self._props and proplist:
            self._ncol = proplist[0].ncol
            self._nrow = proplist[0].nrow
            self._nlay = proplist[0].nlay
        self._props += proplist
        self._names = [p.name for p in self._props]
        self._consistency_check()

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

        return _grid_etc1.get_ijk(
            self, names=names, zerobased=zerobased, asmasked=asmasked
        )

    def get_actnum(self, name="ACTNUM", asmasked=False, mask=None):
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): ACTNUM is returned with all cells
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead.

        Example::

            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> myprops = GridProperties()
            >>> myprops.from_file(
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
        details="Use xtgeo.gridproperties_from_file() instead",
    )
    def from_file(
        self,
        pfile,
        fformat="roff",
        names=None,
        dates=None,
        grid=None,
        namestyle=0,
        strict=(True, False),
    ):
        """Import grid properties from file in one go.

        This class is particulary useful for Eclipse INIT and RESTART files.

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
            >>> import xtgeo
            >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> props = GridProperties()
            >>> props.from_file(
            ...     reek_dir + "/REEK.INIT",
            ...     fformat="init",
            ...     names=["PERMX"],
            ...     grid=grid,
            ... )


        Raises:
            FileNotFoundError: if input file is not found
            DateNotFoundError: The date is not found
            KeywordNotFoundError: The keyword is not found
            KeywordFoundDateNotFoundError: The keyword but not date found

        .. versionadded:: 2.13 Added strict key
        """

        self.append_props(
            list(
                gridproperties_from_file(
                    pfile=pfile,
                    fformat=fformat,
                    names=names,
                    dates=dates,
                    grid=grid,
                    namestyle=namestyle,
                    strict=strict,
                )
            )
        )

    def get_dataframe(
        self, activeonly=False, ijk=False, xyz=False, doubleformat=False, grid=None
    ):
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
            >>> pps.grid_properties_from_file(
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

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use GridProperty.get_dataframe() instead",
    )
    def dataframe(self, *args, **kwargs):
        return self.get_dataframe(*args, **kwargs)

    def _consistency_check(self):
        for p in self._props:
            if (p.ncol, p.nrow, p.nlay) != (self.ncol, self.nrow, self.nlay):
                raise ValueError("Mismatching dimensions in GridProperties members.")

    @staticmethod
    def scan_keywords(
        pfile, fformat="xecl", maxkeys=MAXKEYWORDS, dataframe=False, dates=False
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
            >>> dlist = GridProperties.scan_keywords(reek_dir + "/REEK.UNRST")

        """
        pfile = xtgeo._XTGeoFile(pfile)
        pfile.check_file(raiseerror=ValueError)

        return utils.scan_keywords(
            pfile,
            fformat=fformat,
            maxkeys=maxkeys,
            dataframe=dataframe,
            dates=dates,
        )

    @staticmethod
    def scan_dates(
        pfile, fformat="unrst", maxdates=MAXDATES, dataframe=False, datesonly=False
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
            >>> dlist = GridProperties.scan_dates(reek_dir + "/REEK.UNRST")
            >>> #or getting all dates a simple list:
            >>> dlist = GridProperties.scan_dates(
            ... reek_dir + "/REEK.UNRST",
            ... datesonly=True)

        .. versionchanged:: 2.13 Added datesonly keyword
        """
        logger.info("Format supported as default is %s", fformat)

        pfile = xtgeo._XTGeoFile(pfile)
        pfile.check_file(raiseerror=ValueError)

        dlist = utils.scan_dates(pfile, maxdates=maxdates, dataframe=dataframe)

        if datesonly and dataframe:
            dlist.drop("SEQNUM", axis=1, inplace=True)

        if datesonly and not dataframe:
            dlist = [date for (_, date) in dlist]

        return dlist
