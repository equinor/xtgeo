# -*- coding: utf-8 -*-
"""Import/export of grid properties (cf GridProperties class)"""

from copy import deepcopy

import xtgeo

from . import _grid3d_utils as utils
from ._gridprop_import_eclrun import (
    decorate_name,
    find_gridprop_from_init_file,
    find_gridprops_from_restart_file,
    valid_gridprop_lengths,
)
from .grid_property import GridProperty

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the GridProperties() instance

# On "strict" keyword: Default is (True, False)
# A strict (False, False) simply means that if keyname, optionally with date is not
# found is will just warn and continue to next! If (True, True) it will warn but TRY to
# import anyway, which in turn may raise a KeywordNotError or DateNotFoundError, etc
#
# The (True, False) will be strict on keywords, but sloppy on dates, meaning that
# missing dates will be skipped. However, if all dates are missing an exception will be
# raised
#
# Note that there are keyword and data checks also in _gridprop_import_eclrun


def import_ecl_output(
    self,
    pfile,
    names=None,
    dates=None,
    grid=None,
    namestyle=0,
    strict=(True, False),
):

    strictkeys, strictdates = strict

    if not isinstance(pfile, xtgeo._XTGeoFile):
        raise RuntimeError("BUG kode 84728, pfile is not a _XTGeoFile instance")

    if not grid:
        raise ValueError("Grid Geometry object is missing")

    if not names:
        raise ValueError("Name list cannot be empty (None)")

    if dates is None:
        _import_ecl_output_v2_init(self, pfile, names, grid, strictkeys)

    else:
        _import_ecl_output_v2_rsta(
            self,
            pfile,
            names,
            dates,
            grid,
            strictkeys,
            strictdates,
            namestyle,
        )


def _import_ecl_output_v2_init(self, pfile, names, grid, strict):
    """Import INIT parameters"""

    # scan valid keywords
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=100000, dataframe=True, dates=True
    )

    validnames = list()

    valid_lengths = valid_gridprop_lengths(grid)

    # get a list of valid property names
    for kw in list(kwlist.itertuples(index=False, name=None)):
        kwname, _, nlen, _, _ = kw
        if nlen in valid_lengths and kwname not in validnames:
            validnames.append(kwname)

    if names == "all":
        usenames = deepcopy(validnames)
    else:
        usenames = list(names)

    for name in usenames:
        if name not in validnames:
            if strict:
                raise ValueError(
                    f"Requested keyword {name} is not in INIT file,"
                    f"valid entries are {validnames}, set strict=False to warn instead."
                )
            else:
                logger.warning(
                    "Requested keyword %s is not in INIT file."
                    "Entry will not be read, set strict=True to raise Error instead.",
                    name,
                )

    results = find_gridprop_from_init_file(
        pfile.file,
        names=names,
        grid=grid,
    )
    for result in results:
        prop = GridProperty()
        self._names.append(result["name"])
        result["name"] = decorate_name(result["name"], grid.dualporo, fracture=False)
        for attr, value in result.items():
            setattr(prop, "_" + attr, value)

        self._props.append(prop)
        self._dates.append(prop._date)

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay


def _import_ecl_output_v2_rsta(
    self,
    pfile,
    names,
    dates,
    grid,
    strictkeycomb,
    strictdate,
    namestyle,
):
    """Import RESTART parameters"""

    if dates not in ["all", "first", "last"]:
        # dates may come on form 2020-12-22 or 20201222; process all to latter fmt
        dates = [
            int(str(thedate).replace("-", "")) if isinstance(thedate, str) else thedate
            for thedate in dates
        ]

    # scan valid keywords with dates
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=100000, dataframe=True, dates=True
    )

    validnamedatepairs, validdates = _process_valid_namesdates(kwlist, grid)

    # allow sloppy dates, i.e. remove invalid date entries
    if isinstance(dates, list) and strictdate is False:
        dates = _process_sloppydates(dates, validdates)

    usenamedatepairs = list()
    if names == "all" and dates == "all":
        usenamedatepairs = deepcopy(validnamedatepairs)
        usedates = dates
    else:
        if names == "all" and dates != "all":
            usenames = [namedate[0] for namedate in validnamedatepairs]
            usedates = dates
        elif names != "all" and dates == "all":
            usedates = [namedate[1] for namedate in validnamedatepairs]
            usenames = names
        else:
            usedates = dates
            usenames = names

        for name in usenames:
            for date in usedates:
                usenamedatepairs.append((name, date))

    # Do the actual import
    for namedate in usenamedatepairs:
        name, date = namedate

        if name not in ("SGAS", "SOIL", "SWAT") and namedate not in validnamedatepairs:
            # saturation keywords are a mess in Eclipse and friends; check later
            if strictkeycomb:
                raise ValueError(
                    f"Keyword data combo {name} {date} is not in RESTART file."
                    f"Possible entries are: {validnamedatepairs}"
                )
            else:
                logger.warning(
                    "Keyword data combo %s %s is not in RESTART file."
                    "Possible entries are: %s"
                    "Value will not be imported",
                    name,
                    date,
                    validnamedatepairs,
                )

    results = find_gridprops_from_restart_file(pfile.file, names, dates, grid=grid)
    for result in results:
        prop = GridProperty()

        if namestyle == 1:
            sdate = str(result["date"])
            result["name"] += "--" + sdate[0:4] + "_" + sdate[4:6] + "_" + sdate[6:8]
        else:
            result["name"] = decorate_name(
                result["name"], grid.dualporo, fracture=False, date=result["date"]
            )

        for attr, value in result.items():
            setattr(prop, "_" + attr, value)

        self._props.append(prop)
        self._names.append(prop.name)
        self._dates.append(prop.date)

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay


def _process_valid_namesdates(kwlist, grid):
    """Return lists with valid pairs, dates scanned from RESTART"""
    validnamedatepairs = list()
    validdates = list()
    valid_lengths = valid_gridprop_lengths(grid)
    for kw in list(kwlist.itertuples(index=False, name=None)):
        kwname, kwtyp, nlen, _, date = kw
        if (
            kwtyp != "CHAR"
            and nlen in valid_lengths
            and (kwname, date) not in validnamedatepairs
        ):
            validnamedatepairs.append((kwname, date))
        if kwtyp != "CHAR" and nlen in valid_lengths and date not in validdates:
            validdates.append(date)

    return validnamedatepairs, validdates


def _process_sloppydates(dates, validdates):
    """Allow "sloppy dates", which removes invalid dates from the list"""

    usedates = []
    skipdates = []
    for date in dates:
        if date not in validdates:
            skipdates.append(date)
        else:
            usedates.append(date)
    if not usedates:
        msg = f"No valid dates given (dates: {dates} vs {validdates})"
        xtg.error(msg)
        raise ValueError(msg)

    if skipdates:
        msg = f"Some dates not found: {skipdates}; will continue with dates: {usedates}"
        xtg.warn(msg)

    return usedates
