# -*- coding: utf-8 -*-
"""Import/export of grid properties (cf GridProperties class)"""

from copy import deepcopy
import xtgeo

from xtgeo.grid3d import _gridprop_import_eclrun

from .grid_property import GridProperty
from . import _grid3d_utils as utils

xtg = xtgeo.XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the GridProperties() instance


def import_ecl_output(
    self, pfile, names=None, dates=None, grid=None, namestyle=0, strict=(True, False)
):

    strictname, strictdate = strict  # strict is a tuple e.g. (True, False)

    if not isinstance(pfile, xtgeo._XTGeoFile):
        raise RuntimeError("BUG kode 84728, pfile is not a _XTGeoFile instance")

    if not grid:
        raise ValueError("Grid Geometry object is missing")

    if not names:
        raise ValueError("Name list cannot be empty (None)")

    if dates is None:
        _import_ecl_output_v2_init(self, pfile, names, grid, strictname)

    else:
        _import_ecl_output_v2_rsta(
            self, pfile, names, dates, grid, strictname, strictdate, namestyle
        )


def _import_ecl_output_v2_init(self, pfile, names, grid, strictname):
    """Import INIT parameters"""

    # scan valid keywords
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=100000, dataframe=True, dates=True
    )

    validnames = list()

    nact = grid.nactive
    ntot = grid.ntotal

    if grid.dualporo:
        nact *= 2
        ntot *= 2

    # get a list of valid property names
    for kw in list(kwlist.itertuples(index=False, name=None)):
        kwname, _, nlen, _, _ = kw
        if nlen in (nact, ntot) and kwname not in validnames:
            validnames.append(kwname)

    if names == "all":
        usenames = deepcopy(validnames)
    else:
        usenames = list(names)

    for name in usenames:
        if name not in validnames:
            if strictname:
                msg = f"Requested keyword {name} is not in INIT file,"
                msg += f"valid entries are {validnames}"
                raise ValueError(msg)
            else:
                msg = f"Requested keyword {name} is not in INIT file."
                msg += "Will continue due to keyword <strict> settings."
                xtg.warnuser(msg)
                continue

        prop = GridProperty()
        # use a private GridProperty function, since filehandle
        _gridprop_import_eclrun.import_eclbinary(
            prop, pfile, name=name, grid=grid, etype=1, _kwlist=kwlist,
        )

        self._names.append(name)
        self._props.append(prop)

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay


def _import_ecl_output_v2_rsta(
    self, pfile, names, dates, grid, strictname, strictdate, namestyle
):
    """Import RESTART parameters"""

    # scan valid keywords with dates
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=100000, dataframe=True, dates=True
    )

    validnamedatepairs = list()
    validnames = list()
    validdates = list()

    nact = grid.nactive
    ntot = grid.ntotal

    if grid.dualporo:
        nact *= 2
        ntot *= 2

    for kw in list(kwlist.itertuples(index=False, name=None)):
        kwname, _, nlen, _, date = kw
        if nlen in (nact, ntot) and (kwname, date) not in validnamedatepairs:
            validnamedatepairs.append((kwname, date))
        if nlen in (nact, ntot) and kwname not in validnames:
            validnames.append(kwname)
        if nlen in (nact, ntot) and date not in validdates:
            validdates.append(date)

    usenamedatepairs = list()
    if names == "all" and dates != all:
        usenamedatepairs = deepcopy(validnamedatepairs)

    else:
        if names == "all" and dates != all:
            usenames = [namedate[0] for namedate in validnamedatepairs]
            usedates = dates
        elif names != "all" and dates == all:
            usedates = [namedate[1] for namedate in validnamedatepairs]
            usenames = names
        else:
            usedates = dates
            usenames = names

        for name in usenames:
            for date in usedates:
                usenamedatepairs.append((name, date))

    for namedate in usenamedatepairs:
        name, date = namedate
        skipentry = False

        if name not in ("SGAS", "SOIL", "SWAT") and namedate not in validnamedatepairs:
            # saturation keywords are a mess in Eclipse and friends; check later
            if strictname and strictdate:
                msg = f"Keyword data combo {name} {date} is not in RESTART file."
                msg += f"Possible entries are: {validnamedatepairs}"
                raise ValueError(msg)
            else:
                msg = f"Keyword data combo {name} {date} is not in RESTART file."
                xtg.warnuser(msg)
                skipentry = True

            if strictname and not strictdate and name not in validnames:
                msg = f"Keyword name {name} is not in RESTART file."
                msg += f"Possible entries are: {validnames}"
                raise ValueError(msg)
            elif not strictname and strictdate and date not in validdates:
                msg = f"Keyword date {date} is not in RESTART file."
                msg += f"Possible entries are: {validdates}"
                raise ValueError(msg)
            elif not strictname and name not in validnames:
                msg = f"Keyword name {name} is not in RESTART file."
                xtg.warnuser(msg)
                skipentry = True
            elif not strictdate and date not in validdates:
                msg = f"Keyword date {date} is not in RESTART file."
                xtg.warnuser(msg)
                skipentry = True
            else:
                msg = "Keyword skipped for unknown reasons."
                xtg.warnuser(msg)
                skipentry = True

        if skipentry:
            xtg.warnuser("Will continue due to keyword <strict> settings.")
            continue

        prop = GridProperty()

        usename = name + "_" + str(date)
        if namestyle == 1:
            sdate = str(date)
            usename = name + "--" + sdate[0:4] + "_" + sdate[4:6] + "_" + sdate[6:8]

        # use a private GridProperty function, since filehandle
        _gridprop_import_eclrun.import_eclbinary(
            prop, pfile, name=name, date=date, grid=grid, etype=5, _kwlist=kwlist,
        )

        self._names.append(usename)
        self._props.append(prop)

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay
