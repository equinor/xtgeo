# -*- coding: utf-8 -*-
"""Import/export of grid properties (cf GridProperties class)"""

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import _get_fhandle, _close_fhandle

from xtgeo.grid3d import _gridprop_import

from .grid_property import GridProperty
from . import _grid3d_utils as utils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

XTGDEBUG = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file("NONE")


def import_ecl_output(
    props, pfile, names=None, dates=None, grid=None, namestyle=0
):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    logger.debug("'namestyle' is %s (not in use)", namestyle)

    if not grid:
        raise ValueError("Grid Geometry object is missing")

    if not names:
        raise ValueError("Name list is empty (None)")

    fhandle, pclose = _get_fhandle(pfile)

    # scan valid keywords
    kwlist = utils.scan_keywords(fhandle)

    usenames = list()

    if names == "all":
        nact = grid.nactive
        ntot = grid.ntotal

        for kw in kwlist:
            kwname, _tmp1, nlen, _bs1 = kw
            if nlen in (nact, ntot):
                usenames.append(kwname)
    else:
        usenames = list(names)

    logger.info("NAMES are %s", usenames)

    lookfornames = list(set(usenames))

    possiblekw = []
    for name in lookfornames:
        namefound = False
        for kwitem in kwlist:
            possiblekw.append(kwitem[0])
            if name == kwitem[0]:
                namefound = True
        if not namefound:
            if name == "SOIL":
                pass  # will check for SWAT and SGAS later
            else:
                raise ValueError(
                    "Keyword {} not found. Possible list: {}".format(name, possiblekw)
                )

    # check valid dates, and remove invalid entries (allowing that user
    # can be a bit sloppy on DATES)

    validdates = [None]
    if dates:
        dlist = utils.scan_dates(fhandle)

        validdates = []
        alldates = []
        for date in dates:
            for ditem in dlist:
                alldates.append(str(ditem[1]))
                if str(date) == str(ditem[1]):
                    validdates.append(date)

        if not validdates:
            msg = "No valid dates given (dates: {} vs {})".format(dates, alldates)
            xtg.error(msg)
            raise ValueError(msg)

        if len(dates) > len(validdates):
            invalidddates = list(set(dates).difference(validdates))
            msg = (
                "In file {}: Some dates not found: {}, but will continue "
                "with dates: {}".format(pfile, invalidddates, validdates)
            )
            xtg.warn(msg)
            # raise DateNotFoundError(msg)

    use2names = list(usenames)  # to make copy

    logger.info("Use names: %s", use2names)
    logger.info("Valid dates: %s", validdates)

    # now import each property
    firstproperty = True

    for date in validdates:
        # xprop = dict()
        # soil_ok = False

        for name in use2names:

            if date is None:
                date = None
                propname = name
                etype = 1
            else:
                propname = name + "_" + str(date)
                etype = 5

            prop = GridProperty()

            # use a private GridProperty function here, for convinience
            # (since filehandle)
            ier = _gridprop_import.import_eclbinary(
                prop, fhandle, name=name, date=date, grid=grid, etype=etype
            )
            if ier != 0:
                raise ValueError(
                    "Something went wrong, IER = {} while "
                    "name={}, date={}, etype={}, propname={}".format(
                        ier, name, date, etype, propname
                    )
                )

            if firstproperty:
                ncol = prop.ncol
                nrow = prop.nrow
                nlay = prop.nlay
                firstproperty = False

            logger.info("Appended property %s", propname)
            props._names.append(propname)
            props._props.append(prop)

    props._ncol = ncol
    props._nrow = nrow
    props._nlay = nlay

    if validdates[0] != 0:
        props._dates = validdates

    _close_fhandle(fhandle, pclose)
