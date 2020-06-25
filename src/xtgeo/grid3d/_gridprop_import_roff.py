"""Importing grid props from ROFF, binary"""

from __future__ import print_function, absolute_import

import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

from . import _grid3d_utils as utils
from . import _gridprop_lowlevel
from . import _grid_roff_lowlevel as grl

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(self, pfile, name, grid=None, _roffapiv=1):
    """Import ROFF format"""

    logger.info("Keyword grid is inactive, values is: %s", grid)

    if _roffapiv <= 1:
        _import_roff_v1(self, pfile, name)
    elif _roffapiv == 2:
        _import_roff_v2(self, pfile, name)


def _import_roff_v1(self, pfile, name):
    """Import ROFF format, version 1"""
    # pylint: disable=too-many-locals

    # there is a todo here to get it more robust for various cases,
    # e.g. that a ROFF file may contain both a grid an numerous
    # props

    logger.info("Looking for %s in file %s", name, pfile.name)

    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_ncodes = _cxtgeo.new_intpointer()
    ptr_type = _cxtgeo.new_intpointer()

    ptr_idum = _cxtgeo.new_intpointer()
    ptr_ddum = _cxtgeo.new_doublepointer()

    # read with mode 0, to scan for ncol, nrow, nlay and ndcodes, and if
    # property is found...
    ier, _codenames = _cxtgeo.grd3d_imp_prop_roffbin(
        pfile.name,
        0,
        ptr_type,
        ptr_ncol,
        ptr_nrow,
        ptr_nlay,
        ptr_ncodes,
        name,
        ptr_idum,
        ptr_ddum,
        ptr_idum,
        0,
    )

    if ier == -1:
        msg = "Cannot find property name {}".format(name)
        logger.warning(msg)
        raise SystemExit("Error from ROFF import")

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    self._ncodes = _cxtgeo.intpointer_value(ptr_ncodes)

    ptype = _cxtgeo.intpointer_value(ptr_type)

    ntot = self._ncol * self._nrow * self._nlay

    if self._ncodes <= 1:
        self._ncodes = 1
        self._codes = {0: "undef"}

    logger.debug("Number of codes: %s", self._ncodes)

    # allocate

    if ptype == 1:  # float, assign to double
        ptr_pval_v = _cxtgeo.new_doublearray(ntot)
        ptr_ival_v = _cxtgeo.new_intarray(1)
        self._isdiscrete = False
        self._dtype = "float64"

    elif ptype > 1:
        ptr_pval_v = _cxtgeo.new_doublearray(1)
        ptr_ival_v = _cxtgeo.new_intarray(ntot)
        self._isdiscrete = True
        self._dtype = "int32"

    # number of codes and names
    ptr_ccodes_v = _cxtgeo.new_intarray(self._ncodes)

    # NB! note the SWIG trick to return modified char values; use cstring.i
    # inn the config and %cstring_bounded_output(char *p_codenames_v, NN);
    # Then the argument for *p_codevalues_v in C is OMITTED here!

    ier, cnames = _cxtgeo.grd3d_imp_prop_roffbin(
        pfile.name,
        1,
        ptr_type,
        ptr_ncol,
        ptr_nrow,
        ptr_nlay,
        ptr_ncodes,
        name,
        ptr_ival_v,
        ptr_pval_v,
        ptr_ccodes_v,
        0,
    )

    if self._isdiscrete:
        _gridprop_lowlevel.update_values_from_carray(
            self, ptr_ival_v, np.int32, delete=True
        )
    else:
        _gridprop_lowlevel.update_values_from_carray(
            self, ptr_pval_v, np.float64, delete=True
        )

    # now make dictionary of codes
    if self._isdiscrete:
        cnames = cnames.replace(";", "")
        cname_list = cnames.split("|")
        cname_list.pop()  # some rubbish as last entry
        ccodes = []
        for ino in range(0, self._ncodes):
            ccodes.append(_cxtgeo.intarray_getitem(ptr_ccodes_v, ino))

        self._codes = dict(zip(ccodes, cname_list))

    self._name = name


def _import_roff_v2(self, pfile, name):
    """Import ROFF format, version 2 (improved version)"""

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

    pfile.get_cfhandle()

    kwords = utils.scan_keywords(pfile, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = grl._rkwquery(pfile, kwords, "filedata!byteswaptest", -1)

    ncol = grl._rkwquery(pfile, kwords, "dimensions!nX", byteswap)
    nrow = grl._rkwquery(pfile, kwords, "dimensions!nY", byteswap)
    nlay = grl._rkwquery(pfile, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", ncol, nrow, nlay)

    # get the actual parameter:
    vals = grl._rarraykwquery(
        pfile, kwords, "parameter!name!" + name, byteswap, ncol, nrow, nlay
    )

    self._values = vals
    self._name = name

    # if vals.dtype != "float":
    #     subs = _rkwxlist(pfile, kwords, "subgrids!nLayers", byteswap, strict=False)

    pfile.cfclose()
