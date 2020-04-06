"""Importing grid props from ROFF, binary"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

from . import _grid3d_utils as utils
from . import _gridprop_lowlevel

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
    byteswap = _rkwquery(pfile, kwords, "filedata!byteswaptest", -1)

    ncol = _rkwquery(pfile, kwords, "dimensions!nX", byteswap)
    nrow = _rkwquery(pfile, kwords, "dimensions!nY", byteswap)
    nlay = _rkwquery(pfile, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", ncol, nrow, nlay)

    # get the actual parameter:
    vals = _rarraykwquery(
        pfile, kwords, "parameter!name!" + name, byteswap, ncol, nrow, nlay
    )

    self._values = vals
    self._name = name

    # if vals.dtype != "float":
    #     subs = _rkwxlist(pfile, kwords, "subgrids!nLayers", byteswap, strict=False)

    pfile.cfclose()


def _rkwquery(gfile, kws, name, swap):
    """Local function for _import_roff_v2, single data"""

    kwtypedict = {"int": 1, "float": 2}
    iresult = _cxtgeo.new_intpointer()
    presult = _cxtgeo.new_floatpointer()

    dtype = 0
    reclen = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    logger.debug("DTYPE is %s", dtype)

    if dtype == 0:
        raise ValueError("Cannot find property <{}> in file".format(name))

    if reclen != 1:
        raise SystemError("Stuff is rotten here...")

    _cxtgeo.grd3d_imp_roffbin_data(
        gfile.get_cfhandle(), swap, dtype, bytepos, iresult, presult
    )

    gfile.cfclose()

    # -1 indicates that it is the swap flag which is looked for!
    if dtype == 1:
        xresult = _cxtgeo.intpointer_value(iresult)
        if swap == -1:
            if xresult == 1:
                return 0
            return 1

    elif dtype == 2:
        xresult = _cxtgeo.floatpointer_value(presult)

    return xresult


def _rarraykwquery(gfile, kws, name, swap, ncol, nrow, nlay):
    """
    Local function for _import_roff_v2, 3D parameter arrays.

    This parameters are translated to numpy data for the values
    attribute usage.

    Note from scan:
    parameter!name!PORO   char        1            310
    parameter!data        float   35840            336

    Hence it is the parameter!data which comes after parameter!name!PORO which
    is relevant here, given that name = PORO.

    """

    kwtypedict = {"int": 1, "float": 2, "double": 3, "byte": 5}

    dtype = 0
    reclen = 0
    bytepos = 1
    namefound = False
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            namefound = True
        if "parameter!data" in items[0] and namefound:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    if dtype == 0:
        raise ValueError("Cannot find property <{}> in file".format(name))

    if reclen <= 1:
        raise SystemError("Stuff is rotten here...")

    inumpy = np.zeros(ncol * nrow * nlay, dtype=np.int32)
    fnumpy = np.zeros(ncol * nrow * nlay, dtype=np.float32)

    _cxtgeo.grd3d_imp_roffbin_arr(
        gfile.get_cfhandle(), swap, ncol, nrow, nlay, bytepos, dtype, fnumpy, inumpy
    )

    gfile.cfclose()

    if dtype == 1:
        vals = inumpy
        vals = ma.masked_greater(vals, xtgeo.UNDEF_INT_LIMIT)
        del fnumpy
        del inumpy
    elif dtype == 2:
        vals = fnumpy
        vals = ma.masked_greater(vals, xtgeo.UNDEF_LIMIT)
        vals = vals.astype(np.float64)
        del fnumpy
        del inumpy

    vals = vals.reshape(ncol, nrow, nlay)
    return vals


def _rkwxlist(gfile, kws, name, swap, strict=True):
    """Local function for _import_roff_v2, 1D arrays such as subgrids.

    This parameters are translated to numpy data for the values
    attribute usage.

    Note from scan:
      tag subgrids
      array int nLayers 3
           20           20           16
    """

    kwtypedict = {"int": 1}  # only int lists are supported

    dtype = 0
    reclen = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    if dtype == 0:
        if strict:
            raise ValueError("Cannot find property <{}> in file".format(name))

        return None

    if dtype == 1:
        inumpy = np.zeros(reclen, dtype=np.int32)
        _cxtgeo.grd3d_imp_roffbin_ilist(gfile.get_cfhandle(), swap, bytepos, inumpy)
        gfile.cfclose()
    else:
        raise ValueError("Unsupported data type for lists: {} in file".format(dtype))

    return inumpy


def _rkwxvec(gfile, kws, name, swap, strict=True):
    """Local function for returning swig pointers to C arrays.

    If strict is True, a ValueError will be raised if keyword is not
    found. If strict is False, None will be returned
    """

    kwtypedict = {"int": 1, "float": 2, "double": 3, "char": 4, "bool": 5, "byte": 6}

    dtype = 0
    reclen = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            reclen = items[2]
            bytepos = items[3]
            break

    if dtype == 0:
        if strict:
            raise ValueError("Cannot find property <{}> in file".format(name))

        return None

    if reclen <= 1:
        raise SystemError("Stuff is rotten here...")

    xvec = None
    cfhandle = gfile.get_cfhandle()

    if dtype == 1:
        xvec = _cxtgeo.new_floatarray(reclen)
        _cxtgeo.grd3d_imp_roffbin_ivec(cfhandle, swap, bytepos, xvec, reclen)

    elif dtype == 2:
        xvec = _cxtgeo.new_floatarray(reclen)
        _cxtgeo.grd3d_imp_roffbin_fvec(cfhandle, swap, bytepos, xvec, reclen)

    elif dtype >= 4:
        xvec = _cxtgeo.new_intarray(reclen)  # convert char/byte/bool to int
        _cxtgeo.grd3d_imp_roffbin_bvec(cfhandle, swap, bytepos, xvec, reclen)

    else:
        gfile.cfclose()
        raise ValueError("Unhandled dtype: {}".format(dtype))

    gfile.cfclose()
    return xvec
