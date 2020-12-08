"""Importing or export grid or grid props from ROFF, binary."""

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def _rkwquery(gfile, kws, name, swap):
    """Local function for _import_roff_v2, single data."""

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
    """Local function for _import_roff_v2, 3D parameter arrays.

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
    """Local function for returning swig pointers to C arrays when reading files.

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
    logger.info("Reading %s from file...", name)

    if dtype == 1:
        xvec = _cxtgeo.new_intarray(reclen)
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

    logger.info("Reading %s from file done", name)

    gfile.cfclose()
    return xvec


def _rkwxvec_prop(self, gfile, kws, name, swap, strict=True):
    """Local function for making numpy array directly for a prop while reading roff.

    This is made for xtgversion=2

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

    proparr = None
    cfhandle = gfile.get_cfhandle()
    logger.info("Reading %s from file...", name)

    proparr = None
    if dtype == 1:
        proparr = np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.int32)
        _cxtgeo.grdcp3d_imp_roffbin_prop_ivec(
            cfhandle, swap, bytepos, self._ncol, self._nrow, self._nlay, proparr
        )
    elif dtype in (4, 5):
        proparr = np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.int32)
        _cxtgeo.grdcp3d_imp_roffbin_prop_bvec(
            cfhandle, swap, bytepos, self._ncol, self._nrow, self._nlay, proparr
        )
    else:
        gfile.cfclose()
        raise ValueError("Unhandled dtype: {}".format(dtype))

    logger.info("Reading %s from file done", name)

    gfile.cfclose()
    return proparr


def _rkwxvec_coordsv(
    self,
    gfile,
    kws,
    swap,
    xoffset,
    yoffset,
    zoffset,
    xscale,
    yscale,
    zscale,
):
    """Special for importing ROFF binary for COORD type data when _xtgversion=2."""
    name = "cornerLines!data"

    kwtypedict = {"int": 1, "float": 2, "double": 3, "char": 4, "bool": 5, "byte": 6}

    dtype = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            bytepos = items[3]
            break

    if dtype == 0:
        raise ValueError("COORD not present")

    cfhandle = gfile.get_cfhandle()
    logger.info("Reading %s from file...", name)

    status = _cxtgeo.grdcp3d_imp_roffbin_coordsv(
        cfhandle,
        swap,
        bytepos,
        self._ncol + 1,
        self._nrow + 1,
        xoffset,
        yoffset,
        zoffset,
        xscale,
        yscale,
        zscale,
        self._coordsv,
    )

    if status != 0:
        gfile.cfclose()
        raise RuntimeError("Error running _rkwxvec_coordsv")

    logger.info("Reading %s from file done", name)

    gfile.cfclose()


def _rkwxvec_zcornsv(
    self,
    gfile,
    kws,
    swap,
    xoffset,
    yoffset,
    zoffset,
    xscale,
    yscale,
    zscale,
    p_splitenz_v,
):
    """Special for importing ROFF binary for ZCORNS type data when _xtgversion=2."""
    name = "zvalues!data"

    kwtypedict = {"int": 1, "float": 2, "double": 3, "char": 4, "bool": 5, "byte": 6}

    dtype = 0
    bytepos = 1
    for items in kws:
        if name in items[0]:
            dtype = kwtypedict.get(items[1])
            bytepos = items[3]
            break

    if dtype == 0:
        raise ValueError("COORD not present")

    cfhandle = gfile.get_cfhandle()
    logger.info("Reading %s from file...", name)

    status = _cxtgeo.grdcp3d_imp_roffbin_zcornsv(
        cfhandle,
        swap,
        bytepos,
        self._ncol + 1,
        self._nrow + 1,
        self._nlay + 1,
        xoffset,
        yoffset,
        zoffset,
        xscale,
        yscale,
        zscale,
        p_splitenz_v,
        (self._ncol + 1) * (self._nrow + 1) * (self._nlay + 1),
        self._zcornsv,
    )

    if status != 0:
        gfile.cfclose()
        raise RuntimeError("Error running _rkwxvec_coordsv")

    logger.info("Reading %s from file done", name)

    gfile.cfclose()
