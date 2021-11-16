# coding: utf-8
"""Private module, Grid Import private functions for xtgeo based formats."""

import json
from collections import OrderedDict
from struct import unpack

import h5py
import numpy as np

import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def convert_subgrids(sdict):
    """Convert a simplified ordered dictionary to
        subgrid required by Grid.

    The simplified dictionary is on the form
    {"name1": 3, "name2": 5}

    Note that the input must be an OrderedDict!

    """
    if sdict is None:
        return None

    if not isinstance(sdict, OrderedDict):
        raise ValueError("Input sdict is not an OrderedDict")

    newsub = OrderedDict()

    inn1 = 1
    for name, nsub in sdict.items():
        inn2 = inn1 + nsub
        newsub[name] = range(inn1, inn2)
        inn1 = inn2

    return newsub


def handle_metadata(result, meta, ncol, nrow, nlay):

    # meta _optional_ *may* contain xshift, xscale etc which in case must be taken
    # into account
    coordsv = result["coordsv"]
    zcornsv = result["zcornsv"]
    actnumsv = result["actnumsv"]
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1
    req = meta["_required_"]
    shi = req["xshift"]
    coordsv[0::3] = np.where(shi != 0, coordsv[0::3] + shi, coordsv[0::3])
    sca = req["xscale"]
    coordsv[0::3] = np.where(sca != 1, coordsv[0::3] * sca, coordsv[0::3])
    shi = req["yshift"]
    coordsv[1::3] = np.where(shi != 0, coordsv[1::3] + shi, coordsv[1::3])
    sca = req["yscale"]
    coordsv[1::3] = np.where(sca != 1, coordsv[1::3] * sca, coordsv[1::3])
    shi = req["zshift"]
    coordsv[2::3] = np.where(shi != 0, coordsv[2::3] + shi, coordsv[2::3])
    sca = req["zscale"]
    coordsv[2::3] = np.where(sca != 1, coordsv[2::3] * sca, coordsv[2::3])

    result["coordsv"] = coordsv.reshape((nncol, nnrow, 6)).astype(np.float64)
    result["zcornsv"] = zcornsv.reshape((nncol, nnrow, nnlay, 4)).astype(np.float32)
    result["actnumsv"] = actnumsv.reshape((ncol, nrow, nlay)).astype(np.int32)
    if "subgrids" in req:
        result["subgrids"] = convert_subgrids(req["subgrids"])


def import_xtgcpgeom(
    mfile, mmap=False
):  # pylint: disable=too-many-locals, too-many-statements
    """Using pure python for experimental grid geometry import."""
    #
    offset = 36
    with open(mfile.file, "rb") as fhandle:
        buf = fhandle.read(offset)

    # unpack header
    swap, magic, nformat, ncol, nrow, nlay = unpack("= i i i q q q", buf)
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    if swap != 1 or magic != 1301:
        raise ValueError(f"Error, swap magic are {swap} {magic}, expected is 1 1301")

    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    if nformat not in (444, 844, 841, 881, 884):
        raise ValueError(f"The subformat value {nformat} is not valid")

    coordfmt, zcornfmt, actnumfmt = [int(nbyte) for nbyte in str(nformat)]

    dtype_coordsv = "float" + str(coordfmt * 8)
    dtype_zcornsv = "float" + str(zcornfmt * 8)
    dtype_actnumv = "int" + str(actnumfmt * 8)

    ncoord = nncol * nnrow * 6
    nzcorn = nncol * nnrow * nnlay * 4
    nactnum = ncol * nrow * nlay
    result = dict()
    # read numpy arrays from file
    result["coordsv"] = xsys.npfromfile(
        mfile.file, dtype=dtype_coordsv, count=ncoord, offset=offset, mmap=mmap
    )
    newoffset = offset + ncoord * coordfmt
    result["zcornsv"] = xsys.npfromfile(
        mfile.file, dtype=dtype_zcornsv, count=nzcorn, offset=newoffset, mmap=mmap
    )
    newoffset += nzcorn * zcornfmt
    result["actnumsv"] = xsys.npfromfile(
        mfile.file, dtype=dtype_actnumv, count=nactnum, offset=newoffset, mmap=mmap
    )
    newoffset += nactnum * actnumfmt

    # read metadata which will be at position offet + nfloat*narr +13
    pos = newoffset + 13
    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)

    handle_metadata(result, meta, ncol, nrow, nlay)
    return result


def import_hdf5_cpgeom(mfile, ijkrange=None, zerobased=False):
    """Experimental grid geometry import using hdf5."""
    #
    with h5py.File(mfile.name, "r") as h5h:

        grp = h5h["CornerPointGeometry"]

        idcode = grp.attrs["format-idcode"]
        provider = grp.attrs["provider"]
        if idcode != 1301:
            raise ValueError(f"Wrong id code: {idcode}")
        logger.info("Provider is %s", provider)

        jmeta = grp.attrs["metadata"]
        meta = json.loads(jmeta, object_pairs_hook=OrderedDict)

        req = meta["_required_"]
        ncol = req["ncol"]
        nrow = req["nrow"]
        nlay = req["nlay"]

        if ijkrange is not None:
            incoord, inzcorn, inactnum, ncol, nrow, nlay = _partial_read(
                h5h, req, ijkrange, zerobased
            )
            if (
                "subgrids" in meta["_required_"]
                and meta["_required_"]["subgrids"] is not None
            ):
                meta["_required_"]["subgrids"] = filter_subgrids_partial(
                    meta["_required_"]["subgrids"], *ijkrange[-2:], nlay, zerobased
                )
        else:
            incoord = grp["coord"][:, :, :]
            inzcorn = grp["zcorn"][:, :, :, :]
            inactnum = grp["actnum"][:, :, :]

    result = {}
    result["coordsv"] = incoord.astype("float64")
    result["zcornsv"] = inzcorn.astype("float32")
    result["actnumsv"] = inactnum.astype("float32")

    handle_metadata(result, meta, ncol, nrow, nlay)
    return result


def filter_subgrids_partial(subgrids, k1, k2, nlay, zerobased):
    """
    Filters and truncates the subgrids of the global grid so that they
    refer to the filtered grid.

    >>> filter_subgrids_partial(
    ...   {"subgrid1": 4, "subgrid2": 1, "subgrid3":9},
    ...   4,
    ...   5,
    ...   12,
    ...   True
    ... )
    OrderedDict([('subgrid2', 1), ('subgrid3', 1)])

    Args:
        subgrids: The OrderedDict of subgrids.
        k1: Start of subgrid layers (can be "min" to mean 0 or 1 dependent on zerobased)
        k2: End of subgrid layers (cna be "max" to mean nlay or nlay -1
            dependent on zerobased.
        nlay: Original number of layers
        zerobased: Whether indexing starts with 0 or 1.

    Returns:
        New Orderedict which is the subgrids dictionary with out of range
        subgrids removed or truncated.

    """
    if k1 == "min":
        k1 = 0 if zerobased else 1

    if k2 == "max":
        k2 = nlay - 1 if zerobased else nlay

    # convert k1 and k2 to zero based
    if not zerobased:
        k1 -= 1
        k2 -= 1

    partial_subgrid = OrderedDict()
    start = 0
    for key, value in subgrids.items():
        end = value + start
        partial_start = max(start, k1)
        partial_end = min(end, k2 + 1)
        if partial_end - partial_start > 0:
            partial_subgrid[key] = partial_end - partial_start
        start = end
    return partial_subgrid


def _partial_read(h5h, req, ijkrange, zerobased):
    """Read a partial IJ range."""
    ncol = req["ncol"]
    nrow = req["nrow"]
    nlay = req["nlay"]

    if len(ijkrange) != 6:
        raise ValueError("The ijkrange list must have 6 elements")

    i1, i2, j1, j2, k1, k2 = ijkrange

    if i1 == "min":
        i1 = 0 if zerobased else 1
    if j1 == "min":
        j1 = 0 if zerobased else 1
    if k1 == "min":
        k1 = 0 if zerobased else 1

    if i2 == "max":
        i2 = ncol - 1 if zerobased else ncol
    if j2 == "max":
        j2 = nrow - 1 if zerobased else nrow
    if k2 == "max":
        k2 = nlay - 1 if zerobased else nlay

    if not zerobased:
        i1 -= 1
        i2 -= 1
        j1 -= 1
        j2 -= 1
        k1 -= 1
        k2 -= 1

    ncol2 = i2 - i1 + 1
    nrow2 = j2 - j1 + 1
    nlay2 = k2 - k1 + 1

    if (
        ncol2 < 1
        or ncol2 > ncol
        or nrow2 < 1
        or nrow2 > nrow
        or nlay2 < 1
        or nlay2 > nlay
    ):
        raise ValueError("The ijkrange spesification exceeds boundaries.")

    nncol2 = ncol2 + 1
    nnrow2 = nrow2 + 1
    nnlay2 = nlay2 + 1

    dset = h5h["CornerPointGeometry/coord"]
    cv = dset[i1 : i1 + nncol2, j1 : j1 + nnrow2, :]

    dset = h5h["CornerPointGeometry/zcorn"]
    zv = dset[i1 : i1 + nncol2, j1 : j1 + nnrow2, k1 : k1 + nnlay2, :]

    dset = h5h["CornerPointGeometry/actnum"]
    av = dset[i1 : i1 + ncol2, j1 : j1 + nrow2, k1 : k1 + nlay2]

    return cv, zv, av, ncol2, nrow2, nlay2
