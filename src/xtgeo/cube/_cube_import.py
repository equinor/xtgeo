"""Import Cube data via SegyIO library or XTGeo CLIB."""
import json
from collections import OrderedDict
from struct import unpack
from typing import Dict

import numpy as np
import segyio
import xtgeo
import xtgeo.common.calc as xcalc
import xtgeo.common.sys as xsys
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def import_segy(sfile: xtgeo._XTGeoFile) -> Dict:
    """Import SEGY via the SegyIO library.

    Args:
        sfile (str): File name of SEGY file
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals
    sfile = sfile.file

    logger.debug("Inline sorting %s", segyio.TraceSortingFormat.INLINE_SORTING)

    with segyio.open(sfile, "r") as segyfile:
        segyfile.mmap()

        values = segyio.tools.cube(segyfile)

        logger.info(values.dtype)
        if values.dtype != np.float32:
            xtg.warnuser(
                "Values are converted from {} to {}".format(values.dtype, "float32")
            )

            values = values.astype(np.float32)

        if np.isnan(np.sum(values)):
            raise ValueError("The input contains NaN values which is trouble!")

        ilines = segyfile.ilines
        xlines = segyfile.xlines

        ncol, nrow, nlay = values.shape

        trcode = segyio.TraceField.TraceIdentificationCode
        traceidcodes = segyfile.attributes(trcode)[:].reshape(ncol, nrow)

        logger.info("NRCL  %s %s %s", ncol, nrow, nlay)

        # need positions for all 4 corners
        c1v = xcalc.ijk_to_ib(1, 1, 1, ncol, nrow, 1, forder=False)
        c2v = xcalc.ijk_to_ib(ncol, 1, 1, ncol, nrow, 1, forder=False)
        c3v = xcalc.ijk_to_ib(1, nrow, 1, ncol, nrow, 1, forder=False)
        c4v = xcalc.ijk_to_ib(ncol, nrow, 1, ncol, nrow, 1, forder=False)

        clist = [c1v, c2v, c3v, c4v]

        xori = yori = zori = 0.999
        xvv = rotation = 0.999
        xinc = yinc = zinc = 0.999
        yflip = 1

        for inum, cox in enumerate(clist):
            logger.debug(inum)
            origin = segyfile.header[cox][
                segyio.su.cdpx,
                segyio.su.cdpy,
                segyio.su.scalco,
                segyio.su.delrt,
                segyio.su.dt,
                segyio.su.iline,
                segyio.su.xline,
            ]
            # get the data on SU (seismic unix) format
            cdpx = origin[segyio.su.cdpx]
            cdpy = origin[segyio.su.cdpy]
            scaler = origin[segyio.su.scalco]
            if scaler < 0:
                cdpx = -1 * float(cdpx) / scaler
                cdpy = -1 * float(cdpy) / scaler
            else:
                cdpx = cdpx * scaler
                cdpy = cdpy * scaler

            if inum == 0:
                xori = cdpx
                yori = cdpy
                zori = origin[segyio.su.delrt]
                zinc = origin[segyio.su.dt] / 1000.0

            if inum == 1:
                slen, _, rot1 = xcalc.vectorinfo2(xori, cdpx, yori, cdpy)
                xinc = slen / (ncol - 1)

                rotation = rot1
                xvv = (cdpx - xori, cdpy - yori, 0)

            if inum == 2:
                slen, _, _ = xcalc.vectorinfo2(xori, cdpx, yori, cdpy)
                yinc = slen / (nrow - 1)

                # find YFLIP by cross products
                yvv = (cdpx - xori, cdpy - yori, 0)

                yflip = xcalc.find_flip(xvv, yvv)

        logger.debug("XTGeo rotation is %s", rotation)

    # attributes to update
    return {
        "ilines": ilines,
        "xlines": xlines,
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "xori": xori,
        "xinc": xinc,
        "yori": yori,
        "yinc": yinc,
        "zori": zori,
        "zinc": zinc,
        "rotation": rotation,
        "values": values,
        "yflip": yflip,
        "segyfile": sfile,
        "traceidcodes": traceidcodes,
    }


def _scan_segy_header(sfile, outfile):

    ptr_gn_bitsheader = _cxtgeo.new_intpointer()
    ptr_gn_formatcode = _cxtgeo.new_intpointer()
    ptr_gf_segyformat = _cxtgeo.new_floatpointer()
    ptr_gn_samplespertrace = _cxtgeo.new_intpointer()
    ptr_gn_measuresystem = _cxtgeo.new_intpointer()

    _cxtgeo.cube_scan_segy_hdr(
        sfile,
        ptr_gn_bitsheader,
        ptr_gn_formatcode,
        ptr_gf_segyformat,
        ptr_gn_samplespertrace,
        ptr_gn_measuresystem,
        1,
        outfile,
    )
    gn_bitsheader = _cxtgeo.intpointer_value(ptr_gn_bitsheader)
    logger.info("Scan SEGY header ... %s bytes ... DONE", gn_bitsheader)


def _scan_segy_trace(sfile, outfile):

    ptr_gn_bitsheader = _cxtgeo.new_intpointer()
    ptr_gn_formatcode = _cxtgeo.new_intpointer()
    ptr_gf_segyformat = _cxtgeo.new_floatpointer()
    ptr_gn_samplespertrace = _cxtgeo.new_intpointer()
    ptr_gn_measuresystem = _cxtgeo.new_intpointer()

    _cxtgeo.cube_scan_segy_hdr(
        sfile,
        ptr_gn_bitsheader,
        ptr_gn_formatcode,
        ptr_gf_segyformat,
        ptr_gn_samplespertrace,
        ptr_gn_measuresystem,
        0,
        outfile,
    )

    gn_bitsheader = _cxtgeo.intpointer_value(ptr_gn_bitsheader)
    gn_formatcode = _cxtgeo.intpointer_value(ptr_gn_formatcode)
    gf_segyformat = _cxtgeo.floatpointer_value(ptr_gf_segyformat)
    gn_samplespertrace = _cxtgeo.intpointer_value(ptr_gn_samplespertrace)

    logger.info("Scan SEGY header ... %s bytes ... DONE", gn_bitsheader)
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_xori = _cxtgeo.new_doublepointer()
    ptr_yori = _cxtgeo.new_doublepointer()
    ptr_zori = _cxtgeo.new_doublepointer()
    ptr_xinc = _cxtgeo.new_doublepointer()
    ptr_yinc = _cxtgeo.new_doublepointer()
    ptr_zinc = _cxtgeo.new_doublepointer()
    ptr_rotation = _cxtgeo.new_doublepointer()
    ptr_minval = _cxtgeo.new_doublepointer()
    ptr_maxval = _cxtgeo.new_doublepointer()
    ptr_dummy = _cxtgeo.new_floatpointer()
    ptr_yflip = _cxtgeo.new_intpointer()
    ptr_zflip = _cxtgeo.new_intpointer()

    logger.debug("Scan via C wrapper...")
    _cxtgeo.cube_import_segy(
        sfile,
        # input
        gn_bitsheader,
        gn_formatcode,
        gf_segyformat,
        gn_samplespertrace,
        # result (as pointers)
        ptr_ncol,
        ptr_nrow,
        ptr_nlay,
        ptr_dummy,
        ptr_xori,
        ptr_xinc,
        ptr_yori,
        ptr_yinc,
        ptr_zori,
        ptr_zinc,
        ptr_rotation,
        ptr_yflip,
        ptr_zflip,
        ptr_minval,
        ptr_maxval,
        # options
        1,
        1,
        outfile,
    )

    logger.debug("Scan via C wrapper... done")


def import_stormcube(
    sfile: xtgeo._XTGeoFile,
) -> Dict:
    """Import on StormCube format."""
    # The ASCII header has all the metadata on the form:
    # ---------------------------------------------------------------------
    # storm_petro_binary       // always
    #
    # 0 ModelFile -999 // zonenumber, source_of_file,  undef_value
    #
    # UNKNOWN // name_of_parameter?
    #
    # 452638.45298827 6262.499 6780706.6462283 10762.4999 1800 2500 0 0
    # 700 -0.80039470880765
    #
    # 501 861 140
    # ---------------------------------------------------------------------
    # The rest is float32 binary data, I (column fastest), then J, then K
    # a total of ncol * nrow * nlay

    # Scan the header with Python; then use CLIB for the binary data
    sfile = str(sfile.file)
    with open(sfile, "rb") as stf:

        iline = 0

        ncol = nrow = nlay = nlines = 1
        xori = yori = zori = xinc = yinc = rotation = rot = 0.999
        xlen = ylen = zlen = 0.999

        for line in range(10):
            xline = stf.readline()
            if not xline.strip():
                continue

            iline += 1
            if iline == 1:
                pass
            elif iline == 2:
                _, _, _ = xline.strip().split()
            elif iline == 3:
                pass
            elif iline == 4:
                (xori, xlen, yori, ylen, zori, _, _, _) = xline.strip().split()
            elif iline == 5:
                zlen, rot = xline.strip().split()
            elif iline == 6:
                ncol, nrow, nlay = xline.strip().split()
                nlines = line + 2
                break

    ncol = int(ncol)
    nrow = int(nrow)
    nlay = int(nlay)
    nrcl = ncol * nrow * nlay

    xori = float(xori)
    yori = float(yori)
    zori = float(zori)

    rotation = float(rot)
    if rotation < 0:
        rotation += 360

    xinc = float(xlen) / ncol
    yinc = float(ylen) / nrow
    zinc = float(zlen) / nlay

    yflip = 1

    if yinc < 0:
        yflip = -1
        yinc = yinc * yflip  # not sure if this will ever happen

    ier, values = _cxtgeo.cube_import_storm(ncol, nrow, nlay, sfile, nlines, nrcl, 0)

    if ier != 0:
        raise RuntimeError(
            "Something went wrong in {}, code is {}".format(__name__, ier)
        )

    return {
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "xori": xori,
        "xinc": xinc,
        "yori": yori,
        "yinc": yinc,
        "zori": zori,
        "zinc": zinc,
        "rotation": rotation,
        "values": values.reshape((ncol, nrow, nlay)),
        "yflip": yflip,
    }


def import_xtgregcube(mfile, values=True):
    """Using pure python for experimental cube import, xtgregsurf format."""
    logger.info("Importing cube on xtgregcube format...")

    offset = 36
    with open(mfile.file, "rb") as fhandle:
        buf = fhandle.read(offset)

    # unpack header
    swap, magic, nfloat, ncol, nrow, nlay = unpack("= i i i q q q", buf)

    if swap != 1 or magic != 1201:
        raise ValueError("Invalid file format (wrong swap id or magic number).")

    dtype = np.float32 if nfloat == 4 else np.float64

    vals = None
    narr = ncol * nrow * nlay

    if values:
        vals = xsys.npfromfile(mfile.file, dtype=dtype, count=narr, offset=offset)

    # read metadata which will be at position offet + nfloat*narr +13
    pos = offset + nfloat * narr + 13

    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)
    req = meta["_required_"]

    reqattrs = xtgeo.MetaDataRegularCube.REQUIRED

    results = {myattr: req[myattr] for myattr in reqattrs}

    # For backwards compatability, xtgeo outputs files with the undef field set
    # although we do not support initializing with any other value.
    # As xtgeo-format is only written/read by xtgeo as far as we know, this should
    # be unproblematic for now.
    if results.pop("undef", None) != xtgeo.UNDEF:
        raise ValueError(
            f"File {mfile.file} has non-standard undef, not supported by xtgeo"
        )

    # TODO: dead traces and traceidcodes
    if values:
        results["values"] = vals.reshape(
            results["ncol"], results["nrow"], results["nlay"]
        )

    return results
