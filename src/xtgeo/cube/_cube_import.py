"""Import Cube data via SegyIO library or XTGeo CLIB.

Data model in XTGeo illustrated by example: ncol=3, nrow=4 (non-rotated):

                    3              7                    ^ "J, ~NORTH" direction
  xline=1022       +--------------+--------------+11    |     (if unrotated)
                   |              |              |      |
                   |              |              |      |
                   |              |              |
                   |2             |6             |10
  xline=1020       +--------------+--------------+      Rotation is school angle of
                   |              |              |      "I east" vs X axis
                   |              |              |
                   |              |              |
                   |1             |5             |9
  xline=1018       +--------------+--------------+
                   |              |              |
                   |              |              |
                   |              |              |
                   |0             |4             |8
  xline=1016       +--------------+--------------+      ------> "I, ~EAST" direction

                iline=4400     iline=4401     iline=4402

Indices is fastest along "J" (C order), and xlines and ilines spacing may vary (2 for
xline, 1 for iline in this example) but shall be constant per axis

"""
import json
from collections import OrderedDict, defaultdict
from copy import deepcopy
from struct import unpack
from typing import Dict, List, Tuple
from warnings import warn

import numpy as np
import segyio
import xtgeo
import xtgeo.common.calc as xcalc
import xtgeo.common.sys as xsys
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from segyio import TraceField as TF
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def import_segy(sfile: xtgeo._XTGeoFile) -> Dict:
    """Import SEGY via the SegyIO library.

    Args:
        sfile: File object for SEGY file
    """
    sfile = sfile.file

    attributes = dict()

    try:
        # cube with all traces present
        with segyio.open(sfile, "r") as segyfile:
            attributes = _import_segy_all_traces(segyfile)
    except ValueError as verr:
        if any([word in str(verr) for word in ["Invalid dimensions", "inconsistent"]]):
            # cube with missing traces is now handled but his is complex, hence users
            # shall be warned. With more experience, this warning can be removed.
            warn(
                "Missing or inconsistent traces in SEGY detected, xtgeo will try "
                "to import by infilling, but please check result carefully!",
                UserWarning,
            )

            with segyio.open(sfile, "r", ignore_geometry=True) as segyfile:
                attributes = _import_segy_incomplete_traces(segyfile)
        else:
            raise
    except Exception as anyerror:  # catch the rest
        raise IOError(f"Cannot parse SEGY file: {str(anyerror)}") from anyerror

    if not attributes:
        raise ValueError("Could not get attributes for segy file")

    attributes["segyfile"] = sfile
    return attributes


def _import_segy_all_traces(segyfile: segyio.segy.SegyFile) -> Dict:
    """Import a a full cube SEGY via the SegyIO library to xtgeo format spec.

    Here, the segyio.tools.cube function can be applied

    Args:
        segyfile: Filehandle from segyio
    """
    segyfile.mmap()

    values = _process_cube_values(segyio.tools.cube(segyfile))

    ilines = segyfile.ilines
    xlines = segyfile.xlines

    ncol, nrow, nlay = values.shape

    # get additional but required geometries for xtgeo; xori, yori, xinc, ..., rotation
    attrs = _segy_all_traces_attributes(segyfile, ncol, nrow, nlay)

    attrs["ilines"] = ilines
    attrs["xlines"] = xlines
    attrs["values"] = values

    return attrs


def _process_cube_values(values: np.ndarray) -> np.ndarray:
    """Helper function to validate/check values."""
    if values.dtype != np.float32:
        xtg.warnuser(f"Values are converted from {values.dtype} to float32")
        values = values.astype(np.float32)
    if np.any(np.isnan(values)):
        raise ValueError(
            f"The input values: {values} contains NaN values which is currently "
            "not handled!"
        )

    return values


def _segy_all_traces_attributes(
    segyfile: segyio.segy.SegyFile, ncol, nrow, nlay
) -> Dict:
    """Get the geometrical values xtgeo needs for a cube definition."""
    trcode = segyio.TraceField.TraceIdentificationCode
    traceidcodes = segyfile.attributes(trcode)[:].reshape(ncol, nrow)

    # need positions in corners for making vectors to compute geometries
    c1v = xcalc.ijk_to_ib(1, 1, 1, ncol, nrow, 1, forder=False)
    c2v = xcalc.ijk_to_ib(ncol, 1, 1, ncol, nrow, 1, forder=False)
    c3v = xcalc.ijk_to_ib(1, nrow, 1, ncol, nrow, 1, forder=False)

    xori, yori, zori, zinc = _get_coordinate(segyfile, c1v)
    point_x1, point_y1, _, _ = _get_coordinate(segyfile, c2v)
    point_x2, point_y2, _, _ = _get_coordinate(segyfile, c3v)

    slen1, _, rotation = xcalc.vectorinfo2(xori, point_x1, yori, point_y1)
    xinc = slen1 / (ncol - 1)

    slen2, _, _ = xcalc.vectorinfo2(xori, point_x2, yori, point_y2)
    yinc = slen2 / (nrow - 1)

    # find YFLIP by cross products
    yflip = xcalc.find_flip(
        (point_x1 - xori, point_y1 - yori, 0), (point_x2 - xori, point_y2 - yori, 0)
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
        "yflip": yflip,
        "traceidcodes": traceidcodes,
    }


def _import_segy_incomplete_traces(segyfile: segyio.segy.SegyFile) -> Dict:
    """Import a a cube SEGY with incomplete traces via the SegyIO library.

    Note that the undefined value will be xtgeo.UNDEF (large number)!

    It is also logical to treat missing traces as dead traces, i.e. the should
    get value 2 in traceidcodes

    Args:
        segyfile: Filehandle from segyio
    """
    segyfile.mmap()
    # get data (which will need padding later for missing traces)
    data = segyfile.trace.raw[:]

    trcode = segyio.TraceField.TraceIdentificationCode
    traceidcodes_input = segyfile.attributes(trcode)[:]

    ilines_case = np.array([h[TF.INLINE_3D] for h in segyfile.header])
    xlines_case = np.array([h[TF.CROSSLINE_3D] for h in segyfile.header])

    # detect minimum inline and xline spacing (e.g.sampling could be every second)
    idiff = np.diff(ilines_case)
    xdiff = np.diff(xlines_case)
    ispacing = int(np.abs(idiff[idiff != 0]).min())
    xspacing = int(np.abs(xdiff[xdiff != 0]).min())

    ncol = int(abs(ilines_case.min() - ilines_case.max()) / ispacing) + 1
    nrow = int(abs(xlines_case.min() - xlines_case.max()) / xspacing) + 1
    nlay = data.shape[1]

    values = np.full((ncol, nrow, nlay), xtgeo.UNDEF, dtype=np.float32)
    traceidcodes = np.full((ncol, nrow), 2, dtype=np.int64)

    ilines_shifted = (ilines_case / ispacing).astype(np.int64)
    ilines_shifted -= ilines_shifted.min()
    xlines_shifted = (xlines_case / xspacing).astype(np.int64)
    xlines_shifted -= xlines_shifted.min()

    values[ilines_shifted, xlines_shifted, :] = data
    values = _process_cube_values(values)
    traceidcodes[ilines_shifted, xlines_shifted] = traceidcodes_input

    # generate new ilines, xlines vector with unique values
    ilines = np.array(
        range(ilines_case.min(), ilines_case.max() + ispacing, ispacing), dtype=np.int32
    )
    xlines = np.array(
        range(xlines_case.min(), xlines_case.max() + xspacing, xspacing), dtype=np.int32
    )

    attrs = _geometry_incomplete_traces(
        segyfile,
        ncol,
        nrow,
        ilines,
        xlines,
        ilines_case,
        xlines_case,
        ispacing,
        xspacing,
    )

    attrs["ncol"] = ncol
    attrs["nrow"] = nrow
    attrs["nlay"] = nlay
    attrs["ilines"] = ilines
    attrs["xlines"] = xlines
    attrs["values"] = values
    attrs["traceidcodes"] = traceidcodes
    return attrs


def _inverse_anyline_map(anylines: List[int]) -> Dict:
    """Small helper function to get e.g. inline 2345: [0, 1, 2, .., 70].

    I.e. to get a mapping between inline number and a list of possible indices

    """
    anyll = defaultdict(list)
    for ind, key in enumerate(anylines):
        anyll[key].append(ind)

    return anyll


def _find_long_line(anyll: dict, nany: int) -> List:
    """Helper function; get index of vectors indices to be used for calculations.

    Look for a "sufficiently" long inline/xline, as long distance between points
    increases accuracy.

    """
    minimumlen = int(nany * 0.8)  # get at least 80% length if possible
    maxlenfound = -1

    keepresult = []
    for indices in anyll.values():
        result = [indices[0], indices[-1]]
        indlen = len(indices)
        if indlen > maxlenfound:
            maxlenfound = indlen
            keepresult = deepcopy(result)

        if indlen >= minimumlen:
            break

    if not keepresult or abs(keepresult[1] - keepresult[0]) == 0:
        raise RuntimeError("Not able to get inline or xline vector for geometry")
    return keepresult


def _geometry_incomplete_traces(
    segyfile: segyio.segy.SegyFile,
    ncol: int,
    nrow: int,
    ilines: List[int],
    xlines: List[int],
    ilines_case: List[int],
    xlines_case: List[int],
    ispacing: int,
    xspacing: int,
) -> List:
    """Compute xtgeo attributes (mostly geometries) for incomplete trace cube."""
    attrs = dict()

    ill = _inverse_anyline_map(ilines_case)
    xll = _inverse_anyline_map(xlines_case)

    # need both partial and full reverse lookup of indices vs (iline, xxline)
    # for computing cube origin later
    index_case = {
        ind: (h[TF.INLINE_3D], h[TF.CROSSLINE_3D])
        for ind, h in enumerate(segyfile.header)
    }

    reverseindex_full = {
        (il, xl): (ind, xnd)
        for ind, il in enumerate(ilines)
        for xnd, xl in enumerate(xlines)
    }

    jnd1, jnd2 = _find_long_line(ill, ncol)  # 2 indices along constant iline, aka JY
    ind1, ind2 = _find_long_line(xll, nrow)  # 2 indices along constant xline, aka IX

    il1x, il1y, zori, zinc = _get_coordinate(segyfile, ind1)
    il2x, il2y, _, _ = _get_coordinate(segyfile, ind2)

    jl1x, jl1y, _, _ = _get_coordinate(segyfile, jnd1)
    jl2x, jl2y, _, _ = _get_coordinate(segyfile, jnd2)

    xslen, _, rot1 = xcalc.vectorinfo2(il1x, il2x, il1y, il2y)
    xinc = ispacing * xslen / (abs(ilines_case[ind1] - ilines_case[ind2]))
    yslen, _, _ = xcalc.vectorinfo2(jl1x, jl2x, jl1y, jl2y)
    yinc = xspacing * yslen / (abs(xlines_case[jnd1] - xlines_case[jnd2]))

    yflip = xcalc.find_flip(
        (il2x - il1x, il2y - il1y, 0), (jl2x - jl1x, jl2y - jl1y, 0)
    )

    # need to compute xori and yori from 'case' I J indices with known x y and
    # (iline, xline); use ind1 with assosiated coordinates il1x il1y
    i_use, j_use = reverseindex_full[index_case[ind1]]
    xori, yori = xcalc.xyori_from_ij(
        i_use, j_use, il1x, il1y, xinc, yinc, ncol, nrow, yflip, rot1
    )

    attrs["xori"] = xori
    attrs["yori"] = yori
    attrs["zori"] = zori
    attrs["xinc"] = xinc
    attrs["yinc"] = yinc
    attrs["zinc"] = zinc
    attrs["yflip"] = yflip
    attrs["rotation"] = rot1

    return attrs


def _get_coordinate(
    segyfile: segyio.segy.SegyFile, segyindex: int
) -> Tuple[float, float, float, float]:
    """Helper function to get coordinates given a index."""
    origin = segyfile.header[segyindex][
        segyio.su.cdpx,
        segyio.su.cdpy,
        segyio.su.scalco,
        segyio.su.delrt,
        segyio.su.dt,
        segyio.su.iline,
        segyio.su.xline,
    ]

    point_x = origin[segyio.su.cdpx]
    point_y = origin[segyio.su.cdpy]
    scaler = origin[segyio.su.scalco]
    if scaler < 0:
        point_x = -1 * float(point_x) / scaler
        point_y = -1 * float(point_y) / scaler
    else:
        point_x = point_x * scaler
        point_y = point_y * scaler

    zori = origin[segyio.su.delrt]
    zinc = origin[segyio.su.dt] / 1000.0

    return point_x, point_y, zori, zinc


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
