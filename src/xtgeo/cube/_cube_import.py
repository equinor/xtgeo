"""Import Cube data via SegyIO library or XTGeo CLIB."""
import numpy as np

import segyio
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def import_segy(self, sfile, engine="segyio"):
    if engine == "segyio":
        _import_segy_io(self, sfile)
    else:
        pass
        # _import_segy_xtgeo()


def _import_segy_io(self, sfile):
    """Import SEGY via Statoils FOSS SegyIO library.

    Args:
        self (Cube): Cube object
        sfile (str): File name of SEGY file
        undef (float): If None, dead traces (undef) are read as is, but
            if a a value, than dead traces get this value.
    """

    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals

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

        logger.debug(segyfile.fast)
        logger.debug(segyfile.ilines)
        logger.debug(len(segyfile.ilines))
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
                slen, _rotrad1, rot1 = xcalc.vectorinfo2(xori, cdpx, yori, cdpy)
                xinc = slen / (ncol - 1)

                rotation = rot1
                xvv = (cdpx - xori, cdpy - yori, 0)

            if inum == 2:
                slen, _rotrad2, rot2 = xcalc.vectorinfo2(xori, cdpx, yori, cdpy)
                yinc = slen / (nrow - 1)

                # find YFLIP by cross products
                yvv = (cdpx - xori, cdpy - yori, 0)

                yflip = xcalc.find_flip(xvv, yvv)

        rot2 = segyio.tools.rotation(segyfile)[0]
        logger.debug("SEGYIO rotation is %s", rot2 * 180 / 3.1415)
        logger.debug("XTGeo rotation is %s", rotation)

    # attributes to update
    self._ilines = ilines
    self._xlines = xlines
    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay
    self._xori = xori
    self._xinc = xinc
    self._yori = yori
    self._yinc = yinc
    self._zori = zori
    self._zinc = zinc
    self._rotation = rotation
    self._values = values
    self._yflip = yflip
    self._segyfile = sfile
    self._traceidcodes = traceidcodes


def _import_segy_xtgeo(sfile, scanheadermode=False, scantracemode=False, outfile=None):
    """Import SEGY via XTGeo's C library. OLD NOT UPDATED!!

    Args:
        sfile (str): File name of SEGY file
        scanheadermode (bool, optional): If true, will scan header
        scantracemode (bool, optional): If true, will scan trace headers
        outfile (str, optional): Output file for scan dump (default None)

    Returns:
        A dictionary with relevant data.
    """
    # pylint: disable=too-many-statements, too-many-locals

    sdata = dict()

    logger.info("Import SEGY via XTGeo CLIB")

    if outfile is None:
        outfile = "/dev/null"

    ptr_gn_bitsheader = _cxtgeo.new_intpointer()
    ptr_gn_formatcode = _cxtgeo.new_intpointer()
    ptr_gf_segyformat = _cxtgeo.new_floatpointer()
    ptr_gn_samplespertrace = _cxtgeo.new_intpointer()
    ptr_gn_measuresystem = _cxtgeo.new_intpointer()

    option = 0
    if scantracemode:
        option = 0
    if scanheadermode:
        option = 1

    _cxtgeo.cube_scan_segy_hdr(
        sfile,
        ptr_gn_bitsheader,
        ptr_gn_formatcode,
        ptr_gf_segyformat,
        ptr_gn_samplespertrace,
        ptr_gn_measuresystem,
        option,
        outfile,
    )

    # get values
    gn_bitsheader = _cxtgeo.intpointer_value(ptr_gn_bitsheader)
    gn_formatcode = _cxtgeo.intpointer_value(ptr_gn_formatcode)
    gf_segyformat = _cxtgeo.floatpointer_value(ptr_gf_segyformat)
    gn_samplespertrace = _cxtgeo.intpointer_value(ptr_gn_samplespertrace)

    if scanheadermode:
        logger.info("Scan SEGY header ... %s bytes ... DONE", gn_bitsheader)
        return None

    # next is to scan first and last trace, in order to allocate
    # cube size

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

    optscan = 1

    if scantracemode:
        option = 1

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
        optscan,
        option,
        outfile,
    )

    logger.debug("Scan via C wrapper... done")

    ncol = _cxtgeo.intpointer_value(ptr_ncol)
    nrow = _cxtgeo.intpointer_value(ptr_nrow)
    nlay = _cxtgeo.intpointer_value(ptr_nlay)

    if scantracemode:
        return None

    nrcl = ncol * nrow * nlay

    ptr_cval_v = _cxtgeo.new_floatarray(nrcl)

    # next is to do the actual import of the cube
    optscan = 0

    logger.debug("Import via C wrapper...")
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
        ptr_cval_v,
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
        optscan,
        option,
        outfile,
    )

    logger.debug("Import via C wrapper...")

    sdata["ncol"] = ncol
    sdata["nrow"] = nrow
    sdata["nlay"] = nlay

    sdata["xori"] = _cxtgeo.doublepointer_value(ptr_xori)
    sdata["yori"] = _cxtgeo.doublepointer_value(ptr_yori)
    sdata["zori"] = _cxtgeo.doublepointer_value(ptr_zori)

    sdata["xinc"] = _cxtgeo.doublepointer_value(ptr_xinc)
    sdata["yinc"] = _cxtgeo.doublepointer_value(ptr_yinc)
    sdata["zinc"] = _cxtgeo.doublepointer_value(ptr_zinc)

    sdata["yflip"] = _cxtgeo.intpointer_value(ptr_yflip)
    sdata["zflip"] = _cxtgeo.intpointer_value(ptr_zflip)

    sdata["rotation"] = _cxtgeo.doublepointer_value(ptr_rotation)

    sdata["minval"] = _cxtgeo.doublepointer_value(ptr_minval)
    sdata["maxval"] = _cxtgeo.doublepointer_value(ptr_maxval)

    sdata["zmin"] = sdata["zori"]
    sdata["zmax"] = sdata["zori"] + sdata["zflip"] * sdata["zinc"] * (nlay - 1)

    # the pointer to 1D C array
    sdata["cvalues"] = ptr_cval_v
    sdata["values"] = None

    return sdata


def import_rmsregular(self, sfile):
    """Import on RMS regular format."""
    logger.debug(self, sfile)
    raise NotImplementedError("Sorry, not implemented yet")


def import_stormcube(self, sfile):
    """Import on StormCube format."""

    # pylint: disable=too-many-statements, too-many-locals, too-many-branches

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
    try:
        stf = open(sfile, encoding="ISO-8859-1")  # python 3
    except TypeError:
        stf = open(sfile)

    iline = 0
    for line in range(10):
        xline = stf.readline()
        if not xline.strip():
            continue

        iline += 1
        if iline == 1:
            pass
        elif iline == 2:
            _nn, _modname, _undef_val = xline.strip().split()
        elif iline == 3:
            pass
        elif iline == 4:
            (xori, xlen, yori, ylen, zori, _zmax, _e1, _e2) = xline.strip().split()
        elif iline == 5:
            zlen, rot = xline.strip().split()
        elif iline == 6:
            ncol, nrow, nlay = xline.strip().split()
            nlines = line + 2
            break
    stf.close()

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
            "Something when wrong in {}, code is {}".format(__name__, ier)
        )

    self._ilines = np.array(range(1, ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, nrow + 1), dtype=np.int32)
    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay
    self._xori = xori
    self._xinc = xinc
    self._yori = yori
    self._yinc = yinc
    self._zori = zori
    self._zinc = zinc
    self._rotation = rotation
    self._values = values.reshape((ncol, nrow, nlay))
    self._yflip = yflip
    self._traceidcodes = np.ones((ncol, nrow), dtype=np.int32)
