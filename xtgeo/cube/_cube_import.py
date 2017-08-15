"""Import Cube data via SegyIO library or XTGeo CLIB."""
import logging
import numpy as np

import segyio
import cxtgeo.cxtgeo as _cxtgeo
import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


def find_flip(xv, yv, zv):
    """Find the flip status by computing the cross products."""

    flip = 0

    xv = np.array(xv)
    yv = np.array(yv)
    zv = np.array(zv)

    xycross = np.cross(xv, yv)

    logger.debug('Cross product XY is {}'.format(xycross))

    if xycross[2] < 0:
        flip = 1
    else:
        flip = -1

    return flip


def import_segy_io(sfile):
    """Import SEGY via Statoils FOSS SegyIO library.

    Args:
        sfile (str): File name of SEGY file

    Returns:
        A dictionary with relevant data.
    """

    sdata = dict()

    logger.debug('Inline sorting is {}'
                 .format(segyio.TraceSortingFormat.INLINE_SORTING))

    with segyio.open(sfile, 'r') as segyfile:
        segyfile.mmap()

        values = segyio.tools.cube(segyfile)
        values = np.asfortranarray(values)

        logger.debug(segyfile.fast)
        logger.debug(segyfile.ilines)
        logger.debug(len(segyfile.ilines))

        nx, ny, nz = values.shape

        logger.info('NXYZ  {} {} {}'.format(nx, ny, nz))
        logger.info(len(segyfile.xlines))
        logger.info(len(segyfile.ilines))

        # need positions for all 4 corners
        c1 = xcalc.ijk_to_ib(1, 1, 1, nx, ny, 1, forder=False)
        c2 = xcalc.ijk_to_ib(nx, 1, 1, nx, ny, 1, forder=False)
        c3 = xcalc.ijk_to_ib(1, ny, 1, nx, ny, 1, forder=False)
        c4 = xcalc.ijk_to_ib(nx, ny, 1, nx, ny, 1, forder=False)

        clist = [c1, c2, c3, c4]

        logger.debug('IB to corners are {}'.format(clist))

        for i, co in enumerate(clist):
            logger.debug(i)
            origin = segyfile.header[co][segyio.su.cdpx,
                                         segyio.su.cdpy,
                                         segyio.su.scalco,
                                         segyio.su.delrt,
                                         segyio.su.dt,
                                         segyio.su.iline,
                                         segyio.su.xline]
            # get the data on SU (seismic unix) format
            cdpx = origin[segyio.su.cdpx]
            cdpy = origin[segyio.su.cdpy]
            scaler = origin[segyio.su.scalco]
            iline = origin[segyio.su.iline]
            xline = origin[segyio.su.xline]
            logger.debug('{}: ILINE XLINE is {} {}'.format(i, iline, xline))
            logger.debug(cdpx)
            logger.debug(scaler)
            if (scaler < 0):
                cdpx = -1 * float(cdpx) / scaler
                cdpy = -1 * float(cdpy) / scaler
            else:
                cdpx = cdpx * scaler
                cdpy = cdpy * scaler

            logger.debug(cdpx)
            logger.debug(cdpy)

            if i == 0:
                xori = cdpx
                yori = cdpy
                zori = origin[segyio.su.delrt]
                zinc = origin[segyio.su.dt] / 1000.0

            if i == 1:
                slen, rotrad1, rot1 = xcalc.vectorinfo2(xori, cdpx,
                                                        yori, cdpy)
                xinc = slen / (nx - 1)
                logger.debug(slen)

                rotation = rot1
                xv = [cdpx - xori, cdpy - yori, 0]

            if i == 2:
                slen, rotrad2, rot2 = xcalc.vectorinfo2(xori, cdpx,
                                                        yori, cdpy)
                yinc = slen / (ny - 1)
                logger.debug(slen)

                # find YFLIP by cross products
                yv = [cdpx - xori, cdpy - yori, 0]
                zv = [0, 0, -1]

                yflip = find_flip(xv, yv, zv)
                # due to bug in segyio?
                yflip *= -1

        rot2 = segyio.tools.rotation(segyfile)[0]
        print('SEGYIO rotation is {}'.format(rot2 * 180 / 3.1415))
        print('MY rotation is {}'.format(rotation))

    # data to return
    sdata['values'] = values
    sdata['cvalues'] = None
    sdata['nx'] = nx
    sdata['ny'] = ny
    sdata['nz'] = nz
    sdata['xori'] = xori
    sdata['xinc'] = xinc
    sdata['yori'] = yori
    sdata['yinc'] = yinc
    sdata['zori'] = zori
    sdata['zinc'] = zinc
    sdata['rotation'] = rotation
    sdata['yflip'] = yflip

    return sdata


def import_segy(sfile, scanheadermode=False, scantracemode=False,
                outfile=None):
    """Import SEGY via XTGeo's C library.

    Args:
        sfile (str): File name of SEGY file
        scanheadermode (bool, optional): If true, will scan header
        scantracemode (bool, optional): If true, will scan trace headers
        outfile (str, optional): Output file for scan dump (default None)

    Returns:
        A dictionary with relevant data.
    """

    sdata = dict()

    logger.info('Import SEGY via XTGeo CLIB')

    if outfile is None:
        outfile = '/dev/null'

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

    logger.info('OPTION = {}'.format(option))

    _cxtgeo.cube_scan_segy_hdr(sfile, ptr_gn_bitsheader, ptr_gn_formatcode,
                               ptr_gf_segyformat, ptr_gn_samplespertrace,
                               ptr_gn_measuresystem, option, outfile,
                               xtg_verbose_level)

    # get values
    gn_bitsheader = _cxtgeo.intpointer_value(ptr_gn_bitsheader)
    gn_formatcode = _cxtgeo.intpointer_value(ptr_gn_formatcode)
    gf_segyformat = _cxtgeo.floatpointer_value(ptr_gf_segyformat)
    gn_samplespertrace = _cxtgeo.intpointer_value(ptr_gn_samplespertrace)
    gn_measuresystem = _cxtgeo.intpointer_value(ptr_gn_measuresystem)

    logger.info('GN_BITSHEADER      = {}'.format(gn_bitsheader))
    logger.info('GN_FORMATCODE      = {}'.format(gn_formatcode))
    logger.info('GN_SEGYFORMAT      = {}'.format(gf_segyformat))
    logger.info('GN_SAMPLESPERTRACE = {} (global value)'.
                format(gn_samplespertrace))
    logger.info('GN_MEASURESYSTEM   = {} (code)'.
                format(gn_measuresystem))

    if scanheadermode:
        logger.info('Scan SEGY header ... {} bytes ... DONE'.
                    format(gn_bitsheader))
        return

    # next is to scan first and last trace, in order to allocate
    # cube size

    ptr_nx = _cxtgeo.new_intpointer()
    ptr_ny = _cxtgeo.new_intpointer()
    ptr_nz = _cxtgeo.new_intpointer()
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

    logger.debug('Scan via C wrapper...')
    _cxtgeo.cube_import_segy(sfile,
                             # input
                             gn_bitsheader,
                             gn_formatcode,
                             gf_segyformat,
                             gn_samplespertrace,
                             # result (as pointers)
                             ptr_nx,
                             ptr_ny,
                             ptr_nz,
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
                             xtg_verbose_level)

    logger.debug('Scan via C wrapper... done')

    nx = _cxtgeo.intpointer_value(ptr_nx)
    ny = _cxtgeo.intpointer_value(ptr_ny)
    nz = _cxtgeo.intpointer_value(ptr_nz)

    if scantracemode:
        return

    nxyz = nx * ny * nz

    logger.debug('Allocate number of cells: {}'.format(nxyz))

    ptr_cval_v = _cxtgeo.new_floatarray(nxyz)

    # next is to do the actual import of the cube
    optscan = 0

    logger.debug('Import via C wrapper...')
    _cxtgeo.cube_import_segy(sfile,
                             # input
                             gn_bitsheader,
                             gn_formatcode,
                             gf_segyformat,
                             gn_samplespertrace,
                             # result (as pointers)
                             ptr_nx,
                             ptr_ny,
                             ptr_nz,
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
                             xtg_verbose_level)

    logger.debug('Import via C wrapper...')

    sdata['nx'] = nx
    sdata['ny'] = ny
    sdata['nz'] = nz

    sdata['xori'] = _cxtgeo.doublepointer_value(ptr_xori)
    sdata['yori'] = _cxtgeo.doublepointer_value(ptr_yori)
    sdata['zori'] = _cxtgeo.doublepointer_value(ptr_zori)

    sdata['xinc'] = _cxtgeo.doublepointer_value(ptr_xinc)
    sdata['yinc'] = _cxtgeo.doublepointer_value(ptr_yinc)
    sdata['zinc'] = _cxtgeo.doublepointer_value(ptr_zinc)

    sdata['yflip'] = _cxtgeo.intpointer_value(ptr_yflip)
    sdata['zflip'] = _cxtgeo.intpointer_value(ptr_zflip)

    sdata['rotation'] = _cxtgeo.doublepointer_value(ptr_rotation)

    sdata['minval'] = _cxtgeo.doublepointer_value(ptr_minval)
    sdata['maxval'] = _cxtgeo.doublepointer_value(ptr_maxval)

    sdata['zmin'] = sdata['zori']
    sdata['zmax'] = sdata['zori'] + sdata['zflip'] * sdata['zinc'] * (nz - 1)

    # the pointer to 1D C array
    sdata['cvalues'] = ptr_cval_v
    sdata['values'] = None

    return sdata


def import_stormcube(sfile):
    """Import on StormCube format."""

    # the ASCII header has all the metadata on the form:
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
    # Description:
    #

    sdata = dict()
    # Scan the header with Python; then use CLIB for the binary data
    try:
        sf = open(sfile, encoding='ISO-8859-1')  # python 3
    except TypeError:
        sf = open(sfile)

    iline = 0
    for line in range(10):
        xline = sf.readline()
        if len(xline.strip()) == 0:
            continue
        else:
            iline += 1
            if iline == 1:
                __ = xline
            if iline == 2:
                nn, modname, undef_val = xline.strip().split()
            if iline == 3:
                __ = xline.strip().split()
            if iline == 4:
                (xori, xlen, yori, ylen,
                 zori, zmax, e1, e2) = xline.strip().split()
            if iline == 5:
                zlen, rot = xline.strip().split()
            if iline == 6:
                nx, ny, nz = xline.strip().split()
                dataline = line + 2
    sf.close()

    logger.debug('BINARY data starts at line  {}'.format(dataline))

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    nxyz = nx * ny * nz

    xori = float(xori)
    yori = float(yori)
    zori = float(zori)

    rotation = float(rot)
    if rotation < 0:
        rotation += 360

    xinc = float(xlen) / (nx - 1 + 1)
    yinc = float(ylen) / (ny - 1 + 1)
    zinc = float(zlen) / (nz - 1 + 1)

    yflip = 1

    logger.debug('NX NY NZ {} {} {}'.
                 format(nx, ny, nz))

    logger.debug('XINC, YINC, ZINC {} {} {}'.
                 format(xinc, yinc, zinc))

    logger.debug('ROT  {}'.format(rotation))

    xtg_verbose_level = xtg.get_syslevel()

    cvalues = _cxtgeo.new_floatarray(nxyz)

    _cxtgeo.cube_import_storm(sfile, dataline, int(undef_val), nxyz,
                              cvalues, 0, xtg_verbose_level)

    sdata['nx'] = nx
    sdata['ny'] = ny
    sdata['nz'] = nz

    sdata['xori'] = xori
    sdata['yori'] = yori
    sdata['zori'] = zori

    sdata['xinc'] = xinc
    sdata['yinc'] = yinc
    sdata['zinc'] = zinc

    sdata['yflip'] = yflip

    sdata['rotation'] = rotation

    sdata['cvalues'] = cvalues
    sdata['values'] = None

    return sdata
