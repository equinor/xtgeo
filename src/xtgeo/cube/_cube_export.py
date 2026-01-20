"""Export Cube data via SegyIO library or XTGeo CLIB."""

import json
import shutil
import struct

import numpy as np
import segyio

import xtgeo
from xtgeo import _cxtgeo
from xtgeo._cxtgeo import XTGeoCLibError
from xtgeo.common import XTGeoDialog, null_logger

logger = null_logger(__name__)
xtg = XTGeoDialog()


def export_segy(
    cube: xtgeo.Cube,
    sfile: str,
    template: str | None = None,
    pristine: bool = False,
    engine: str = "xtgeo",
) -> None:
    """Export on SEGY using segyio library.

    Args:
        cube (:class:`xtgeo.cube.Cube`): The instance
        sfile : File name to export to.
        template : Use an existing file a template.
        pristine : Make SEGY from scrtach if True; otherwise use an
            existing SEGY file.
        engine : Use 'xtgeo' or (later?) 'segyio'
    """
    if engine == "segyio":
        _export_segy_segyio(cube, sfile, template=template, pristine=pristine)
    else:
        _export_segy_xtgeo(cube, sfile)


def _export_segy_segyio(
    cube: xtgeo.Cube,
    sfile: str,
    template: str | None = None,
    pristine: bool = False,
) -> None:
    """Export on SEGY using segyio library.

    Args:
        cube (:class:`xtgeo.cube.Cube`): The instance
        sfile : File name to export to.
        template : Use an existing file a template.
        pristine : Make SEGY from scrtach if True; otherwise use an
            existing SEGY file.
    """
    logger.debug("Export segy format using segyio...")

    if template is None and cube._segyfile is None:
        raise NotImplementedError("Error, template=None is not yet made!")

    # There is an existing _segyfile attribute, in this case the current SEGY
    # headers etc are applied for the new data. Requires that shapes etc are
    # equal.
    if template is None and cube._segyfile is not None:
        template = cube._segyfile

    cvalues = cube.values

    if template is not None and not pristine:
        try:
            shutil.copyfile(cube._segyfile, sfile)
        except Exception as errormsg:
            xtg.warn(f"Error message: {errormsg}")
            raise

        logger.debug("Input segy file copied ...")

        with segyio.open(sfile, "r+") as segyfile:
            logger.debug("Output segy file is now open...")

            if segyfile.sorting == 1:
                logger.info("xline sorting")
                for xll, xline in enumerate(segyfile.xlines):
                    segyfile.xline[xline] = cvalues[xll]  # broadcasting
            else:
                logger.info("iline sorting")
                ixv, jyv, kzv = cvalues.shape
                for ill, iline in enumerate(segyfile.ilines):
                    if ixv != jyv != kzv or ixv != kzv != jyv:
                        segyfile.iline[iline] = cvalues[ill]  # broadcasting
                    else:
                        # safer but a bit slower than broadcasting
                        segyfile.iline[iline] = cvalues[ill, :, :]

    else:
        # NOT FINISHED!
        logger.debug("Input segy file from scratch ...")

        # sintv = int(self.zinc * 1000)
        spec = segyio.spec()

        spec.sorting = 2
        spec.format = 1

        spec.samples = np.arange(cube.nlay)
        spec.ilines = np.arange(cube.ncol)
        spec.xlines = np.arange(cube.nrow)

        with segyio.create(sfile, spec) as fseg:
            # write the line itself to the file and the inline number
            # in all this line's headers
            for ill, ilno in enumerate(spec.ilines):
                fseg.iline[ilno] = cvalues[ill]
                # f.header.iline[ilno] = {
                #     segyio.TraceField.INLINE_3D: ilno,
                #     segyio.TraceField.offset: 0,
                #     segyio.TraceField.TRACE_SAMPLE_INTERVAL: sintv
                # }

            # # then do the same for xlines
            # for xlno in spec.xlines:
            #     f.header.xline[xlno] = {
            #         segyio.TraceField.CROSSLINE_3D: xlno,
            #         segyio.TraceField.TRACE_SAMPLE_INTERVAL: sintv
            #     }


def _export_segy_xtgeo(cube: xtgeo.Cube, sfile: str) -> None:
    """Export SEGY via XTGeo internal C routine."""

    values1d = cube.values.reshape(-1)

    ilinesp = _cxtgeo.new_intarray(len(cube._ilines))
    xlinesp = _cxtgeo.new_intarray(len(cube._xlines))
    tracidp = _cxtgeo.new_intarray(cube.ncol * cube.nrow)

    ilns = cube._ilines.astype(np.int32)
    xlns = cube._xlines.astype(np.int32)
    trid = cube._traceidcodes.flatten().astype(np.int32)

    _cxtgeo.swig_numpy_to_carr_i1d(ilns, ilinesp)
    _cxtgeo.swig_numpy_to_carr_i1d(xlns, xlinesp)
    _cxtgeo.swig_numpy_to_carr_i1d(trid, tracidp)

    status = _cxtgeo.cube_export_segy(
        sfile,
        cube.ncol,
        cube.nrow,
        cube.nlay,
        values1d,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        cube.zori,
        cube.zinc,
        cube.rotation,
        cube.yflip,
        1,
        ilinesp,
        xlinesp,
        tracidp,
        0,
    )

    if status != 0:
        raise XTGeoCLibError("Error when exporting to SEGY (xtgeo engine)")

    _cxtgeo.delete_intarray(ilinesp)
    _cxtgeo.delete_intarray(xlinesp)


def export_rmsreg(cube: xtgeo.Cube, sfile: str) -> None:
    """Export on RMS regular format."""

    logger.debug("Export to RMS regular format...")
    values1d = cube.values.reshape(-1)

    status = _cxtgeo.cube_export_rmsregular(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.xori,
        cube.yori,
        cube.zori,
        cube.xinc,
        cube.yinc * cube.yflip,
        cube.zinc,
        cube.rotation,
        cube.yflip,
        values1d,
        sfile,
    )

    if status != 0:
        raise RuntimeError("Error when exporting to RMS regular")


def export_xtgregcube(cube: xtgeo.Cube, mfile: str) -> None:
    """Export to experimental xtgregcube format, python version."""
    logger.info("Export as xtgregcube...")
    cube.metadata.required = cube

    prevalues = (1, 1201, 4, cube.ncol, cube.nrow, cube.nlay)
    mystruct = struct.Struct("= i i i q q q")
    pre = mystruct.pack(*prevalues)

    meta = cube.metadata.get_metadata()

    jmeta = json.dumps(meta).encode()

    with open(mfile, "wb") as fout:
        fout.write(pre)

    with open(mfile, "ab") as fout:
        # TODO. Treat dead traces as undef
        cube.values.tofile(fout)

    with open(mfile, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(mfile, "ab") as fout:
        fout.write(jmeta)

    logger.info("Export as xtgregcube... done")
