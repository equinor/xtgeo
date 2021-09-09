"""Export Cube data via SegyIO library or XTGeo CLIB."""
import shutil
import struct
import json
import numpy as np

import segyio

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo import XTGeoCLibError

xtg = XTGeoDialog()


logger = xtg.functionlogger(__name__)


def export_segy(self, sfile, template=None, pristine=False, engine="xtgeo"):
    """Export on SEGY using segyio library.

    Args:
        self (:class:`xtgeo.cube.Cube`): The instance
        sfile (str): File name to export to.
        template (str): Use an existing file a template.
        pristine (bool): Make SEGY from scrtach if True; otherwise use an
            existing SEGY file.
        engine (str): Use 'xtgeo' or (later?) 'segyio'
    """
    if not isinstance(self, xtgeo.cube.Cube):
        raise ValueError("first argument is not a Cube instance")

    if engine == "segyio":
        _export_segy_segyio(self, sfile, template=template, pristine=pristine)
    else:
        _export_segy_xtgeo(self, sfile)


def _export_segy_segyio(self, sfile, template=None, pristine=False):
    """Export on SEGY using segyio library.

    Args:
        self (:class:`xtgeo.cube.Cube`): The instance
        sfile (str): File name to export to.
        template (str): Use an existing file a template.
        pristine (bool): Make SEGY from scrtach if True; otherwise use an
            existing SEGY file.
    """
    logger.debug("Export segy format using segyio...")

    if template is None and self._segyfile is None:
        raise NotImplementedError("Error, template=None is not yet made!")

    # There is an existing _segyfile attribute, in this case the current SEGY
    # headers etc are applied for the new data. Requires that shapes etc are
    # equal.
    if template is None and self._segyfile is not None:
        template = self._segyfile

    cvalues = self.values

    if template is not None and not pristine:

        try:
            shutil.copyfile(self._segyfile, sfile)
        except Exception as errormsg:
            xtg.warn("Error message: {}".format(errormsg))
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

        spec.samples = np.arange(self.nlay)
        spec.ilines = np.arange(self.ncol)
        spec.xlines = np.arange(self.nrow)

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


def _export_segy_xtgeo(self, sfile):
    """Export SEGY via XTGeo internal C routine."""

    values1d = self.values.reshape(-1)

    ilinesp = _cxtgeo.new_intarray(len(self._ilines))
    xlinesp = _cxtgeo.new_intarray(len(self._xlines))
    tracidp = _cxtgeo.new_intarray(self.ncol * self.nrow)

    ilns = self._ilines.astype(np.int32)
    xlns = self._xlines.astype(np.int32)
    trid = self._traceidcodes.flatten().astype(np.int32)

    _cxtgeo.swig_numpy_to_carr_i1d(ilns, ilinesp)
    _cxtgeo.swig_numpy_to_carr_i1d(xlns, xlinesp)
    _cxtgeo.swig_numpy_to_carr_i1d(trid, tracidp)

    status = _cxtgeo.cube_export_segy(
        sfile,
        self.ncol,
        self.nrow,
        self.nlay,
        values1d,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.zori,
        self.zinc,
        self.rotation,
        self.yflip,
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


def export_rmsreg(self, sfile):
    """Export on RMS regular format."""

    logger.debug("Export to RMS regular format...")
    values1d = self.values.reshape(-1)

    status = _cxtgeo.cube_export_rmsregular(
        self.ncol,
        self.nrow,
        self.nlay,
        self.xori,
        self.yori,
        self.zori,
        self.xinc,
        self.yinc * self.yflip,
        self.zinc,
        self.rotation,
        self.yflip,
        values1d,
        sfile,
    )

    if status != 0:
        raise RuntimeError("Error when exporting to RMS regular")


def export_xtgregcube(self, mfile):
    """Export to experimental xtgregcube format, python version."""
    logger.info("Export as xtgregcube...")
    self.metadata.required = self

    prevalues = (1, 1201, 4, self.ncol, self.nrow, self.nlay)
    mystruct = struct.Struct("= i i i q q q")
    pre = mystruct.pack(*prevalues)

    meta = self.metadata.get_metadata()

    jmeta = json.dumps(meta).encode()

    with open(mfile, "wb") as fout:
        fout.write(pre)

    with open(mfile, "ab") as fout:
        # TODO. Treat dead traces as undef
        self.values.tofile(fout)

    with open(mfile, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(mfile, "ab") as fout:
        fout.write(jmeta)

    logger.info("Export as xtgregcube... done")
