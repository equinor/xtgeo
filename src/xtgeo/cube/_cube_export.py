"""Export Cube data via SegyIO library or XTGeo CLIB."""
import numpy as np
import shutil
import segyio

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')


def export_segy(self, sfile, template=None):
    """Export on SEGY using segyio library"""

    if template is None and self._segyfile is None:
        raise NotImplementedError('Error, template=None is not yet made!')

    # There is an existing _segyfile attribute, in this case the current SEGY
    # headers etc are applied for the new data. Requires that shapes etc are
    # equal.
    if self._segyfile is not None:
        newvalues = np.asanyarray(self.values, order='C')

        try:
            shutil.copyfile(self._segyfile, sfile)
        except Exception as errormsg:
            xtg.warn('Error message: '.format(errormsg))
            raise

        with segyio.open(sfile, 'r+') as dst:
            if dst.sorting == 1:
                logger.info('xline sorting')
                for xl, xline in enumerate(dst.xlines):
                    dst.xline[xline] = newvalues[xl]   # broadcasting
            else:
                logger.info('iline sorting')
                for il, iline in enumerate(dst.ilines):
                    dst.iline[iline] = newvalues[il]  # broadcasting

    else:
        raise NotImplementedError('Error, SEGY export is not properly made!')


def export_rmsreg(nx, ny, nz, xori, yori, zori, xinc, yinc, zinc,
                  rotation, yflip, values, sfile, xtg_verbose_level):
    """Export on RMS regular format."""

    values1d = np.ravel(values, order='F')

    yinc = yinc * yflip
    status = _cxtgeo.cube_export_rmsregular(nx, ny, nz,
                                            xori, yori, zori,
                                            xinc, yinc, zinc,
                                            rotation, yflip,
                                            values1d,
                                            sfile, xtg_verbose_level)

    if status != 0:
        raise RuntimeError('Error when exporting to RMS regular')
