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

    logger.debug('Export segy format using segyio...')

    if template is None and self._segyfile is None:
        raise NotImplementedError('Error, template=None is not yet made!')

    # There is an existing _segyfile attribute, in this case the current SEGY
    # headers etc are applied for the new data. Requires that shapes etc are
    # equal.
    if self._segyfile is not None:
        newvalues = self.values
#        newvalues = np.asanyarray(self.values, order='C')

        try:
            shutil.copyfile(self._segyfile, sfile)
        except Exception as errormsg:
            xtg.warn('Error message: '.format(errormsg))
            raise

        logger.debug('Input segy file copied...')
        with segyio.open(sfile, 'r+') as segyfile:

            logger.debug('Output segy file is now open...')
            if segyfile.sorting == 1:
                logger.info('xline sorting')
                for xl, xline in enumerate(segyfile.xlines):
                    segyfile.xline[xline] = newvalues[xl]   # broadcasting
            else:
                logger.info('iline sorting')
                logger.debug('ilines object: {}'.format(segyfile.ilines))
                logger.debug('iline object: {}'.format(segyfile.iline))
                logger.debug('newvalues shape {}'.format(newvalues.shape))
                ix, jy, kz = newvalues.shape
                for il, iline in enumerate(segyfile.ilines):
                    logger.debug('il={}, iline={}'.format(il, iline))
                    if ix != jy != kz or ix != kz != jy :
                        segyfile.iline[iline] = newvalues[il]  # broadcasting
                    else:
                        # safer but a bit slower than broadcasting
                        segyfile.iline[iline] = newvalues[il, :, :]

    else:
        raise NotImplementedError('Error, SEGY export is not properly made!')


def export_rmsreg(self, sfile):
    """Export on RMS regular format."""

    logger.debug('Export to RMS regular format...')
    values1d = self.values.reshape(-1)

    yinc = self.yinc * self.yflip
    status = _cxtgeo.cube_export_rmsregular(self.nx, self.ny, self.nz,
                                            self.xori, self.yori, self.zori,
                                            self.xinc, yinc, self.zinc,
                                            self.rotation, self.yflip,
                                            values1d,
                                            sfile, xtg_verbose_level)

    if status != 0:
        raise RuntimeError('Error when exporting to RMS regular')
