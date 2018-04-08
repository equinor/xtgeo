"""Export Cube data via SegyIO library or XTGeo CLIB."""
import shutil
import numpy as np

import segyio

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')


def export_segy(self, sfile, template=None, pristine=False):
    """Export on SEGY using segyio library.

    Args:
        self (Cube): The instance
        sfile (str): File name to export to.
        template (str): Use an existing file a template.
        pristine (bool): Make SEGY from scrtach if True; otherwise use an
            existing SEGY file.
    """

    logger.debug('Export segy format using segyio...')

    if template is None and self._segyfile is None:
        raise NotImplementedError('Error, template=None is not yet made!')

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
            xtg.warn('Error message: '.format(errormsg))
            raise

        logger.debug('Input segy file copied ...')

        with segyio.open(sfile, 'r+') as segyfile:

            logger.debug('Output segy file is now open...')

            if segyfile.sorting == 1:
                logger.info('xline sorting')
                for xl, xline in enumerate(segyfile.xlines):
                    segyfile.xline[xline] = cvalues[xl]   # broadcasting
            else:
                logger.info('iline sorting')
                logger.debug('ilines object: {}'.format(segyfile.ilines))
                logger.debug('iline object: {}'.format(segyfile.iline))
                logger.debug('cvalues shape {}'.format(cvalues.shape))
                ix, jy, kz = cvalues.shape
                for il, iline in enumerate(segyfile.ilines):
                    logger.debug('il={}, iline={}'.format(il, iline))
                    if ix != jy != kz or ix != kz != jy:
                        segyfile.iline[iline] = cvalues[il]  # broadcasting
                    else:
                        # safer but a bit slower than broadcasting
                        segyfile.iline[iline] = cvalues[il, :, :]

    else:
        logger.debug('Input segy file from scratch ...')

        sintv = int(self.zinc * 1000)


        spec = segyio.spec()

        spec.sorting = 2
        spec.format = 1

        spec.samples = np.arange(self.nlay)
        spec.ilines = np.arange(self.ncol)
        spec.xlines = np.arange(self.nrow)


        with segyio.create(sfile, spec) as f:

            # write the line itself to the file and the inline number
            # in all this line's headers
            for il, ilno in enumerate(spec.ilines):
                logger.debug('il={}, iline={}'.format(il, ilno))
                f.iline[ilno] = cvalues[il]
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
