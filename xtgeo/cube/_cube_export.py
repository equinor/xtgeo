"""Export Cube data via SegyIO library or XTGeo CLIB."""
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

logger = logging.getLogger('xtgeo.cube._cube_export')
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


def export_rmsreg(nx, ny, nz, xori, yori, zori, xinc, yinc, zinc,
                  rotation, yflip, cvalues, sfile, xtg_verbose_level):
    """Export on RMS regular format."""

    yinc = yinc * yflip
    status = _cxtgeo.cube_export_rmsregular(nx, ny, nz,
                                            xori, yori, zori,
                                            xinc, yinc, zinc,
                                            rotation, yflip,
                                            cvalues,
                                            sfile, xtg_verbose_level)

    if status != 0:
        raise RuntimeError('Error when exporting to RMS regular')
