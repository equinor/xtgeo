# coding: utf-8
"""Roxar API functions for XTGeo Cube"""
from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

XTGDEBUG = xtg.get_syslevel()


def import_cube_roxapi(self, project, name):
    """Import (transfer) a Cube via ROXAR API container to XTGeo."""
    import roxar  # pylint: disable=import-error

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            _roxapi_import_cube(self, proj, name)
    else:
        _roxapi_import_cube(self, project, name)


def _roxapi_import_cube(self, proj, name):
    # note that name must be in brackets
    if [name] not in proj.seismic.data.keys():
        raise ValueError(
            "Name {} is not within RMS Seismic Cube container".format(name)
        )
    try:
        rcube = proj.seismic.data[[name]]
        _roxapi_cube_to_xtgeo(self, rcube)
    except KeyError as emsg:
        logger.error(emsg)
        raise


def _roxapi_cube_to_xtgeo(self, rcube):
    """Tranforming cube from ROXAPI to XTGeo object."""
    logger.info("Cube from roxapi to xtgeo...")
    self._xori, self._yori = rcube.origin
    self._zori = rcube.first_z
    self._ncol, self._nrow, self._nlay = rcube.dimensions
    self._xinc, self._yinc = rcube.increment
    self._zinc = rcube.sample_rate
    self._rotation = rcube.rotation
    self._yflip = -1
    if rcube.handedness == "left":
        self._yflip = 1

    ilstart = rcube.get_inline(0)
    xlstart = rcube.get_crossline(0)
    ilincr, xlincr = rcube.inline_crossline_increment

    self._ilines = np.array(
        range(ilstart, self._ncol + ilstart, ilincr), dtype=np.int32
    )
    self._xlines = np.array(
        range(xlstart, self._nrow + xlstart, xlincr), dtype=np.int32
    )

    # roxar API does not store traceid codes, assume 1
    self._traceidcodes = np.ones((self._ncol, self._nrow), dtype=np.int32)

    if rcube.is_empty:
        xtg.warn("Cube has no data; assume 0")
    else:
        self._values = rcube.get_values()


def export_cube_roxapi(
    self, project, name, folder=None, domain="time", compression=("wavelet", 5)
):
    """Export (store) a Seismic cube to RMS via ROXAR API spec."""
    import roxar  # pylint: disable=import-error

    logger.debug("TODO: folder %s", folder)
    logger.debug("TODO: compression %s", compression)

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            _roxapi_export_cube(
                self, roxar, proj, name, domain=domain, compression=compression
            )
    else:
        _roxapi_export_cube(
            self, roxar, project, name, domain=domain, compression=compression
        )


def _roxapi_export_cube(
    self, roxar, proj, name, folder=None, domain="time", compression=("wavelet", 5)
):

    logger.info(
        "There are issues with compression%s, hence it {} is ignored", compression
    )

    if folder is None:
        rcube = proj.seismic.data.create_cube(name)
    else:
        rcube = proj.seismic.data.create_cube(name, folder)

    # populate
    origin = (self.xori, self.yori)
    first_z = self.zori
    increment = (self.xinc, self.yinc)
    sample_rate = self.zinc
    rotation = self.rotation
    vertical_domain = roxar.VerticalDomain.time
    if domain == "depth":
        vertical_domain = roxar.VerticalDomain.depth

    values = self.values.copy()  # copy() needed?

    handedness = roxar.Direction.right
    if self.yflip == 1:
        handedness = roxar.Direction.left

    # inline xline vector
    ilstart = self.ilines[0]
    xlstart = self.xlines[0]
    ilincr = self.ilines[1] - self.ilines[0]
    xlincr = self.xlines[1] - self.xlines[0]

    rcube.set_cube(
        values,
        origin,
        increment,
        first_z,
        sample_rate,
        rotation,
        vertical_domain=vertical_domain,
        handedness=handedness,
        inline_crossline_start=(ilstart, xlstart),
        inline_crossline_increment=(ilincr, xlincr),
    )
