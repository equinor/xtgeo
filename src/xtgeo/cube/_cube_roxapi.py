# coding: utf-8
"""Roxar API functions for XTGeo Cube

Note on rotation:

xtgeo uses rotation of "columns" whick is xline direction counterclockwise
measured from X axis.

roxarapi uses rotation of inline direction (rows) relative to Y axis.
api < 1.4: counterclockwise "rotation"
api >= 1.4 clockwise "orientation"

Seems like self._rotation == roxar.orientation * -1 anyway @ reverse engineering/testing

"""

import numpy as np

from xtgeo.common import XTGeoDialog
from xtgeo import RoxUtils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_cube_roxapi(self, project, name, folder=None):  # pragma: no cover
    """Import (transfer) a Cube via ROXAR API container to XTGeo.

    .. versionadded:: 2.1
    """
    rox = RoxUtils(project, readonly=True)

    proj = rox.project

    _roxapi_import_cube(self, rox, proj, name, folder)


def _roxapi_import_cube(self, rox, proj, name, folder):  # pragma: no cover
    """Short summary.

    Args:
        proj (object): RMS magic project.
        name (str): Name of cube.
        folder (str): Cube folder in RMS.

    """
    # note that name must be in brackets
    path = [name]
    if folder is not None:
        fld = folder.split("/")
        path = fld + path

    if path not in proj.seismic.data.keys():
        raise ValueError(
            "Path {} is not within RMS Seismic Cube container".format(path)
        )
    try:
        rcube = proj.seismic.data[path]
        _roxapi_cube_to_xtgeo(self, rox, rcube)
    except KeyError as emsg:
        logger.error(emsg)
        raise


def _roxapi_cube_to_xtgeo(self, rox, rcube):  # pragma: no cover
    """Tranforming cube from ROXAPI to XTGeo object."""
    logger.info("Cube from roxapi to xtgeo...")

    # roxrotation is cube rotation clockwise from azimuth but not consistent
    if rox.version_required("1.4"):
        roxrotation = rcube.orientation
    else:
        roxrotation = rcube.rotation * -1

    roxhandedness = str(rcube.handedness)

    self._xori, self._yori = rcube.origin
    self._zori = rcube.first_z
    self._zinc = rcube.sample_rate
    self._ncol, self._nrow, self._nlay = rcube.dimensions
    self._xinc, self._yinc = rcube.increment

    self._rotation = roxrotation * -1

    if self._rotation < 0:
        self._rotation += 360
    elif self._rotation > 360:
        self._rotation -= 360

    self._yflip = 1
    if roxhandedness == "right":
        self._yflip = -1

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
        self.values = rcube.get_values()


def export_cube_roxapi(
    self, project, name, folder=None, domain="time", compression=("wavelet", 5)
):  # pragma: no cover
    """Export (store) a Seismic cube to RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

    logger.debug("TODO: compression %s", compression)

    _roxapi_export_cube(
        self,
        rox.project,
        rox,
        name,
        folder=folder,
        domain=domain,
        compression=compression,
    )

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_export_cube(
    self, proj, rox, name, folder=None, domain="time", compression=("wavelet", 5)
):  # type: ignore # pragma: no cover

    roxar = None
    try:
        import roxar  # type: ignore
    except ImportError:
        pass

    logger.info(
        "There are issues with compression %s, hence it is ignored", compression
    )

    path = []
    if folder is not None:
        fld = folder.split("/")
        path = fld + path

    rcube = proj.seismic.data.create_cube(name, path=path)

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

    handedness = roxar.Direction.left
    if self.yflip == -1:
        handedness = roxar.Direction.right

    # inline xline vector
    ilstart = self.ilines[0]
    xlstart = self.xlines[0]
    ilincr = self.ilines[1] - self.ilines[0]
    xlincr = self.xlines[1] - self.xlines[0]

    if rox.version_required("1.4"):
        rcube.set_seismic(
            values,
            origin,
            increment,
            first_z,
            sample_rate,
            rotation * -1,
            vertical_domain=vertical_domain,
            handedness=handedness,
            inline_crossline_start=(ilstart, xlstart),
            inline_crossline_increment=(ilincr, xlincr),
        )
    else:
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
