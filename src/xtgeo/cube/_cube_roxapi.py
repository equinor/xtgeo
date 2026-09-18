# coding: utf-8
"""Roxar API functions for XTGeo Cube

Note on rotation:

xtgeo uses rotation of "columns" which is xline direction counterclockwise
measured from X axis.

roxarapi uses rotation of inline direction (rows) relative to Y axis.
api < 1.4: counterclockwise "rotation"
api >= 1.4 clockwise "orientation"

Seems like cube._rotation == roxar.orientation * -1 anyway @ reverse engineering/testing

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.roxutils import RoxUtils
from xtgeo.roxutils._roxar_loader import roxar

if TYPE_CHECKING:
    from .cube1 import Cube

xtg = XTGeoDialog()

logger = null_logger(__name__)


def import_cube_roxapi(
    cube: Cube, project: Any, name: str, folder: str | None = None
) -> None:  # pragma: no cover
    """Import (transfer) a Cube via ROXAR API container to XTGeo.

    .. versionadded:: 2.1
    """
    rox = RoxUtils(project, readonly=True)

    proj = rox.project

    _roxapi_import_cube(cube, rox, proj, name, folder)


def _roxapi_import_cube(
    cube: Cube, rox: RoxUtils, proj: Any, name: str, folder: str | None
) -> None:  # pragma: no cover
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

    if path not in proj.seismic.data:
        raise ValueError(f"Path {path} is not within RMS Seismic Cube container")
    try:
        rcube = proj.seismic.data[path]
        _roxapi_cube_to_xtgeo(cube, rox, rcube)
    except KeyError as emsg:
        logger.error(emsg)
        raise


def _roxapi_cube_to_xtgeo(
    cube: Cube, rox: RoxUtils, rcube: Any
) -> None:  # pragma: no cover
    """Transforming cube from ROXAPI to XTGeo object."""
    logger.info("Cube from roxapi to xtgeo...")

    # roxrotation is cube rotation clockwise from azimuth but not consistent
    roxrotation = rcube.orientation

    roxhandedness = str(rcube.handedness)

    cube._xori, cube._yori = rcube.origin
    cube._zori = rcube.first_z
    cube._zinc = rcube.sample_rate
    cube._ncol, cube._nrow, cube._nlay = rcube.dimensions
    cube._xinc, cube._yinc = rcube.increment

    cube._rotation = roxrotation * -1

    if cube._rotation < 0:
        cube._rotation += 360
    elif cube._rotation > 360:
        cube._rotation -= 360

    cube._yflip = 1
    if roxhandedness == "right":
        cube._yflip = -1

    il_start = rcube.get_inline(0)
    xl_start = rcube.get_crossline(0)
    il_incr, xl_incr = rcube.inline_crossline_increment
    il_end = (cube._ncol) * il_incr + il_start
    xl_end = (cube._nrow) * xl_incr + xl_start

    cube._ilines = np.array(range(il_start, il_end, il_incr), dtype=np.int32)
    cube._xlines = np.array(range(xl_start, xl_end, xl_incr), dtype=np.int32)

    # roxar API does not store traceid codes, assume 1
    cube._traceidcodes = np.ones((cube._ncol, cube._nrow), dtype=np.int32)

    if rcube.is_empty:
        xtg.warn("Cube has no data; assume 0")
    else:
        cube.values = rcube.get_values()


def export_cube_roxapi(
    cube: Cube,
    project: Any,
    name: str,
    folder: str | None = None,
    domain: str = "time",
    compression: tuple[str, float] = ("wavelet", 5),
) -> None:  # pragma: no cover
    """Export (store) a Seismic cube to RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

    logger.debug("TODO: compression %s", compression)

    _roxapi_export_cube(
        cube,
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
    cube: Cube,
    proj: Any,
    rox: RoxUtils,
    name: str,
    folder: str | None = None,
    domain: str = "time",
    compression: tuple[str, float] = ("wavelet", 5),
) -> None:  # pragma: no cover
    logger.info(
        "There are issues with compression %s, hence it is ignored", compression
    )

    if roxar is None:
        raise RuntimeError(
            "The 'roxar'/'rmsapi' module is not available. This function can "
            "only be run inside an RMS environment."
        )

    path: list[str] = []
    if folder is not None:
        fld = folder.split("/")
        path = fld + path

    rcube = proj.seismic.data.create_cube(name, path=path)

    # populate
    origin = (float(cube.xori), float(cube.yori))
    first_z = cube.zori
    increment = (cube.xinc, cube.yinc)
    sample_rate = cube.zinc
    rotation = cube.rotation
    vertical_domain = roxar.VerticalDomain.time
    if domain == "depth":
        vertical_domain = roxar.VerticalDomain.depth

    values = cube.values.copy()  # copy() needed?

    handedness = roxar.Direction.left
    if cube.yflip == -1:
        handedness = roxar.Direction.right

    # inline xline vector
    ilstart = cube.ilines[0]
    xlstart = cube.xlines[0]
    ilincr = cube.ilines[1] - cube.ilines[0]
    xlincr = cube.xlines[1] - cube.xlines[0]

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
