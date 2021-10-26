"""Import multiple surfaces"""
# pylint: disable=protected-access

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_grid3d(grid, subgrids, rfactor):
    """Get surfaces, subtype and order from 3D grid, including subgrids"""

    logger.info("Extracting surface from 3D grid...")

    # determine layers
    layers = []
    names = []
    if subgrids and grid.subgrids is not None:
        last = ""
        for sgrd, srange in grid.subgrids.items():
            layers.append(str(srange[0]) + "_top")
            names.append(sgrd + "_top")
            last = str(srange[-1])
            lastname = sgrd
        # base of last layer
        layers.append(last + "_base")
        names.append(lastname + "_base")
    else:
        layers.append("top")
        names.append("top")
        layers.append("base")
        names.append("base")

    # next extract these layers
    layerstack = []
    for inum, lay in enumerate(layers):
        layer = xtgeo.surface_from_grid3d(
            grid, template=None, where=lay, rfactor=rfactor
        )
        layer.name = names[inum]
        layerstack.append(layer)

    logger.info("Extracting surface from 3D grid... DONE")

    return layerstack, "tops", "stratigraphic"
