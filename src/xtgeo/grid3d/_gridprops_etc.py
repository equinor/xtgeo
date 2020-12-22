"""Various methods for grid properties (cf GridProperties class)"""

from collections import OrderedDict

import pandas as pd
import numpy as np

from xtgeo.common import XTGeoDialog

from ._grid3d import _Grid3D

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


#

# self = GridProperties instance


def dataframe(
    self, activeonly=True, ijk=False, xyz=False, doubleformat=False, grid=None
):  # pylint: disable=too-many-branches, too-many-statements
    """Returns a Pandas dataframe table for the properties."""

    proplist = OrderedDict()

    if grid is not None and isinstance(grid, _Grid3D):
        master = grid
        logger.info("Grid is present")
    else:
        master = self
        logger.info("No Grid instance")

    if ijk:
        logger.info("IJK is active")
        if activeonly:
            logger.info("Active cells only")
            ix, jy, kz = master.get_ijk(asmasked=True)
            proplist["IX"] = ix.get_active_npvalues1d()
            proplist["JY"] = jy.get_active_npvalues1d()
            proplist["KZ"] = kz.get_active_npvalues1d()
        else:
            logger.info("All cells (1)")
            act = master.get_actnum(dual=True)
            ix, jy, kz = grid.get_ijk(asmasked=False)
            proplist["ACTNUM"] = act.values1d
            proplist["IX"] = ix.values1d
            proplist["JY"] = jy.values1d
            proplist["KZ"] = kz.values1d

    if xyz:
        if grid is None:
            raise ValueError("You ask for xyz but no Grid is present. Use " "grid=...")

        logger.info("XYZ is active")
        option = False
        if activeonly:
            logger.info("Active cells only")
            option = True

        xc, yc, zc = grid.get_xyz(asmasked=option)
        if activeonly:
            proplist["X_UTME"] = xc.get_active_npvalues1d()
            proplist["Y_UTMN"] = yc.get_active_npvalues1d()
            proplist["Z_TVDSS"] = zc.get_active_npvalues1d()
        else:
            logger.info("All cells (2)")
            proplist["X_UTME"] = xc.values1d
            proplist["Y_UTMN"] = yc.values1d
            proplist["Z_TVDSS"] = zc.values1d

    logger.info("Proplist: %s", proplist)

    if self.props is not None:
        for prop in self.props:
            logger.info("Getting property %s", prop.name)
            if activeonly:
                vector = prop.get_active_npvalues1d()
            else:
                vector = prop.values1d.copy()
                # mask values not supported in Pandas:
                if prop.isdiscrete:
                    vector = vector.filled(fill_value=0)
                else:
                    vector = vector.filled(fill_value=np.nan)

            if doubleformat:
                vector = vector.astype(np.float64)
            else:
                vector = vector.astype(np.float32)

            proplist[prop.name] = vector
    else:
        logger.info("No properties to data frame")

    for key, prop in proplist.items():
        logger.info("Property %s has length %s", key, prop.shape)
    mydataframe = pd.DataFrame.from_dict(proplist)
    logger.debug("Dataframe: \n%s", mydataframe)

    return mydataframe
