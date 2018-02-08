"""Various methods for grid properties (cf GridProperties class"""

import pandas as pd
import numpy as np

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')

# self = GridProperties instance


def dataframe(self, activeonly=True, ijk=False, xyz=False,
              doubleformat=False):
    """Returns a Pandas dataframe table for the properties."""

    colnames = []
    proplist = []

    if ijk:
        if activeonly:
            ix, jy, kz = self._grid.get_ijk(mask=True)
            proplist.extend([ix.get_active_values(),
                             jy.get_active_values(),
                             kz.get_active_values()])
        else:
            ix, jy, kz = self._grid.get_ijk(mask=False)
            proplist.extend([ix.values, jy.values, kz.values])

        colnames.extend(['IX', 'JY', 'KZ'])

    if xyz:
        option = False
        if activeonly:
            option = True

        xc, yc, zc = self._grid.get_xyz(mask=option)
        colnames.extend(['X_UTME', 'Y_UTMN', 'Z_TVDSS'])
        if activeonly:
            proplist.extend([xc.get_active_values(),
                             yc.get_active_values(),
                             zc.get_active_values()])
        else:
            proplist.extend([xc.values, yc.values, zc.values])

    for prop in self.props:
        self.logger.info('Getting property {}'.format(prop.name))
        colnames.append(prop.name)
        if activeonly:
            vector = prop.get_active_values()
        else:
            vector = prop.values.copy()
            # mask values not supported in Pandas:
            if prop.isdiscrete:
                vector = vector.filled(fill_value=0)
            else:
                vector = vector.filled(fill_value=np.nan)

        if doubleformat:
            vector = vector.astype(np.float64)
        else:
            vector = vector.astype(np.float32)

        proplist.append(vector)

    dataframe = pd.DataFrame.from_items(zip(colnames, proplist))

    return dataframe
