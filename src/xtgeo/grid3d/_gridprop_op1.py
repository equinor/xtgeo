"""Various grid property operations"""

from __future__ import print_function, absolute_import

import logging
import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_xy_value_lists(self, **kwargs):
    """Get values for webportal format

    Two cells:
    [[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
    [(x5,y5), (x6,y6), (x7,y7), (x8,y8)]]]

    If mask is True then inactive cells are ommited from the lists,
    else the active cells corners will be present while the property
    will have a -999 value.

    """

    grid = kwargs.get('grid', None)

    mask = kwargs.get('mask', True)

    if grid is None:
        raise RuntimeError('Missing grid object')

    if not isinstance(grid, xtgeo.grid3d.Grid):
        raise RuntimeError('The input grid is not a XTGeo Grid instance')

    if not isinstance(self, xtgeo.grid3d.GridProperty):
        raise RuntimeError('The property is not a XTGeo GridProperty instance')

    clist = grid.get_xyz_corners()
    actnum = grid.get_actnum()

    # set value 0 if actnum is 0 to facilitate later operations
    if mask:
        for i in range(len(clist)):
            clist[i].values[actnum.values == 0] = 0

    # now some numpy operations (coffee?, any?)
    xy0 = np.column_stack((clist[0].values1d, clist[1].values1d))
    xy1 = np.column_stack((clist[3].values1d, clist[4].values1d))
    xy2 = np.column_stack((clist[6].values1d, clist[7].values1d))
    xy3 = np.column_stack((clist[9].values1d, clist[10].values1d))

    xyc = np.column_stack((xy0, xy1, xy2, xy3))
    xyc = xyc.reshape(grid.nlay, grid.ncol * grid.nrow, 4, 2)

    coordlist = xyc.tolist()

    # remove cells that are undefined ("marked" as coordinate [0, 0] if mask)
    coordlist = [[[tuple(xy) for xy in cell if xy[0] > 0]
                  for cell in lay] for lay in coordlist]

    coordlist = [[cell for cell in lay if len(cell) > 1] for lay in coordlist]

    pval = self.values1d.reshape((grid.nlay, grid.ncol * grid.nrow))
    valuelist = pval.tolist(fill_value=-999.0)
    if mask:
        valuelist = [[val for val in lay if val != -999.0]
                     for lay in valuelist]

    return coordlist, valuelist
