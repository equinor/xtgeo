# coding: utf-8
"""Various operations on XYZ data"""
from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

import shapely.geometry as sg

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access


def operation_polygons(self, poly, value, opname='add', inside=True,
                       where=True):
    """
    Operations restricted to closed polygons, for points or polyline points.
    """

    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError('The poly input is not a Polygons instance')

    # make a temporary column in the input dataframe as "filter" or "proxy"
    # value will be 1 inside polygons, 0 outside. If where is applied and is
    # False, the -1 is used as flag

    idgroups = poly.dataframe.groupby(poly.pname)

    xcor = self._df[self.xname].values
    ycor = self._df[self.yname].values
    proxy = np.zeros(xcor.shape, dtype='int')

    points = sg.MultiPoint(np.stack([xcor, ycor], axis=1))
    for inum, point in enumerate(points):

        for id_, grp in idgroups:
            pxcor = grp[poly.xname].values
            pycor = grp[poly.yname].values
            spoly = sg.Polygon(np.stack([pxcor, pycor], axis=1))

            if point.within(spoly):
                proxy[inum] = 1

    if not isinstance(where, pd.Series):
        if where:
            where_array = np.ones(proxy.shape, dtype=bool)
        else:
            where_array = np.zeros(proxy.shape, dtype=bool)
    else:
        where_array = where.values

    proxy[~where_array] = -1

    dfwork = self._df.copy()
    dfwork['_PROXY'] = proxy

    proxytarget = 1
    if not inside:
        proxytarget = 0

    cond = dfwork['_PROXY'] == proxytarget

    if opname == 'add':
        dfwork.loc[cond, self.zname] += value

    elif opname == 'sub':
        dfwork.loc[cond, self.zname] -= value

    elif opname == 'mul':
        dfwork.loc[cond, self.zname] *= value

    elif opname == 'div':
        if value != 0.0:
            dfwork.loc[cond, self.zname] /= value
        else:
            dfwork.loc[cond, self.zname] *= 0.0

    elif opname == 'set':
        dfwork.loc[cond, self.zname] = value

    elif opname == 'eli':
        dfwork = dfwork[~cond]

    dfwork.drop(['_PROXY'], inplace=True, axis=1)

    self._df = dfwork.reset_index(drop=True)
