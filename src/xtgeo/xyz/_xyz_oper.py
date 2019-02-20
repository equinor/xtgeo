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


def rescale_polygons(self, distance=10):
    """Rescale (resample) a polygons segment"""

    if not self._ispolygons:
        raise ValueError('Not a Polygons object')

    idgroups = self.dataframe.groupby(self.pname)

    dfrlist = []
    for idx, grp in idgroups:
        pxcor = grp[self.xname].values
        pycor = grp[self.yname].values
        pzcor = grp[self.zname].values
        spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))

        new_spoly = _redistribute_vertices(spoly, distance)

        dfr = pd.DataFrame(np.array(new_spoly),
                           columns=[self.xname, self.yname, self.zname])
        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.dataframe = dfr.reset_index(drop=True)


def _redistribute_vertices(geom, distance):
    """Local function to interpolate in a polyline"""
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return sg.LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [_redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('Unhandled geometry %s', (geom.geom_type,))
