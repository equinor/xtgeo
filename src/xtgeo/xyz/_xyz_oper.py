# coding: utf-8
"""Various operations on XYZ data"""
from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline

import shapely.geometry as sg

import xtgeo
from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()

# pylint: disable=protected-access


def operation_polygons(self, poly, value, opname="add", inside=True, where=True):
    """
    Operations re restricted to closed polygons, for points or polyline points.

    If value is not float but 'poly', then the avg of each polygon Z value will
    be used instead.

    'Inside' several polygons will become a union, while 'outside' polygons
    will be the intersection.

    The "where" filter is reserved for future use.
    """

    logger.warning("Where is not imeplented: %s", where)

    oper = {"set": 1, "add": 2, "sub": 3, "mul": 4, "div": 5, "eli": 11}

    insidevalue = 0
    if inside:
        insidevalue = 1

    logger.info("Operations of points inside polygon(s)...")
    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError("The poly input is not a Polygons instance")

    idgroups = poly.dataframe.groupby(poly.pname)

    xcor = self._df[self.xname].values
    ycor = self._df[self.yname].values
    zcor = self._df[self.zname].values

    usepoly = False
    if isinstance(value, str) and value == "poly":
        usepoly = True

    for id_, grp in idgroups:
        pxcor = grp[poly.xname].values
        pycor = grp[poly.yname].values
        pvalue = value
        if usepoly:
            pvalue = grp[poly.zname].values.mean()
        else:
            pvalue = value

        logger.info("C function for polygon %s...", id_)

        ies = _cxtgeo.pol_do_points_inside(
            xcor, ycor, zcor, pxcor, pycor, pvalue, oper[opname], insidevalue, XTGDEBUG
        )
        logger.info("C function for polygon %s... done", id_)

        if ies != 0:
            raise RuntimeError("Something went wrong, code {}".format(ies))

    zcor[zcor > xtgeo.UNDEF_LIMIT] = np.nan
    self._df[self.zname] = zcor
    # removing rows where Z column is undefined
    self._df.dropna(how="any", subset=[self.zname], inplace=True)
    self._df.reset_index(inplace=True, drop=True)
    logger.info("Operations of points inside polygon(s)... done")


def rescale_polygons(self, distance=10, addhlen=True, kind="slinear"):
    """Rescale (resample) a polygons segment"""

    if kind in ("slinear", "cubic"):
        _rescale_v2(self, distance, addhlen, kind=kind)

    else:
        _rescale_v1(self, distance, addhlen)


def _rescale_v1(self, distance, addhlen):
    # version 1, simple approach

    if not self._ispolygons:
        raise ValueError("Not a Polygons object")

    idgroups = self.dataframe.groupby(self.pname)

    dfrlist = []
    for idx, grp in idgroups:
        pxcor = grp[self.xname].values
        pycor = grp[self.yname].values
        pzcor = grp[self.zname].values
        spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))

        new_spoly = _redistribute_vertices(spoly, distance)

        dfr = pd.DataFrame(
            np.array(new_spoly), columns=[self.xname, self.yname, self.zname]
        )

        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.dataframe = dfr.reset_index(drop=True)

    if addhlen:
        self.hlen()
    else:
        self.delete_columns([self.hname, self.dhname])

    return None


def _redistribute_vertices(geom, distance):
    """Local function to interpolate in a polyline using Shapely"""
    if geom.geom_type == "LineString":
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return sg.LineString(
            [
                geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )

    if geom.geom_type == "MultiLineString":
        parts = [_redistribute_vertices(part, distance) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])

    raise ValueError("Unhandled geometry {}".format(geom.geom_type))


def _rescale_v2(self, distance, addhlen, kind="slinear"):

    # Rescaling to constant increment is perhaps impossible, but this is
    # is quite close

    self.hlen()

    idgroups = self.dataframe.groupby(self.pname)

    dfrlist = []
    for idx, grp in idgroups:

        points = [grp[self.xname], grp[self.yname], grp[self.zname]]
        print(points)
        lenh = grp[self.hname].iloc[-1]
        nstep = int(lenh / distance)
        alpha = np.linspace(0, lenh, num=nstep, endpoint=True)

        if kind == "slinear":
            points = np.array(points).T
            interpolator = interp1d(
                grp[self.hname], points, kind="slinear", axis=0, assume_sorted=True
            )
            ip = interpolator(alpha)
        elif kind == "cubic":
            splines = [
                UnivariateSpline(grp[self.hname], crd) for crd in points
            ]
            for spl in splines:
                print(spl(alpha))

            ip = np.vstack(spl(alpha) for spl in splines).T
            print(ip)
        else:
            raise ValueError("Invalid kind chosen: {}".format(kind))

        dfr = pd.DataFrame(np.array(ip), columns=[self.xname, self.yname, self.zname])

        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.dataframe = dfr.reset_index(drop=True)

    if addhlen:
        self.hlen()
    else:
        self.delete_columns([self.hname, self.dhname])


def get_fence(
    self,
    distance=20,
    atleast=5,
    nextend=2,
    name=None,
    asnumpy=True,
    polyid=None,
    _version=2,
):
    """Get a fence suitable for plotting xsections, either as a numpy or as a
    new Polygons instance.

    The atleast parameter will win over the distance, meaning that if total length
    horizontally is 50, and distance is set to 20, the actual length will be 50/5=10

    """
    if _version == 1:
        return _fence_v1(self, distance, atleast, nextend, name, asnumpy)

    return _fence_v2(self, distance, atleast, nextend, name, asnumpy, polyid)


def _fence_v1(self, distance, atleast, nextend, name, asnumpy):

    # POLYID NOT IMPLEMENTED!

    if len(self._df) < 2:
        xtg.warn("Well does not enough points in interval, outside range?")
        return False

    hxlen = self.get_shapely_objects()[0].length

    if hxlen / float(atleast) < distance:
        distance = hxlen / float(atleast)

    nbuf = 1000000

    logger.info("%s %s %s %s %s ", distance, atleast, nextend, name, asnumpy)
    npxarr = np.zeros(nbuf, dtype=np.float64)
    npyarr = np.zeros(nbuf, dtype=np.float64)
    npzarr = np.zeros(nbuf, dtype=np.float64)
    npharr = np.zeros(nbuf, dtype=np.float64)

    logger.info("Calling C routine...")
    # C function:
    ier, npxarr, npyarr, npzarr, npharr, nlen = _cxtgeo.pol_resampling(
        self._df[self.xname].values,
        self._df[self.yname].values,
        self._df[self.zname].values,
        distance,
        distance * nextend,
        nbuf,
        nbuf,
        nbuf,
        nbuf,
        0,
        XTGDEBUG,
    )
    logger.info("Calling C routine... DONE")

    if ier != 0:
        raise RuntimeError("Nonzero code from_cxtgeo.pol_resampling")

    npxarr = npxarr[:nlen]
    npyarr = npyarr[:nlen]
    npzarr = npzarr[:nlen]
    npharr = npharr[:nlen]

    npdharr = np.subtract(npharr[1:], npharr[0 : nlen - 1])
    npdharr = np.insert(npdharr, 0, [0.0])

    if asnumpy is True:
        rval = np.concatenate((npxarr, npyarr, npzarr, npharr, npdharr), axis=0)
        rval = np.reshape(rval, (nlen, 5), order="F")
    else:
        rval = xtgeo.xyz.Polygons()
        arr = np.vstack(
            [npxarr, npyarr, npzarr, npharr, npdharr, np.zeros(nlen, dtype=np.int32)]
        )
        col = ["X_UTME", "Y_UTMN", "Z_TVDSS", "HLEN", "DHLEN", "ID"]
        dfr = pd.DataFrame(arr.T, columns=col, dtype=np.float64)
        dfr = dfr.astype({"ID": int})
        if name:
            dfr = dfr.assign(NAME=name)

        rval.dataframe = dfr
        if name:
            rval.name = name

    return rval


def _fence_v2(self, distance, atleast, nextend, name, asnumpy, polyid):

    new = self.copy()
    new.hlen()

    if len(new.dataframe) < 2:
        xtg.warn("Well does not enough points in interval, outside range?")
        return False

    new.filter_byid(polyid)

    hxlen = new.get_shapely_objects()[0].length

    if hxlen / float(atleast) < distance:
        distance = hxlen / float(atleast)

    new.rescale(distance)
    new.extend(distance, nsamples=nextend)

    if name:
        new.name = name

    if asnumpy is True:
        rval = np.concatenate(
            (
                new.dataframe[new.xname].values,
                new.dataframe[new.yname].values,
                new.dataframe[new.zname].values,
                new.dataframe[new.hname].values,
                new.dataframe[new.dhname].values,
            ),
            axis=0,
        )
        return np.reshape(rval, (new.nrow, 5), order="F")

    return new


def snap_surface(self, surf, activeonly=True):
    """Snap (or transfer) operation.

    Points that falls outside the surface will be UNDEF, and they will be removed
    if activeonly. Otherwise, the old values will be kept.
    """

    if not isinstance(surf, xtgeo.RegularSurface):
        raise ValueError("Input object of wrong data type, must be RegularSurface")

    zval = self._df[self.zname].values.copy()

    ier = _cxtgeo.surf_get_zv_from_xyv(
        self._df[self.xname].values,
        self._df[self.yname].values,
        zval,
        surf.ncol,
        surf.nrow,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.yflip,
        surf.rotation,
        surf.get_values1d(),
        XTGDEBUG,
    )

    if ier != 0:
        raise RuntimeError(
            "Error code from C routine surf_get_zv_from_xyv is {}".format(ier)
        )
    if activeonly:
        self._df[self.zname] = zval
        self._df = self._df[self._df[self.zname] < xtgeo.UNDEF_LIMIT]
        self._df.reset_index(inplace=True, drop=True)
    else:
        out = np.where(zval < xtgeo.UNDEF_LIMIT, zval, self._df[self.zname].values)
        self._df[self.zname] = out


def hlen(self, hname="H_CUMLEN", dhname="H_DELTALEN", atindex=0):
    """Get the distance (cumulative and delta) between points in polygons.

    The properties hname and dhname will be updated.

    Note that DH at first location will be set equal to dHat location 1
    """

    if not isinstance(self, xtgeo.Polygons):
        raise ValueError("Input object of wrong data type, must be Polygons")

    # delete existing self.hname and self.dhname columns
    self.delete_columns([self.hname, self.dhname])

    idgroups = self._df.groupby(self.pname)

    hdist = np.array([])
    dhdist = np.array([])
    for _id, grp in idgroups:
        print(grp)
        ier, hlenv, dhlenv = _cxtgeo.pol_geometrics(
            grp[self.xname].values,
            grp[self.yname].values,
            grp[self.zname].values,
            len(grp),
            len(grp),
            XTGDEBUG,
        )
        if ier != 0:
            raise RuntimeError(
                "Error code from _cxtgeo.pol_geometrics is {}".format(ier)
            )

        dhlenv[0] = dhlenv[1]
        if atindex > 0:
            cumval = hlenv[atindex]
            hlenv -= cumval

        hdist = np.append(hdist, hlenv)
        dhdist = np.append(dhdist, dhlenv)

    self._df[hname] = hdist
    self._df[dhname] = dhdist

    self.hname = hname
    self.dhname = dhname


def extend(self, distance, nsamples, addhlen=True):
    """Extend polygon by distance, nsamples times.

    It is default to recompute HLEN from nsamples
    """

    if not isinstance(self, xtgeo.Polygons):
        raise ValueError("Input object of wrong data type, must be Polygons")

    for _nsam in range(nsamples):

        # beginning of poly
        row0 = self._df.iloc[0]
        row1 = self._df.iloc[1]

        rown = row0.copy()

        ier, newx, newy, _newz = _cxtgeo.x_vector_linint2(
            row1[0], row1[1], row1[2], row0[0], row0[1], row0[2], distance, 2, XTGDEBUG
        )

        if ier != 0:
            raise RuntimeError(
                "Error code from _cxtgeo.x_vector_linint2 is {}".format(ier)
            )

        rown[self.xname] = newx
        rown[self.yname] = newy

        df_to_add = rown.to_frame().T

        self._df = pd.concat([df_to_add, self._df]).reset_index(drop=True)

        # end of poly
        row0 = self._df.iloc[-2]
        row1 = self._df.iloc[-1]

        rown = row1.copy()

        ier, newx, newy, _newz = _cxtgeo.x_vector_linint2(
            row0[0], row0[1], row0[2], row1[0], row1[1], row1[2], distance, 1, XTGDEBUG
        )

        rown[self.xname] = newx
        rown[self.yname] = newy

        df_to_add = rown.to_frame().T

        self._df = pd.concat([self._df, df_to_add]).reset_index(drop=True)

    if addhlen:
        self.hlen(atindex=nsamples)
