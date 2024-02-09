"""Operations along a well, private module."""

from copy import deepcopy

import numpy as np
import pandas as pd

from xtgeo import _cxtgeo
from xtgeo.common._xyz_enum import _AttrType
from xtgeo.common.constants import UNDEF_INT, UNDEF_INT_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.common.sys import _get_carray
from xtgeo.xyz.points import Points

logger = null_logger(__name__)


def rescale(self, delta=0.15, tvdrange=None):
    """Rescale by using a new MD increment.

    The rescaling is technically done by interpolation in the Pandas dataframe
    """
    pdrows = pd.options.display.max_rows
    pd.options.display.max_rows = 999

    dfrcolumns0 = self.get_dataframe(copy=False).columns

    if self.mdlogname is None:
        self.geometrics()

    dfrcolumns1 = self.get_dataframe(copy=False).columns
    columnsadded = list(set(dfrcolumns1) - set(dfrcolumns0))  # new tmp columns, if any

    dfr = self.get_dataframe().set_index(self.mdlogname)

    logger.debug("Initial dataframe\n %s", dfr)

    start = dfr.index[0]
    stop = dfr.index[-1]
    startt = start
    stopt = stop

    if tvdrange and isinstance(tvdrange, tuple) and len(tvdrange) == 2:
        tvd1, tvd2 = tvdrange

        try:
            startt = dfr.index[dfr[self._wdata.zname] >= tvd1][0]
        except IndexError:
            startt = start

        try:
            stopt = dfr.index[dfr[self._wdata.zname] >= tvd2][0]
        except IndexError:
            stopt = stop

    dfr1 = dfr[start:startt]
    dfr2 = dfr[stopt:stop]

    nentry = int(round((stopt - startt) / delta))

    dfr = dfr.reindex(dfr.index.union(np.linspace(startt, stopt, num=nentry)))
    dfr = dfr.interpolate("index", limit_area="inside").loc[
        np.linspace(startt, stopt, num=nentry)
    ]

    dfr = pd.concat([dfr1, dfr, dfr2], sort=False)

    dfr.drop_duplicates(inplace=True)
    dfr[self.mdlogname] = dfr.index
    dfr.reset_index(inplace=True, drop=True)

    for lname in dfr.columns:
        if lname in self.wlogtypes:
            ltype = self.wlogtypes[lname]
            if ltype == _AttrType.DISC.value:
                dfr = dfr.round({lname: 0})

    logger.debug("Updated dataframe:\n%s", dfr)

    pd.options.display.max_rows = pdrows  # reset

    self.set_dataframe(dfr)
    if columnsadded:
        self.delete_log(columnsadded)


def make_zone_qual_log(self, zqname):
    """Make a flag log based on stratigraphic relations."""
    if zqname in self.get_dataframe(copy=False):
        logger.warning("Quality log %s exists, will be overwritten", zqname)

    if not self.zonelogname or self.zonelogname not in self.get_dataframe(copy=False):
        raise ValueError("Cannot find a zonelog")

    dff = self.get_filled_dataframe()
    dff["ztmp"] = dff[self.zonelogname].copy()
    dff["ztmp"] = (dff.ztmp != dff.ztmp.shift()).cumsum()

    sgrp = dff.groupby("ztmp")

    dff[zqname] = dff[self.zonelogname] * 0

    idlist = []
    seq = []
    for idx, grp in sgrp:
        izns = int(grp[self.zonelogname].mean())
        seq.append(izns)
        idlist.append(idx)

    codes = {
        0: "UNDETERMINED",
        1: "INCREASE",
        2: "DECREASE",
        3: "U_TURN",
        4: "INV_U_TURN",
        9: "INCOMPLETE",
    }

    code = []
    for ind, iseq in enumerate(seq):
        if ind in (0, len(seq) - 1):
            code.append(0)
        else:
            prev_ = seq[ind - 1]
            next_ = seq[ind + 1]
            if prev_ > UNDEF_INT_LIMIT or next_ > UNDEF_INT_LIMIT:
                code.append(9)
            elif next_ > iseq > prev_:
                code.append(1)
            elif next_ < iseq < prev_:
                code.append(2)
            elif next_ < iseq > prev_:
                code.append(3)
            elif next_ > iseq < prev_:
                code.append(4)
    dcode = dict(zip(idlist, code))

    # now create the new log
    self.create_log(zqname, logtype=_AttrType.DISC.value, logrecord=codes)
    dataframe = self.get_dataframe()
    for key, val in dcode.items():
        dataframe.loc[dff["ztmp"] == key, zqname] = val

    self.set_dataframe(dataframe)
    # set the metadata
    self.set_logtype(zqname, _AttrType.DISC.value)
    self.set_logrecord(zqname, codes)
    self._ensure_consistency()

    del dff


def make_ijk_from_grid(self, grid, grid_id="", algorithm=1, activeonly=True):
    """Make an IJK log from grid indices."""
    logger.debug("Using algorithm %s in %s", algorithm, __name__)

    if algorithm == 1:
        _make_ijk_from_grid_v1(self, grid, grid_id=grid_id)
    else:
        _make_ijk_from_grid_v2(self, grid, grid_id=grid_id, activeonly=activeonly)

    logger.debug("Using algorithm %s in %s done", algorithm, __name__)


def _make_ijk_from_grid_v1(self, grid, grid_id=""):
    """Getting IJK from a grid and make as well logs.

    This is the first version, using _cxtgeo.grd3d_well_ijk from C
    """
    logger.debug("Using algorithm 1 in %s", __name__)

    wxarr = _get_carray(self.get_dataframe(copy=False), self.wlogtypes, self.xname)
    wyarr = _get_carray(self.get_dataframe(copy=False), self.wlogtypes, self.yname)
    wzarr = _get_carray(self.get_dataframe(copy=False), self.wlogtypes, self.zname)

    nlen = self.nrow
    wivec = _cxtgeo.new_intarray(nlen)
    wjvec = _cxtgeo.new_intarray(nlen)
    wkvec = _cxtgeo.new_intarray(nlen)

    onelayergrid = grid.copy()
    onelayergrid.reduce_to_one_layer()

    cstatus = _cxtgeo.grd3d_well_ijk(
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._coordsv,
        grid._zcornsv,
        grid._actnumsv,
        onelayergrid._zcornsv,
        onelayergrid._actnumsv,
        self.nrow,
        wxarr,
        wyarr,
        wzarr,
        wivec,
        wjvec,
        wkvec,
        0,
    )

    if cstatus != 0:
        raise RuntimeError(f"Error from C routine, code is {cstatus}")

    indarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wivec).astype("float")
    jndarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wjvec).astype("float")
    kndarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wkvec).astype("float")

    indarray[indarray == 0] = np.nan
    jndarray[jndarray == 0] = np.nan
    kndarray[kndarray == 0] = np.nan

    icellname = "ICELL" + grid_id
    jcellname = "JCELL" + grid_id
    kcellname = "KCELL" + grid_id

    self._wdata.data[icellname] = indarray
    self._wdata.data[jcellname] = jndarray
    self._wdata.data[kcellname] = kndarray

    for cellname in [icellname, jcellname, kcellname]:
        self.set_logtype(cellname, _AttrType.DISC.value)

    self.set_logrecord(icellname, {ncel: str(ncel) for ncel in range(1, grid.ncol + 1)})
    self.set_logrecord(jcellname, {ncel: str(ncel) for ncel in range(1, grid.nrow + 1)})
    self.set_logrecord(kcellname, {ncel: str(ncel) for ncel in range(1, grid.nlay + 1)})

    _cxtgeo.delete_intarray(wivec)
    _cxtgeo.delete_intarray(wjvec)
    _cxtgeo.delete_intarray(wkvec)
    _cxtgeo.delete_doublearray(wxarr)
    _cxtgeo.delete_doublearray(wyarr)
    _cxtgeo.delete_doublearray(wzarr)

    del onelayergrid


def _make_ijk_from_grid_v2(self, grid, grid_id="", activeonly=True):
    """Getting IJK from a grid and make as well logs.

    This is a newer version using grid.get_ijk_from_points. This one
    is believed to be more precise!
    """
    # establish a Points instance and make points dataframe from well trajectory X Y Z
    wpoints = Points()
    wpdf = self.get_dataframe().loc[:, [self.xname, self.yname, self.zname]]
    wpdf.reset_index(inplace=True, drop=True)
    wpoints.set_dataframe(wpdf)

    # column names
    cna = ("ICELL" + grid_id, "JCELL" + grid_id, "KCELL" + grid_id)

    df = grid.get_ijk_from_points(
        wpoints,
        activeonly=activeonly,
        zerobased=False,
        dataframe=True,
        includepoints=False,
        columnnames=cna,
        fmt="float",
        undef=np.nan,
    )

    # The resulting df shall have same length as the well's dataframe,
    # but the well index may not start from one. So first ignore index, then
    # re-establish
    wellindex = self.get_dataframe(copy=False).index

    newdf = pd.concat([self.get_dataframe().reset_index(drop=True), df], axis=1)
    newdf.index = wellindex

    self.set_dataframe(newdf)


def get_gridproperties(self, gridprops, grid=("ICELL", "JCELL", "KCELL"), prop_id=""):
    """Getting gridproperties as logs.

    The routine will make grid_coordinates from grid with make_ijk_from_grid(), or reuse
    existing vectors if grid is a tuple (much faster).
    """
    from xtgeo.grid3d.grid import Grid
    from xtgeo.grid3d.grid_properties import GridProperties
    from xtgeo.grid3d.grid_property import GridProperty

    if not isinstance(gridprops, (GridProperty, GridProperties)):
        raise ValueError('"gridprops" not a GridProperties or GridProperty instance')

    if isinstance(gridprops, GridProperty):
        gprops = GridProperties()
        gprops.append_props([gridprops])
    else:
        gprops = gridprops

    ijk_logs_created_tmp = False
    if isinstance(grid, tuple):
        icl, jcl, kcl = grid
    elif isinstance(grid, Grid):
        self.make_ijk_from_grid(grid, grid_id="_tmp", algorithm=2)
        icl, jcl, kcl = ("ICELL_tmp", "JCELL_tmp", "KCELL_tmp")
        ijk_logs_created_tmp = True
    else:
        raise ValueError("The 'grid' is of wrong type, must be a tuple or a Grid")

    # let grid values have base 1 when looking up cells for gridprops
    iind = self.get_dataframe(copy=False)[icl].to_numpy(copy=True) - 1
    jind = self.get_dataframe(copy=False)[jcl].to_numpy(copy=True) - 1
    kind = self.get_dataframe(copy=False)[kcl].to_numpy(copy=True) - 1

    xind = iind.copy()

    iind[np.isnan(iind)] = 0
    jind[np.isnan(jind)] = 0
    kind[np.isnan(kind)] = 0

    iind = iind.astype("int")
    jind = jind.astype("int")
    kind = kind.astype("int")
    dfr = self.get_dataframe()

    pnames = {}
    for prop in gprops.props:
        arr = prop.values[iind, jind, kind].astype("float")
        arr = np.ma.filled(arr, fill_value=np.nan)
        arr[np.isnan(xind)] = np.nan
        pname = prop.name + prop_id
        dfr[pname] = arr
        pnames[pname] = (prop.isdiscrete, deepcopy(prop.codes))

    self.set_dataframe(dfr)
    for pname, isdiscrete_codes in pnames.items():
        isdiscrete, codes = isdiscrete_codes
        if isdiscrete:
            self.set_logtype(pname, _AttrType.DISC.value)
            self.set_logrecord(pname, codes)
        else:
            self.set_logtype(pname, _AttrType.CONT.value)
            self.set_logrecord(pname, ("", ""))

    if ijk_logs_created_tmp:
        self.delete_logs(["ICELL_tmp", "JCELL_tmp", "KCELL_tmp"])


def report_zonation_holes(self, threshold=5):
    """Reports if well has holes in zonation, less or equal to N samples."""

    if self.zonelogname is None:
        raise RuntimeError("No zonelog present for well")

    wellreport = []

    zlog = self._wdata.data[self.zonelogname].values.copy()

    mdlog = None
    if self.mdlogname:
        mdlog = self._wdata.data[self.mdlogname].values

    xvv = self._wdata.data[self.xname].values
    yvv = self._wdata.data[self.yname].values
    zvv = self._wdata.data[self.zname].values
    zlog[np.isnan(zlog)] = UNDEF_INT

    ncv = 0
    first = True
    hole = False
    for ind, zone in np.ndenumerate(zlog):
        ino = ind[0]
        if zone > UNDEF_INT_LIMIT and first:
            continue

        if zone < UNDEF_INT_LIMIT and first:
            first = False
            continue

        if zone > UNDEF_INT_LIMIT:
            ncv += 1
            hole = True

        if zone > UNDEF_INT_LIMIT and ncv > threshold:
            logger.debug("Restart first (bigger hole)")
            hole = False
            first = True
            ncv = 0
            continue

        if hole and zone < UNDEF_INT_LIMIT and ncv <= threshold:
            # here we have a hole that fits criteria
            if mdlog is not None:
                entry = (
                    ino,
                    xvv[ino],
                    yvv[ino],
                    zvv[ino],
                    int(zone),
                    self.xwellname,
                    mdlog[ino],
                )
            else:
                entry = (ino, xvv[ino], yvv[ino], zvv[ino], int(zone), self.xwellname)

            wellreport.append(entry)

            # restart count
            hole = False
            ncv = 0

        if hole and zone < UNDEF_INT_LIMIT and ncv > threshold:
            hole = False
            ncv = 0

    if not wellreport:  # ie length is 0
        return None

    if mdlog is not None:
        clm = ["INDEX", self.xname, self.yname, self.zname, "Zone", "Well", "MD"]
    else:
        clm = ["INDEX", self.xname, self.yname, self.zname, "Zone", "Well"]

    return pd.DataFrame(wellreport, columns=clm)


def mask_shoulderbeds(self, inputlogs, targetlogs, nsamples, strict):
    """Mask targetlogs around discrete boundaries.

    Returns True if inputlog(s) and targetlog(s) are present; otherwise False.
    """
    logger.debug("Mask shoulderbeds for some logs...")

    useinputs, usetargets, use_numeric = _mask_shoulderbeds_checks(
        self, inputlogs, targetlogs, nsamples, strict
    )

    if not useinputs or not usetargets:
        logger.debug("Mask shoulderbeds for some logs... nothing done")
        return False

    for inlog in useinputs:
        inseries = self._wdata.data[inlog]
        if use_numeric:
            bseries = _get_bseries(inseries, nsamples)
        else:
            mode, value = list(nsamples.items())[0]

            depth = self._wdata.data[self.zname]
            if mode == "md" and self.mdlogname is not None:
                depth = self._wdata.data[self.mdlogname]
            elif mode == "md" and self.mdlogname is None:
                raise ValueError("There is no mdlogname attribute present.")

            bseries = _get_bseries_by_distance(depth, inseries, value)

        for target in usetargets:
            self._wdata.data.loc[bseries, target] = np.nan

    logger.debug("Mask shoulderbeds for some logs... done")
    return True


def _mask_shoulderbeds_checks(self, inputlogs, targetlogs, nsamples, strict):
    """Checks/validates input for mask targetlogs around discrete boundaries."""
    # check that inputlogs exists and that they are discrete, and targetlogs
    useinputs = []
    for inlog in inputlogs:
        if inlog not in self.wlogtypes and strict is True:
            raise ValueError(f"Input log {inlog} is missing and strict=True")
        if inlog in self.wlogtypes and self.wlogtypes[inlog] != _AttrType.DISC.value:
            raise ValueError(f"Input log {inlog} is not of type DISC")
        if inlog in self.wlogtypes:
            useinputs.append(inlog)

    usetargets = []
    for target in targetlogs:
        if target not in self.wlogtypes and strict is True:
            raise ValueError(f"Target log {target} is missing and strict=True")
        if target in self.wlogtypes:
            usetargets.append(target)

    use_numeric = True
    if isinstance(nsamples, int):
        maxlen = self.nrow // 2
        if nsamples < 1 or nsamples > maxlen:
            raise ValueError(f"Keyword nsamples must be an int > 1 and < {maxlen}")
    elif isinstance(nsamples, dict):
        if len(nsamples) == 1 and any(key in nsamples for key in ["md", "tvd"]):
            use_numeric = False
        else:
            raise ValueError(f"Keyword nsamples is incorrect in some way: {nsamples}")
    else:
        raise ValueError("Keyword nsamples is not an int or a dictionary")

    # return a list of input logs to be used (useinputs), a list of target logs to
    # be used (usetargets) and a use_numeric bool (True if nsamples is an int)
    return useinputs, usetargets, use_numeric


def _get_bseries(inseries, nsamples):
    """Return a bool filter based on number of samples."""
    if not isinstance(inseries, pd.Series):
        raise RuntimeError("Bug, input must be a pandas Series() instance.")

    if len(inseries) == 0:
        return pd.Series([], dtype=bool)

    # nsmaples < 1 or input series with <= 1 element will not be prosessed
    if nsamples < 1 or len(inseries) <= 1:
        return pd.Series(inseries, dtype=bool).replace(True, False)

    def _growfilter(bseries, nleft):
        if not nleft:
            return bseries

        return _growfilter(bseries | bseries.shift(-1) | bseries.shift(1), nleft - 1)

    # make a tmp mask log (series) based input logs and use that for mask filterings
    transitions = inseries.diff().abs() > 0
    bseries = transitions | transitions.shift(-1)

    return _growfilter(bseries, nsamples - 1)


def _get_bseries_by_distance(depth, inseries, distance):
    """Return a bool filter defined by distance to log breaks."""
    if not isinstance(inseries, pd.Series):
        raise RuntimeError("BUG: input must be a pandas Series() instance.")

    if len(inseries) == 0:
        return pd.Series([], dtype=bool)

    # Input series with <= 1 element will not be prosessed
    if len(inseries) <= 1:
        return pd.Series(inseries, dtype=bool).replace(True, False)

    bseries = pd.Series(np.zeros(inseries.values.size), dtype="int32").values
    try:
        inseries = np.nan_to_num(inseries.values, nan=UNDEF_INT).astype("int32")
    except TypeError:
        # for older numpy version
        inseries = inseries.values
        inseries[np.isnan(inseries)] = UNDEF_INT
        inseries = inseries.astype("int32")

    res = _cxtgeo.well_mask_shoulder(
        depth.values.astype("float64"), inseries, bseries, distance
    )

    if res != 0:
        raise RuntimeError("BUG: return from _cxtgeo.well_mask_shoulder not zero")

    return np.array(bseries, dtype=bool)


def create_surf_distance_log(self, surf, name):
    """Create a log which is vertical distance from a RegularSurface."""
    from xtgeo.surface.regular_surface import RegularSurface

    logger.debug("Create a log which is distance to surface")

    if not isinstance(surf, RegularSurface):
        raise ValueError("Input surface is not a RegularSurface instance.")

    # make a Points instance since points has the snap
    zvalues = self.get_dataframe()[self.zname]
    points = Points()
    dframe = self.get_dataframe().iloc[:, 0:3]
    points.set_dataframe(dframe)
    points.snap_surface(surf)
    snapped = points.get_dataframe(copy=False)[self.zname]
    diff = snapped - zvalues

    # create log (default is force overwrite if it exists)
    self.create_log(name)
    dframe = self.get_dataframe()
    dframe[name] = diff
    self.set_dataframe(dframe)
