"""Operations along a well, private module."""

import copy
from distutils.version import StrictVersion

import numpy as np
import pandas as pd

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import constants as const

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def delete_log(self, lname):
    """Delete/remove an existing log, or list of logs."""
    self._ensure_consistency()

    if not isinstance(lname, list):
        lname = [lname]

    lcount = 0
    for logn in lname:
        if logn not in self._wlognames:
            logger.info("Log does no exist: %s", logn)
            continue

        logger.info("Log exist and will be deleted: %s", logn)
        lcount += 1
        del self._wlogtypes[logn]
        del self._wlogrecords[logn]

        self._df.drop(logn, axis=1, inplace=True)
        self._ensure_consistency()

        if self._mdlogname == logn:
            self._mdlogname = None
        if self._zonelogname == logn:
            self._zonelogname = None

    self._ensure_consistency()
    return lcount


def rescale(self, delta=0.15, tvdrange=None):
    """Rescale by using a new MD increment.

    The rescaling is technically done by interpolation in the Pandas dataframe
    """
    pdrows = pd.options.display.max_rows
    pd.options.display.max_rows = 999

    dfrcolumns0 = self._df.columns

    if self.mdlogname is None:
        self.geometrics()

    dfrcolumns1 = self._df.columns
    columnsadded = list(set(dfrcolumns1) - set(dfrcolumns0))  # new tmp columns, if any

    dfr = self._df.copy().set_index(self.mdlogname)

    logger.debug("Initial dataframe\n %s", dfr)

    start = dfr.index[0]
    stop = dfr.index[-1]
    startt = start
    stopt = stop

    if tvdrange and isinstance(tvdrange, tuple) and len(tvdrange) == 2:
        tvd1, tvd2 = tvdrange

        try:
            startt = dfr.index[dfr["Z_TVDSS"] >= tvd1][0]
        except IndexError:
            startt = start

        try:
            stopt = dfr.index[dfr["Z_TVDSS"] >= tvd2][0]
        except IndexError:
            stopt = stop

    dfr1 = dfr[start:startt]
    dfr2 = dfr[stopt:stop]

    nentry = int(round((stopt - startt) / delta))

    dfr = dfr.reindex(dfr.index.union(np.linspace(startt, stopt, num=nentry)))
    dfr = dfr.interpolate("index", limit_area="inside").loc[
        np.linspace(startt, stopt, num=nentry)
    ]

    if StrictVersion(pd.__version__) > StrictVersion("0.23.0"):
        dfr = pd.concat([dfr1, dfr, dfr2], sort=False)
    else:
        dfr = pd.concat([dfr1, dfr, dfr2])

    dfr.drop_duplicates(inplace=True)
    dfr[self.mdlogname] = dfr.index
    dfr.reset_index(inplace=True, drop=True)

    for lname in dfr.columns:
        if lname in self._wlogtypes:
            ltype = self._wlogtypes[lname]
            if ltype == "DISC":
                dfr = dfr.round({lname: 0})

    logger.debug("Updated dataframe:\n%s", dfr)

    pd.options.display.max_rows = pdrows  # reset

    self._df = dfr
    if columnsadded:
        self.delete_log(columnsadded)


def make_zone_qual_log(self, zqname):
    """Make a flag log based on stratigraphic relations."""
    if zqname in self.dataframe:
        logger.warning("Quality log %s exists, will be overwritten", zqname)

    if not self.zonelogname or self.zonelogname not in self.dataframe:
        raise ValueError("Cannot find a zonelog")

    dff = self.get_filled_dataframe()
    dff["ztmp"] = dff[self.zonelogname]
    dff["ztmp"] = (dff.ztmp != dff.ztmp.shift()).cumsum()

    sgrp = dff.groupby("ztmp")

    dff[zqname] = dff[self.zonelogname] * 0

    idlist = list()
    seq = list()
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

    code = list()
    for ind, iseq in enumerate(seq):
        if ind in (0, len(seq) - 1):
            code.append(0)
        else:
            prev_ = seq[ind - 1]
            next_ = seq[ind + 1]
            if prev_ > const.UNDEF_INT_LIMIT or next_ > const.UNDEF_INT_LIMIT:
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
    self.create_log(zqname, logtype="DISC", logrecord=codes)
    for key, val in dcode.items():
        self._df[zqname][dff["ztmp"] == key] = val

    # set the metadata
    self.set_logtype(zqname, "DISC")
    self.set_logrecord(zqname, codes)

    del dff


def make_ijk_from_grid(self, grid, grid_id="", algorithm=1, activeonly=True):
    """Make an IJK log from grid indices."""
    logger.info("Using algorithm %s in %s", algorithm, __name__)

    if algorithm == 1:
        _make_ijk_from_grid_v1(self, grid, grid_id=grid_id)
    else:
        _make_ijk_from_grid_v2(self, grid, grid_id=grid_id, activeonly=activeonly)

    logger.info("Using algorithm %s in %s done", algorithm, __name__)


def _make_ijk_from_grid_v1(self, grid, grid_id=""):
    """Getting IJK from a grid and make as well logs.

    This is the first version, using _cxtgeo.grd3d_well_ijk from C
    """
    logger.info("Using algorithm 1 in %s", __name__)

    wxarr = self.get_carray("X_UTME")
    wyarr = self.get_carray("Y_UTMN")
    wzarr = self.get_carray("Z_TVDSS")

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
        raise RuntimeError("Error from C routine, code is {}".format(cstatus))

    indarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wivec).astype("float")
    jndarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wjvec).astype("float")
    kndarray = _cxtgeo.swig_carr_to_numpy_i1d(nlen, wkvec).astype("float")

    indarray[indarray == 0] = np.nan
    jndarray[jndarray == 0] = np.nan
    kndarray[kndarray == 0] = np.nan

    icellname = "ICELL" + grid_id
    jcellname = "JCELL" + grid_id
    kcellname = "KCELL" + grid_id

    self._df[icellname] = indarray
    self._df[jcellname] = jndarray
    self._df[kcellname] = kndarray

    for cellname in [icellname, jcellname, kcellname]:
        self._wlogtypes[cellname] = "DISC"

    self._wlogrecords[icellname] = {ncel: str(ncel) for ncel in range(1, grid.ncol + 1)}
    self._wlogrecords[jcellname] = {ncel: str(ncel) for ncel in range(1, grid.nrow + 1)}
    self._wlogrecords[kcellname] = {ncel: str(ncel) for ncel in range(1, grid.nlay + 1)}

    _cxtgeo.delete_intarray(wivec)
    _cxtgeo.delete_intarray(wjvec)
    _cxtgeo.delete_intarray(wkvec)
    _cxtgeo.delete_doublearray(wxarr)
    _cxtgeo.delete_doublearray(wyarr)
    _cxtgeo.delete_doublearray(wzarr)

    del onelayergrid


def _make_ijk_from_grid_v2(self, grid, grid_id="", activeonly=True):
    """Getting IJK from a grid and make as well logs.

    This is a newer version, using grid.get_ijk_from_points which in turn
    use the from C method x_chk_point_in_hexahedron, while v1 use the
    x_chk_point_in_cell. This one is believed to be more precise!
    """
    # establish a Points instance and make points dataframe from well trajectory X Y Z
    wpoints = xtgeo.Points()
    wpdf = self.dataframe.loc[:, ["X_UTME", "Y_UTMN", "Z_TVDSS"]].copy()
    wpoints.dataframe = wpdf
    wpoints.dataframe.reset_index(inplace=True, drop=True)

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
    wellindex = self.dataframe.index

    newdf = pd.concat([self.dataframe.reset_index(drop=True), df], axis=1)
    newdf.index = wellindex

    self.dataframe = newdf


def get_gridproperties(self, gridprops, grid=("ICELL", "JCELL", "KCELL"), prop_id=""):
    """Getting gridproperties as logs."""
    if not isinstance(gridprops, (xtgeo.GridProperty, xtgeo.GridProperties)):
        raise ValueError('"gridprops" not a GridProperties or GridProperty instance')

    if isinstance(gridprops, xtgeo.GridProperty):
        gprops = xtgeo.GridProperties()
        gprops.append_props([gridprops])
    else:
        gprops = gridprops

    if isinstance(grid, tuple):
        icl, jcl, kcl = grid
    elif isinstance(grid, xtgeo.Grid):
        self.make_ijk_from_grid(grid, grid_id="_tmp", algorithm=2)
        icl, jcl, kcl = ("ICELL_tmp", "JCELL_tmp", "KCELL_tmp")
    else:
        raise ValueError('The "grid" is of wrong type, must be a tuple or ' "a Grid")

    iind = self.dataframe[icl].values - 1
    jind = self.dataframe[jcl].values - 1
    kind = self.dataframe[kcl].values - 1

    xind = iind.copy()

    iind[np.isnan(iind)] = 0
    jind[np.isnan(jind)] = 0
    kind[np.isnan(kind)] = 0

    #    iind = np.ma.masked_where(iind[~np.isnan(iind)].astype('int')
    iind = iind.astype("int")
    jind = jind.astype("int")
    kind = kind.astype("int")

    for prop in gprops.props:
        arr = prop.values[iind, jind, kind].astype("float")
        arr[np.isnan(xind)] = np.nan
        pname = prop.name + prop_id
        self.dataframe[pname] = arr
        self._wlognames.append(pname)
        if prop.isdiscrete:
            self._wlogtypes[pname] = "DISC"
            self._wlogrecords[pname] = copy.deepcopy(prop.codes)
    self._ensure_consistency()
    self.delete_logs(["ICELL_tmp", "JCELL_tmp", "KCELL_tmp"])


def report_zonation_holes(self, threshold=5):
    """Reports if well has holes in zonation, less or equal to N samples."""
    # pylint: disable=too-many-branches, too-many-statements

    if self.zonelogname is None:
        raise RuntimeError("No zonelog present for well")

    wellreport = []

    zlog = self._df[self.zonelogname].values.copy()

    mdlog = None
    if self.mdlogname:
        mdlog = self._df[self.mdlogname].values

    xvv = self._df["X_UTME"].values
    yvv = self._df["Y_UTMN"].values
    zvv = self._df["Z_TVDSS"].values
    zlog[np.isnan(zlog)] = const.UNDEF_INT

    ncv = 0
    first = True
    hole = False
    for ind, zone in np.ndenumerate(zlog):
        ino = ind[0]
        if zone > const.UNDEF_INT_LIMIT and first:
            continue

        if zone < const.UNDEF_INT_LIMIT and first:
            first = False
            continue

        if zone > const.UNDEF_INT_LIMIT:
            ncv += 1
            hole = True

        if zone > const.UNDEF_INT_LIMIT and ncv > threshold:
            logger.info("Restart first (bigger hole)")
            hole = False
            first = True
            ncv = 0
            continue

        if hole and zone < const.UNDEF_INT_LIMIT and ncv <= threshold:
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

        if hole and zone < const.UNDEF_INT_LIMIT and ncv > threshold:
            hole = False
            ncv = 0

    if not wellreport:  # ie length is 0
        return None

    if mdlog is not None:
        clm = ["INDEX", "X_UTME", "Y_UTMN", "Z_TVDSS", "Zone", "Well", "MD"]
    else:
        clm = ["INDEX", "X_UTME", "Y_UTMN", "Z_TVDSS", "Zone", "Well"]

    return pd.DataFrame(wellreport, columns=clm)


def mask_shoulderbeds(self, inputlogs, targetlogs, nsamples, strict):
    """Mask targetlogs around discrete boundaries.

    Returns True if inputlog(s) and targetlog(s) are present; otherwise False.
    """
    logger.info("Mask shoulderbeds for some logs...")

    useinputs, usetargets, use_numeric = _mask_shoulderbeds_checks(
        self, inputlogs, targetlogs, nsamples, strict
    )

    if not useinputs or not usetargets:
        logger.info("Mask shoulderbeds for some logs... nothing done")
        return False

    for inlog in useinputs:
        inseries = self._df[inlog]
        if use_numeric:
            bseries = _get_bseries(inseries, nsamples)
        else:
            mode, value = list(nsamples.items())[0]

            depth = self._df["Z_TVDSS"]
            if mode == "md" and self.mdlogname is not None:
                depth = self._df[self.mdlogname]
            elif mode == "md" and self.mdlogname is None:
                raise ValueError("There is no mdlogname attribute present.")

            bseries = _get_bseries_by_distance(depth, inseries, value)

        for target in usetargets:
            self._df.loc[bseries, target] = np.nan

    logger.info("Mask shoulderbeds for some logs... done")
    return True


def _mask_shoulderbeds_checks(self, inputlogs, targetlogs, nsamples, strict):
    """Checks/validates input for mask targetlogs around discrete boundaries."""
    # check that inputlogs exists and that they are discrete, and targetlogs
    useinputs = []
    for inlog in inputlogs:
        if inlog not in self._wlogtypes.keys() and strict is True:
            raise ValueError(f"Input log {inlog} is missing and strict=True")
        if inlog in self._wlogtypes.keys() and self._wlogtypes[inlog] != "DISC":
            raise ValueError(f"Input log {inlog} is not of type DISC")
        if inlog in self._wlogtypes.keys():
            useinputs.append(inlog)

    usetargets = []
    for target in targetlogs:
        if target not in self._wlogtypes.keys() and strict is True:
            raise ValueError(f"Target log {target} is missing and strict=True")
        if target in self._wlogtypes.keys():
            usetargets.append(target)

    use_numeric = True
    if isinstance(nsamples, int):
        maxlen = len(self._df) // 2
        if nsamples < 1 or nsamples > maxlen:
            raise ValueError(f"Keyword nsamples must be an int > 1 and < {maxlen}")
    elif isinstance(nsamples, dict):
        if len(nsamples) == 1 and any(key in nsamples.keys() for key in ["md", "tvd"]):
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
        inseries = np.nan_to_num(inseries.values, nan=xtgeo.UNDEF_INT).astype("int32")
    except TypeError:
        # for older numpy version
        inseries = inseries.values
        inseries[np.isnan(inseries)] = xtgeo.UNDEF_INT
        inseries = inseries.astype("int32")

    res = _cxtgeo.well_mask_shoulder(
        depth.values.astype("float64"), inseries, bseries, distance
    )

    if res != 0:
        raise RuntimeError("BUG: return from _cxtgeo.well_mask_shoulder not zero")

    res = np.array(bseries, dtype=bool)
    return res


def create_surf_distance_log(self, surf, name):
    """Create a log which is vertical distance from a RegularSurface."""
    logger.info("Create a log which is distance to surface")

    if not isinstance(surf, xtgeo.RegularSurface):
        raise ValueError("Input surface is not a RegularSurface instance.")

    # make a Points instance since points has the snap
    zvalues = self.dataframe["Z_TVDSS"]
    points = xtgeo.Points()
    points.dataframe = self.dataframe.iloc[:, 0:3]
    points.snap_surface(surf)
    snapped = points.dataframe["Z_TVDSS"]
    diff = snapped - zvalues

    # create log (default is force overwrite if it exists)
    self.create_log(name)
    self.dataframe[name] = diff
