# -*- coding: utf-8 -*-
"""Operations along a well, private module"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import constants as const

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTG_DEBUG = xtg.get_syslevel()


def rescale(self, delta=0.15):
    """Rescale by using a new MD increment

    The rescaling is technically done by interpolation in the Pandas dataframe
    """

    pdrows = pd.options.display.max_rows
    pd.options.display.max_rows = 999

    if self.mdlogname is None:
        self.geometrics()

    dfr = self._df.copy().set_index(self.mdlogname)

    logger.debug("Initial dataframe\n %s", dfr)

    start = dfr.index[0]
    stop = dfr.index[-1]

    nentry = int(round((stop - start) / delta))

    dfr = dfr.reindex(dfr.index.union(np.linspace(start, stop, num=nentry)))
    dfr = dfr.interpolate("index", limit_area="inside").loc[
        np.linspace(start, stop, num=nentry)
    ]

    dfr[self.mdlogname] = dfr.index
    dfr.reset_index(inplace=True, drop=True)

    for lname in dfr.columns:
        if lname in self._wlogtype:
            ltype = self._wlogtype[lname]
            if ltype == "DISC":
                dfr = dfr.round({lname: 0})

    logger.debug("Updated dataframe:\n%s", dfr)

    pd.options.display.max_rows = pdrows  # reset

    self._df = dfr


def make_zone_qual_log(self, zqname):
    """Make a flag log based on stratigraphic relations"""

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


def get_ijk_from_grid(self, grid, grid_id=""):
    """Getting IJK from a grid as well logs."""

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
        grid._p_coord_v,
        grid._p_zcorn_v,
        grid._p_actnum_v,
        onelayergrid._p_zcorn_v,
        onelayergrid._p_actnum_v,
        self.nrow,
        wxarr,
        wyarr,
        wzarr,
        wivec,
        wjvec,
        wkvec,
        0,
        XTG_DEBUG,
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
        self._wlogtype[cellname] = "DISC"

    self._wlogrecord[icellname] = {ncel: str(ncel) for ncel in range(1, grid.ncol + 1)}
    self._wlogrecord[jcellname] = {ncel: str(ncel) for ncel in range(1, grid.nrow + 1)}
    self._wlogrecord[kcellname] = {ncel: str(ncel) for ncel in range(1, grid.nlay + 1)}

    _cxtgeo.delete_intarray(wivec)
    _cxtgeo.delete_intarray(wjvec)
    _cxtgeo.delete_intarray(wkvec)
    _cxtgeo.delete_doublearray(wxarr)
    _cxtgeo.delete_doublearray(wyarr)
    _cxtgeo.delete_doublearray(wzarr)

    del onelayergrid


def get_gridproperties(self, gridprops, grid=("ICELL", "JCELL", "KCELL"), prop_id=""):
    """In prep! Getting gridprops as logs"""

    raise NotImplementedError

    # if not isinstance(gridprops, GridProperties):
    #     raise ValueError('"gridprops" are not a GridProperties instance')

    # if isinstance(grid, tuple):
    #     icl, jcl, kcl = grid
    # elif isinstance(grid, Grid):
    #     self.get_ijk_from_grid(grid, grid_id="_tmp")
    #     # icl, jcl, kcl = ('ICELL_tmp', 'JCELL_tmp', 'KCELL_tmp')
    # else:
    #     raise ValueError('The "grid" is of wrong type, must be a tuple or ' "a Grid")


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
