"""Private module for grid vs well zonelog checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.grid3d.grid_property import GridProperty
from xtgeo.well import Well

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid


xtg = XTGeoDialog()
logger = null_logger(__name__)


def report_zone_mismatch(
    self: Grid,
    well: Well | None = None,
    zonelogname: str = "ZONELOG",
    zoneprop: GridProperty | None = None,
    zonelogrange: tuple[int, int] = (0, 9999),
    zonelogshift: int = 0,
    depthrange: tuple[int | float, int | float] | None = None,
    perflogname: str | None = None,
    perflogrange: tuple[int | float, int | float] = (1, 9999),
    filterlogname: str | None = None,
    filterlogrange: tuple[int | float, int | float] = (1e-32, 9999.0),
    resultformat: Literal[1, 2] = 1,
) -> dict[str, float | int | Well] | tuple[float, int, int] | None:  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Reports well to zone mismatch; this works together with a Well object.

    The idea is to sample the current zone property for the well in the grid as fast as
    possible.

    Then the sampled zonelog is compared with the actual zonelog, and the difference
    is reported.

    One can apply a perforation log as a mask, meaning that we filter zonelog
    match in intervals with a perforation log only if requested.

    This method was completely redesigned in version 2.8
    """
    self._xtgformat1()

    if not isinstance(well, Well):
        raise ValueError("Input well is not a Well() instance")

    if zoneprop is None or not isinstance(zoneprop, GridProperty):
        raise ValueError("Input zoneprop is missing or not a GridProperty() instance")

    if zonelogname not in well.get_dataframe(copy=False).columns:
        logger.warning("Zonelog %s is missing for well %s", zonelogname, well.name)
        return None

    if perflogname == "None" or perflogname is None:  # "None" for backwards compat
        logger.info("No perforation log as filter")
        perflogname = None

    # get the IJK along the well as logs; use a copy of the well instance
    wll = well.copy()
    wll_df = wll.get_dataframe()
    wll_df[zonelogname] += zonelogshift

    if depthrange:
        d1, d2 = depthrange
        wll_df = wll_df[(d1 < wll_df.Z_TVDSS) & (d2 > wll_df.Z_TVDSS)]
    wll.set_dataframe(wll_df)

    wll.get_gridproperties(zoneprop, self)
    zonename = zoneprop.name if zoneprop.name is not None else "Zone"
    zmodel = zonename + "_model"

    # from here, work with the dataframe only
    df = wll.get_dataframe()

    # zonelogrange
    z1, z2 = zonelogrange
    zmin = zmax = 0
    try:
        zmin = int(df[zonelogname].min())
    except ValueError as verr:
        if "cannot convert" in str(verr):
            msg = f"TVD range {depthrange} is possibly to narrow? ({str(verr)})"
            raise ValueError(msg)
    try:
        zmax = int(df[zonelogname].max())
    except ValueError as verr:
        if "cannot convert" in str(verr):
            msg = f"TVD range {depthrange} is possibly to narrow? ({str(verr)})"
            raise ValueError(msg)

    skiprange = list(range(zmin, z1)) + list(range(z2 + 1, zmax + 1))

    for zname in (zonelogname, zmodel):
        if skiprange:  # needed check; du to a bug in pandas version 0.21 .. 0.23
            df[zname] = df[zname].replace(skiprange, -888)
        df[zname] = df[zname].fillna(-999)
        if perflogname:
            if perflogname in df.columns:
                df[perflogname] = df[perflogname].replace(np.nan, -1)
                pfr1, pfr2 = perflogrange
                df[zname] = np.where(df[perflogname] < pfr1, -899, df[zname])
                df[zname] = np.where(df[perflogname] > pfr2, -899, df[zname])
            else:
                return None
        if filterlogname:
            if filterlogname in df.columns:
                df[filterlogname] = df[filterlogname].replace(np.nan, -1)
                ffr1, ffr2 = filterlogrange
                df[zname] = np.where(df[filterlogname] < ffr1, -919, df[zname])
                df[zname] = np.where(df[filterlogname] > ffr2, -919, df[zname])
            else:
                return None

    # now there are various variotions on how to count mismatch:
    # dfuse 1: count matches when zonelogname is valid (exclude -888)
    # dfuse 2: count matches when zonelogname OR zmodel are valid (exclude < -888
    # or -999)
    # The first one is the original approach

    dfuse1 = df.copy(deep=True)
    dfuse1 = dfuse1.loc[dfuse1[zonelogname] > -888]

    dfuse1["zmatch1"] = np.where(dfuse1[zmodel] == dfuse1[zonelogname], 1, 0)
    mcount1 = dfuse1["zmatch1"].sum()
    tcount1 = dfuse1["zmatch1"].count()
    if not np.isnan(mcount1):
        mcount1 = int(mcount1)
    if not np.isnan(tcount1):
        tcount1 = int(tcount1)

    res1 = dfuse1["zmatch1"].mean() * 100

    if resultformat == 1:
        return (res1, mcount1, tcount1)

    dfuse2 = df.copy(deep=True)
    dfuse2 = dfuse2.loc[(df[zmodel] > -888) | (df[zonelogname] > -888)]
    dfuse2["zmatch2"] = np.where(dfuse2[zmodel] == dfuse2[zonelogname], 1, 0)
    mcount2 = dfuse2["zmatch2"].sum()
    tcount2 = dfuse2["zmatch2"].count()
    if not np.isnan(mcount2):
        mcount2 = int(mcount2)
    if not np.isnan(tcount2):
        tcount2 = int(tcount2)

    res2 = dfuse2["zmatch2"].mean() * 100

    # update Well() copy (segment only)
    wll.set_dataframe(dfuse2)

    return {
        "MATCH1": res1,
        "MCOUNT1": mcount1,
        "TCOUNT1": tcount1,
        "MATCH2": res2,
        "MCOUNT2": mcount2,
        "TCOUNT2": tcount2,
        "WELLINTV": wll,
    }
