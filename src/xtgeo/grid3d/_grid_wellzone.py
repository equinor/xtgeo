"""Private module for grid vs well zonelog checks"""

from __future__ import print_function, absolute_import, division

import numpy as np
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def report_zone_mismatch(
    self,
    well=None,
    zonelogname="ZONELOG",
    zoneprop=None,
    onelayergrid=None,  # redundant; will be computed internally
    zonelogrange=(0, 9999),
    zonelogshift=0,
    depthrange=None,
    perflogname=None,
    filterlogname=None,
    resultformat=1,
):  # pylint: disable=too-many-locals
    """
    Reports well to zone mismatch; this works together with a Well object.

    The idea is to sample the current zone property for the well in the grid as fast as
    possible.

    Then the sampled zonelog is compared with the actual zonelog, and the difference
    is reported.

    One can apply a perforation log as a mask, meaning that we filter zonelog
    match in intervals with a perforation log only if requested.

    This method was completely redesigned in version 2.8
    """

    if onelayergrid is not None:
        xtg.warndeprecated("Using key 'onelayergrid' is redundant and can be skipped")

    if not isinstance(well, xtgeo.Well):
        raise ValueError("Input well is not a Well() instance")

    if not zonelogname in well.dataframe.columns:
        logger.warning("Zonelog %s is missing for well %s", zonelogname, well.name)
        return None

    if perflogname == "None" or perflogname is None:  # "None" for backwards compat
        logger.info("No perforation log as filter")
        perflogname = None

    # get the IJK along the well as logs; use a copy of the well instance
    wll = well.copy()
    wll._df[zonelogname] += zonelogshift

    if depthrange:
        d1, d2 = depthrange
        wll._df = wll._df[(d1 < wll._df.Z_TVDSS) & (wll._df.Z_TVDSS < d2)]

    wll.get_gridproperties(zoneprop, self)
    zmodel = zoneprop.name + "_model"

    # from here, work with the dataframe only
    df = wll._df

    # zonelogrange
    z1, z2 = zonelogrange
    zmin = int(df[zonelogname].min())
    zmax = int(df[zonelogname].max())
    skiprange = list(range(zmin, z1)) + list(range(z2 + 1, zmax + 1))

    for zname in (zonelogname, zmodel):
        df[zname].replace(skiprange, -888, inplace=True)
        df[zname].fillna(-999, inplace=True)
        if perflogname:
            if perflogname in df.columns:
                df[perflogname].replace(np.nan, -1, inplace=True)
                df[zname] = np.where(df[perflogname] <= 0, -899, df[zname])
            else:
                return None
        if filterlogname:
            if filterlogname in df.columns:
                df[filterlogname].replace(np.nan, -1, inplace=True)
                df[zname] = np.where(df[filterlogname] <= 0, -919, df[zname])
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
    mcount1 = int(dfuse1["zmatch1"].sum())
    tcount1 = int(dfuse1["zmatch1"].count())
    res1 = dfuse1["zmatch1"].mean() * 100

    dfuse2 = df.copy(deep=True)
    dfuse2 = dfuse2.loc[(df[zmodel] > -888) | (df[zonelogname] > -888)]
    dfuse2["zmatch2"] = np.where(dfuse2[zmodel] == dfuse2[zonelogname], 1, 0)
    mcount2 = int(dfuse2["zmatch2"].sum())
    tcount2 = int(dfuse2["zmatch2"].count())
    res2 = dfuse2["zmatch2"].mean() * 100

    # update Well() copy (segment only)
    wll.dataframe = dfuse2

    if resultformat == 1:
        return (res1, mcount1, tcount1)

    res = {
        "MATCH1": res1,
        "MCOUNT1": mcount1,
        "TCOUNT1": tcount1,
        "MATCH2": res2,
        "MCOUNT2": mcount2,
        "TCOUNT2": tcount2,
        "WELLINTV": wll,
    }
    return res
