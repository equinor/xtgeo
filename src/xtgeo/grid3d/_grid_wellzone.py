"""Private module for grid vs well zonelog checks"""

from __future__ import print_function, absolute_import, division

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
):
    """
    Reports well to zone mismatch; this works together with a Well object.

    The idea is to sample the current zone property for the well in the grid as fast as
    possible.

    Then the sampled zonelog is compared with the actual zonelog in the dataframe.

    One can apply a perforation log as a mask, meaning that we filter zonelog
    match in intervals with a perforation log only if requested.

    This method was completely redesigned in version 2.8
    """

    if onelayergrid is not None:
        xtg.warndeprecated("Using key 'onelayergrid' is redundant and can be skipped")

    if not isinstance(well, xtgeo.Well):
        raise ValueError("Input well is not a Well() instance")

    if not zonelogname in well.dataframe:
        raise ValueError("Input zonelogname {} not in Well".format(zonelogname))

    if perflogname == "None" or perflogname is None:  # "None" for backwards compat
        logger.info("No perforation log as filter")

    # get the IJK along the well as logs; use a copy of the well instance
    wll = well.copy()
    # wll._df.loc[(wll._df[zonelogname] == 0)] = np.nan
    wll._df[zonelogname] += zonelogshift

    # wll._df[zonelogname].replace(0, -777, inplace=True)

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
        wll._df[zname].replace(skiprange, -888, inplace=True)
        df[zname].fillna(-999, inplace=True)

    df = df[(df[zmodel] > -888) | (df[zonelogname] > -888)]

    #
    df.insert(df.shape[1], "zmatch", 0.0, True)
    df["zmatch"][df[zmodel] == df[zonelogname]] = 1

    res = df["zmatch"].mean()

    return res
