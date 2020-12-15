# -*- coding: utf-8 -*-
"""Well marker data; private module"""


from collections import OrderedDict
import numpy as np
import pandas as pd

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
import xtgeo.common.constants as const

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def get_zonation_points(self, tops, incl_limit, top_prefix, zonelist, use_undef):
    """
    Getting zonation tops (private routine)

    Args, see calling routine
    """

    zlist = []
    # get the relevant logs:

    self.geometrics()  # note the caller has made a copy of the true self

    # as zlog is float64; need to convert to int array with high
    # number as undef
    if self.zonelogname is not None:
        if use_undef:
            self._df.dropna(subset=[self.zonelogname], inplace=True)
        zlog = self._df[self.zonelogname].values
        zlog[np.isnan(zlog)] = const.UNDEF_INT
        zlog = np.rint(zlog).astype(int)
    else:
        return None

    xvv = self._df["X_UTME"].values
    yvv = self._df["Y_UTMN"].values
    zvv = self._df["Z_TVDSS"].values
    incl = self._df["Q_INCL"].values
    mdv = self._df["Q_MDEPTH"].values

    if self.mdlogname is not None:
        mdv = self._df[self.mdlogname].values

    if zonelist is None:
        # need to declare as list; otherwise Py3 will get dict.keys
        zonelist = list(self.get_logrecord(self.zonelogname).keys())

    logger.info("Find values for %s", zonelist)

    ztops, ztopnames, zisos, zisonames = _extract_ztops(
        self,
        zonelist,
        xvv,
        yvv,
        zvv,
        zlog,
        mdv,
        incl,
        tops=tops,
        incl_limit=incl_limit,
        prefix=top_prefix,
        use_undef=use_undef,
    )

    if tops:
        zlist = ztops
    else:
        zlist = zisos

    logger.debug(zlist)

    if tops:
        dfr = pd.DataFrame(zlist, columns=ztopnames)
    else:
        dfr = pd.DataFrame(zlist, columns=zisonames)

    return dfr


def _extract_ztops(
    self,
    zonelist,
    xcv,
    ycv,
    zcv,
    zlog,
    mdv,
    incl,
    tops=True,
    incl_limit=80,
    prefix="Top",
    use_undef=False,
):
    """Extract a list of tops for a zone.

    Args:
        zonelist (list-like): The zonelog list numbers to apply; either
            as a list, or a tuple; 2 entries forms a range [start, stop]
        xcv (np): X Position numpy array
        ycv (np): Y Position numpy array
        zcv (np): Z Position numpy array
        zlog (np): Zonelog array
        mdv (np): MDepth log numpy array
        incl (np): Inclination log numpy array
        tops (bool): Compute tops or thickness (zone) points (default True)
        incl_limit (float): Limitation of zone computation (angle, degrees)
        use_undef (bool): If True, then transition from UNDEF is also
            used.
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    # The wellpoints will be a list of tuples (one tuple per hit)
    wpts = []
    zlogname = self.zonelogname

    if not tops and incl_limit is None:
        incl_limit = 80

    azi = -999.0  # tmp so far

    usezonerange = range(0, 99999)
    if isinstance(zonelist, tuple) and len(zonelist) == 2:
        usezonerange = range(zonelist[0], zonelist[1] + 1)
    elif isinstance(zonelist, list) and len(zonelist) > 1:
        usezonerange = zonelist
    else:
        raise ValueError("Something is wrong with zonelist input")

    # check if increasing monotonic and with no jumps:
    if not all(i + 1 == j for i, j in zip(usezonerange, usezonerange[1:])):
        raise ValueError("The zonelist is not valid! C.f. documentation")

    iundef = const.UNDEF_INT
    iundeflimit = const.UNDEF_INT_LIMIT
    pzone = iundef

    if use_undef:
        pzone = usezonerange[0] - 1

    for ind, zone in np.ndenumerate(zlog):

        ino = ind[0]  # since ind is a tuple...

        if pzone != zone and pzone < iundeflimit and zone < iundeflimit:
            logger.debug("Found break in zonation")
            if pzone < zone:
                logger.debug(
                    "Found match, increasing zonation at %s < %s (MD %s)",
                    pzone,
                    zone,
                    mdv[ino],
                )
                for kzv in range(pzone + 1, zone + 1):
                    if kzv in usezonerange:
                        zname = self.get_logrecord_codename(zlogname, kzv)
                        zname = prefix + zname
                        ztop = (
                            xcv[ino],
                            ycv[ino],
                            zcv[ino],
                            mdv[ino],
                            incl[ino],
                            azi,
                            kzv,
                            zname,
                            self.xwellname,
                        )
                        wpts.append(ztop)
            if pzone > zone and ino > 0:
                logger.debug(
                    "Found match, decreasing zonation at %s > %s (MD %s)",
                    pzone,
                    zone,
                    mdv[ino - 1],
                )
                for kzv in range(pzone, zone, -1):
                    if kzv in usezonerange:
                        zname = self.get_logrecord_codename(zlogname, kzv)
                        zname = prefix + zname
                        ztop = (
                            xcv[ino - 1],
                            ycv[ino - 1],
                            zcv[ino - 1],
                            mdv[ino - 1],
                            incl[ino - 1],
                            azi,
                            kzv,
                            zname,
                            self.xwellname,
                        )
                        wpts.append(ztop)
        pzone = zone

    wpts_names = [
        "X_UTME",
        "Y_UTMN",
        "Z_TVDSS",
        self.mdlogname,
        "Q_INCL",
        "Q_AZI",
        "Zone",
        "TopName",
        "WellName",
    ]

    if tops:
        return wpts, wpts_names, None, None

    # next get a MIDPOINT zthickness (DZ)
    llen = len(wpts) - 1

    zwpts_names = [
        "X_UTME",
        "Y_UTMN",
        "Z_TVDSS",
        self.mdlogname + "_AVG",
        "Q_MD1",
        "Q_MD2",
        "Q_INCL",
        "Q_AZI",
        "Zone",
        "ZoneName",
        "WellName",
    ]

    zwpts = []
    for ino in range(llen):
        i1v = ino
        i2v = ino + 1
        xx1, yy1, zz1, md1, incl1, _azi1, zk1, zn1, wn1 = wpts[i1v]
        xx2, yy2, zz2, md2, incl2, _azi2, zk2, zn2, _wn2 = wpts[i2v]

        # mid point
        xx_avg = (xx1 + xx2) / 2
        yy_avg = (yy1 + yy2) / 2
        md_avg = (md1 + md2) / 2
        incl_avg = (incl1 + incl2) / 2

        azi_avg = -999.0  # to be fixed later

        zzp = round(abs(zz2 - zz1), 4)

        useok = False

        if incl_avg < incl_limit:
            useok = True

        if useok and zk2 != zk1:
            usezk = zk1
            usezn = zn1
            if zk1 > zk2:
                usezk = zk2
                usezn = zn2
            usezn = usezn[len(prefix) :]

            zzone = (
                xx_avg,
                yy_avg,
                zzp,
                md_avg,
                md1,
                md2,
                incl_avg,
                azi_avg,
                usezk,
                usezn,
                wn1,
            )
            zwpts.append(zzone)

    return wpts, wpts_names, zwpts, zwpts_names


def get_fraction_per_zone(
    self,
    dlogname,
    dvalues,
    zonelist=None,
    incl_limit=80,
    count_limit=3,
    zonelogname=None,
):  # pylint: disable=too-many-branches, too-many-statements

    """Fraction of e.g. a facies in a zone segment.

        X_UTME       Y_UTMN    Z_TVDSS  Zonelog  Facies  M_INCL
    464011.719  5931757.257  1663.1079      3.0     1.0    10.2
    464011.751  5931757.271  1663.6084      3.0     1.0    10.3
    464011.783  5931757.285  1664.1090      3.0     2.0    11.2
    464011.815  5931757.299  1664.6097      3.0     2.0    11.4
    464011.847  5931757.313  1665.1105      3.0     2.0    11.5
    464011.879  5931757.326  1665.6114      3.0     2.0    12.0
    464011.911  5931757.340  1666.1123      3.0     1.0    12.2
    464011.943  5931757.354  1666.6134      3.0     1.0    13.4

    Count fraction of one or more facies (dvalues list)
    filtered on a zone, given that Inclination is below limit all over.
    Since a zone can be repeated, it is important to split
    into segments by POLY_ID. When fraction is determined, the
    AVG X Y coord is applied.

    If there are one or more occurences of undef for the dlogname
    in that interval, no value shall be computed.

    Args:
        dlogname (str): Name of discrete log e.g. Facies
        dvalues (list): List of codes to sum fraction upon
        zonelist (list): List of zones to compute over
        incl_limit (float): Skip if max inclination found > incl_limit
        count_limit (int): Minimum number of samples required per segment

    Returns:
        A dataframe with relevant data...

    """
    logger.info("The zonelist is %s", zonelist)
    logger.info("The dlogname is %s", dlogname)
    logger.info("The dvalues are %s", dvalues)

    if zonelogname is not None:
        usezonelogname = zonelogname
        self.zonelogname = zonelogname
    else:
        usezonelogname = self.zonelogname

    if usezonelogname is None:
        raise RuntimeError("Stop, zonelogname is None")

    self.make_zone_qual_log("_QFLAG")

    if zonelist is None:
        # need to declare as list; otherwise Py3 will get dict.keys
        zonelist = list(self.get_logrecord(self.zonelogname).keys())

    useinclname = "Q_INCL"
    if "M_INCL" in self._df:
        useinclname = "M_INCL"
    else:
        self.geometrics()

    result = OrderedDict()
    result["X_UTME"] = []
    result["Y_UTMN"] = []
    result["DFRAC"] = []
    result["Q_INCL"] = []
    result["ZONE"] = []
    result["WELLNAME"] = []
    result[dlogname] = []

    svalues = str(dvalues).rstrip("]").lstrip("[").replace(", ", "+")

    xtralogs = [dlogname, useinclname, "_QFLAG"]
    for izon in zonelist:
        logger.info("The zone number is %s", izon)
        logger.info("The extralogs are %s", xtralogs)

        dfr = self.get_zone_interval(izon, extralogs=xtralogs)

        if dfr is None:
            continue

        dfrx = dfr.groupby("POLY_ID")

        for _polyid, dframe in dfrx:
            qinclmax = dframe["Q_INCL"].max()
            qinclavg = dframe["Q_INCL"].mean()
            qflag = dframe["_QFLAG"].mean()
            dseries = dframe[dlogname]
            if qflag < 0.5 or qflag > 2.5:  # 1 or 2 is OK
                logger.debug("Skipped due to zone %s", qflag)
                continue
            if qinclmax > incl_limit:
                logger.debug("Skipped due to max inclination %s", qinclmax)
                continue
            if dseries.size < count_limit:  # interval too short for fraction
                logger.debug("Skipped due to too few values %s", dseries.size)
                continue
            if dseries.max() > const.UNDEF_INT_LIMIT:
                logger.debug("Skipped due to too missing/undef value(s)")
                continue

            xavg = dframe["X_UTME"].mean()
            yavg = dframe["Y_UTMN"].mean()

            dfrac = 0.0
            for dval in dvalues:
                if any(dseries.isin([dval])):
                    dfrac += dseries.value_counts(normalize=True)[dval]

            result["X_UTME"].append(xavg)
            result["Y_UTMN"].append(yavg)
            result["DFRAC"].append(dfrac)
            result["Q_INCL"].append(qinclavg)
            result["ZONE"].append(izon)
            result["WELLNAME"].append(self.xwellname)
            result[dlogname].append(svalues)

    # make the dataframe and return it
    if result["X_UTME"]:
        return pd.DataFrame.from_dict(result)

    self.delete_log("_QFLAG")

    return None


def get_surface_picks(self, surf):
    """get Surface picks"""

    xcor = self._df["X_UTME"].values
    ycor = self._df["Y_UTMN"].values
    zcor = self._df["Z_TVDSS"].values

    if self.mdlogname:
        mcor = self._df[self.mdlogname].values
    else:
        mcor = np.zeros(xcor.size, dtype=np.float64) + xtgeo.UNDEF

    nval, xres, yres, zres, mres, dres = _cxtgeo.well_surf_picks(
        xcor,
        ycor,
        zcor,
        mcor,
        surf.ncol,
        surf.nrow,
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.yflip,
        surf.rotation,
        surf.npvalues1d,
        xcor.size,
        xcor.size,
        xcor.size,
        xcor.size,
        xcor.size,
    )

    if nval > 0:
        poi = xtgeo.Points()

        mres[mres > xtgeo.UNDEF_LIMIT] = np.nan

        res = OrderedDict()
        res[poi.xname] = xres[:nval]
        res[poi.yname] = yres[:nval]
        res[poi.zname] = zres[:nval]
        if self.mdlogname:
            res[self.mdlogname] = mres[:nval]
        res["DIRECTION"] = dres[:nval]
        res["WELLNAME"] = self.wellname

        poi.dataframe = pd.DataFrame.from_dict(res)

        return poi

    return None

    # return a xtgeo Poinst() object with points as dataframe, given that nval > 0
