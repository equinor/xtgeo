# -*- coding: utf-8 -*-
"""Well marker data; private module"""

from __future__ import print_function, absolute_import

import numpy as np

from xtgeo.common import XTGeoDialog
import xtgeo.common.constants as const

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def extract_ztops(self, zonelist, xcv, ycv, zcv, zlog, mdv, incl,
                  tops=True, incl_limit=80, prefix='Top',
                  use_undef=False):
    """Extract a list of tops for a zone.

    Args:
        zonelist (list-like): The zonelog list numbers to apply; either
            as a list, or a tuple; 2 entries forms a range [start, stop)
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

    logger.debug(zlog)
    logger.info('Well name is %s', self.wellname)

    if not tops and incl_limit is None:
        incl_limit = 80

    azi = -999.0  # tmp so far

    usezonerange = range(0, 99999)
    if isinstance(zonelist, tuple) and len(zonelist) == 2:
        usezonerange = range(zonelist[0], zonelist[1] + 1)
    elif isinstance(zonelist, list) and len(zonelist) > 1:
        usezonerange = zonelist
    else:
        raise ValueError('Something is wrong with zonelist input')

    iundef = const.UNDEF_INT
    iundeflimit = const.UNDEF_INT_LIMIT
    pzone = iundef

    if use_undef:
        pzone = zlog.min() - 1

    for ind, zone in np.ndenumerate(zlog):

        ino = ind[0]  # since ind is a tuple...

        if use_undef and zone > iundeflimit:
            zone = zlog.min() - 1

        if pzone != zone and pzone < iundeflimit and zone < iundeflimit:
            logger.debug('Found break in zonation')
            if pzone < zone:
                logger.debug('Found match with zonation increasing')
                for kzv in range(pzone + 1, zone + 1):
                    if kzv in usezonerange:
                        zname = self.get_logrecord_codename(zlogname, kzv)
                        zname = prefix + zname
                        ztop = (xcv[ino], ycv[ino], zcv[ino], mdv[ino],
                                incl[ino], azi, kzv, zname, self.xwellname)
                        wpts.append(ztop)
            if pzone > zone and ino > 0:
                logger.info('Found match, decreasing zonation')
                for kzv in range(pzone, zone, -1):
                    if kzv in usezonerange:
                        zname = self.get_logrecord_codename(zlogname, kzv)
                        zname = prefix + zname
                        ztop = (xcv[ino - 1], ycv[ino - 1], zcv[ino - 1],
                                mdv[ino - 1], incl[ino - 1], azi, kzv, zname,
                                self.xwellname)
                        wpts.append(ztop)
        pzone = zone

    mdname = 'Q_MDEPTH'
    if self.mdlogname is not None:
        mdname = 'M_MDEPTH'

    wpts_names = ['X_UTME', 'Y_UTMN', 'Z_TVDSS', mdname, 'Q_INCL', 'Q_AZI',
                  'Zone', 'TopName', 'WellName']

    if tops:
        return wpts, wpts_names, None, None

    # next get a MIDPOINT zthickness (DZ)
    llen = len(wpts) - 1

    zwpts_names = ['X_UTME', 'Y_UTMN', 'Z_TVDSS', mdname + '_AVG',
                   'Q_MD1', 'Q_MD2', 'Q_INCL',
                   'Q_AZI', 'Zone', 'ZoneName', 'WellName']

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
            usezn = usezn[len(prefix):]

            zzone = (xx_avg, yy_avg, zzp, md_avg, md1, md2, incl_avg,
                     azi_avg, usezk, usezn, wn1)
            zwpts.append(zzone)

    return wpts, wpts_names, zwpts, zwpts_names


def get_fraction_per_zone(self, dlogname, dvalue, zonelist=None,
                          incl_limit=80):

    #     X_UTME       Y_UTMN    Z_TVDSS  Zonelog  Facies  M_INCL
    # 464011.719  5931757.257  1663.1079      3.0     1.0    10.2
    # 464011.751  5931757.271  1663.6084      3.0     1.0    10.3
    # 464011.783  5931757.285  1664.1090      3.0     2.0    11.2
    # 464011.815  5931757.299  1664.6097      3.0     2.0    11.4
    # 464011.847  5931757.313  1665.1105      3.0     2.0    11.5
    # 464011.879  5931757.326  1665.6114      3.0     2.0    12.0
    # 464011.911  5931757.340  1666.1123      3.0     1.0    12.2
    # 464011.943  5931757.354  1666.6134      3.0     1.0    13.4
    #
    # Count number of one facies filtered on a zone, given that
    # Inclination is below limit all over. Since a zone can be
    # repeated, it is important to split into segments. When
    # fraction is determined, the AVG X Y coord is applied.

    if zonelist is None:
        # need to declare as list; otherwise Py3 will get dict.keys
        zonelist = list(self.get_logrecord(self.zonelogname).keys())

    useinclname = 'Q_INCL'
    if 'M_INCL' in self._df:
        useinclname = 'M_INCL'
    else:
        self.geometrics()

    xtralogs = [dlogname, useinclname]
    for izon in zonelist:
        print(izon)
        dfr = self.get_zone_interval(izon, extralogs=xtralogs)
        vvv = dfr[dlogname].value_counts(normalize=True)[dvalue]
        xxx = dfr['X_UTME'].mean()
        yyy = dfr['Y_UTMN'].mean()
        print(xxx, yyy, vvv)
