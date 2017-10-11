# -*- coding: utf-8 -*-
"""Well marker data; private module"""

from __future__ import print_function, absolute_import

import logging
import numpy as np

from xtgeo.common import XTGeoDialog
import cxtgeo.cxtgeo as _cxtgeo

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


def extract_ztops(well, zonelist, xc, yc, zc, zlog, md, incl, zlogname,
                  tops=True, incl_limit=80, prefix='Top'):
    """Extract a list of tops for a zone.

    Args:
        zonelist (list-like): The zonelog list numbers to apply; either
            as a list, or a tuple; 2 entries forms a range [start, stop)
        xc (np): X Position numpy array
        yc (np): Y Position numpy array
        zc (np): Z Position numpy array
        zlog (np): Zonelog array
        md (np): MDepth log numpy array
        incl (np): Inclination log numpy array
        zlogname (str): Name of zonation log
        tops (bool): Compute tops or thickness (zone) points (default True)
        incl_limit (float): Limitation of zone computation (angle, degrees)
    """

    # The wellpoints will be a list of tuples (one tuple per hit)
    wpts = []

    logger.debug(zlog)
    logger.info('Well name is {}'.format(well.wellname))

    if not tops and incl_limit is None:
        incl_limit = 80

    azi = -999.0  # tmp so far

    usezonerange = range(0, 99999)
    if isinstance(zonelist, tuple) and len(zonelist) == 2:
        usezonerange = range(zonelist)
    elif isinstance(zonelist, list) and len(zonelist) > 1:
        usezonerange = zonelist
    else:
        raise ValueError('Something is wrong with zonelist input')

    iundef = _cxtgeo.UNDEF_INT
    iundeflimit = _cxtgeo.UNDEF_INT_LIMIT

    pzone = iundef

    for ind, zone in np.ndenumerate(zlog):

        i = ind[0]  # since ind is a tuple...

        logger.debug('PZONE is {} and zone is {}'
                     .format(pzone, zone))

        if pzone != zone and pzone < iundeflimit and zone < iundeflimit:
            logger.debug('Found break in zonation')
            if pzone < zone:
                logger.debug('Found match with zonation increasing')
                for kz in range(pzone + 1, zone + 1):
                    if kz in usezonerange:
                        zname = well.get_logrecord_codename(zlogname, kz)
                        zname = prefix + zname
                        ztop = (xc[i], yc[i], zc[i], md[i], incl[i], azi,
                                kz, zname, well.xwellname)
                        wpts.append(ztop)
            if pzone > zone and i > 0:
                logger.info('Found match, decreasing zonation')
                for kz in range(pzone, zone, -1):
                    if kz in usezonerange:
                        zname = well.get_logrecord_codename(zlogname, kz)
                        zname = prefix + zname
                        ztop = (xc[i - 1], yc[i - 1], zc[i - 1], md[i - 1],
                                incl[i - 1], azi, kz, zname,
                                well.xwellname)
                        wpts.append(ztop)
        pzone = zone

    wpts_names = ['X', 'Y', 'Z', 'QMD', 'QINCL', 'QAZI', 'Zone', 'TopName',
                  'WellName']

    if tops:
        return wpts, wpts_names, None, None

    # next get a MIDPOINT zthickness (DZ)
    llen = len(wpts) - 1

    zwpts_names = ['X', 'Y', 'Z', 'QMD_AVG', 'QMD1', 'QMD2', 'QINCL',
                   'QAZI', 'Zone', 'ZoneName', 'WellName']

    zwpts = []
    for i in range(llen):
        i1 = i
        i2 = i + 1
        xx1, yy1, zz1, md1, incl1, azi1, zk1, zn1, wn1 = wpts[i1]
        xx2, yy2, zz2, md2, incl2, azi2, zk2, zn2, wn2 = wpts[i2]

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
            logger.debug(' -- Zone {} {} ---> {}'
                         .format(zk1, zk2, zzp))
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
