# -*- coding: utf-8 -*-
"""Well input and ouput, private module"""

from __future__ import print_function, absolute_import

import logging
import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


# Import RMS ascii
# -------------------------------------------------------------------------
def import_rms_ascii(wfile, mdlogname=None, zonelogname=None,
                     strict=True):

    attr = dict()

    wlogtype = dict()
    wlogrecord = dict()

    lognames_all = ['X_UTME', 'Y_UTMN', 'Z_TVDSS']
    lognames = []

    lnum = 1
    with open(wfile, 'r') as f:
        for line in f:
            if lnum == 1:
                ffver = line.strip()
            elif lnum == 2:
                wtype = line.strip()
            elif lnum == 3:
                row = line.strip().split()
                rkb = float(row[-1])
                ypos = float(row[-2])
                xpos = float(row[-3])
                wname = row[-4]

            elif lnum == 4:
                nlogs = int(line)
                nlogread = 1

            else:
                row = line.strip().split()
                lname = row[0]
                ltype = row[1].upper()

                rx = row[2:]

                lognames_all.append(lname)
                lognames.append(lname)

                wlogtype[lname] = ltype

                if ltype == 'DISC':
                    xdict = {int(rx[i]): rx[i + 1] for i in
                             range(0, len(rx), 2)}
                    wlogrecord[lname] = xdict
                else:
                    wlogrecord[lname] = rx

                nlogread += 1

                if nlogread > nlogs:
                    break

            lnum += 1

    # now import all logs as pandas framework

    df = pd.read_csv(wfile, delim_whitespace=True, skiprows=lnum,
                     header=None, names=lognames_all,
                     dtype=np.float64, na_values=-999)

    # undef values have a high float number? or keep Nan?
    # df.fillna(Well.UNDEF, inplace=True)

    # check for MD log:
    if mdlogname is not None:
        if mdlogname in df.columns:
            mdlogname = mdlogname
        else:
            msg = ('mdlogname={} was requested but no such log '
                   'found for well {}'.format(mdlogname, wname))

            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)

    # check for zone log:
    if zonelogname is not None:
        if zonelogname in df.columns:
            zonelogname = zonelogname
        else:
            msg = ('zonelogname={} was requested but no such log '
                   'found for well {}'.format(zonelogname, wname))

            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)

    logger.debug(df.head())

    attr['wlogtype'] = wlogtype
    attr['wlogrecord'] = wlogrecord
    attr['lognames_all'] = lognames_all
    attr['lognames'] = lognames
    attr['ffver'] = ffver
    attr['wtype'] = wtype
    attr['rkb'] = rkb
    attr['xpos'] = xpos
    attr['ypos'] = ypos
    attr['wname'] = wname
    attr['nlogs'] = nlogs
    attr['df'] = df
    attr['mdlogname'] = mdlogname
    attr['zonelogname'] = zonelogname

    return attr


def export_rms_ascii(well, wfile, precision=4):
    """Export to RMS well format."""

    with open(wfile, 'w') as f:
        print('{}'.format(well._ffver), file=f)
        print('{}'.format(well._wtype), file=f)
        print('{} {} {} {}'.format(well._wname, well._xpos, well._ypos,
                                   well._rkb), file=f)
        for lname in well.lognames:
            wrec = []
            if type(well._wlogrecord[lname]) is dict:
                for key in well._wlogrecord[lname]:
                    wrec.append(key)
                    wrec.append(well._wlogrecord[lname][key])

            else:
                wrec = well._wlogrecord[lname]

            wrec = ' '.join(str(x) for x in wrec)
            print(wrec)

            print('{} {} {}'.format(lname, well._wlogtype[lname],
                                    wrec), file=f)

    # now export all logs as pandas framework
    tmpdf = well._df.copy()
    tmpdf.fillna(value=-999, inplace=True)

    # make the disc as is np.int
    for lname in well._wlogtype:
        if well._wlogtype[lname] == 'DISC':
            tmpdf[[lname]] = tmpdf[[lname]].astype(int)

    cformat = '%-.' + str(precision) + 'f'
    tmpdf.to_csv(wfile, sep=' ', header=False, index=False,
                 float_format=cformat, escapechar=' ', mode='a')
