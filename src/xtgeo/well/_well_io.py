# -*- coding: utf-8 -*-
"""Well input and ouput, private module"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_rms_ascii(self, wfile, mdlogname=None, zonelogname=None,
                     strict=True):
    """Import RMS ascii table well"""
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    wlogtype = dict()
    wlogrecord = dict()

    lognames_all = ['X_UTME', 'Y_UTMN', 'Z_TVDSS']
    lognames = []

    lnum = 1
    with open(wfile, 'r') as fwell:
        for line in fwell:
            if lnum == 1:
                _ffver = line.strip()  # noqa, file version
            elif lnum == 2:
                _wtype = line.strip()  # noqa, well type
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

                rxv = row[2:]

                lognames_all.append(lname)
                lognames.append(lname)

                wlogtype[lname] = ltype

                if ltype == 'DISC':
                    xdict = {int(rxv[i]): rxv[i + 1] for i in
                             range(0, len(rxv), 2)}
                    wlogrecord[lname] = xdict
                else:
                    wlogrecord[lname] = rxv

                nlogread += 1

                if nlogread > nlogs:
                    break

            lnum += 1

    # now import all logs as pandas framework

    dfr = pd.read_csv(wfile, delim_whitespace=True, skiprows=lnum,
                      header=None, names=lognames_all,
                      dtype=np.float64, na_values=-999)

    # undef values have a high float number? or keep Nan?
    # df.fillna(Well.UNDEF, inplace=True)

    # check for MD log:
    if mdlogname is not None:
        if mdlogname in dfr.columns:
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
        if zonelogname in dfr.columns:
            zonelogname = zonelogname
        else:
            msg = ('zonelogname={} was requested but no such log '
                   'found for well {}'.format(zonelogname, wname))

            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)

    logger.debug(dfr.head())

    self._wlogtype = wlogtype
    self._wlogrecord = wlogrecord
    self._rkb = rkb
    self._xpos = xpos
    self._ypos = ypos
    self._wname = wname
    self._df = dfr
    self._mdlogname = mdlogname
    self._zonelogname = zonelogname


def export_rms_ascii(self, wfile, precision=4):
    """Export to RMS well format."""

    with open(wfile, 'w') as fwell:
        print('{}'.format('1.0'), file=fwell)
        print('{}'.format('Unknown'), file=fwell)
        print('{} {} {} {}'.format(self._wname, self._xpos, self._ypos,
                                   self._rkb), file=fwell)
        print('{}'.format(len(self.lognames)), file=fwell)
        for lname in self.lognames:
            usewrec = 'linear'
            wrec = []
            if isinstance(self._wlogrecord[lname], dict):
                for key in self._wlogrecord[lname]:
                    wrec.append(key)
                    wrec.append(self._wlogrecord[lname][key])
                usewrec = ' '.join(str(x) for x in wrec)

            print('{} {} {}'.format(lname, self._wlogtype[lname],
                                    usewrec), file=fwell)

    # now export all logs as pandas framework
    tmpdf = self._df.copy()
    tmpdf.fillna(value=-999, inplace=True)

    # make the disc as is np.int
    for lname in self._wlogtype:
        if self._wlogtype[lname] == 'DISC':
            tmpdf[[lname]] = tmpdf[[lname]].astype(int)

    cformat = '%-.' + str(precision) + 'f'
    tmpdf.to_csv(wfile, sep=' ', header=False, index=False,
                 float_format=cformat, escapechar=' ', mode='a')
