# -*- coding: utf-8 -*-
"""Well input and ouput, private module"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_rms_ascii(
    self,
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
):
    """Import RMS ascii table well"""
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    wlogtype = dict()
    wlogrecord = dict()

    xlognames_all = ["X_UTME", "Y_UTMN", "Z_TVDSS"]
    xlognames = []

    lnum = 1
    with open(wfile, "r") as fwell:
        for line in fwell:
            if lnum == 1:
                _ffver = line.strip()  # noqa, file version
            elif lnum == 2:
                _wtype = line.strip()  # noqa, well type
            elif lnum == 3:
                # usually 4 fields, but last (rkb) can be missing. A
                # complication is that first field (well name) may have spaces,
                # hence some clever guessing is needed. However, this cannot be
                # 100% foolproof... if Ycoord < 1000 and last item of a well
                # name with spaces is a number, then this may fail.
                assume_rkb = False
                row = line.strip().split()
                newrow = []
                if len(row) > 3:
                    for item in row:
                        try:
                            item = float(item)
                        except ValueError:
                            item = str(item)
                        newrow.append(item)
                    if all(isinstance(var, float) for var in newrow[-3:]):
                        if abs(newrow[-1] < 1000.0):
                            assume_rkb = True

                if assume_rkb:
                    rkb = float(row.pop())
                else:
                    rkb = None
                ypos = float(row.pop())
                xpos = float(row.pop())
                wname = " ".join(map(str, row))

            elif lnum == 4:
                nlogs = int(line)
                nlogread = 1
                logger.debug("Number of logs: %s", nlogs)

            else:
                row = line.strip().split()
                lname = row[0]

                # if i_index etc, make uppercase to I_INDEX
                # however it is most practical to treat indexes as CONT logs
                if "_index" in lname:
                    lname = lname.upper()

                ltype = row[1].upper()

                rxv = row[2:]

                xlognames_all.append(lname)
                xlognames.append(lname)

                wlogtype[lname] = ltype

                logger.debug("Reading log name %s of type %s", lname, ltype)

                if ltype == "DISC":
                    xdict = {int(rxv[i]): rxv[i + 1] for i in range(0, len(rxv), 2)}
                    wlogrecord[lname] = xdict
                else:
                    wlogrecord[lname] = rxv

                nlogread += 1

                if nlogread > nlogs:
                    break

            lnum += 1

    # now import all logs as pandas framework

    dfr = pd.read_csv(
        wfile,
        delim_whitespace=True,
        skiprows=lnum,
        header=None,
        names=xlognames_all,
        dtype=np.float64,
        na_values=-999,
    )

    # undef values have a high float number? or keep Nan?
    # df.fillna(Well.UNDEF, inplace=True)

    dfr = _trim_on_lognames(dfr, lognames, lognames_strict, wname)
    mdlogname, zonelogname = _check_special_logs(
        dfr, mdlogname, zonelogname, strict, wname
    )

    self._wlogtype = wlogtype
    self._wlogrecord = wlogrecord
    self._rkb = rkb
    self._xpos = xpos
    self._ypos = ypos
    self._wname = wname
    self._df = dfr
    self._mdlogname = mdlogname
    self._zonelogname = zonelogname


def _trim_on_lognames(dfr, lognames, lognames_strict, wname):
    """Reduce the dataframe based on provided list of lognames"""
    if lognames == "all":
        return dfr

    uselnames = ["X_UTME", "Y_UTMN", "Z_TVDSS"]
    if isinstance(lognames, str):
        uselnames.append(lognames)
    elif isinstance(lognames, list):
        uselnames.extend(lognames)

    newdf = pd.DataFrame()
    for lname in uselnames:
        if lname in dfr.columns:
            newdf[lname] = dfr[lname]
        else:
            if lognames_strict:
                msg = "Logname <{0}> is not present for <{1}>".format(lname, wname)
                msg += " (required log under condition lognames_strict=True)"
                raise ValueError(msg)

    return newdf


def _check_special_logs(dfr, mdlogname, zonelogname, strict, wname):
    """Check for MD log and Zonelog, if requested"""

    mname = mdlogname
    zname = zonelogname

    if mdlogname is not None:
        if mdlogname not in dfr.columns:
            msg = (
                "mdlogname={} was requested but no such log "
                "found for well {}".format(mdlogname, wname)
            )
            mname = None
            if strict:
                raise ValueError(msg)

            logger.warning(msg)

    # check for zone log:
    if zonelogname is not None:
        if zonelogname not in dfr.columns:
            msg = (
                "zonelogname={} was requested but no such log "
                "found for well {}".format(zonelogname, wname)
            )
            zname = None
            if strict:
                raise ValueError(msg)

            logger.warning(msg)

    return mname, zname


def export_rms_ascii(self, wfile, precision=4):
    """Export to RMS well format."""

    with open(wfile, "w") as fwell:
        print("{}".format("1.0"), file=fwell)
        print("{}".format("Unknown"), file=fwell)
        if self._rkb is None:
            print("{} {} {}".format(self._wname, self._xpos, self._ypos), file=fwell)
        else:
            print(
                "{} {} {} {}".format(self._wname, self._xpos, self._ypos, self._rkb),
                file=fwell,
            )
        print("{}".format(len(self.lognames)), file=fwell)
        for lname in self.lognames:
            usewrec = "linear"
            wrec = []
            if isinstance(self._wlogrecord[lname], dict):
                for key in self._wlogrecord[lname]:
                    wrec.append(key)
                    wrec.append(self._wlogrecord[lname][key])
                usewrec = " ".join(str(x) for x in wrec)

            print("{} {} {}".format(lname, self._wlogtype[lname], usewrec), file=fwell)

    # now export all logs as pandas framework
    tmpdf = self._df.copy()
    tmpdf.fillna(value=-999, inplace=True)

    # make the disc as is np.int
    for lname in self._wlogtype:
        if self._wlogtype[lname] == "DISC":
            tmpdf[[lname]] = tmpdf[[lname]].astype(int)

    cformat = "%-." + str(precision) + "f"
    tmpdf.to_csv(
        wfile,
        sep=" ",
        header=False,
        index=False,
        float_format=cformat,
        escapechar=" ",
        mode="a",
    )
