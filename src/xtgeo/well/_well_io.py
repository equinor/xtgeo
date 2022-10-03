# -*- coding: utf-8 -*-
"""Well input and ouput, private module"""
import json
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_rms_ascii(
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
    wlogrecords = dict()

    xlognames_all = ["X_UTME", "Y_UTMN", "Z_TVDSS"]
    xlognames = []

    lnum = 1
    with open(wfile.file, "r") as fwell:
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
                    wlogrecords[lname] = xdict
                else:
                    wlogrecords[lname] = rxv

                nlogread += 1

                if nlogread > nlogs:
                    break

            lnum += 1

    # now import all logs as pandas framework

    dfr = pd.read_csv(
        wfile.file,
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

    return {
        "wlogtypes": wlogtype,
        "wlogrecords": wlogrecords,
        "rkb": rkb,
        "xpos": xpos,
        "ypos": ypos,
        "wname": wname,
        "df": dfr,
        "mdlogname": mdlogname,
        "zonelogname": zonelogname,
    }


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
            if isinstance(self._wlogrecords[lname], dict):
                for key in self._wlogrecords[lname]:
                    wrec.append(key)
                    wrec.append(self._wlogrecords[lname][key])
                usewrec = " ".join(str(x) for x in wrec)

            print("{} {} {}".format(lname, self._wlogtypes[lname], usewrec), file=fwell)

    # now export all logs as pandas framework
    tmpdf = self._df.copy()
    tmpdf.fillna(value=-999, inplace=True)

    # make the disc as is np.int
    for lname in self._wlogtypes:
        if self._wlogtypes[lname] == "DISC":
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


def export_hdf5_well(self, wfile, compression="lzf"):
    """Save to HDF5 format."""
    logger.info("Export to hdf5 format...")

    self._ensure_consistency()

    self.metadata.required = self

    meta = self.metadata.get_metadata()
    jmeta = json.dumps(meta)

    complib = "zlib"  # same as default lzf
    complevel = 5
    if compression and compression == "blosc":
        complib = "blosc"
    else:
        complevel = 0

    with pd.HDFStore(wfile.file, "w", complevel=complevel, complib=complib) as store:
        logger.info("export to HDF5 %s", wfile.name)
        store.put("Well", self._df)
        store.get_storer("Well").attrs["metadata"] = jmeta
        store.get_storer("Well").attrs["provider"] = "xtgeo"
        store.get_storer("Well").attrs["format_idcode"] = 1401

    logger.info("Export to hdf5 format... done!")


def import_wlogs(wlogs: OrderedDict):
    """
    This converts joined wlogtypes/wlogrecords such as found in
    the hdf5 format to the format used in the Well object.

    >>> import_wlogs(OrderedDict())
    {'wlogtypes': {}, 'wlogrecords': {}}
    >>> import_wlogs(OrderedDict([("X_UTME", ("CONT", None))]))
    {'wlogtypes': {'X_UTME': 'CONT'}, 'wlogrecords': {'X_UTME': None}}

    Returns:
        dictionary with "wlogtypes" and "wlogrecords" as keys
        and corresponding values.
    """
    wlogtypes = dict()
    wlogrecords = dict()
    for key in wlogs.keys():
        typ, rec = wlogs[key]

        if typ in {"DISC", "CONT"}:
            wlogtypes[key] = deepcopy(typ)
        else:
            raise ValueError(f"Invalid log type found in input: {typ}")

        if rec is None or isinstance(rec, dict):
            wlogrecords[key] = deepcopy(rec)
        else:
            raise ValueError(f"Invalid log record found in input: {rec}")
    return {"wlogtypes": wlogtypes, "wlogrecords": wlogrecords}


def import_hdf5_well(wfile, **kwargs):
    """Load from HDF5 format."""
    logger.info("The kwargs may be unused: %s", kwargs)
    reqattrs = xtgeo.MetaDataWell.REQUIRED

    with pd.HDFStore(wfile.file, "r") as store:
        data = store.get("Well")
        wstore = store.get_storer("Well")
        jmeta = wstore.attrs["metadata"]
        # provider = wstore.attrs["provider"]
        # format_idcode = wstore.attrs["format_idcode"]

    if isinstance(jmeta, bytes):
        jmeta = jmeta.decode()

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)
    req = meta["_required_"]
    result = dict()
    for myattr in reqattrs:
        if myattr == "wlogs":
            result.update(import_wlogs(req[myattr]))
        elif myattr == "name":
            result["wname"] = req[myattr]
        else:
            result[myattr] = req[myattr]

    result["df"] = data
    return result
