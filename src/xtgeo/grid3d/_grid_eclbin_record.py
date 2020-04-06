"""Generic function to import ECL binary records for EGRID, INIT etc"""

from __future__ import print_function, absolute_import


import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def eclbin_record(gfile, kwname, kwlen, kwtype, kwbyte):
    # generic: read a single binary Eclipse record via cxtgeo

    ilen = flen = dlen = 1

    if kwtype == "INTE":
        ilen = kwlen
        kwntype = 1
    elif kwtype == "REAL":
        flen = kwlen
        kwntype = 2
    elif kwtype == "DOUB":
        dlen = kwlen
        kwntype = 3
    elif kwtype == "LOGI":
        ilen = kwlen
        kwntype = 5
    else:
        raise ValueError(
            "Wrong type of kwtype {} for {}, must be INTE, REAL "
            "or DOUB".format(kwtype, kwname)
        )

    npint = np.zeros((ilen), dtype=np.int32)
    npflt = np.zeros((flen), dtype=np.float32)
    npdbl = np.zeros((dlen), dtype=np.float64)

    _cxtgeo.grd3d_read_eclrecord(
        # int(kwbyte) .. to solve a deep type issue in pandas < 0.21
        gfile.get_cfhandle(),
        int(kwbyte),
        kwntype,
        npint,
        npflt,
        npdbl,
    )
    gfile.cfclose()

    npuse = None
    if kwtype == "INTE":
        npuse = npint
        del npflt
        del npdbl
    if kwtype == "REAL":
        npuse = npflt
        del npint
        del npdbl
    if kwtype == "DOUB":
        npuse = npdbl
        del npint
        del npflt
    if kwtype == "LOGI":
        npuse = npint
        del npdbl
        del npflt

    return npuse
