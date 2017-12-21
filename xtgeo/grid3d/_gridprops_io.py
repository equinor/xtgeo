"""Import/export or scans of grid properties (cf GridProperties class"""
import pandas as pd

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')


def scan_keywords(pfile, fformat='xecl', maxkeys=10000, dataframe=False):

    ultramax = int(1000000 / 9)  # cf *swig_bnd_char_1m in cxtgeo.i
    if maxkeys > ultramax:
        raise ValueError('maxkeys value is too large, must be < {}'
                         .format(ultramax))

    rectypes = _cxtgeo.new_intarray(maxkeys)
    reclens = _cxtgeo.new_longarray(maxkeys)
    recstarts = _cxtgeo.new_longarray(maxkeys)

    nkeys, keywords = _cxtgeo.grd3d_scan_eclbinary(pfile, rectypes, reclens,
                                                   recstarts, maxkeys,
                                                   xtg_verbose_level)

    keywords = keywords.replace(' ', '')
    keywords = keywords.split('|')

    # record types translation (cf: grd3d_scan_eclbinary.c in cxtgeo)
    rct = {'1': 'INTE', '2': 'REAL', '3': 'DOUB', '4': 'CHAR', '5': 'LOGI',
           '6': 'MESS', '-1': '????'}

    rc = []
    rl = []
    rs = []
    for i in range(nkeys):
        rc.append(rct[str(_cxtgeo.intarray_getitem(rectypes, i))])
        rl.append(_cxtgeo.longarray_getitem(reclens, i))
        rs.append(_cxtgeo.longarray_getitem(recstarts, i))

    _cxtgeo.delete_intarray(rectypes)
    _cxtgeo.delete_longarray(reclens)
    _cxtgeo.delete_longarray(recstarts)

    result = list(zip(keywords, rc, rl, rs))

    if dataframe:
        cols = ['KEYWORD', 'TYPE', 'NITEMS', 'BYTESTART']
        df = pd.DataFrame.from_records(result, columns=cols)
        return df
    else:
        return result


def scan_dates(pfile, fformat='unrst', maxdates=1000, dataframe=False):

    seq = _cxtgeo.new_intarray(maxdates)
    day = _cxtgeo.new_intarray(maxdates)
    mon = _cxtgeo.new_intarray(maxdates)
    yer = _cxtgeo.new_intarray(maxdates)

    nstat = _cxtgeo.grd3d_ecl_tsteps(pfile, seq, day, mon, yer, maxdates,
                                     xtg_verbose_level)

    sq = []
    da = []
    for i in range(nstat):
        sq.append(_cxtgeo.intarray_getitem(seq, i))
        dday = _cxtgeo.intarray_getitem(day, i)
        dmon = _cxtgeo.intarray_getitem(mon, i)
        dyer = _cxtgeo.intarray_getitem(yer, i)
        date = '{0:4}{1:02}{2:02}'.format(dyer, dmon, dday)
        da.append(int(date))

    for item in [seq, day, mon, yer]:
        _cxtgeo.delete_intarray(item)

    zdates = list(zip(sq, da))  # list for PY3

    if dataframe:
        cols = ['SEQNUM', 'DATE']
        df = pd.DataFrame.from_records(zdates, columns=cols)
        return df
    else:
        return zdates
