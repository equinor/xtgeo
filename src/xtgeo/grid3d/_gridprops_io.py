"""Import/export or scans of grid properties (cf GridProperties class"""
import warnings

import pandas as pd

import xtgeo
import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')


def _get_fhandle(pfile):
    """Examine for file or filehandle and return filehandle + a bool"""

    pclose = True
    if "Swig Object of type 'FILE" in str(pfile):
        fhandle = pfile
        pclose = False
    else:
        fhandle = _cxtgeo.xtg_fopen(pfile, 'rb')

    return fhandle, pclose


def _close_fhandle(fh, flag):
    """Close file if flag is True"""

    if flag:
        _cxtgeo.xtg_fclose(fh)
        logger.debug('File is now closed')
    else:
        logger.debug('File remains open')


def scan_keywords(pfile, fformat='xecl', maxkeys=10000, dataframe=False):

    # In case pfile is not a file name but a swig pointer to a file handle,
    # the file must not be closed

    ultramax = int(1000000 / 9)  # cf *swig_bnd_char_1m in cxtgeo.i
    if maxkeys > ultramax:
        raise ValueError('maxkeys value is too large, must be < {}'
                         .format(ultramax))

    rectypes = _cxtgeo.new_intarray(maxkeys)
    reclens = _cxtgeo.new_longarray(maxkeys)
    recstarts = _cxtgeo.new_longarray(maxkeys)

    fhandle, pclose = _get_fhandle(pfile)

    nkeys, keywords = _cxtgeo.grd3d_scan_eclbinary(fhandle, rectypes, reclens,
                                                   recstarts, maxkeys,
                                                   xtg_verbose_level)

    _close_fhandle(fhandle, pclose)

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

    fhandle, pclose = _get_fhandle(pfile)

    nstat = _cxtgeo.grd3d_ecl_tsteps(fhandle, seq, day, mon, yer, maxdates,
                                     xtg_verbose_level)

    _close_fhandle(fhandle, pclose)

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


def import_ecl_output_v2(props, pfile, names=None, dates=None,
                         grid=None, namestyle=0):

    if not grid:
        raise ValueError('Grid Geometry object is missing')
    else:
        props._grid = grid

    if not names:
        raise ValueError('Name list is missing')

    fhandle = _cxtgeo.xtg_fopen(pfile, 'rb')

    # check valid dates, and remove invalid entries (allowing that user
    # can be a bit sloppy on DATES)

    validdates = [None]
    if dates:
        dlist = props.scan_dates(fhandle)

        validdates = []
        for date in dates:
            for ditem in dlist:
                if str(date) == str(ditem[1]):
                    validdates.append(date)

        if len(validdates) < 1:
            raise ValueError('No valid dates given (dates: {} vs {})'
                             .format(dates, dlist))

        if len(dates) > len(validdates):
            invalidddates = list(set(dates).difference(validdates))
            msg = 'Some dates not found: {}'.format(invalidddates)
            warnings.warn(RuntimeWarning(msg))

    usenames = list(names)  # to make copy

    # special treatement of SOIL since it is not present in restarts, but
    # has to be computed from SWAT and SGAS as SOIL = 1 - SWAT - SGAS

    qsoil = False
    if 'SOIL' in set(names):
        usenames.insert(0, 'SWAT')
        usenames.insert(0, 'SGAS')
        usenames.remove('SOIL')
        qsoil = True

    logger.info('Use names: {}'.format(usenames))
    logger.info('Valid dates: {}'.format(validdates))

    # now import each property
    firstproperty = True

    for date in validdates:
        xprop = dict()
        soil_ok = False

        for name in usenames:

            if date is None:
                date = None
                propname = name
                etype = 1
            else:
                propname = name + '_' + str(date)
                etype = 5

            prop = xtgeo.grid3d.GridProperty()

            # use a private GridProperty function here, for convinience
            # (since filehandle)
            ier = prop._import_ecl_output(fhandle, name=name, date=date,
                                          grid=grid, apiversion=2, etype=etype)
            if ier != 0:
                raise ValueError('Something went wrong, IER = {} while '
                                 'name={}, date={}, etype={}, propname={}'
                                 .format(ier, name, date, etype, propname))

            if firstproperty:
                ncol = prop.ncol
                nrow = prop.nrow
                nlay = prop.nlay
                firstproperty = False

            if qsoil and not soil_ok:
                if name in set(['SWAT', 'SGAS']):
                    xprop[name] = prop.values
                    logger.info('Made xprop for {}'.format(name))

                if len(xprop) == 2:
                    soilv = xprop['SWAT'].copy() * 0 + 1
                    soilv = soilv - xprop['SWAT'] - xprop['SGAS']
                    soil_ok = True
                    propname = 'SOIL' + '_' + str(date)

                    prop = xtgeo.grid3d.GridProperty(ncol=ncol, nrow=nrow,
                                                     nlay=nlay, name=propname,
                                                     discrete=False,
                                                     values=soilv)
                else:
                    logger.info('Length xprop is {}'.format(len(xprop)))
                    continue

            logger.info('Appended property {}'.format(propname))
            props._names.append(propname)
            props._props.append(prop)

    props._ncol = ncol
    props._nrow = nrow
    props._nlay = nlay
    if validdates[0] != 0:
        props._dates = validdates
