"""Import/export or scans of grid properties (cf GridProperties class"""

import numpy as np

import xtgeo
import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')


def import_ecl_output_v1(props, pfile, etype=1, dates=[],
                         grid=None, names=[], namestyle=0):

    # this code is barely hopeless spagetti and shall be replaced with _v2

    if not grid:
        raise ValueError('Grid Geometry object is missing')
    else:
        props._grid = grid

    # for handling of SOIL... (need SGAS and SWAT)
    qsoil = False
    qsgas = False
    qswat = False

    # sort dates if any, and special treatment of SOIL keyword
    if etype == 5:
        if dates:
            dates.sort()
            logger.debug(dates)
        else:
            raise ValueError('Restart file indicated, but no date(s)')

        if 'SGAS' in names:
            qsgas = True
        if 'SWAT' in names:
            qswat = True
        if 'SOIL' in names:
            qsoil = True

    if qsoil:
        names.remove('SOIL')
    if qsoil and not qsgas:
        names.append('SGAS')
    if qsoil and not qswat:
        names.append('SWAT')

    logger.info("Scanning NX NY NZ for checking...")
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()

    _cxtgeo.grd3d_scan_ecl_init_hd(1, ptr_ncol, ptr_nrow, ptr_nlay,
                                   pfile, xtg_verbose_level)

    ncol0 = _cxtgeo.intpointer_value(ptr_ncol)
    nrow0 = _cxtgeo.intpointer_value(ptr_nrow)
    nlay0 = _cxtgeo.intpointer_value(ptr_nlay)

    if grid.ncol != ncol0 or grid.nrow != nrow0 or \
            grid.nlay != nlay0:
        logger.error("Errors in dimensions property vs grid")
        return

    props._ncol = ncol0
    props._nrow = nrow0
    props._nlay = nlay0

    # split date and populate array
    if not dates:
        dates = [99998877]  # just a value if INIT file
    else:
        ndates = len(dates)

    ptr_day = _cxtgeo.new_intarray(len(dates))
    ptr_month = _cxtgeo.new_intarray(len(dates))
    ptr_year = _cxtgeo.new_intarray(len(dates))

    idate = 0
    for date in dates:
        date = str(date)
        logger.debug("DATE is {}".format(date))
        day = int(date[6:8])
        mon = int(date[4:6])
        yer = int(date[0:4])

        logger.debug("DD MM YYYY input is {} {} {}".
                     format(day, mon, yer))

        _cxtgeo.intarray_setitem(ptr_day, idate, day)
        _cxtgeo.intarray_setitem(ptr_month, idate, mon)
        _cxtgeo.intarray_setitem(ptr_year, idate, yer)
        idate += 1

    nklist = len(names)

    if dates:
        nmult = len(dates)
    else:
        nmult = 1

    ptr_dvec_v = _cxtgeo.new_doublearray(nmult * nklist * ncol0 *
                                         nrow0 * nlay0)

    ptr_nktype = _cxtgeo.new_intarray(nmult * nklist)
    ptr_norder = _cxtgeo.new_intarray(nmult * nklist)
    ptr_dsuccess = _cxtgeo.new_intarray(nmult)

    useprops = ""
    for name in names:
        useprops = useprops + "{0:8s}|".format(name)
        logger.debug("<{}>".format(useprops))

    if etype == 1:
        ndates = 0

    logger.debug("NKLIST and NDATES is {} {}".
                 format(nklist, ndates))

    _cxtgeo.grd3d_import_ecl_prop(etype,
                                  ncol0 * nrow0 * nlay0,
                                  grid._p_actnum_v,
                                  nklist,
                                  useprops,
                                  ndates,
                                  ptr_day,
                                  ptr_month,
                                  ptr_year,
                                  pfile,
                                  ptr_dvec_v,
                                  ptr_nktype,
                                  ptr_norder,
                                  ptr_dsuccess,
                                  xtg_verbose_level,
                                  )

    # Drink a mix of coffee and whisky before you read this:
    # a list is returned, this list is nktype, and can be like this
    # nktype=(1,2,1,2,2,3) meaning that keyword no 1,3 are INT (type 1),
    # 2,4,5 are FLOAT (type 2), and 6 is DOUBLE (type 3)

    # In addition we have nkorder, which specifies the _actual_ order
    # hence (PORO PORV) may have nkorder (1 0), meaning that PORV is first

    # and on top of that, we have dates... (the trick is to avoid madness)

    # hence the input array may be like this for a restart
    #  .........20140101.............   ...........20150201............
    # [...SGAS....SWAT....PRESSURE...   ...SGAS.....SWAT....PRESSURE...]

    # the issue is now to convert to XTGeo storage

    # scan number of keywords that where got successfully
    dateswarning = []
    nkeysgot = 0
    for kn in range(nklist):
        order = _cxtgeo.intarray_getitem(ptr_norder, kn)
        logger.debug("ORDER = {}".format(order))
        if order >= 0:
            nkeysgot += 1
        else:
            logger.error('Did not find property'
                         ' <{}>'.format(names[kn]))
            raise ValueError('Property not found: {}'.
                             format(names[kn]))

    if nkeysgot == 0:
        logger.error("No keywords found. STOP!")
        raise ValueError('No property keywords found')

    logger.info(
        "Number of keys successfully read: {}".format(nkeysgot))

    nloop = 1
    if ndates >= 1:
        nloop = ndates

    dcounter = 0

    for idate in range(nloop):

        dsuccess = _cxtgeo.intarray_getitem(ptr_dsuccess, idate)

        usedatetag = ""

        logger.debug("Date tag is <{}> and success was {}"
                     .format(dates[idate], dsuccess))

        if ndates > 0 and dsuccess == 1:
            if namestyle == 1:
                dtag = str(dates[idate])
                usedatetag = "--" + dtag[0:4] + "_" + dtag[4:6] + \
                             "_" + dtag[6:8]
            else:
                usedatetag = "_" + str(dates[idate])
            logger.debug("Date tag is <{}> and success was {}"
                         .format(usedatetag, dsuccess))
            dcounter += 1

        elif ndates > 0 and dsuccess == 0:
            dateswarning.append(dates[idate])

        else:
            # INIT props:
            dsuccess = 1
            dcounter = 1

        # the actual order is dependent on time step.
        # E.g. tstep and kweywords order
        # DATE:     2001-02-03           2001-08-01       2014-01-01
        # Keyw: SWAT PRESSURE SOIL SWAT PRESSURE SOIL SWAT PRESSURE SOIL
        # Order:   1      0     2  |  1      0     2  |  1     0      2
        # Actorder 1      0     2  |  4      3     5  |  7     6      8
        # which means ... actorder = order + (dcounter-1)*nkeysgot

        if dsuccess > 0:

            if ndates > 0:
                props._dates.append(dates[idate])

            for kn in range(nklist):
                nktype = _cxtgeo.intarray_getitem(ptr_nktype, kn)
                norder = _cxtgeo.intarray_getitem(ptr_norder, kn)

                aorder = norder + (dcounter - 1) * nkeysgot

                pname = names[kn]
                ppname = pname + usedatetag

                # create the object
                xelf = xtgeo.grid3d.GridProperty()

                if nktype == 1:
                    xelf._cvalues = _cxtgeo.new_intarray(ncol0 * nrow0 *
                                                         nlay0)

                    xelf._isdiscrete = True
                    xelf._undef = _cxtgeo.UNDEF_INT
                    xelf._undef_limit = _cxtgeo.UNDEF_INT_LIMIT
                    xelf._ptype = 2
                    xelf._dtype = np.int32
                    xelf._name = ppname
                    xelf._ncol = ncol0
                    xelf._nrow = nrow0
                    xelf._nlay = nlay0
                    if ndates > 0:
                        xelf._date = dates[idate]

                    _cxtgeo.grd3d_strip_anint(ncol0 * nrow0 * nlay0,
                                              aorder,
                                              ptr_dvec_v, xelf._cvalues,
                                              xtg_verbose_level)

                else:
                    xelf._cvalues = _cxtgeo.new_doublearray(
                        ncol0 * nrow0 * nlay0)
                    xelf._isdiscrete = False
                    xelf._undef = _cxtgeo.UNDEF
                    xelf._undef_limit = _cxtgeo.UNDEF_LIMIT
                    xelf._ptype = 1
                    xelf._name = ppname
                    xelf._ncol = ncol0
                    xelf._nrow = nrow0
                    xelf._nlay = nlay0
                    if ndates > 0:
                        xelf._date = dates[idate]

                    _cxtgeo.grd3d_strip_adouble(ncol0 * nrow0 * nlay0,
                                                aorder,
                                                ptr_dvec_v, xelf._cvalues,
                                                xtg_verbose_level)

                xelf._update_values()

                props._names.append(ppname)
                props._props.append(xelf)

            # end of KN loop

            # SOIL: OK, now I have the following cases:
            # I ask for SOIL, but not SGAS, but SWAT
            # I ask for SOIL, but not SWAT, but SGAS
            # I ask for SOIL, but also SWAT + SGAS
            # I ask for SOIL, but none of SGAS/SWAT

            if qsoil:
                logger.info("Getting SOIL from SGAS and SWAT...")
                soilname = 'SOIL' + usedatetag
                sgasname = 'SGAS' + usedatetag
                swatname = 'SWAT' + usedatetag

                # create the oil object by copying
                myswat = props.get_prop_by_name(swatname)
                mysgas = props.get_prop_by_name(sgasname)

                logger.debug(mysgas.values)

                mysoil = myswat.copy(newname=soilname)

                mysoil.values = mysoil.values * -1 - mysgas.values + 1.0

                # now store the SOIL in the GridProperties class
                props._names.append(soilname)
                props._props.append(mysoil)

                # now we may neewd to remove SWAT and/or SGAS
                # if it was not asked for...

                if not qsgas:
                    props._names.remove(sgasname)
                    props._props.remove(mysgas)

                if not qswat:
                    props._names.remove(swatname)
                    props._props.remove(myswat)

    if len(dateswarning) > 0:
        raise RuntimeWarning('Some dates not found: {}'.
                             format(dateswarning))
