# #############################################################################
# NAME:
#    grid_properties.py
#
# AUTHOR(S):
#    Jan C. Rivenaes
#
# DESCRIPTION:
#    XTGeo class for a group of grid properties. See GridProperty for
#    individual properties.
#
#    In methods, "prop/props" is _the_ shorthand of property/properties!
#
# TODO/ISSUES/BUGS:
#
# LICENCE:
#    Statoil property
#
# CODING STANDARD:
#    PEP8 / Googe style (http://google.github.io/styleguide/pyguide.html)
#    (http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html)
#
# #############################################################################
import numpy as np
import os.path
import logging
import cxtgeo.cxtgeo as _cxtgeo
import sys
from xtgeo.common import XTGeoDialog
from .grid3d import Grid3D

from .grid_property import GridProperty


class GridProperties(Grid3D):
    """
    Class for a collection of 3D grid properties, belonging to the same grid.
    See also the GridProperty() class
    """

    def __init__(self):
        """
        The __init__ (constructor) method.

        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._nx = 10
        self._ny = 12
        self._nz = 14

        self._props = []            # list of GridProperty objects
        self._names = []            # list of GridProperty names

        self._xtg = XTGeoDialog()

# ==============================================================================
# Import and export
# This class can importies several properties in one go, which is efficient
# for some file types such as Eclipse INIT and UNRST, and Roff
# ==============================================================================

    def from_file(self, pfile, fformat='roff', names=[],
                  dates=[], grid=None, namestyle=0):
        """
        Import grid properties from file in one go. This class is particurlary
        useful for Eclipse INIT and RESTART files

        Args:
            pfile (str): Name of file with properties
            fformat (str): roff/init/unrst
            names: list of property names, e.g. ['PORO', 'PERMX']
            dates: list of dates on YYYYMMDD format, for restart files
            grid (obj): The grid geometry object (optional?)
            namestyle (int): 0 (default) for style SWAT_20110223,
                1 for SWAT--2011_02_23 (applies to restart only)

        Example::
            >>> props = GridProperties()
            >>> props.from_file('ECL.UNRST', fformat='unrst',
                dates=[20110101, 20141212], names=['PORO', 'DZ']

        Raises:
            FileNotFoundError: if file is not found

        """

        # work on file extension
        froot, fext = os.path.splitext(pfile)
        if not fext:
            # file extension is missing, guess from format
            self.logger.info("File extension missing; guessing...")

            useext = ''
            if fformat == 'init':
                useext = '.INIT'
            elif fformat == 'unrst':
                useext = '.UNRST'
            elif fformat == 'roff':
                useext = '.roff'

            pfile = froot + useext

        self.logger.info("File name to be used is {}".format(pfile))

        if os.path.isfile(pfile):
            self.logger.info('File {} exists OK'.format(pfile))
        else:
            self.logger.warning('No such file: {}'.format(pfile))
            sys.exit(1)

        if (fformat.lower() == "roff"):
            self._import_roff(pfile, names)
        elif (fformat.lower() == "init"):
            self._import_ecl_output(pfile, names=names, etype=1,
                                    grid=grid)
        elif (fformat.lower() == "unrst"):
            self._import_ecl_output(pfile, names=names, etype=5,
                                    dates=dates, grid=grid,
                                    namestyle=namestyle)
        else:
            self.logger.warning("Invalid file format")
            sys.exit(1)

    def to_file(self, pfile, fformat="roff"):
        """Export grid property to file. NB not working!

        Args:
            pfile (str): file name
            fformat (str): file format to be used (roff is the only supported)
            mode (int): 0 for binary ROFF, 1 for ASCII
        """
        pass

    # =========================================================================
    # Properties, NB decorator only works when class is inherited from "object"
    # =========================================================================

    @property
    def names(self):
        """
        Returns a list of property names

        Example::

            proplist = props.names
            for prop in proplist:
                print ('Property is {}'.format(prop))

        """

        return self._names

    @property
    def props(self):
        """
        Returns a list of property objects

        Example::

            proplist = props.properties
            for prop in proplist:
                print ('Property object ID is {}'.format(prop))

        """

        return self._props

    # =========================================================================
    # Other setters and getters
    # =========================================================================

    def get_prop_by_name(self, name):
        """
        Find and return a property object (GridProperty) by name

        Example::

            poro = props.property('PORO')
            # add 0.1 to poro (via numpy)
            poro.values3d = poro.values3d + 0.1

        """

        for p in self._props:
            self.logger.debug("Look for {}, actual is {}".format(name, p.name))
            if p.name == name:
                self.logger.debug(repr(p))
                return p

        raise Exception('Cannot find property with name <{}>'.format(name))

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import methods for various formats
    # -------------------------------------------------------------------------

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import Eclipse's INIT and UNRST formats
    # type: 1 binary INIT
    #       5 binary unified restart
    # These Eclipse files need the NX NY NZ and the ACTNUM array as they
    # only store the active cells for most vectors. Hence, the grid sizes
    # and the actnum array shall be provided, and that is done via the grid
    # geometry object directly. Note that this import only takes may
    # take several properties at the time.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _import_ecl_output(self, pfile, etype=1, dates=[],
                           grid=None, names=[], namestyle=0):

        if not grid:
            raise Exception('Grid Geometry object is missing')

        _cxtgeo.xtg_verbose_file("NONE")

        # for handling of SOIL... (need SGAS and SWAT)
        qsoil = False
        qsgas = False
        qswat = False

        # sort dates if any, and special treatment of SOIL keyword
        if etype == 5:
            if dates:
                dates.sort()
                self.logger.debug(dates)
            else:
                raise Exception('Oh... Restart file indicated, but no date(s)')

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

        xtg_verbose_level = self._xtg.get_syslevel()

        self.logger.info("Scanning NX NY NZ for checking...")
        ptr_nx = _cxtgeo.new_intpointer()
        ptr_ny = _cxtgeo.new_intpointer()
        ptr_nz = _cxtgeo.new_intpointer()

        _cxtgeo.grd3d_scan_ecl_init_hd(1, ptr_nx, ptr_ny, ptr_nz,
                                       pfile, xtg_verbose_level)

        nx0 = _cxtgeo.intpointer_value(ptr_nx)
        ny0 = _cxtgeo.intpointer_value(ptr_ny)
        nz0 = _cxtgeo.intpointer_value(ptr_nz)

        if grid.nx != nx0 or grid.ny != ny0 or \
                grid.nz != nz0:
            self.logger.error("Errors in dimensions property vs grid")
            return

        self._nx = nx0
        self._ny = ny0
        self._nz = nz0

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
            self.logger.debug("DATE is {}".format(date))
            day = int(date[6:8])
            mon = int(date[4:6])
            yer = int(date[0:4])

            self.logger.debug("DD MM YYYY input is {} {} {}".
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

        ptr_dvec_v = _cxtgeo.new_doublearray(nmult * nklist * nx0 * ny0 * nz0)

        ptr_nktype = _cxtgeo.new_intarray(nmult * nklist)
        ptr_norder = _cxtgeo.new_intarray(nmult * nklist)
        ptr_dsuccess = _cxtgeo.new_intarray(nmult)

        useprops = ""
        for name in names:
            useprops = useprops + "{0:8s}|".format(name)
            self.logger.debug("<{}>".format(useprops))

        if etype == 1:
            ndates = 0

        self.logger.debug("NKLIST and NDATES is {} {}".
                          format(nklist, ndates))

        _cxtgeo.grd3d_import_ecl_prop(etype,
                                      nx0 * ny0 * nz0,
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
        nkeysgot = 0
        for kn in range(nklist):
            order = _cxtgeo.intarray_getitem(ptr_norder, kn)
            self.logger.debug("ORDER = {}".format(order))
            if order >= 0:
                nkeysgot += 1
            else:
                self.logger.warning(
                    "Did not find property <{}>".format(
                        names[kn]))

        if nkeysgot == 0:
            self.logger.error("No keywords found. STOP!")
            raise Exception("No keywords found")

        self.logger.info(
            "Number of keys successfully read: {}".format(nkeysgot))

        nloop = 1
        if ndates >= 1:
            nloop = ndates

        dcounter = 0

        for idate in range(nloop):

            dsuccess = _cxtgeo.intarray_getitem(ptr_dsuccess, idate)

            usedatetag = ""

            self.logger.debug("Date tag is <{}> and success was {}"
                              .format(dates[idate], dsuccess))

            if ndates > 0 and dsuccess == 1:
                if namestyle == 1:
                    dtag = str(dates[idate])
                    usedatetag = "--" + dtag[0:4] + "_" + dtag[4:6] + \
                                 "_" + dtag[6:8]
                else:
                    usedatetag = "_" + str(dates[idate])
                self.logger.debug("Date tag is <{}> and success was {}"
                                  .format(usedatetag, dsuccess))
                dcounter += 1

            elif ndates > 0 and dsuccess == 0:
                self.logger.critical("Date <{}> not found in file <{}>!"
                                     .format(dates[idate], pfile))
                self.logger.critical("Date tag is <{}> and success was {}"
                                     .format(usedatetag, dsuccess))
                raise RuntimeError("Date {} not found in restart file".
                                   format(dates[idate]))
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

            if dsuccess:
                for kn in range(nklist):
                    nktype = _cxtgeo.intarray_getitem(ptr_nktype, kn)
                    norder = _cxtgeo.intarray_getitem(ptr_norder, kn)

                    aorder = norder + (dcounter - 1) * nkeysgot

                    pname = names[kn]
                    ppname = pname + usedatetag

                    # create the object
                    xelf = GridProperty()

                    if nktype == 1:
                        xelf._cvalues = _cxtgeo.new_intarray(nx0 * ny0 * nz0)
                        xelf._isdiscrete = True
                        xelf._undef = _cxtgeo.UNDEF_INT
                        xelf._undef_limit = _cxtgeo.UNDEF_INT_LIMIT
                        xelf._ptype = 2
                        xelf._dtype = np.int32
                        xelf._name = ppname
                        xelf._nx = nx0
                        xelf._ny = ny0
                        xelf._nz = nz0

                        _cxtgeo.grd3d_strip_anint(nx0 * ny0 * nz0, aorder,
                                                  ptr_dvec_v, xelf._cvalues,
                                                  xtg_verbose_level)

                    else:
                        xelf._cvalues = _cxtgeo.new_doublearray(
                            nx0 * ny0 * nz0)
                        xelf._isdiscrete = False
                        xelf._undef = _cxtgeo.UNDEF
                        xelf._undef_limit = _cxtgeo.UNDEF_LIMIT
                        xelf._ptype = 1
                        xelf._name = ppname
                        xelf._nx = nx0
                        xelf._ny = ny0
                        xelf._nz = nz0

                        _cxtgeo.grd3d_strip_adouble(nx0 * ny0 * nz0, aorder,
                                                    ptr_dvec_v, xelf._cvalues,
                                                    xtg_verbose_level)

                    xelf._update_values()

                    self._names.append(ppname)
                    self._props.append(xelf)

                # end of KN loop

                # SOIL: OK, now I have the following cases:
                # I ask for SOIL, but not SGAS, but SWAT
                # I ask for SOIL, but not SWAT, but SGAS
                # I ask for SOIL, but also SWAT + SGAS
                # I ask for SOIL, but none of SGAS/SWAT

                if qsoil:
                    self.logger.info("Getting SOIL from SGAS and SWAT...")
                    soilname = 'SOIL' + usedatetag
                    sgasname = 'SGAS' + usedatetag
                    swatname = 'SWAT' + usedatetag

                    # create the oil object by copying
                    myswat = self.get_prop_by_name(swatname)
                    mysgas = self.get_prop_by_name(sgasname)

                    print(mysgas.values)

                    mysoil = myswat.copy(newname=soilname)

                    mysoil.values = mysoil.values * -1 - mysgas.values + 1.0

                    # now store the SOIL in the GridProperties class
                    self._names.append(soilname)
                    self._props.append(mysoil)

                    # now we may neewd to remove SWAT and/or SGAS
                    # if it was not asked for...

                    if not qsgas:
                        self._names.remove(sgasname)
                        self._props.remove(mysgas)

                    if not qswat:
                        self._names.remove(swatname)
                        self._props.remove(myswat)
