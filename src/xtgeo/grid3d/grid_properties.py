# -*- coding: utf-8 -*-

"""GridProperties class"""
from __future__ import division, absolute_import
from __future__ import print_function

import warnings

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import XTGDescription

from ._grid3d import Grid3D
from .grid_property import GridProperty

from . import _gridprops_io
from . import _grid3d_utils as utils
from . import _gridprops_etc
from . import _grid_etc1

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# --------------------------------------------------------------------------------------
# Comment on 'asmasked' vs 'activeonly:
#
# 'asmasked'=True will return a np.ma array, while 'asmasked' = False will
# return a np.ndarray
#
# The 'activeonly' will filter out masked entries, or use None or np.nan
# if 'activeonly' is False.
#
# Use word 'zerobased' for a bool regrading startcell basis is 1 or 0
#
# For functions with mask=... ,they should be replaced with asmasked=...
# --------------------------------------------------------------------------------------


class GridProperties(Grid3D):
    """Class for a collection of 3D grid props, belonging to the same grid.

    See also the :class:`GridProperty` class.
    """

    def __init__(self, *args, **kwargs):
        super(GridProperties, self).__init__(*args, **kwargs)

        self._ncol = kwargs.get("ncol", 5)
        self._nrow = kwargs.get("nrow", 12)
        self._nlay = kwargs.get("nlay", 2)

        self._props = []  # list of GridProperty objects
        self._names = []  # list of GridProperty names
        self._dates = []  # list of dates (_after_ import) YYYYDDMM

    def __repr__(self):
        myrp = (
            "{0.__class__.__name__} (id={1}) ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, nlay={0._nlay!r}, "
            "filesrc={0._names!r}".format(self, id(self))
        )
        return myrp

    def __str__(self):
        # user friendly print
        return self.describe(flush=False)

    # Properties:

    @property
    def names(self):
        """Returns a list of property names

        Example::

            namelist = props.names
            for prop in namelist:
                print ('Property name is {}'.format(name))

        """

        return self._names

    @property
    def props(self):
        """Returns a list of XTGeo GridProperty objects, None if empty.

        Example::

            gg = xtgeo.grid3d.Grid('TEST1.EGRID')
            myprops = XTGeo.grid3d.GridProperties()
            from_file('TEST1.INIT', fformat='init', names=['PORO']         )
            proplist = myprops.props
            for prop in proplist:
                print ('Property object ID is {}'.format(prop))

            # adding a property, e.g. get ACTNUM as a property from the grid
            actn = gg.get_actnum()  # this will get actn as a GridProperty
            myprops.add_props([actn])
        """
        if not self._props:
            return None

        return self._props

    @props.setter
    def props(self, propslist):
        self._props = propslist

    @property
    def dates(self):
        """Returns a list of valid (found) dates after import.

        Returns None if no dates present

        Example::

            datelist = props.dates
            for date in datelist:
                print ('Date applied is {}'.format(date))

        """
        if not self._dates:
            return None

        return self._dates

    # Copy, and etc aka setters and getters

    def copy(self):
        """Copy a GridProperties instance to a new unique instance.

        Note that the GridProperty instances will also be unique.
        """

        new = GridProperties()
        new._ncol = self._ncol
        new._nrow = self._nrow
        new._nlay = self._nlay

        for prp in self._props:
            newprp = prp.copy()
            new._props.append(newprp)

        for name in self._names:
            new._names.append(name)

        for date in self._dates:
            new._dates.append(date)

        return new

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = XTGDescription()
        dsc.title("Description of GridProperties instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        dsc.txt("Attached grid props objects (names)", self._names)

        if flush:
            dsc.flush()
            return None
        return dsc.astext()

    def get_prop_by_name(self, name):
        """Find and return a property object (GridProperty) by name."""

        for prop in self._props:
            logger.debug("Look for %s, actual is %s", name, prop.name)
            if prop.name == name:
                logger.debug(repr(prop))
                return prop

        raise ValueError("Cannot find property with name <{}>".format(name))

    def append_props(self, proplist):
        """Adds a list of GridProperty objects to the current
        GridProperties instance."""

        for prop in proplist:
            if isinstance(prop, GridProperty):
                # an prop instance can only occur once
                if prop not in self._props:

                    self._props.append(prop)
                    self._names.append(prop.name)
                    self._dates.append(prop.date)
                else:
                    logger.info("Not added. GridProperty instance already present")
            else:
                raise ValueError("Input property is not a valid GridProperty " "object")

    def get_ijk(
        self, names=("IX", "JY", "KZ"), zerobased=False, asmasked=False, mask=None
    ):
        """Returns 3 xtgeo.grid3d.GridProperty objects: I counter,
        J counter, K counter.

        Args:
            names: a 3 x tuple of names per property (default IX, JY, KZ).
            mask: If True, then active cells only.
            zerobased: If True, counter start from 0, otherwise 1 (default=1).
        """

        if mask is not None:
            asmasked = super(GridProperties, self)._evaluate_mask(mask)

        # resuse method from grid
        ixc, jyc, kzc = _grid_etc1.get_ijk(
            self, names=names, zerobased=zerobased, asmasked=asmasked
        )

        # return the objects
        return ixc, jyc, kzc

    def get_actnum(self, name="ACTNUM", asmasked=False, mask=None):
        """Return an ACTNUM GridProperty object.

        Args:
            name (str): name of property in the XTGeo GridProperty object.
            asmasked (bool): ACTNUM is returned with all cells
                as default. Use asmasked=True to make 0 entries masked.
            mask (bool): Deprecated, use asmasked instead.

        Example::

            act = myprops.get_actnum()
            print('{}% cells are active'.format(act.values.mean() * 100))

        Returns:
            A GridProperty instance of ACTNUM, or None if no props present.
        """

        if mask is not None:
            asmasked = super(GridProperties, self)._evaluate_mask(mask)

        # borrow function from GridProperty class:
        if self._props:
            return self._props[0].get_actnum(name=name, mask=asmasked)

        warnings.warn("No gridproperty in list", UserWarning)
        return None

    # Import and export
    # This class can importies several properties in one go, which is efficient
    # for some file types such as Eclipse INIT and UNRST, and Roff

    def from_file(
        self, pfile, fformat="roff", names=None, dates=None, grid=None, namestyle=0
    ):
        """Import grid properties from file in one go.

        This class is particulary useful for Eclipse INIT and RESTART files.

        In case of names='all' then all vectors which have a valid length
        (number of total or active cells in the grid) will be read

        Args:
            pfile (str or Path): Name of file with properties
            fformat (str): roff/init/unrst
            names: list of property names, e.g. ['PORO', 'PERMX'] or 'all'
            dates: list of dates on YYYYMMDD format, for restart files
            grid (obj): The grid geometry object (optional if ROFF)
            namestyle (int): 0 (default) for style SWAT_20110223,
                1 for SWAT--2011_02_23 (applies to restart only)

        Example::
            >>> props = GridProperties()
            >>> props.from_file('ECL.UNRST', fformat='unrst',
                dates=[20110101, 20141212], names=['PORO', 'DZ']

        Raises:
            FileNotFoundError: if input file is not found
            ValueError: if a property is not found
            RuntimeWarning: if some dates are not found

        """

        pfile = xtgeo._XTGeoFile(pfile, mode="rb")

        # work on file extension
        froot, fext = pfile.splitext(lower=True)
        if not fext:
            # file extension is missing, guess from format
            logger.info("File extension missing; guessing...")

            useext = ""
            if fformat == "init":
                useext = ".INIT"
            elif fformat == "unrst":
                useext = ".UNRST"
            elif fformat == "roff":
                useext = ".roff"

            pfile = froot + useext

        logger.info("File name to be used is %s", pfile)

        pfile.check_file(raiseerror=OSError)

        if fformat.lower() == "roff":
            lst = list()
            for name in names:
                lst.append(GridProperty(pfile, fformat="roff", name=name))
            self.append_props(lst)

        elif fformat.lower() in ("init", "unrst"):
            _gridprops_io.import_ecl_output(
                self, pfile, dates=dates, grid=grid, names=names, namestyle=namestyle
            )
        else:
            raise OSError("Invalid file format")

    def to_file(self, pfile, fformat="roff"):
        """Export grid property to file. NB not working!

        Args:
            pfile (str): file name
            fformat (str): file format to be used (roff is the only supported)
            mode (int): 0 for binary ROFF, 1 for ASCII
        """
        raise NotImplementedError("Not implented yet!")

    def get_dataframe(
        self, activeonly=False, ijk=False, xyz=False, doubleformat=False, grid=None
    ):
        """Returns a Pandas dataframe table for the properties

        Args:
            activeonly (bool): If True, return only active cells, NB!
                If True, will require a grid instance (see grid key)
            ijk (bool): If True, show cell indices, IX JY KZ columns
            xyz (bool): If True, show cell center coordinates (needs grid).
            doubleformat (bool): If True, floats are 64 bit, otherwise 32 bit.
                Note that coordinates (if xyz=True) is always 64 bit floats.
            grid (Grid): The grid geometry object. This is required for the
                xyz option.

        Returns:
            Pandas dataframe object

        Examples::

            grd = Grid(gfile1, fformat='egrid')
            xpr = GridProperties()

            names = ['SOIL', 'SWAT', 'PRESSURE']
            dates = [19991201]
            xpr.from_file(rfile1, fformat='unrst', names=names, dates=dates,
                        grid=grd)
            df = x.dataframe(activeonly=False, ijk=True, xyz=True, grid=grd)

        """

        dfr = _gridprops_etc.dataframe(
            self,
            activeonly=activeonly,
            ijk=ijk,
            xyz=xyz,
            doubleformat=doubleformat,
            grid=grid,
        )

        return dfr

    dataframe = get_dataframe  # for compatibility, but deprecated

    # Static methods (scans etc)
    # Don't make a GridProperties instance inside other XTGeo classes
    # as it make cyclic imports. I.e. use only these functions in clients
    # if needed...

    @staticmethod
    def scan_keywords(
        pfile, fformat="xecl", maxkeys=100000, dataframe=False, dates=False
    ):
        """Quick scan of keywords in Eclipse binary restart/init/... file,
        or ROFF binary files.

        For Eclipse files:
        Returns a list of tuples (or dataframe), e.g. ('PRESSURE',
        'REAL', 355299, 3582700), where (keyword, type, no_of_values,
        byteposition_in_file)

        For ROFF files
        Returns a list of tuples (or dataframe), e.g.
        ('translate!xoffset', 'float', 1, 3582700),
        where (keyword, type, no_of_values, byteposition_in_file).

        For Eclipse, the byteposition is to the KEYWORD, while for ROFF
        the byte position is to the beginning of the actual data.

        Args:
            pfile (str): Name or a filehandle to file with properties
            fformat (str): xecl (Eclipse INIT, RESTART, ...) or roff for
                ROFF binary,
            maxkeys (int): Maximum number of keys
            dataframe (bool): If True, return a Pandas dataframe instead
            dates (bool): if True, the date is the last column (only
                menaingful for restart files). Default is False.

        Return:
            A list of tuples or dataframe with keyword info

        Example::
            >>> props = GridProperties()
            >>> dlist = props.scan_keywords('ECL.UNRST')

        """

        pfile = xtgeo._XTGeoFile(pfile)

        dlist = utils.scan_keywords(
            pfile, fformat=fformat, maxkeys=maxkeys, dataframe=dataframe, dates=dates
        )

        return dlist

    @staticmethod
    def scan_dates(pfile, fformat="unrst", maxdates=1000, dataframe=False):
        """Quick scan dates in a simulation restart file.

        Args:
            pfile (str): Name of file or file handle with properties
            fformat (str): unrst (so far)
            maxdates (int): Maximum number of dates to collect
            dataframe (bool): If True, return a Pandas dataframe instead

        Return:
            A list of tuples or a dataframe with (seqno, date),
            date is on YYYYMMDD form.

        Example::
            >>> props = GridProperties()
            >>> dlist = props.scan_dates('ECL.UNRST')

        """
        pfile = xtgeo._XTGeoFile(pfile)

        logger.info("Format supported as default is %s", fformat)

        dlist = utils.scan_dates(pfile, maxdates=maxdates, dataframe=dataframe)

        return dlist
