# -*- coding: utf-8 -*-
"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a columns called 'POLY_ID'
# which identifies each polygon piece.

from __future__ import print_function, absolute_import
import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog
from ._xyz import XYZ
from ._xyz_io import _convert_idbased_xyz
from . import _xyz_oper

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# =============================================================================
# METHODS as wrappers to class init + import


def polygons_from_file(wfile, fformat="xyz"):
    """Make an instance of a Polygons object directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`Polygons.from_file`

    Example::

        import xtgeo
        mypoly = xtgeo.polygons_from_file('somefile.xyz')
    """

    obj = Polygons()

    obj.from_file(wfile, fformat=fformat)

    return obj


def polygons_from_roxar(project, name, category, stype="horizons", realisation=0):
    """This makes an instance of a Polygons directly from roxar input.

    For arguments, see :meth:`Polygons.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mypolys = xtgeo.polygons_from_roxar(project, 'TopEtive', 'DL_polys')

    """

    obj = Polygons()

    obj.from_roxar(project, name, category, stype=stype, realisation=realisation)

    return obj


# =============================================================================
# CLASS
class Polygons(XYZ):  # pylint: disable=too-many-public-methods
    """Class for a polygons (connected points) in the XTGeo framework.

    The term Polygons is hereused in a wider context, as it includes
    polylines that do not connect into closed polygons. A Polygons
    instance may contain several polylines/polygons, which are
    identified by POLY_ID

    The polygons are stored in Python as a Pandas dataframe, which
    allow for flexible manipulation and fast execution.

    A Polygons instance will have 4 mandatory columns; here by default names:

    * X_UTME - for X UTM coordinate (Easting)
    * Y_UTMN - For Y UTM coordinate (Northing)
    * Z_TVDSS - For depth or property from mean SeaLevel; Depth positive down
    * POLY_ID - for polygon ID as there may be several polylines segments

    Each Polygons instance can also a name (through the name attribute).
    Default is 'poly'. E.g. if a well fence, it is logical to name the
    instance to be the same as the well name.

    """

    def __init__(self, *args, **kwargs):
        self._df = None
        self._xname = "X_UTME"
        self._yname = "Y_UTMN"
        self._zname = "Z_TVDSS"
        self._pname = "POLY_ID"
        self._mname = "M_MDEPTH"
        self._name = "poly"  # the name of the Polygons() instance
        super(Polygons, self).__init__(*args, **kwargs)

        self._ispolygons = True

    def __str__(self):
        # user friendly print
        return self.describe(flush=False)

    def __del__(self):
        logger.info("Polygons object %s deleted", id(self))

    # ----------------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------------

    @property
    def nrow(self):
        """ Returns the Pandas dataframe object number of rows"""
        if self._df is None:
            return 0

        return len(self._df.index)

    @property
    def name(self):
        """ Returns or sets the name of the instance."""
        return self._name

    @name.setter
    def name(self, newname):
        self._name = newname

    @property
    def xname(self):
        """ Returns the name of the X column"""
        return self._xname

    @property
    def yname(self):
        """ Returns the name of the Y column"""
        return self._yname

    @property
    def zname(self):
        """ Returns or set the name of the Z/VALUE column"""
        return self._zname

    @zname.setter
    def zname(self, zname):
        if isinstance(zname, str):
            self._zname = zname
        else:
            raise ValueError("Wrong type of input to zname; must be string")

    @property
    def pname(self):
        """ Returns the name of the POLY_ID column"""
        return self._pname

    @property
    def dataframe(self):
        """ Returns or set the Pandas dataframe object"""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    # ----------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------

    def describe(self, flush=True):
        super(Polygons, self).describe(flush=flush)

    def from_file(self, pfile, fformat="xyz"):
        """Import Polygons from a file.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'guess': Try to choose file format based on extension

        Args:
            pfile (str): Name of file
            fformat (str): File format, see list above

        Returns:
            Object instance (needed optionally)

        Raises:
            OSError: if file is not present or wrong permissions.

        """

        super(Polygons, self).from_file(pfile, fformat=fformat)

        # for polygons, a seperate column with POLY_ID is required;
        # however this may lack if the input is on XYZ format

        if self._pname not in self._df.columns:
            pxn = self._pname
            self._df[pxn] = self._df.isnull().all(axis=1).cumsum().dropna()
            self._df.dropna(axis=0, inplace=True)
            self._df.reset_index(inplace=True, drop=True)

    def from_roxar(
        self, project, name, category, stype="horizons", realisation=0, attributes=False
    ):
        """Load a polygons item from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that a shortform to::

          import xtgeo
          mypoly = xtgeo.xyz.Polygons()
          mypoly.from_roxar(project, 'TopAare', 'DepthPolys')

        is::

          import xtgeo
          mypoly = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        If the project is a folder, then the project will be closed
        automatically after each job. For big projects (long load time),
        this is inconvinient, and instead, one  can preload the project data as
        this::

          import xtgeo
          projpath = '/some/path/to/rms_project'
          myp = xtgeo.RoxUtils(projpath)
          poly1 = xtgeo.polygons_from_roxar(myp.project, 'Top1', 'DepthPoly')
          poly2 = xtgeo.polygons_from_roxar(myp.project, 'Top2', 'DepthPoly')
          poly3 = xtgeo.polygons_from_roxar(myp.project, 'Top3', 'DepthPoly')
          poly4 = xtgeo.polygons_from_roxar(myp.project, 'Top4', 'DepthPoly')
          # ...
          myp.safe_close()

        Args:
            project (str or special): Name of project (as folder) if
               outside RMS, or use a special RoxApi(path).project variable
               (see example above). If the current project inside RMS,
               just use the magic ``project`` word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones/faults: for example 'DL_depth',
                for clipboard use folders such as 'somefolder/somesubfolder' or
                just empty '' if no folder. Alternatively, use '|' instead
                of '/'.
            stype (str): RMS folder type, 'horizons' (default) or 'zones',
                'faults', 'clipboard'.
            realisation (int): Realisation number, default is 0
            attributes (bool): Not in use

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        """
        logger.info("Skip attributes: %s", attributes)

        super(Polygons, self).from_roxar(
            project, name, category, stype=stype, realisation=realisation
        )

    def to_roxar(
        self, project, name, category, stype="horizons", realisation=0, attributes=False
    ):
        """Export/save/store a polygons item to a Roxar RMS project.

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones only: for example 'DL_depth'
            stype (str): RMS folder type, 'horizons' (default) or 'zones'
            realisation (int): Realisation number, default is 0

        Raises:
            ValueError: Various types of invalid inputs.

        """

        super(Polygons, self).to_roxar(
            project, name, category, stype=stype, realisation=realisation
        )

    def to_file(
        self,
        pfile,
        fformat="xyz",
        attributes=None,
        pfilter=None,
        wcolumn=None,
        hcolumn=None,
        mdcolumn=None,
    ):
        """Export Polygons to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
            attributes (list): List of extra columns to export (some formats)
            pfilter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            wcolumn (str): Name of well column (rms_wellpicks format only)
            hcolumn (str): Name of horizons column (rms_wellpicks format only)
            mdcolumn (str): Name of MD column (rms_wellpicks format only)

        Returns:
            Number of points exported

        Note that the rms_wellpicks will try to output to:

        * HorizonName, WellName, MD  if a MD (mdcolumn) is present,
        * HorizonName, WellName, X, Y, Z  otherwise

        Raises:
            KeyError if pfilter is set and key(s) are invalid

        """

        super(Polygons, self).to_file(
            pfile,
            fformat=fformat,
            attributes=attributes,
            pfilter=pfilter,
            wcolumn=wcolumn,
            hcolumn=hcolumn,
            mdcolumn=mdcolumn,
        )

    def from_wells(self, wells, zone, resample=1):

        """Get line segments from a list of wells and a zone number

        Args:
            wells (list): List of XTGeo well objects
            zone (int): Which zone to apply
            resample (int): If given, resample every N'th sample to make
                polylines smaller in terms of bit and bytes.
                1 = No resampling.

        Returns:
            None if well list is empty; otherwise the number of wells that
            have one or more line segments to return

        Raises:
            Todo
        """

        if not wells:
            return None

        dflist = []
        maxid = 0
        for well in wells:
            wp = well.get_zone_interval(zone, resample=resample)
            if wp is not None:
                # as well segments may have overlapping POLY_ID:
                wp[self._pname] += maxid
                maxid = wp[self._pname].max() + 1
                dflist.append(wp)

        if dflist:
            self._df = pd.concat(dflist, ignore_index=True)
            self._df.reset_index(inplace=True, drop=True)
        else:
            return None

        return len(dflist)

    def get_xyz_dataframe(self):
        """Convert from POLY_ID based to XYZ, where a new polygon is marked
        with a 999.0 value as flag"""

        logger.info(self.dataframe)

        return _convert_idbased_xyz(self, self.dataframe)

    def get_boundary(self):
        """Get the XYZ window (boundaries) of the instance.

        Returns:
            (xmin, xmax, ymin, ymax, zmin, zmax)
        """

        xmin = np.nanmin(self.dataframe[self.xname].values)
        xmax = np.nanmax(self.dataframe[self.xname].values)
        ymin = np.nanmin(self.dataframe[self.yname].values)
        ymax = np.nanmax(self.dataframe[self.yname].values)
        zmin = np.nanmin(self.dataframe[self.zname].values)
        zmax = np.nanmax(self.dataframe[self.zname].values)

        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def rescale(self, distance):
        """Rescale (resample) by using a new horizontal increment.

        As usual, the instance is updated in-place.

        If the distance is larger than the total input poly-line length,
        nothing is done. Note that the result distance may differ from then
        requested distance to rounding to fit original length.

        Args:
             distance (float): New distance between points
        """

        _xyz_oper.rescale_polygons(self, distance=distance)

    # =========================================================================
    # Operations restricted to inside/outside polygons
    # =========================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True):
        """A generic function for doing points operations restricted to inside
        or outside polygon(s).

        The points are parts of polygons.

        Args:
            poly (Polygons): A XTGeo Polygons instance
            value(float): Value to add, subtract etc
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside polygons; else outside.
        """

        _xyz_oper.operation_polygons(self, poly, value, opname=opname, inside=inside)

    # shortforms
    def add_inside(self, poly, value):
        """Add a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=True)

    def add_outside(self, poly, value):
        """Add a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=False)

    def sub_inside(self, poly, value):
        """Subtract a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=True)

    def sub_outside(self, poly, value):
        """Subtract a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=False)

    def mul_inside(self, poly, value):
        """Multiply a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=True)

    def mul_outside(self, poly, value):
        """Multiply a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=False)

    def div_inside(self, poly, value):
        """Divide a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="div", inside=True)

    def div_outside(self, poly, value):
        """Divide a value (scalar) outside polygons (value 0.0 will give
        result 0)"""
        self.operation_polygons(poly, value, opname="div", inside=False)

    def set_inside(self, poly, value):
        """Set a value (scalar) inside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=True)

    def set_outside(self, poly, value):
        """Set a value (scalar) outside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=False)

    def eli_inside(self, poly):
        """Eliminate current map values inside polygons"""
        self.operation_polygons(poly, 0, opname="eli", inside=True)

    def eli_outside(self, poly):
        """Eliminate current map values outside polygons"""
        self.operation_polygons(poly, 0, opname="eli", inside=False)
