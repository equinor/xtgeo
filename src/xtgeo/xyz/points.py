# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class"""

from __future__ import print_function, absolute_import

from collections import OrderedDict

import numpy as np
import numpy.ma as ma
import pandas as pd

import xtgeo

# from xtgeo.common import XTGeoDialog
# from xtgeo.surface import RegularSurface
from ._xyz import XYZ
from . import _xyz_oper

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


# =============================================================================
# METHODS as wrappers to class init + import


def points_from_file(wfile, fformat="xyz"):
    """Make an instance of a Points object directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`Points.from_file`

    Example::

        import xtgeo
        mypoints = xtgeo.points_from_file('somefile.xyz')
    """

    obj = Points()

    obj.from_file(wfile, fformat=fformat)

    return obj


def points_from_roxar(
    project, name, category, stype="horizons", realisation=0, attributes=False
):
    """This makes an instance of a Points directly from roxar input.

    For arguments, see :meth:`Points.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mypoints = xtgeo.points_from_roxar(project, 'TopEtive', 'DP_seismic')

    """

    obj = Points()

    obj.from_roxar(
        project,
        name,
        category,
        stype=stype,
        realisation=realisation,
        attributes=attributes,
    )

    return obj


class Points(XYZ):  # pylint: disable=too-many-public-methods
    """Points: Class for a points set in the XTGeo framework.

    The Points class is a subclass of the :class:`.XYZ` class,
    and the point set itself is a `pandas <http://pandas.pydata.org>`_
    dataframe object.

    The instance can be made either from file or by a spesification,
    e.g. from file::

        xp = Points().from_file('somefilename', fformat='xyz')
        # or perhaps better
        xp = xtgeo.points_from_file('somefilename', fformat='xyz')
        # show the Pandas dataframe
        print(xp.dataframe)

    You can also make points from list of tuples in Python::

        plist = [(234, 556, 12), (235, 559, 14), (255, 577, 12)]
        mypoints = Points(plist)

    Default column names in the dataframe:

    * X_UTME: UTM X coordinate  as self._xname
    * Y_UTMN: UTM Y coordinate  as self._yname
    * Z_TVDSS: Z coordinate, often depth below TVD SS, but may also be
      something else! Use zname attribute
    * M_MDEPTH: measured depth, (if present)
    * Q_*: Quasi geometrical measures, such as Q_MDEPTH, Q_AZI, Q_INCL

    """

    def __init__(self, *args, **kwargs):

        # instance variables listed
        self._df = None
        self._ispolygons = False
        self._xname = "X_UTME"
        self._yname = "Y_UTMN"
        self._zname = "Z_TVDSS"
        self._pname = "POLY_ID"
        self._mname = "M_MDEPTH"

        super(Points, self).__init__(*args, **kwargs)

        if len(args) == 1:
            if isinstance(args[0], xtgeo.surface.RegularSurface):
                self.from_surface(args[0])
        logger.info("Initiated Points")

    def __repr__(self):
        # should be able to newobject = eval(repr(thisobject))
        myrp = "{0.__class__.__name__} (filesrc={0._filesrc!r}, " "ID={1})".format(
            self, id(self)
        )
        return myrp

    def __str__(self):
        # user friendly print
        return self.describe(flush=False)

    def __del__(self):
        logger.info("Points object %s deleted", id(self))

    def __eq__(self, value):
        return self.dataframe[self.zname] == value

    def __gt__(self, value):
        return self.dataframe[self.zname] > value

    def __ge__(self, value):
        return self.dataframe[self.zname] >= value

    def __lt__(self, value):
        return self.dataframe[self.zname] < value

    def __le__(self, value):
        return self.dataframe[self.zname] <= value

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
    def dataframe(self):
        """ Returns or set the Pandas dataframe object"""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    @property
    def xname(self):
        return super(Points, self).xname

    @xname.setter
    def xname(self, newname):
        super(Points, self)._name_setter(newname)

    @property
    def yname(self):
        return super(Points, self).yname

    @yname.setter
    def yname(self, newname):
        super(Points, self)._name_setter(newname)

    @property
    def zname(self):
        return super(Points, self).zname

    @zname.setter
    def zname(self, newname):
        super(Points, self)._name_setter(newname)

    @property
    def pname(self):
        """ Return or set the name of the POLY_ID column"""
        return self._pname

    @pname.setter
    def pname(self, pname):
        if isinstance(pname, str):
            self._pname = pname
        else:
            raise ValueError("The pname property must be a string")

    # ----------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------

    def _random(self, nrandom=10):
        """Generate nrandom random points within the range 0..1

        Args:
            nrandom (int): Number of random points (default 10)

        .. versionadded:: 2.3.0
        """

        # currently a non-published method

        self._df = pd.DataFrame(
            np.random.rand(nrandom, 3), columns=[self._xname, self._yname, self._zname]
        )

    def copy(self, stype="points"):
        """Deep copy of a Points instance.

        Example::

            myp = xtgeo.points_from_file("somepoints.poi")
            mypcopy = myp.copy()

        .. versionadded:: 2.1.0
        """

        stype = "points"
        return super(Points, self).copy(stype)

    def describe(self, flush=True):
        """Describe a Points instance"""
        return super(Points, self).describe(flush=flush)

    def from_file(self, pfile, fformat="xyz"):
        """Import points.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'rms_attr' or 'rmsattr': RMS points formats with attributes (extra columns)

        * 'guess': Try to choose file format based on extension

        Args:
            pfile (str): Name of file
            fformat (str): File format, see list above

        Returns:
            Object instance (needed optionally)

        Raises:
            OSError: if file is not present or wrong permissions.


        """
        super(Points, self).from_file(pfile, fformat=fformat)

        self._df.dropna(inplace=True)

    def from_list(self, plist):
        """Import points from list.

        [(x1, y1, z1, <id1>), (x2, y2, z2, <id2>), ...]

        It is currently not much error checking that lists/tuples are consistent, e.g.
        if there always is either 3 or 4 elements per tuple, or that 4 number is
        an integer.

        Args:
            plist (str): List of tuples, each tuple is length 3 or 4

        Raises:
            ValueError: If something is wrong with input

        .. versionadded: 2.6

        """
        super(Points, self).from_list(plist)
        self._df.dropna(inplace=True)

    def to_file(
        self,
        pfile,
        fformat="xyz",
        attributes=True,
        pfilter=None,
        filter=None,  # deprecated use (for backwards compatibility)
        wcolumn=None,
        hcolumn=None,
        mdcolumn="M_MDEPTH",
    ):  # pylint: disable=redefined-builtin
        """Export XYZ (Points/Polygons) to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
            attributes (bool or list): List of extra columns to export (some formats)
            pfilter (dict): (or filter) Filter on e.g. top name(s) with
                keys TopName
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
        if filter is not None and pfilter is None:
            xtg.warndeprecated(
                "Keyword 'filter' is deprecated, use 'pfilter' instead (will continue)"
            )
            pfilter = filter
            filter = None

        elif filter is not None and pfilter is not None:
            xtg.warndeprecated(
                "Keyword 'filter' is deprecated, using 'pfilter' instead"
            )
            filter = None

        return super(Points, self).to_file(
            pfile,
            fformat=fformat,
            attributes=attributes,
            pfilter=pfilter,
            wcolumn=wcolumn,
            hcolumn=hcolumn,
            mdcolumn=mdcolumn,
            filter=None,
        )

    def from_roxar(
        self, project, name, category, stype="horizons", realisation=0, attributes=False
    ):
        """Load a points item from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that a shortform to::

          import xtgeo
          mypoly = xtgeo.xyz.Points()
          mypoints.from_roxar(project, 'TopAare', 'DepthPoints')

        is::

          import xtgeo
          mypoints = xtgeo.points_from_roxar(project, 'TopAare', 'DepthPoints')

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of Points item
            category (str): For horizons/zones only: for example 'DL_depth'
            stype (str): RMS 'super type' i.e. 'horizons' (default), 'zones', ...
            realisation (int): Realisation number, default is 0
            attributes (bool or list): If attributes should be included (requires
                a workaround in the library code)

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        """
        super(Points, self).from_roxar(
            project,
            name,
            category,
            stype=stype,
            realisation=realisation,
            attributes=attributes,
        )

    def to_roxar(
        self,
        project,
        name,
        category,
        stype="horizons",
        pfilter=None,
        realisation=0,
        attributes=False,
    ):
        """Export/save/store a points item to a Roxar RMS project.

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        For stype=clipboard, category will be created automatically if not existing.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of horizons/zone item or well picks set
            category (str): For horizons/zones only: for example 'DL_depth', for
                clipboard, see examples.
            stype (str): RMS "super type": 'horizons', 'zones', 'clipboard'
            pfilter (dict): Filter on e.g. top name(s) with
                keys TopName or ZoneName as e.g. {'TopName': ['Top1', 'Top2']}
            realisation (int): Realisation number, default is 0 (partly implemented)
            attributes (bool or list): If attributes should be included (requires
                a workaround in the library code)

        Examples::

            # store in Horizons:
            wpoints.to_roxar(project, "MyTop", "MyCategory", stype="horizons")

        For Clipboard, the category can have a nested structure. The following forms are
        all legal::

            p.to_roxar(project, "thename", None, stype="clipboard")  # no subfolder
            p.to_roxar(project, "thename", "", stype="clipboard")  # no subfolder
            p.to_roxar(project, "thename", "dir1", stype="clipboard")  # one level
            # two levels examples, all forms possible:
            p.to_roxar(project, "thename", "dir1/dir2", stype="clipboard")
            p.to_roxar(project, "thename", "dir1|dir2", stype="clipboard")
            p.to_roxar(project, "thename", ["dir1", "dir2"], stype="clipboard")


        Raises:
            ValueError: Various types of invalid inputs.

        .. versionchanged:: 2.1.0 added pfilter option
        """

        super(Points, self).to_roxar(
            project,
            name,
            category,
            stype=stype,
            pfilter=pfilter,
            realisation=realisation,
            attributes=attributes,
        )

    def from_wells(
        self,
        wells,
        tops=True,
        incl_limit=None,
        top_prefix="Top",
        zonelist=None,
        use_undef=False,
    ):

        """Get tops or zone points data from a list of wells.

        Args:
            wells (list): List of XTGeo well objects
            tops (bool): Get the tops if True (default), otherwise zone
            incl_limit (float): Inclination limit for zones (thickness points)
            top_prefix (str): Prefix used for Tops
            zonelist (list-like): Which zone numbers to apply.
            use_undef (bool): If True, then transition from UNDEF is also
                used.

        Returns:
            None if well list is empty; otherwise the number of wells.

        Raises:
            Todo
        """

        if not wells:
            return None

        dflist = []
        for well in wells:
            wp = well.get_zonation_points(
                tops=tops,
                incl_limit=incl_limit,
                top_prefix=top_prefix,
                zonelist=zonelist,
                use_undef=use_undef,
            )

            if wp is not None:
                dflist.append(wp)

        if dflist:
            self._df = pd.concat(dflist, ignore_index=True)
        else:
            return None

        for col in self._df.columns:
            if col == "Zone":
                self._attrs[col] = "int"
            elif col == "ZoneName":
                self._attrs[col] = "str"
            elif col == "WellName":
                self._attrs[col] = "str"
            else:
                self._attrs[col] = "float"

        return len(dflist)

    def dfrac_from_wells(
        self,
        wells,
        dlogname,
        dcodes,
        incl_limit=90,
        count_limit=3,
        zonelist=None,
        zonelogname=None,
    ):

        """Get fraction of discrete code(s) (e.g. facies) per zone.

        Args:
            wells (list): List of XTGeo well objects
            dlogname (str): Name of discrete log (e.g. Facies)
            dcodes (list of int): Code(s) to get fraction for, e.g. [3]
            incl_limit (float): Inclination limit for zones (thickness points)
            count_limit (int): Min. no of counts per segment for valid result
            zonelist (list of int): Whihc zones to compute for (default None
                means that all zones will be evaluated)
            zonelogname (str): Name of zonelog; if None than the
                well.zonelogname property will be applied.

        Returns:
            None if well list is empty; otherwise the number of wells.

        Raises:
            Todo
        """

        if not wells:
            return None

        dflist = []  # will be a list of pandas dataframes
        for well in wells:
            wpf = well.get_fraction_per_zone(
                dlogname,
                dcodes,
                zonelist=zonelist,
                incl_limit=incl_limit,
                count_limit=count_limit,
                zonelogname=zonelogname,
            )

            if wpf is not None:
                dflist.append(wpf)

        if dflist:
            self._df = pd.concat(dflist, ignore_index=True)
            self.zname = "DFRAC"  # name of third column
        else:
            return None

        for col in self._df.columns[3:]:
            if col == "Zone":
                self._attrs[col] = "int"
            elif col == "ZoneName":
                self._attrs[col] = "str"
            elif col == "WellName":
                self._attrs[col] = "str"
            else:
                self._attrs[col] = "float"

        return len(dflist)

    def from_surface(self, surf, zname="Z_TVDSS"):
        """Get points as X Y Value from a surface object nodes.

        Note that undefined surface nodes will not be included.

        Args:
            surf (RegularSurface): A XTGeo RegularSurface object instance.
            zname (str): Name of value column (3'rd column)

        Example::

            topx = RegularSurface('topx.gri')
            topx_aspoints = Points()
            topx_aspoints.from_surface(topx)

            # alternative shortform:
            topx_aspoints = Points(topx)  # get an instance directly

            topx_aspoints.to_file('mypoints.poi')  # export as XYZ file
        """

        # check if surf is instance from RegularSurface
        if not isinstance(surf, xtgeo.surface.RegularSurface):
            raise ValueError("Given surf is not a RegularSurface object")

        val = surf.values
        xc, yc = surf.get_xy_values()

        coor = []
        for vv in [xc, yc, val]:
            vv = ma.filled(vv.flatten(order="C"), fill_value=np.nan)
            vv = vv[~np.isnan(vv)]
            coor.append(vv)

        # now populate the dataframe:
        xc, yc, val = coor  # pylint: disable=unbalanced-tuple-unpacking
        ddatas = OrderedDict()
        ddatas[self._xname] = xc
        ddatas[self._yname] = yc
        ddatas[self._zname] = val
        self._df = pd.DataFrame(ddatas)
        self.zname = zname

    # ==================================================================================
    # Operations vs surfaces and possibly other
    # ==================================================================================

    def snap_surface(self, surf, activeonly=True):
        """Snap (transfer) the points Z values to a RegularSurface

        Args:
            surf (~xtgeo.surface.regular_surface.RegularSurface): Surface to snap to.
            activeonly (bool): If True (default), the points outside the defined surface
                will be removed. If False, these points will keep the original values.

        Returns:
            None (instance is updated inplace)

        Raises:
            ValueError: Input object of wrong data type, must be RegularSurface
            RuntimeError: Error code from C routine surf_get_zv_from_xyv is ...

        .. versionadded:: 2.1.0

        """
        _xyz_oper.snap_surface(self, surf, activeonly=activeonly)

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # ==================================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True, where=True):
        """A generic function for doing points operations restricted to inside
        or outside polygon(s).

        Args:
            poly (Points): A XTGeo Points instance
            value(float or str): Value to add, subtract etc. If 'poly'
                then use the avg Z value from each polygon.
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside Points; else outside.
            where (bool): A logical filter (current not implemented)

        Examples::

            # assume a point set where you want eliminate number inside
            # Points, given that the points Z value inside this polygon is
            # larger than 1700:
            poi = Points(POINTSET2)
            pol = Points(POLSET2)

            poi.operation_polygons(pol, 0, opname='eli', inside=True)



        """

        _xyz_oper.operation_polygons(
            self, poly, value, opname=opname, inside=inside, where=where
        )

    # shortforms
    def add_inside(self, poly, value, where=True):
        """Add a value (scalar) inside Points (see `operation_polygons`)"""
        self.operation_polygons(poly, value, opname="add", inside=True, where=where)

    def add_outside(self, poly, value, where=True):
        """Add a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="add", inside=False, where=where)

    def sub_inside(self, poly, value, where=True):
        """Subtract a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="sub", inside=True, where=where)

    def sub_outside(self, poly, value, where=True):
        """Subtract a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="sub", inside=False, where=where)

    def mul_inside(self, poly, value, where=True):
        """Multiply a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="mul", inside=True, where=where)

    def mul_outside(self, poly, value, where=True):
        """Multiply a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="mul", inside=False, where=where)

    def div_inside(self, poly, value, where=True):
        """Divide a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="div", inside=True, where=where)

    def div_outside(self, poly, value, where=True):
        """Divide a value (scalar) outside Points (value 0.0 will give
        result 0)"""
        self.operation_polygons(poly, value, opname="div", inside=False, where=where)

    def set_inside(self, poly, value, where=True):
        """Set a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="set", inside=True, where=where)

    def set_outside(self, poly, value, where=True):
        """Set a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="set", inside=False, where=where)

    def eli_inside(self, poly, where=True):
        """Eliminate current map values inside Points"""
        self.operation_polygons(poly, 0, opname="eli", inside=True, where=where)

    def eli_outside(self, poly, where=True):
        """Eliminate current map values outside Points"""
        self.operation_polygons(poly, 0, opname="eli", inside=False, where=where)
