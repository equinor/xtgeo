# -*- coding: utf-8 -*-
"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a columns called 'POLY_ID'
# which identifies each polygon piece.

from __future__ import print_function, absolute_import
import numpy as np
import pandas as pd
import shapely.geometry as sg

import xtgeo
from ._xyz import XYZ
from ._xyz_io import _convert_idbased_xyz
from . import _xyz_oper

xtg = xtgeo.common.XTGeoDialog()
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


def polygons_from_roxar(
    project, name, category, stype="horizons", realisation=0, attributes=False
):
    """This makes an instance of a Polygons directly from roxar input.

    For arguments, see :meth:`Polygons.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mypolys = xtgeo.polygons_from_roxar(project, 'TopEtive', 'DL_polys')
    """

    obj = Polygons()

    obj.from_roxar(
        project,
        name,
        category,
        stype=stype,
        realisation=realisation,
        attributes=attributes,
    )

    return obj


# =============================================================================
# CLASS
class Polygons(XYZ):  # pylint: disable=too-many-public-methods
    """Class for a polygons object (connected points) in the XTGeo framework.

    The term Polygons is here used in a wider context, as it includes
    polylines that do not connect into closed polygons. A Polygons
    instance may contain several pieces of polylines/polygons, which are
    identified by POLY_ID.

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
        self._hname = "H_CUMLEN"
        self._dhname = "H_DELTALEN"
        self._tname = "T_CUMLEN"
        self._dtname = "T_DELTALEN"
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
        """ Returns the Pandas dataframe object number of rows (read only)."""
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
        return super(Polygons, self).xname

    @xname.setter
    def xname(self, newname):
        super(Polygons, self)._name_setter(newname)

    @property
    def yname(self):
        return super(Polygons, self).yname

    @yname.setter
    def yname(self, newname):
        super(Polygons, self)._name_setter(newname)

    @property
    def zname(self):
        return super(Polygons, self).zname

    @zname.setter
    def zname(self, newname):
        super(Polygons, self)._name_setter(newname)

    @property
    def pname(self):
        """ Returns or set the name of the POLY_ID column"""
        return self._pname

    @pname.setter
    def pname(self, newname):
        if isinstance(newname, str):
            self._pname = newname
        else:
            raise ValueError("Wrong type of input to pname; must be string")

    @property
    def hname(self):
        """ Returns or set the name of the cumulative horizontal length column,
        if it exists.

        .. versionadded:: 2.1.0
        """
        if isinstance(self._hname, str):
            if self._hname not in self._df.columns:
                self._hname = None

        return self._hname

    @hname.setter
    def hname(self, myhname):
        if isinstance(myhname, str):
            self._hname = myhname
        else:
            raise ValueError("Wrong type of input to hname; must be string")

    @property
    def dhname(self):
        """ Returns or set the name of the delta horizontal length column if
        it exists.

        .. versionadded:: 2.1.0
        """
        if isinstance(self._dhname, str):
            if self._dhname not in self._df.columns:
                self._dhname = None

        return self._dhname

    @dhname.setter
    def dhname(self, mydhname):
        if isinstance(mydhname, str):
            self._dhname = mydhname
        else:
            raise ValueError("Wrong type of input to dhname; must be string")

    @property
    def tname(self):
        """ Returns or set the name of the cumulative total length column,
        if it exists.

        .. versionadded:: 2.1.0
        """
        if isinstance(self._tname, str):
            if self._tname not in self._df.columns:
                self._tname = None

        return self._tname

    @tname.setter
    def tname(self, mytname):
        if isinstance(mytname, str):
            self._tname = mytname
        else:
            raise ValueError("Wrong type of input to tname; must be string")

    @property
    def dtname(self):
        """ Returns or set the name of the delta total length column if
        it exists.

        .. versionadded:: 2.1.0
        """
        if isinstance(self._dtname, str):
            if self._dtname not in self._df.columns:
                self._dtname = None

        return self._dtname

    @dtname.setter
    def dtname(self, mydtname):
        if isinstance(mydtname, str):
            self._dtname = mydtname
        else:
            raise ValueError("Wrong type of input to dhname; must be string")

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

    def copy(self, stype="polygons"):
        """Deep copy of a Polygons instance.

        .. versionadded:: 2.1.0
        """
        return super(Polygons, self).copy(stype)

    def describe(self, flush=True):
        """Describe a Polygons instance"""
        return super(Polygons, self).describe(flush=flush)

    def delete_columns(self, clist, strict=False):
        """Delete one or more columns by name in a safe way.

        Note that the coordinate columns will be protected, as well as then
        POLY_ID column (pname atribute).

        Args:
            clist (list): Name of columns
            strict (bool): I False, will not trigger exception if a column is not
                found. Otherways a ValueError will be raised.

        Raises:
            ValueError: If strict is True and columnname not present

        Example::

            mypoly.delete_columns(["WELL_ID", self.hname, self.dhname])

        .. versionadded:: 2.1.0

        """

        for cname in clist:
            if cname in (self.xname, self.yname, self.zname, self.pname):
                xtg.warnuser(
                    "The column {} is protected and will not be deleted".format(cname)
                )
                continue

            if cname not in self._df:
                if strict is True:
                    raise ValueError("The column {} is not present".format(cname))

            if cname in self._df:
                self._df.drop(cname, axis=1, inplace=True)

    def from_file(self, pfile, fformat="xyz"):
        """Import Polygons from a file.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ or RMS text format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'rms_attr': RMS points formats with attributes (extra columns)

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

    def from_list(self, plist):
        """Import points from list.

        [(x1, y1, z1, id1), (x2, y2, z2, id2), ...]

        It is currently not much error checking that lists/tuples are consistent, e.g.
        if there always is either 3 or 4 elements per tuple, or that 4 number is
        an integer.

        Args:
            plist (str): List of tuples, each tuple is length 3 or 4

        Raises:
            ValueError: If something is wrong with input

        .. versionadded: 2.6

        """
        super(Polygons, self).from_list(plist)
        self._df.dropna(inplace=True)

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
            attributes (bool): Currently inactive option

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        """

        super(Polygons, self).from_roxar(
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
        """Export/save/store a polygons item to a Roxar RMS project.

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised. This is not the case for
        stype="clipboard" where "folders" will be created on demand.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones only: for example 'DL_depth'
            pfilter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            stype (str): RMS folder type, 'horizons' (default) or 'zones'
            realisation (int): Realisation number, default is 0
            attributes (bool): Also use attributes, if True

        Raises:
            ValueError: Various types of invalid inputs.

        .. seealso::
           The equivalent ``Points`` method :meth:`~xtgeo.xyz.points.Points.to_roxar()`
           for examples.

        """

        super(Polygons, self).to_roxar(
            project,
            name,
            category,
            stype=stype,
            pfilter=pfilter,
            realisation=realisation,
            attributes=attributes,
        )

    def to_file(
        self,
        pfile,
        fformat="xyz",
        attributes=False,
        pfilter=None,
        filter=None,  # deprecated
        wcolumn=None,
        hcolumn=None,
        mdcolumn=None,
    ):  # pylint: disable=redefined-builtin
        """Export Polygons to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
            attributes (bool): Not is use for polygons
            pfilter (dict): Filter on e.g. top name(s) with keys
                 TopName or ZoneName as {'TopName': ['Top1', 'Top2']}
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

        elif filter is not None and pfilter is not None:
            xtg.warndeprecated(
                "Keyword 'filter' is deprecated, using 'pfilter' instead"
            )

        super(Polygons, self).to_file(
            pfile,
            fformat=fformat,
            attributes=attributes,
            pfilter=pfilter,
            filter=None,
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

        logger.debug("Data frame: \n%s", self.dataframe)

        return _convert_idbased_xyz(self, self.dataframe)

    def get_shapely_objects(self):
        """Returns a list of Shapely LineString objects, one per POLY_ID.

        .. versionadded:: 2.1.0

        """
        spolys = []
        idgroups = self.dataframe.groupby(self.pname)

        for _idx, grp in idgroups:
            pxcor = grp[self.xname].values
            pycor = grp[self.yname].values
            pzcor = grp[self.zname].values
            spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))
            spolys.append(spoly)

        return spolys

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

    def filter_byid(self, polyid=None):
        """Remove all line segments not in polyid.

        The instance is updated in-place.

        Args:
            polyid (int or list of int): Which ID(s) to keep, None means use first.

        Example::

            mypoly.filter_byid(polyid=[2, 4])  # keep POLY_ID 2 and 4

        .. versionadded:: 2.1.0
        """
        if polyid is None:
            polyid = int(self.dataframe[self.pname].iloc[0])

        if not isinstance(polyid, list):
            polyid = [polyid]

        dflist = []
        for pid in polyid:
            dflist.append(self.dataframe[self.dataframe[self.pname] == pid])

        self.dataframe = pd.concat(dflist)

    def tlen(self, tname="T_CUMLEN", dtname="T_DELTALEN", atindex=0):
        """Compute and add or replace columns for cum. total 3D length and delta length.

        The instance is updated in-place.

        Args:
            tname (str): Name of cumulative total length. Default is T_CUMLEN.
            dtname (str): Name of delta length column. Default is T_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1.0
        """

        _xyz_oper.tlen(self, tname=tname, dtname=dtname, atindex=atindex)

    def hlen(self, hname="H_CUMLEN", dhname="H_DELTALEN", atindex=0):
        """Compute and add or replace columns for cum. horizontal length
        and delta length.

        The instance is updated in-place.

        Args:
            hname (str): Name of cumulative horizontal length. Default is H_CUMLEN.
            dhname (str): Name of delta length column. Default is H_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1.0
        """

        _xyz_oper.hlen(self, hname=hname, dhname=dhname, atindex=atindex)

    def extend(self, distance, nsamples=1, mode2d=True):
        """Extend polyline by `distance` at both ends, nsmaples times.

        The instance is updated in-place.

        Args:
            distance (float): The horizontal distance (sampling) to extend
            nsamples (int): Number of samples to extend.
            mode2d (bool): XY extension (only True is supported)

        .. versionadded:: 2.1.0
        """

        _xyz_oper.extend(self, distance, nsamples, mode2d)

    def rescale(self, distance, addlen=False, kind="simple", mode2d=True):
        """Rescale (resample) by using a new increment.

        The increment (distance) may be a horizontal or a True 3D
        distance dependent on mode2d.

        The instance is updated in-place.

        If the distance is larger than the total input poly-line length,
        nothing is done. Note that the result distance may differ from then
        requested distance to rounding to fit original length.

        Args:
            distance (float): New distance between points
            addhlen (str): If True, total and horizontal cum. and delta length
                columns will be added.
            kind (str): What kind of rescaling: slinear/cubic/simple
            mode2d (bool): The distance may be a 2D (XY) ora 3D (XYZ) mode.

        .. versionchanged:: 2.1.0 a new algorithm

        """

        _xyz_oper.rescale_polygons(
            self, distance=distance, addlen=addlen, kind=kind, mode2d=mode2d
        )

    def get_fence(
        self, distance=20, atleast=5, nextend=2, name=None, asnumpy=True, polyid=None
    ):
        """Extracts a fence with constant horizontal sampling and
        additonal H_CUMLEN and H_DELTALEN vectors, suitable for X sections.

        Args:
            distance (float): New horizontal distance between points
            atleast (int): Minimum number of point. If the true length/atleast is
                less than distance, than distance will be be reset to
                length/atleast.
            nextend (int): Number of samples to extend at each end. Note that
                in case of internal resetting of distance (due to 'atleast'), then
                nextend internally will be modified in order to fulfill the
                initial intention. Hence keep distance*nextend as target.
            name (str): Name of polygon (if asnumpy=False)
            asnumpy (bool): Return a [:, 5] numpy array with
                columns X.., Y.., Z.., HLEN, dH
            polyid (int): Which POLY_ID to use. Default (if None) is to use the
                first found.

        Returns:
            A numpy array (if asnumpy=True) or a new Polygons() object

        .. versionadded:: 2.1.0
        """
        logger.info("Getting fence within a Polygons instance...")
        return _xyz_oper.get_fence(
            self,
            distance=distance,
            atleast=atleast,
            nextend=nextend,
            name=name,
            asnumpy=asnumpy,
            polyid=polyid,
        )

    # ==================================================================================
    # Plotting
    # ==================================================================================

    def quickplot(
        self,
        others=None,
        filename=None,
        title="QuickPlot for Polygons",
        subtitle=None,
        infotext=None,
        linewidth=1.0,
        color="r",
    ):
        """Simple plotting of polygons using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            others (list of Polygons): List of other polygon instances to plot
            title (str): Title of plot
            subtitle (str): Subtitle of plot
            infotext (str): Additonal info on plot.
            color (str): Name of color (may use matplotib shortcuts, e.g. 'r' for 'red')
        """
        mymap = xtgeo.plot.Map()
        mymap.canvas(title=title, subtitle=subtitle, infotext=infotext)

        if others:
            for other in others:
                lwid = linewidth / 2.0
                mymap.plot_polygons(
                    other, idname=other.pname, linewidth=lwid, color="black"
                )

        mymap.plot_polygons(self, idname=self.pname, linewidth=linewidth, color=color)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # ==================================================================================

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
