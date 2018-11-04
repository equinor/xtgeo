# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma
import pandas as pd
import xtgeo
from xtgeo.xyz import XYZ
# import xtgeo.xyz._xyz_roxapi as _xyz_roxapi


class Points(XYZ):
    """Points: Class for a points set in the XTGeo framework.

    The Points class is a subclass of the :class:`.XYZ` class,
    and the point set itself is a `pandas <http://pandas.pydata.org>`_
    dataframe object.

    The instance can be made either from file or by a spesification,
    e.g. from file::

        xp = Points(xp.from_file('somefilename', fformat='xyz')
        # show the Pandas dataframe
        print(xp.dataframe)

    Magic column names in the dataframe:

    * X_UTME: UTM X coordinate
    * Y_UTMN: UTM Y coordinate
    * Z_TVDSS: Z coordinate, often depth below TVD SS, but may also be
      something else!
    * M_MDEPTH: measured depth, (if present)
    * Q_*: Quasi geometrical measures, such as MD, AZIMUTH, INCL

    """

    def __init__(self, *args, **kwargs):

        # instance variables listed
        self._df = None
        self._ispolygons = False

        super(Points, self).__init__(*args, **kwargs)

        if len(args) == 1:
            if isinstance(args[0], xtgeo.surface.RegularSurface):
                self.from_surface(args[0])

    @property
    def nrow(self):
        """ Returns the Pandas dataframe object number of rows"""
        if self._df is None:
            return 0
        else:
            return len(self._df.index)

    @property
    def dataframe(self):
        """ Returns or set the Pandas dataframe object"""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    def from_file(self, pfile, fformat='xyz'):
        """Import points.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

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

    def to_file(self, pfile, fformat='xyz', attributes=None, filter=None,
                wcolumn=None, hcolumn=None, mdcolumn='M_MDEPTH'):
        """Export XYZ (Points/Polygons) to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
            attributes (list): List of extra columns to export (some formats)
            filter (dict): Filter on e.g. top name(s) with keys TopName
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
            KeyError if filter is set and key(s) are invalid

        """

        super(Points, self).to_file(pfile, fformat=fformat,
                                    attributes=attributes, filter=filter,
                                    wcolumn=wcolumn, hcolumn=hcolumn,
                                    mdcolumn=mdcolumn)

    def from_wells(self, wells, tops=True, incl_limit=None, top_prefix='Top',
                   zonelist=None, use_undef=False):

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

        if len(wells) == 0:
            return None

        dflist = []
        for well in wells:
            wp = well.get_zonation_points(tops=tops, incl_limit=incl_limit,
                                          top_prefix=top_prefix,
                                          zonelist=zonelist,
                                          use_undef=use_undef)

            if wp is not None:
                dflist.append(wp)

        if len(dflist) > 0:
            self._df = pd.concat(dflist, ignore_index=True)
        else:
            return None

        return len(dflist)

    def dfrac_from_wells(self, wells, dlogname, dcodes, incl_limit=90,
                         zonelist=None):

        """Get fraction of discrete code(s) (e.g. facies) per zone.

        Args:
            wells (list): List of XTGeo well objects
            dlogname (str): Name of discrete log (e.g. Facies)
            dcodes (list of int): Code(s) to get fraction for, e.g. [3]
            incl_limit (float): Inclination limit for zones (thickness points)

        Returns:
            None if well list is empty; otherwise the number of wells.

        Raises:
            Todo
        """

        if len(wells) == 0:
            return None

        if zonelist is None:
            zonelist = [1]

        dflist = []
        for well in wells:
            wpf = well.get_fraction_per_zone(
                dlogname, dcodes, zonelist=zonelist,
                incl_limit=incl_limit)

            if wpf is not None:
                dflist.append(wpf)

        if len(dflist) > 0:
            self._df = pd.concat(dflist, ignore_index=True)
        else:
            return None

        return len(dflist)

    def from_surface(self, surf):
        """Get points as X Y Value from a surface object nodes.

        Note that undefined surface nodes will not be included.

        Args:
            surf (RegularSurface): A XTGeo RegularSurface object instance.

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
            raise ValueError('Given surf is not a RegularSurface object')

        val = surf.values
        xc, yc = surf.get_xy_values()

        coor = []
        for vv in [xc, yc, val]:
            vv = ma.filled(vv.flatten(order='C'), fill_value=np.nan)
            vv = vv[~np.isnan(vv)]
            coor.append(vv)

        # now populate the dataframe:
        xc, yc, val = coor
        ddatas = {'X_UTME': xc, 'Y_UTMN': yc, 'Z_TVDSS': val}
        self._df = pd.DataFrame(ddatas)

    def from_roxar(self, project, name, category, stype='horizons',
                   realisation=0):
        """Load a points set from a Roxar RMS project.

        Syntax is::

          import xtgeo
          mysurf = xtgeo.surface_from_roxar(project, 'TopAare', 'DepthSurface')

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic `project` word if
                within RMS.
            name (str): Name of surface/map
            category (str): For horizons/zones only: for example 'DP_extracted'
            stype (str): RMS folder type, 'horizons' (default) or 'zones'
            realisation (int): Realisation number, default is 0

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

            >>> mypoints = Points()
            >>> mypoints.from_roxar(project, 'TopAare', 'DepthPoints')

        """
        pass

        # stype = stype.lower()
        # valid_stypes = ['horizons', 'zones']

        # if stype not in valid_stypes:
        #     raise ValueError('Invalid stype, only {} stypes is supported.'
        #                      .format(valid_stypes))

        # _xyz_roxapi.import_points_roxapi(
        #     self, project, name, category, stype, realisation)
