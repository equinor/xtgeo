# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma
import pandas as pd
import xtgeo
from xtgeo.xyz import XYZ
import cxtgeo.cxtgeo as _cxtgeo

UNDEF = _cxtgeo.UNDEF
UNDEF_LIMIT = _cxtgeo.UNDEF_LIMIT


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

    .. autoclass:: XYZ:: members:: inherited-members

    """

    def __init__(self, *args, **kwargs):

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
                wcolumn=None, hcolumn=None, mdcolumn=None):
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
