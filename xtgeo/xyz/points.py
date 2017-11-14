# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class"""

from __future__ import print_function, absolute_import

import pandas as pd
from xtgeo.xyz import XYZ


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

    """

    def __init__(self, *args, **kwargs):

        super(Points, self).__init__(*args, **kwargs)

    @property
    def nrows(self):
        """Cf :py:attr:`.XYZ.nrows`"""
        return super(Points, self).nrows

    @property
    def dataframe(self):
        """Cf :py:attr:`.XYZ.dataframe`"""
        return super(Points, self).dataframe

    @dataframe.setter
    def dataframe(self, df):
        super(Points, self).dataframe = df

    def from_file(self, pfile, fformat='xyz'):
        """Cf :meth:`.XYZ.from_file`"""
        super(Points, self).from_file(pfile, fformat=fformat)

    def to_file(self, pfile, fformat='xyz', attributes=None, filter=None,
                wcolumn=None, hcolumn=None, mdcolumn=None):
        """Cf :meth:`.XYZ.to_file`"""
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
