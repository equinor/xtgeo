# -*- coding: utf-8 -*-
"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a columns called 'ID'
# which identifies each polygon piece.

from __future__ import print_function, absolute_import
import pandas as pd

from xtgeo.xyz import XYZ


class Polygons(XYZ):
    """Class for a polygons (connected points) in the XTGeo framework."""

    def __init__(self, *args, **kwargs):

        super(Polygons, self).__init__(*args, **kwargs)

        self._ispolygons = True

    @property
    def nrow(self):
        """Cf :py:attr:`.XYZ.nrow`"""
        return super(Polygons, self).nrow

    @property
    def dataframe(self):
        """Cf :py:attr:`.XYZ.dataframe`"""
        return super(Polygons, self).dataframe

    @dataframe.setter
    def dataframe(self, df):
        super(Polygons, self).dataframe = df

    def from_file(self, pfile, fformat='xyz'):
        """Cf :meth:`.XYZ.from_file`"""
        super(Polygons, self).from_file(pfile, fformat=fformat)

        # for polygons, a seperate column with ID is required; however this may
        # lack if the input is on XYZ format

        if 'ID' not in self._df.columns:
            self._df['ID'] = self._df.isnull().all(axis=1).cumsum().dropna()
            self._df.dropna(axis=0, inplace=True)
            self._df.reset_index(inplace=True, drop=True)

    def to_file(self, pfile, fformat='xyz', attributes=None, filter=None,
                wcolumn=None, hcolumn=None, mdcolumn=None):
        """Cf :meth:`.XYZ.to_file`"""
        super(Polygons, self).to_file(pfile, fformat=fformat,
                                      attributes=attributes, filter=filter,
                                      wcolumn=wcolumn, hcolumn=hcolumn,
                                      mdcolumn=mdcolumn)

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

        if len(wells) == 0:
            return None

        dflist = []
        maxid = 0
        for well in wells:
            wp = well.get_zone_interval(zone, resample=resample)
            if wp is not None:
                # as well segments may have overlapping ID:
                wp['ID'] += maxid
                maxid = wp['ID'].max() + 1
                dflist.append(wp)

        if len(dflist) > 0:
            self._df = pd.concat(dflist, ignore_index=True)
            self._df.reset_index(inplace=True, drop=True)
        else:
            return None

        return len(dflist)
