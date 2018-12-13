# -*- coding: utf-8 -*-

"""Wells module, which has the Wells class (collection of Well objects)"""
from __future__ import division, absolute_import
from __future__ import print_function


import xtgeo

from xtgeo.well import _wells_utils

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Wells(object):
    """Class for a collection of Well objects, for operations that involves
    a number of wells.

    See also the :class:`xtgeo.well.Well` class.
    """

    def __init__(self):

        self._wells = []            # list of Well objects

    @property
    def names(self):
        """Returns a list of well names (read only).

        Example::

            namelist = wells.names
            for prop in namelist:
                print ('Well name is {}'.format(name))

        """

        wlist = []
        for wel in self._wells:
            wlist.append(wel.name)

        return wlist

    @property
    def wells(self):
        """Returns or sets a list of XTGeo Well objects, None if empty."""
        if len(self._wells) == 0:
            return None

        return self._wells

    @wells.setter
    def wells(self, well_list):

        for well in well_list:
            if not isinstance(well, xtgeo.well.Well):
                raise ValueError('Well in list not valid Well object')

        self._wells = well_list

    def copy(self):
        """Copy a Wells instance to a new unique instance (a deep copy)."""

        new = Wells()

        for well in self._wells:
            newwell = well.copy()
            new._props.append(newwell)

        return new

    def from_files(self, filelist, fformat='rms_ascii', mdlogname=None,
                   zonelogname=None, strict=True, append=True):

        """Import wells from a list of files (filelist).

        Args:
            filelist (list of str): List with file names
            fformat (str): File format, rms_ascii (rms well) is
                currently supported and default format.
            mdlogname (str): Name of measured depth log, if any
            zonelogname (str): Name of zonation log, if any
            strict (bool): If True, then import will fail if
                zonelogname or mdlogname are asked for but not present
                in wells.
            append (bool): If True, new wells will be added to existing
                wells.

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mywells = Wells(['31_2-6.w', '31_2-7.w', '31_2-8.w'])
        """

        if not append:
            self._wells = []

        # file checks are done within the Well() class
        for wfile in filelist:
            self._wells.append(xtgeo.well.Well(wfile))

    def quickplot(self, filename=None, title='QuickPlot'):
        """Fast plot of wells using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            title (str): Title of plot

        """

        mymap = xtgeo.plot.Map()

        mymap.canvas(title=title)

        mymap.plot_wells(self)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)

    def wellintersections(self, sampling=None, tvdrange=(None, None)):
        """Get intersections between wells, return as dataframe table.

        Args:
            sampling (float): Sampling interval. Default is well log sampling,
                but a sparser sampling may speed up a lot.

            tvdrange (tuple of floats): Search interval. One is often just
                interested in the reservoir section.

        Returns:
            A Pandas dataframe object, with columns WELL, CWELL and UTMX UTMY
                TVD coordinates where CWELL crosses WELL.
        """

        dfr = _wells_utils.wellintersections(self, sampling=sampling,
                                             tvdrange=tvdrange)

        return dfr
