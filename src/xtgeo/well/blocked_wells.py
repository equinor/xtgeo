# -*- coding: utf-8 -*-

"""BlockedWells module, (collection of BlockedWell objects)"""
from __future__ import division, absolute_import
from __future__ import print_function

import pandas as pd

import xtgeo
from . import _blockedwells_roxapi

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def blockedwells_from_roxar(project, gname, bwname, lognames=None, ijk=True):

    """This makes an instance of a BlockedWells directly from Roxar RMS.

    For arguments, see :meth:`BlockedWells.from_roxar`.

    Note the difference between classes BlockedWell and BlockedWells.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mybws = xtgeo.blockedwells_from_roxar(project, 'Simgrid', 'BW',
                                            lognames=mylogs)

    """

    obj = BlockedWells()

    obj.from_roxar(project, gname, bwname, ijk=ijk,
                   lognames=lognames)

    return obj


class BlockedWells(object):
    """Class for a collection of BlockedWell objects, for operations that
    involves a number of wells.

    See also the :class:`xtgeo.well.BlockedWell` class.
    """

    def __init__(self):

        self._bwells = []            # list of Well objects

    @property
    def names(self):
        """Returns a list of blocked well names (read only).

        Example::

            namelist = wells.names
            for prop in namelist:
                print ('Well name is {}'.format(name))

        """

        wlist = []
        for wel in self._bwells:
            wlist.append(wel.name)

        return wlist

    @property
    def wells(self):
        """Returns or sets a list of XTGeo BlockedWell objects, None if
        empty.
        """
        if len(self._bwells) == 0:
            return None

        return self._bwells

    @wells.setter
    def wells(self, bwell_list):

        for well in bwell_list:
            if not isinstance(well, xtgeo.well.BlockedWell):
                raise ValueError('Well in list not valid BlockedWell object')

        self._wells = bwell_list

    def copy(self):
        """Copy a BlockedWells instance to a new unique instance."""

        new = BlockedWells()

        for well in self._bwells:
            newwell = well.copy()
            new._props.append(newwell)

        return new

    def from_files(self, filelist, fformat='rms_ascii', mdlogname=None,
                   zonelogname=None, strict=True, append=True):

        """Import blocked wells from a list of files (filelist).

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
            self._bwells = []

        # file checks are done within the Well() class
        for wfile in filelist:
            try:
                wll = xtgeo.well.BlockedWell(wfile, fformat=fformat,
                                             mdlogname=mdlogname,
                                             zonelogname=zonelogname,
                                             strict=strict)
                self._bwells.append(wll)
            except ValueError as err:
                xtg.warn('SKIP this well: {}'.format(err))
                continue
        if not self._bwells:
            xtg.warn('No wells imported!')

    def from_roxar(self, project, gname, bwname, lognames=None,
                   ijk=True, realisation=0):
        """Import (retrieve) blocked wells from roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        All the wells present in the bwname icon will be imported.

        Args:
            project (str): Magic string 'project' or file path to project
            gname (str): Name of GridModel icon in RMS
            bwname (str): Name of Blocked Well icon in RMS, usually 'BW'
            lognames (list): List of lognames to include, or use 'all' for
                all current blocked logs for this well.
            ijk (bool): If True, then logs with grid IJK as I_INDEX, etc
            realisation (int): Realisation index (0 is default)
        """

        _blockedwells_roxapi.import_bwells_roxapi(self, project, gname, bwname,
                                                  lognames=lognames,
                                                  ijk=ijk)

    def get_dataframe(self, filled=False, fill_value1=-999, fill_value2=-9999):
        """Get a big dataframe for all wells in instance, with well name
        as first column

        Args:
            filled (bool): If True, then NaN's are replaces with values
            fill_value1 (int): Only applied if filled=True, for logs that
                have missing values
            fill_value2 (int): Only applied if filled=True, when logs
                are missing completely for that well.
        """

        bigdf = []
        for well in self._bwells:
            dfr = well.dataframe.copy()
            dfr['WELLNAME'] = well.name
            if filled:
                dfr.fillna(fill_value1)
            bigdf.append(dfr)

        dfr = pd.concat(bigdf, ignore_index=True, axis=1)

        # the concat itself may lead to NaN's:
        if filled:
            dfr = dfr.fillna(fill_value2)

        spec_order = ['WELLNAME', 'X_UTME', 'Y_UTMN', 'Z_TVDSS']
        dfr = dfr[spec_order + [col for col in dfr if col not in spec_order]]

        return dfr

    def quickplot(self, filename=None, title='QuickPlot'):
        """Fast plot of blocked wells using matplotlib.

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
