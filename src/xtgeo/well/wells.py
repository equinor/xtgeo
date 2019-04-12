# -*- coding: utf-8 -*-

"""Wells module, which has the Wells class (collection of Well objects)"""
from __future__ import division, absolute_import
from __future__ import print_function

from distutils.version import StrictVersion

import pandas as pd

import xtgeo

from . import _wells_utils

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Wells(object):
    """Class for a collection of Well objects, for operations that involves
    a number of wells.

    See also the :class:`xtgeo.well.Well` class.
    """

    def __init__(self, *args, **kwargs):

        self._wells = []  # list of Well objects
        self._bw = False
        self._props = None

        if args:
            # make instance from file import
            wfiles = args[0]
            fformat = kwargs.get("fformat", "rms_ascii")
            mdlogname = kwargs.get("mdlogname", None)
            zonelogname = kwargs.get("zonelogname", None)
            strict = kwargs.get("strict", True)
            self.from_files(
                wfiles,
                fformat=fformat,
                mdlogname=mdlogname,
                zonelogname=zonelogname,
                strict=strict,
                append=False,
            )

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
        if not self._wells:
            return None

        return self._wells

    @wells.setter
    def wells(self, well_list):

        for well in well_list:
            if not isinstance(well, xtgeo.well.Well):
                raise ValueError("Well in list not valid Well object")

        self._wells = well_list

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = xtgeo.common.XTGDescription()
        dsc.title("Description of {} instance".format(self.__class__.__name__))
        dsc.txt("Object ID", id(self))

        dsc.txt("Wells", self.names)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def copy(self):
        """Copy a Wells instance to a new unique instance (a deep copy)."""

        new = Wells()

        for well in self._wells:
            newwell = well.copy()
            new._props.append(newwell)

        return new

    def get_well(self, name):
        """Get a Well() instance by name, or None"""

        logger.info("Asking for a well with name %s", name)
        for well in self._wells:
            if well.name == name:
                return well
        return None

    def from_files(
        self,
        filelist,
        fformat="rms_ascii",
        mdlogname=None,
        zonelogname=None,
        strict=True,
        append=True,
    ):

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
            try:
                wll = xtgeo.well.Well(
                    wfile,
                    fformat=fformat,
                    mdlogname=mdlogname,
                    zonelogname=zonelogname,
                    strict=strict,
                )
                self._wells.append(wll)
            except ValueError as err:
                xtg.warn("SKIP this well: {}".format(err))
                continue
        if not self._wells:
            xtg.warn("No wells imported!")

    def from_roxar(self, *args, **kwargs):
        """Import (retrieve) all wells (or based on a filter) from
        roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        All the wells present in the bwname icon will be imported.

        Args:
            project (str): Magic string 'project' or file path to project
            lognames (list): List of lognames to include, or use 'all' for
                all current blocked logs for this well.
            wfilter (str): This is a regular expression to tell which wells
                that shall be included.
            ijk (bool): If True, then logs with grid IJK as I_INDEX, etc
            realisation (int): Realisation index (0 is default)

        Example::

            import xtgeo
            mywells = xtgeo.Wells()
            mywells.from_roxar(project, lognames='all', wfilter='31.*')

        """
        raise NotImplementedError("In prep...")

    # not having this as property but a get_ .. is intended, for flexibility
    def get_dataframe(self, filled=False, fill_value1=-999, fill_value2=-9999):
        """Get a big dataframe for all wells or blocked wells in instance,
        with well name as first column

        Args:
            filled (bool): If True, then NaN's are replaces with values
            fill_value1 (int): Only applied if filled=True, for logs that
                have missing values
            fill_value2 (int): Only applied if filled=True, when logs
                are missing completely for that well.
        """
        logger.info("Ask for big dataframe for all wells")

        bigdf = []
        for well in self._wells:
            dfr = well.dataframe.copy()
            dfr["WELLNAME"] = well.name
            logger.info(well.name)
            if filled:
                dfr = dfr.fillna(fill_value1)
            bigdf.append(dfr)

        if StrictVersion(pd.__version__) > StrictVersion("0.23.0"):
            # pylint: disable=unexpected-keyword-arg
            dfr = pd.concat(bigdf, ignore_index=True, sort=True)
        else:
            dfr = pd.concat(bigdf, ignore_index=True)

        # the concat itself may lead to NaN's:
        if filled:
            dfr = dfr.fillna(fill_value2)

        spec_order = ["WELLNAME", "X_UTME", "Y_UTMN", "Z_TVDSS"]
        dfr = dfr[spec_order + [col for col in dfr if col not in spec_order]]

        return dfr

    def quickplot(self, filename=None, title="QuickPlot"):
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

    def limit_tvd(self, tvdmin, tvdmax):
        """Limit TVD to be in range tvdmin, tvdmax for all wells"""
        for well in self.wells:
            well.limit_tvd(tvdmin, tvdmax)

    def downsample(self, interval=4, keeplast=True):
        """Downsample by sampling every N'th element (coarsen only), all
        wells.
        """

        for well in self.wells:
            well.downsample(interval=interval, keeplast=keeplast)

    def wellintersections(self, wfilter=None, showprogress=False):
        """Get intersections between wells, return as dataframe table.

        Notes on wfilter: A wfilter is settings to improve result. In
        particular to remove parts of trajectories that are parallel.

        wfilter = {'parallel': {'xtol': 4.0, 'ytol': 4.0, 'ztol':2.0,
                                'itol':10, 'atol':2}}

        Here xtol is tolerance in X coordinate; further Y tolerance,
        Z tolerance, (I)nclination tolerance, and (A)zimuth tolerance.

        Args:
            tvdrange (tuple of floats): Search interval. One is often just
                interested in the reservoir section.
            wfilter (dict): A dictionrary for filter options, in order to
                improve result. See example above.
            showprogress (bool): Will show progress to screen if enabled.

        Returns:
            A Pandas dataframe object, with columns WELL, CWELL and UTMX UTMY
                TVD coordinates for CWELL where CWELL crosses WELL,
                and also MDEPTH for the WELL.
        """

        dfr = _wells_utils.wellintersections(
            self, wfilter=wfilter, showprogress=showprogress
        )

        return dfr
