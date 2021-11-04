# -*- coding: utf-8 -*-

"""Wells module, which has the Wells class (collection of Well objects)"""


from distutils.version import StrictVersion

import deprecation
import numpy as np
import pandas as pd
import shapely.geometry as sg

import xtgeo
from xtgeo.common import XTGShowProgress

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def wells_intersections(
    well_list, wfilter=None, showprogress=False
):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Get intersections between wells, return as dataframe table.

    Notes on wfilter: A wfilter is settings to improve result. In
    particular to remove parts of trajectories that are parallel.

    wfilter = {'parallel': {'xtol': 4.0, 'ytol': 4.0, 'ztol':2.0,
                            'itol':10, 'atol':2}}

    Here xtol is tolerance in X coordinate; further Y tolerance,
    Z tolerance, (I)nclination tolerance, and (A)zimuth tolerance.

    Args:
        well_list (list of wells): The list of wells to find intersections of.
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

    xpoints = []

    # make a dict if nocrossings
    nox = {}

    progress = XTGShowProgress(
        len(well_list), show=showprogress, leadtext="progress: ", skip=5
    )

    for iwell, well in enumerate(well_list):

        progress.flush(iwell)

        logger.info("Work with %s", well.name)
        try:
            well.geometrics()
        except ValueError:
            logger.info("Skip %s (cannot compute geometrics)", well.name)
            continue

        welldfr = well.dataframe.copy()

        xcor = welldfr["X_UTME"].values
        ycor = welldfr["Y_UTMN"].values
        mcor = welldfr[well.mdlogname].values
        logger.info("The mdlogname property is: %s", well.mdlogname)

        if xcor.size < 2:
            continue

        thisline1 = sg.LineString(np.stack([xcor, ycor], axis=1))
        thisline2 = sg.LineString(np.stack([xcor, ycor, mcor], axis=1))

        nox[well.name] = list()
        # loop over other wells
        for other in well_list:

            if other.name == well.name:
                continue  # same well

            if not well.may_overlap(other):
                nox[well.name].append(other.name)
                continue  # a quick check; no chance for overlap

            logger.info("Consider crossing with %s ...", other.name)

            # try to be smart to skip entries that earlier have beenn tested
            # for crossing. If other does not cross well, then well does not
            # cross other...
            if other.name in nox.keys() and well.name in nox[other.name]:
                continue

            # truncate away the paralell part on a copy
            owell = other.copy()

            # wfilter = None
            if wfilter is not None and "parallel" in wfilter:
                xtol = wfilter["parallel"].get("xtol")
                ytol = wfilter["parallel"].get("ytol")
                ztol = wfilter["parallel"].get("ztol")
                itol = wfilter["parallel"].get("itol")
                atol = wfilter["parallel"].get("atol")
                owell.truncate_parallel_path(
                    well, xtol=xtol, ytol=ytol, ztol=ztol, itol=itol, atol=atol
                )

            xcorc = owell.dataframe["X_UTME"].values
            ycorc = owell.dataframe["Y_UTMN"].values
            zcorc = owell.dataframe["Z_TVDSS"].values

            if xcorc.size < 2:
                continue

            otherline = sg.LineString(np.stack([xcorc, ycorc, zcorc], axis=1))

            if not thisline1.crosses(otherline):
                nox[well.name].append(other.name)
                continue

            ixx = thisline1.intersection(otherline)

            if ixx.is_empty:
                nox[well.name].append(other.name)
                continue

            # need this trick to get mdepth
            other2 = sg.LineString(np.stack([xcorc, ycorc], axis=1))
            ixx2 = thisline2.intersection(other2)

            logger.debug("==> Intersects with %s", other.name)

            if isinstance(ixx, sg.Point):
                xcor, ycor, zcor = ixx.coords[0]
                _x, _y, mcor = ixx2.coords[0]
                xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

            elif isinstance(ixx, sg.MultiPoint):
                pxx2 = list(ixx2)
                for ino, pxx in enumerate(list(ixx)):
                    xcor, ycor, zcor = pxx.coords[0]
                    _x, _y, mcor = pxx2[ino].coords[0]
                    xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

            elif isinstance(ixx, sg.GeometryCollection):
                gxx2 = list(ixx2)
                for ino, gxx in enumerate(list(ixx)):
                    if isinstance(gxx, sg.Point):
                        xcor, ycor, zcor = gxx.coords[0]
                        _x, _y, mcor = gxx2[ino].coords[0]
                        xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

    dfr = pd.DataFrame(
        xpoints, columns=["WELL", "MDEPTH", "CWELL", "X_UTME", "Y_UTMN", "Z_TVDSS"]
    )

    progress.finished()

    logger.info("All intersections found!")
    return dfr


def wells_dataframe(well_list, filled=False, fill_value1=-999, fill_value2=-9999):
    """Get a big dataframe for all wells or blocked wells in well list,
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
    for well in well_list:
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


def plot_wells(well_list, filename=None, title="QuickPlot"):
    """Fast plot of wells using matplotlib.

    Args:
        filename (str): Name of plot file; None will plot to screen.
        title (str): Title of plot

    """

    mymap = xtgeo.plot.Map()

    mymap.canvas(title=title)

    mymap.plot_wells(well_list)

    if filename is None:
        mymap.show()
    else:
        mymap.savefig(filename)


class Wells(object):
    """Deprecated collection of wells, use a list of wells instead"""

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="The wells class is deprecated in favor of using a list",
    )
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
        return [w.name for w in self._wells]

    def __iter__(self):
        return iter(self._wells)

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

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use [w.copy() for w in wells] instead",
    )
    def copy(self):
        """Copy a Wells instance to a new unique instance (a deep copy)."""

        new = Wells()
        new.wells = [w.copy() for w in self._wells]

        return new

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use [w for w in wells if w.name == name] instead",
    )
    def get_well(self, name):
        """Get a Well() instance by name, or None"""

        logger.info("Asking for a well with name %s", name)
        for well in self._wells:
            if well.name == name:
                return well
        return None

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use [xtgeo.well_from_file(f) for f in filelist] instead",
    )
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
        raise NotImplementedError(
            "Not implemented and the Wells class"
            " is deprecated, use a list of wells instead."
        )

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.wells_dataframe() instead",
    )
    def get_dataframe(self, *args, **kwargs):
        return wells_dataframe(self, *args, **kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.plot_wells() instead",
    )
    def quickplot(self, *args, **kwargs):
        return plot_wells(self, *args, **kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use [w.limit_tvd() for w in wells] instead.",
    )
    def limit_tvd(self, tvdmin, tvdmax):
        for well in self.wells:
            well.limit_tvd(tvdmin, tvdmax)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use [w.downsample() for w in wells] instead.",
    )
    def downsample(self, interval=4, keeplast=True):
        for well in self.wells:
            well.downsample(interval=interval, keeplast=keeplast)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.wells_intersections() instead",
    )
    def wellintersections(self, wfilter=None, showprogress=False):
        return wells_intersections(self, wfilter=wfilter, showprogress=showprogress)
