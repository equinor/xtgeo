"""Module for map plots of surfaces, using matplotlib."""


import matplotlib.pyplot as plt
import matplotlib.patches as mplp
from matplotlib import ticker
import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog
from .baseplot import BasePlot

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Map(BasePlot):
    """Class for plotting a map, using matplotlib."""

    def __init__(self):
        """The constructor method for a Map object."""
        super().__init__()

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        logger.info(clsname)

        self._wells = None
        self._surface = None
        self._tight = False

        self._wfence = None
        self._showok = True  # to indicate if plot is OK to show
        self._legendtitle = "Map"

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def pagesize(self):
        """Returns page size."""
        return self._pagesize

    # =========================================================================
    # Functions methods (public)
    # =========================================================================

    def plot_surface(
        self,
        surf,
        minvalue=None,
        maxvalue=None,
        contourlevels=None,
        xlabelrotation=None,
        colormap=None,
        logarithmic=False,
    ):  # pylint: disable=too-many-statements
        """Input a surface and plot it."""
        # need a deep copy to avoid changes in the original surf

        logger.info("The key contourlevels %s is not in use", contourlevels)

        usesurf = surf.copy()
        if usesurf.yflip < 0:
            usesurf.swapaxes()

        if abs(surf.rotation) > 0.001:
            usesurf.unrotate()

        xi, yi, zi = usesurf.get_xyz_values()

        zimask = ma.getmaskarray(zi).copy()  # yes need a copy!

        legendticks = None
        if minvalue is not None and maxvalue is not None:
            minv = float(minvalue)
            maxv = float(maxvalue)

            step = (maxv - minv) / 10.0
            legendticks = []
            for i in range(10 + 1):
                llabel = float("{0:9.4f}".format(minv + step * i))
                legendticks.append(llabel)

            zi.unshare_mask()
            zi[zi < minv] = minv
            zi[zi > maxv] = maxv

            # need to restore the mask:
            zi.mask = zimask

            # note use surf.min, not usesurf.min here ...
            notetxt = (
                "Note: map values are truncated from ["
                + str(surf.values.min())
                + ", "
                + str(surf.values.max())
                + "] "
                + "to interval ["
                + str(minvalue)
                + ", "
                + str(maxvalue)
                + "]"
            )

            self._fig.text(0.99, 0.02, notetxt, ha="right", va="center", fontsize=8)

        logger.info("Legendticks: %s", legendticks)

        if minvalue is None:
            minvalue = usesurf.values.min()

        if maxvalue is None:
            maxvalue = usesurf.values.max()

        # this will override current instance colormap locally, and is
        # therefore reset afterwards
        keepcolor = self.colormap
        if colormap is not None:
            self.colormap = colormap

        levels = np.linspace(minvalue, maxvalue, self.contourlevels)
        logger.debug("Number of contour levels: %s", levels)

        plt.setp(self._ax.xaxis.get_majorticklabels(), rotation=xlabelrotation)

        # zi = ma.masked_where(zimask, zi)
        # zi = ma.masked_greater(zi, xtgeo.UNDEF_LIMIT)
        logger.info("Current colormap is %s, requested is %s", self.colormap, colormap)
        logger.info("Current colormap name is %s", self.colormap.name)

        if ma.std(zi) > 1e-07:
            uselevels = levels
        else:
            uselevels = 1

        try:
            if logarithmic is False:
                locator = None
                ticks = legendticks
                im = self._ax.contourf(
                    xi, yi, zi, uselevels, locator=locator, cmap=self.colormap
                )

            else:
                logger.info("use LogLocator")
                locator = ticker.LogLocator()
                ticks = None
                uselevels = None
                im = self._ax.contourf(xi, yi, zi, locator=locator, cmap=self.colormap)

            self._fig.colorbar(im, ticks=ticks)
        except ValueError as err:
            logger.warning("Could not make plot: %s", err)

        plt.gca().set_aspect("equal", adjustable="box")
        self.colormap = keepcolor

    def plot_faults(
        self,
        fpoly,
        idname="POLY_ID",
        color="k",
        edgecolor="k",
        alpha=0.7,
        linewidth=0.8,
    ):
        """Plot the faults.

        Args:
            fpoly (object): A XTGeo Polygons object
            idname (str): Name of column which has the faults ID
            color (c): Fill color model c according to Matplotlib_
            edgecolor (c): Edge color according to Matplotlib_
            alpha (float): Degree of opacity
            linewidth (float): Line width

        .. _Matplotlib: http://matplotlib.org/api/colors_api.html
        """
        aff = fpoly.dataframe.groupby(idname)

        for name, _group in aff:

            # make a dataframe sorted on faults (groupname)
            myfault = aff.get_group(name)

            # make a list [(X,Y) ...];
            af = list(zip(myfault["X_UTME"].values, myfault["Y_UTMN"].values))

            px = mplp.Polygon(af, alpha=alpha, color=color, ec=edgecolor, lw=linewidth)

            if px.get_closed():
                self._ax.add_artist(px)
            else:
                IOError(f"A polygon is not closed: {px}")

    def plot_polygons(self, fpoly, idname="POLY_ID", color="k", linewidth=0.8):
        """Plot a polygons instance.

        Args:
            fpoly (object): A XTGeo Polygons object
            idname (str): Name of column which has the faults ID
            color (c): Line color model c according to Matplotlib_
            linewidth (float): Line width

        .. _Matplotlib: http://matplotlib.org/api/colors_api.html
        """
        aff = fpoly.dataframe.groupby(idname)

        for _name, group in aff:

            # make a dataframe sorted on groupname
            pname = fpoly.name

            xarr = group[fpoly.xname].values
            yarr = group[fpoly.yname].values

            self._ax.plot(xarr, yarr, label=pname, lw=linewidth, color=color)
            self._ax.legend()

    def plot_points(self, points):
        """Plot a points set on the map.

        This can be be useful e.g. for plotting the underlying point set
        that makes a gridded map.

        Args:
            points (Points): A XTGeo Points object X Y VALUE

        """
        # This function is "in prep"

        dataframe = points.dataframe

        self._ax.scatter(
            dataframe["X_UTME"].values, dataframe["Y_UTMN"].values, marker="x"
        )

    def plot_wells(self, wells):
        """Plot wells on the map.

        Args:
            wells (Wells): A XTGeo Wells object (contains a number of Well
                instances).

        """
        for well in wells.wells:
            dataframe = well.dataframe

            xval = dataframe["X_UTME"].values
            yval = dataframe["Y_UTMN"].values
            self._ax.plot(xval, yval)
            self._ax.annotate(well.name, xy=(xval[-1], yval[-1]))
