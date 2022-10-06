"""The baseplot module."""
from distutils.version import LooseVersion

import deprecation
import matplotlib as mpl
import matplotlib.pyplot as plt
import xtgeo
from matplotlib.colors import LinearSegmentedColormap
from xtgeo.common import XTGeoDialog

from . import _colortables as _ctable

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def _get_colormap(name):
    """For matplotlib compatibility."""
    if LooseVersion(mpl.__version__) < LooseVersion("3.6"):
        return plt.cm.get_cmap(name)
    else:
        return mpl.colormaps[name]


class BasePlot(object):
    """Base class for plots, providing some functions to share."""

    def __init__(self):
        """Init method."""
        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        logger.info(clsname)

        self._contourlevels = 3
        self._colormap = _get_colormap("viridis")

        self._ax = None
        self._tight = False
        self._showok = True
        self._fig = None
        self._allfigs = []
        self._pagesize = "A4"

        logger.info("Ran __init__ ...")

    @property
    def contourlevels(self):
        """Get or set the number of contour levels."""
        return self._contourlevels

    @contourlevels.setter
    def contourlevels(self, num):
        self._contourlevels = num

    @property
    def colormap(self):
        """Get or set the color table as a matplot cmap object."""
        return self._colormap

    @colormap.setter
    def colormap(self, cmap):

        if isinstance(cmap, LinearSegmentedColormap):
            self._colormap = cmap
        elif isinstance(cmap, str):
            logger.info("Definition of a colormap from string name: %s", cmap)
            self.define_colormap(cmap)
        else:
            raise ValueError("Input incorrect")

        logger.info("Colormap: %s", self._colormap)

    @property
    def pagesize(self):
        """Returns page size."""
        return self._pagesize

    @staticmethod
    def define_any_colormap(cfile, colorlist=None):
        """Defines any color map from file or a predefined name.

        This is a static method, which returns a matplotlib CM object.

        Args:
            cfile (str): File name (RMS format) or an alias for a predefined
                map name, e.g. 'xtgeo', or one of matplotlibs numerous tables.
            colorlist (list, int, optional): List of integers redefining
                color entries per zone and/or well, which starts
                from 0 index. Default is just keep the linear sequence as is.

        """
        valid_maps = sorted(m for m in plt.cm.datad)

        logger.info("Valid color maps: %s", valid_maps)

        colors = []

        cmap = _get_colormap("rainbow")

        if cfile is None:
            cfile = "rainbow"
            cmap = _get_colormap("rainbow")

        elif cfile == "xtgeo":
            colors = _ctable.xtgeocolors()
            cmap = LinearSegmentedColormap.from_list(cfile, colors, N=len(colors))
            cmap.name = "xtgeo"
        elif cfile == "random40":
            colors = _ctable.random40()
            cmap = LinearSegmentedColormap.from_list(cfile, colors, N=len(colors))
            cmap.name = "random40"

        elif cfile == "randomc":
            colors = _ctable.randomc(256)
            cmap = LinearSegmentedColormap.from_list(cfile, colors, N=len(colors))
            cmap.name = "randomc"

        elif isinstance(cfile, str) and "rms" in cfile:
            colors = _ctable.colorsfromfile(cfile)
            cmap = LinearSegmentedColormap.from_list("rms", colors, N=len(colors))
            cmap.name = cfile
        elif cfile in valid_maps:
            cmap = _get_colormap(cfile)
            for i in range(cmap.N):
                colors.append(cmap(i))
        else:
            xtg.warnuser(
                "Trying to access as color map not installed in "
                "this version of matplotlib: <{}>. Revert to <rainbow>".format(cfile)
            )
            cmap = _get_colormap("rainbow")
            for i in range(cmap.N):
                colors.append(cmap(i))

        ctable = []

        if colorlist:
            for entry in colorlist:
                if entry < len(colors):
                    ctable.append(colors[entry])
                else:
                    logger.warning("Color list out of range")
                    ctable.append(colors[0])

            cmap = LinearSegmentedColormap.from_list(ctable, colors, N=len(colors))
            cmap.name = "user"

        return cmap

    @staticmethod
    def get_any_colormap_as_table(cmap):
        """Returns the given color map cmap as a list of RGB tuples."""
        cmaplist = [cmap(i) for i in range(cmap.N)]
        return cmaplist

    @deprecation.deprecated(
        deprecated_in="1.16",
        removed_in="3.0",
        current_version=xtgeo.version,
        details=(
            "set_colortable is deprecated, use the colormap"
            " property or define_colormap instead"
        ),
    )
    def set_colortable(self, cname, colorlist=None):
        """Deprecated function, use define_colormap or the colormap property instead"""
        if colorlist is None:
            self.colormap = cname
        else:
            self.define_colormap(cname, colorlist=colorlist)

    def get_colormap_as_table(self):
        """Get the current color map as a list of RGB tuples."""
        return self.get_any_colormap_as_table(self._colormap)

    def define_colormap(self, cfile, colorlist=None):
        """Defines a color map from file or a predefined name.

        Args:
            cfile (str): File name (RMS format) or an alias for a predefined
                map name, e.g. 'xtgeo', or one of matplotlibs numerous tables.
            colorlist (list, int, optional): List of integers redefining
                color entries per zone and/or well, which starts
                from 0 index. Default is just keep the linear sequence as is.

        """
        logger.info("Defining colormap")

        cmap = self.define_any_colormap(cfile, colorlist=colorlist)

        self.contourlevels = cmap.N
        self._colormap = cmap

    def canvas(self, title=None, subtitle=None, infotext=None, figscaling=1.0):
        """Prepare the canvas to plot on, with title and subtitle.

        Args:
            title (str, optional): Title of plot.
            subtitle (str, optional): Sub title of plot.
            infotext (str, optional): Text to be written as info string.
            figscaling (str, optional): Figure scaling, default is 1.0


        """
        # self._fig, (ax1, ax2) = plt.subplots(2, figsize=(11.69, 8.27))
        self._fig, self._ax = plt.subplots(
            figsize=(11.69 * figscaling, 8.27 * figscaling)
        )
        self._allfigs.append(self._fig)
        if title is not None:
            self._fig.suptitle(title, fontsize=18)
        if subtitle is not None:
            self._ax.set_title(subtitle, size=14)
        if infotext is not None:
            self._fig.text(0.01, 0.02, infotext, ha="left", va="center", fontsize=8)

    def show(self):
        """Call to matplotlib.pyplot show method.

        Returns:
            True of plotting is done; otherwise False
        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            logger.info("Calling plt show method...")
            plt.show()
            return True

        logger.warning("Nothing to plot (well outside Z range?)")
        return False

    def close(self):
        """
        Explicitly closes the plot, meaning that memory will be cleared.

        After close is called, no more operations can be performed on the plot.

        """
        for fig in self._allfigs:
            plt.close(fig)

    def savefig(self, filename, fformat="png", last=True, **kwargs):
        """Call to matplotlib.pyplot savefig method.

        Args:
            filename (str): File to plot to
            fformat (str): Plot format, e.g. png (default), jpg, svg
            last (bool): Default is true, calls close on the plot, let last
                be False for all except the last plots.
            kwargs: Additional keyword arguments that are passed
                to matplotlib when saving the figure

        Returns:
            True of plotting is done; otherwise False

        Example::

            myplot.savefig('layerslice.svg', fformat='svg', last=False)
            myplot.savefig('layerslice.png')

        .. versionchanged:: 2.4 added kwargs option

        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            plt.savefig(filename, format=fformat, **kwargs)
            if last:
                self.close()
            return True

        logger.warning("Nothing to plot (well outside Z range?)")
        return False
