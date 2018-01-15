import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from xtgeo.plot import _colortables as _ctable


class BasePlot(object):
    """Base class for plots, providing some functions to share"""
    def __init__(self):

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._contourlevels = 3
        self._colortable = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]

    @property
    def contourlevels(self):
        """Get the number of contour levels"""
        return self._contourlevels

    @contourlevels.setter
    def contourlevels(self, n):
        self._contourlevels = n

    @property
    def colortable(self):
        """Get or set the color table as a list of RGB tuples."""
        return self._colortable

    @colortable.setter
    def colortable(self, list):
        # checking is missing...
        self._colortable = list

    @property
    def colormap(self):
        """Get or set the color table as a matplot cmap object."""
        return self._colormap

    @colormap.setter
    def colormap(self, cmap):
        if isinstance(cmap, LinearSegmentedColormap):
            self._colormap = cmap
        else:
            raise ValueError('Input not correct Matplotlib cmap instance')

    def set_colortable(self, cfile, colorlist=None):
        """Defines a color table from file or a predefined name.

        Args:
            cfile (str): File name (RMS format) or an alias for a predefined
                map name, e.g. 'xtgeo', or one of matplotlibs numerous tables.
            colorlist (list, int, optional): List of integers redefining
                color entries per zone and/or well, which starts
                from 0 index. Default is just keep the linear sequence as is.

        """
        valid_maps = sorted(m for m in plt.cm.datad)

        colors = []

        cmap = plt.get_cmap('rainbow')

        if cfile is None:
            cfile = 'rainbow'
            cmap = plt.get_cmap('rainbow')

        if cfile == 'xtgeo':
            colors = _ctable.xtgeocolors()
            self.contourlevels = len(colors)

            cmap = LinearSegmentedColormap.from_list(cfile, colors,
                                                     N=len(colors))

        elif 'rms' in cfile:
            colors = _ctable.colorsfromfile(cfile)
            self.contourlevels = len(colors)

            cmap = LinearSegmentedColormap.from_list('rms', colors,
                                                     N=len(colors))

        elif cfile in valid_maps:
            cmap = plt.get_cmap(cfile)
            self.logger.info(cmap.N)
            for i in range(cmap.N):
                colors.append(cmap(i))
            self.contourlevels = cmap.N

        else:
            cmap = plt.get_cmap('rainbow')
            self.logger.info(cmap.N)
            for i in range(cmap.N):
                colors.append(cmap(i))
            self.contourlevels = cmap.N

        ctable = []

        if colorlist:
            for entry in colorlist:
                if entry < len(colors):
                    ctable.append(colors[entry])
                else:
                    self.logger.warn('Color list out of range')
                    ctable.append(colors[0])
            self._colortable = ctable
        else:
            self._colortable = colors

        self._colormap = cmap

    def show(self):
        """Call to matplotlib.pyplot show().

        Returns:
            True of plotting is done; otherwise False
        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            self.logger.info('Calling plt show method...')
            plt.show()
            return True
        else:
            self.logger.warning("Nothing to plot (well outside Z range?)")
            return False

    def savefig(self, filename, fformat='png', last=True):
        """Call to matplotlib.pyplot savefig().

        Args:
            filename (str): File to plot to
            fformat (str): Plot format, e.g. png (default), jpg, svg
            last (bool): Default is true, meaning that memory will be cleared;
                however if several plot types for the same instance, let last
                be False fora all except the last plots.

        Returns:
            True of plotting is done; otherwise False

        Example::
            myplot.savefig('TMP/layerslice.svg', fformat='svg', last=False)
            myplot.savefig('TMP/layerslice.png')

        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            plt.savefig(filename, format=fformat)
            if last:
                plt.close(self._fig)
            return True
        else:
            self.logger.warning("Nothing to plot (well outside Z range?)")
            return False
