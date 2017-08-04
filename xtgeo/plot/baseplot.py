import logging
import matplotlib.pyplot as plt

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
        if cfile == 'xtgeo':
            colors = _ctable.xtgeocolors()
            self.contourlevels = len(colors)

        elif cfile in valid_maps:
            cmap = plt.get_cmap(cfile)
            self.logger.info(cmap.N)
            for i in range(cmap.N):
                colors.append(cmap(i))
            self.contourlevels = cmap.N

        else:
            colors = _ctable.colorsfromfile(cfile)
            self.contourlevels = len(colors)

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
