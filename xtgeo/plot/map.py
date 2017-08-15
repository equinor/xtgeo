"""Module for map plots of surfaces, using matplotlib."""

from __future__ import print_function

import os
import matplotlib as mpl
# if os.environ.get('DISPLAY', '') == '':
#     print('No display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import matplotlib.pyplot as plt
import logging
import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog
from xtgeo.plot import _colortables as _ctable
from xtgeo.plot.baseplot import BasePlot


class Map(BasePlot):
    """Class for plotting a map."""

    def __init__(self):
        """The __init__ (constructor) method for a Map object."""

        super(Map, self).__init__()

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        self._wells = None
        self._surface = None
        self._tight = False

        self._pagesize = 'A4'
        self._wfence = None
        self._showok = True  # to indicate if plot is OK to show
        self._legendtitle = "Map"

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def pagesize(self):
        """ Returns page size."""
        return self._pagesize


    # =========================================================================
    # Functions methods (public)
    # =========================================================================

    def canvas(self, title=None, subtitle=None, infotext=None,
               figscaling=1.0):
        """Prepare the canvas to plot on, with title and subtitle.

        Args:
            title (str, optional): Title of plot.
            subtitle (str, optional): Sub title of plot.
            infotext (str, optional): Text to be written as info string.
            figscaling (str, optional): Figure scaling, default is 1.0


        """
        # self._fig, (ax1, ax2) = plt.subplots(2, figsize=(11.69, 8.27))
        self._fig, self._ax = plt.subplots(figsize=(11.69 * figscaling,
                                                    8.27 * figscaling))
        if title is not None:
            plt.title(title, fontsize=18)
        if subtitle is not None:
            self._ax.set_title(subtitle, size=14)

    def plot_surface(self, surf, minvalue=None, maxvalue=None,
                     contourlevels=None, xlabelrotation=None,
                     colortable=None):
        """Input a surface and plot it."""

        xmax = surf.xori + surf.xinc * surf.nx
        ymax = surf.yori + surf.yinc * surf.ny
        xi = np.linspace(surf.xori, xmax, surf.nx)
        yi = np.linspace(surf.yori, ymax, surf.ny)

        zi = ma.transpose(surf.values.copy())

        if minvalue is None:
            minvalue = surf.values.min()

        if maxvalue is None:
            maxvalue = surf.values.max()

        # zi[zi < minvalue] = minvalue
        # zi[zi > maxvalue] = maxvalue

        if colortable is not None:
            self.set_colortable(colortable)
        else:
            self.set_colortable('rainbow')

        levels = np.linspace(minvalue, maxvalue, self.contourlevels)

        plt.setp(self._ax.xaxis.get_majorticklabels(), rotation=xlabelrotation)
        im = self._ax.contourf(xi, yi, zi, levels, colors=self.colortable)
        self._fig.colorbar(im)
        plt.gca().set_aspect('equal', adjustable='box')

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

    def savefig(self, filename):
        """Call to matplotlib.pyplot savefig().

        Returns:
            True of plotting is done; otherwise False
        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            plt.savefig(filename)
            plt.close(self._fig)
            return True
        else:
            self.logger.warning("Nothing to plot (well outside Z range?)")
            return False
