"""Module for map plots of surfaces, using matplotlib."""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import logging
import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog
from xtgeo.plot.baseplot import BasePlot
import cxtgeo.cxtgeo as _cxtgeo


class Map(BasePlot):
    """Class for plotting a map, using matplotlib."""

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
        if infotext is not None:
            self._fig.text(0.01, 0.02, infotext, ha='left', va='center',
                           fontsize=8)

    def plot_surface(self, surf, minvalue=None, maxvalue=None,
                     contourlevels=None, xlabelrotation=None,
                     colortable=None):
        """Input a surface and plot it."""

        xmax = surf.xori + surf.xinc * surf.nx
        ymax = surf.yori + surf.yinc * surf.ny
        xi = np.linspace(surf.xori, xmax, surf.nx)
        yi = np.linspace(surf.yori, ymax, surf.ny)

        # make a copy so original numpy is not altered!
        zi = ma.transpose(surf.values.copy())

        # store the current mask:
        zimask = ma.getmask(zi).copy()

        legendticks = None
        if minvalue is not None and maxvalue is not None:
            minv = float(minvalue)
            maxv = float(maxvalue)

            step = (maxv - minv) / 10.0
            legendticks = []
            for i in range(10 + 1):
                llabel = float('{0:9.4f}'.format(minv + step * i))
                legendticks.append(llabel)

            zi[zi < minv] = minv
            zi[zi > maxv] = maxv

            notetxt = ('Note: map values are truncated from [' +
                       str(surf.values.min()) + ', ' +
                       str(surf.values.max()) + '] ' +
                       'to interval [' +
                       str(minvalue) + ', ' + str(maxvalue) + ']')

            self._fig.text(0.99, 0.02, notetxt, ha='right', va='center',
                           fontsize=8)

        self.logger.info('Legendticks: {}'.format(legendticks))

        if minvalue is None:
            minvalue = surf.values.min()

        if maxvalue is None:
            maxvalue = surf.values.max()

        if colortable is not None:
            self.set_colortable(colortable)
        else:
            self.set_colortable('rainbow')

        levels = np.linspace(minvalue, maxvalue, self.contourlevels)
        self.logger.debug('Number of contour levels: {}'.format(levels))

        plt.setp(self._ax.xaxis.get_majorticklabels(), rotation=xlabelrotation)

        zi = ma.masked_where(zimask, zi)
        zi = ma.masked_greater(zi, _cxtgeo.UNDEF_LIMIT)

        if ma.std(zi) > 1e-07:
            uselevels = levels
        else:
            uselevels = 1

        try:
            im = self._ax.contourf(xi, yi, zi, uselevels,
                                   colors=self.colortable)
            self._fig.colorbar(im, ticks=legendticks)
        except ValueError as err:
            self.logger.warning('Could not make plot: {}'.format(err))

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
