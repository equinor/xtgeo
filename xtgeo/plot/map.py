"""Module for map plots of surfaces, using matplotlib."""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import matplotlib.patches as mplp
from matplotlib import ticker
import logging
import numpy as np
import numpy.ma as ma
import six

import xtgeo
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
                     colortable=None, logarithmic=False):
        """Input a surface and plot it."""

        # need a deep copy to avoid changes in the original surf

        usesurf = surf.copy()
        if (abs(surf.rotation) > 0.001):
            # resample the surface to something nonrotated
            self.logger.info('Resampling to non-rotated surface...')
            xlen = surf.xmax - surf.xmin
            ylen = surf.ymax - surf.ymin
            ncol = surf.ncol * 2
            nrow = surf.nrow * 2
            xinc = xlen / (ncol - 1)
            yinc = ylen / (nrow - 1)
            vals = ma.zeros((ncol, nrow), order='F')

            nonrot = xtgeo.surface.RegularSurface(xori=surf.xmin,
                                                  yori=surf.ymin,
                                                  xinc=xinc, yinc=yinc,
                                                  ncol=ncol, nrow=nrow,
                                                  values=vals)
            nonrot.resample(surf)
            usesurf = nonrot

        # make a copy so original numpy is not altered!
        self.logger.info('Transpose values...')
        zi = ma.transpose(usesurf.values)

        # store the current mask:
        zimask = ma.getmask(zi).copy()

        xi = np.linspace(usesurf.xmin, usesurf.xmax, zi.shape[1])
        yi = np.linspace(usesurf.ymin, usesurf.ymax, zi.shape[0])

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

            # note use surf.min, not usesurf.min here ...
            notetxt = ('Note: map values are truncated from [' +
                       str(surf.values.min()) + ', ' +
                       str(surf.values.max()) + '] ' +
                       'to interval [' +
                       str(minvalue) + ', ' + str(maxvalue) + ']')

            self._fig.text(0.99, 0.02, notetxt, ha='right', va='center',
                           fontsize=8)

        self.logger.info('Legendticks: {}'.format(legendticks))

        if minvalue is None:
            minvalue = usesurf.values.min()

        if maxvalue is None:
            maxvalue = usesurf.values.max()

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
            if logarithmic is False:
                locator = None
                ticks=legendticks
                im = self._ax.contourf(xi, yi, zi, uselevels, locator=locator,
                                       cmap=self.colormap)

            else:
                self.logger.info('use LogLocator')
                locator = ticker.LogLocator()
                ticks = None
                uselevels = None
                im = self._ax.contourf(xi, yi, zi, locator=locator,
                                       cmap=self.colormap)

            self._fig.colorbar(im, ticks=ticks)
        except ValueError as err:
            self.logger.warning('Could not make plot: {}'.format(err))

        plt.gca().set_aspect('equal', adjustable='box')

    def plot_faults(self, fpoly, idname='ID', color='k', edgecolor='k',
                    alpha=0.7, linewidth=0.8):
        """Plot the faults

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

        for name, group in aff:

            # make a dataframe sorted on faults (groupname)
            myfault = aff.get_group(name)

            # make a list [(X,Y) ...]; note PY3 need the
            # list before the zip!
            if six.PY3:
                af = list(zip(myfault['X'].values,
                              myfault['Y'].values))
            else:
                # make a numpy (X,Y) list from pandas series
                af = myfault[['X', 'Y']].values

            p = mplp.Polygon(af, alpha=0.7, color=color, ec=edgecolor,
                             lw=linewidth)

            if p.get_closed():
                self._ax.add_artist(p)
            else:
                print("A polygon is not closed...")

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

    def savefig(self, filename, fformat='png'):
        """Call to matplotlib.pyplot savefig().

        Returns:
            True of plotting is done; otherwise False
        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            plt.savefig(filename, format=fformat)
            plt.close(self._fig)
            return True
        else:
            self.logger.warning("Nothing to plot (well outside Z range?)")
            return False
