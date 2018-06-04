"""Module for fast XSection plots of wells/surfaces, using matplotlib."""

from __future__ import print_function

import numpy.ma as ma
import matplotlib.pyplot as plt

from xtgeo.common import XTGeoDialog
from xtgeo.plot import BasePlot

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class XSection(BasePlot):
    """Class for plotting a cross-section of a well.

    Args:
        zmin (float): Upper level of the plot (top Y axis)
        zmax (float): Lower level of the plot (bottom Y axis)
        well (obj): XTgeo well object
        surfaces (list of obj): List of XTGeo RegularSurface objects
        surfacenames (list of str): List of surface names for legend
        colormap (str): Name of colormap, e.g. 'Set1'. Default is 'xtgeo'
        outline (obj): XTGeo Polygons object
        tight (bool): True for tight_layout (False is default)

    """

    def __init__(self, zmin=0, zmax=9999, well=None, surfaces=None,
                 colormap=None, zonelogshift=0, surfacenames=None,
                 outline=None, tight=False):

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        logger = xtg.functionlogger(clsname)
        self._xtg = XTGeoDialog()

        self._zmin = zmin
        self._zmax = zmax
        self._well = well
        self._surfaces = surfaces
        self._surfacenames = surfacenames
        self._zonelogshift = zonelogshift
        self._tight = tight
        self._outline = outline

        self._pagesize = 'A4'
        self._wfence = None
        self._showok = True  # to indicate if plot is OK to show
        self._surfaceplot_count = 0
        self._legendtitle = 'Zones'

        if colormap is None:
            self._colormap = plt.cm.viridis
        else:
            self.define_colormap(colormap)

        logger.info('Ran __init__ ...')
        logger.info('Colormap is {}'.format(self._colormap))

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
        self._fig, __ = plt.subplots(figsize=(11.69 * figscaling,
                                              8.27 * figscaling))
        ax1 = []

        ax1.append(plt.subplot2grid((20, 28), (0, 0), rowspan=20, colspan=22))

        ax2 = plt.subplot2grid((20, 28), (17, 22), rowspan=3,
                               colspan=3)
        ax3 = plt.subplot2grid((20, 28), (17, 25), rowspan=3,
                               colspan=3)
        # indicate A to B
        plt.text(0.02, 0.98, 'A', ha='left', va='top',
                 transform=ax1[0].transAxes, fontsize=8)
        plt.text(0.98, 0.98, 'B', ha='right', va='top',
                 transform=ax1[0].transAxes, fontsize=8)

        # title her:
        if title is not None:
            plt.text(0.5, 1.09, title, ha='center', va='center',
                     transform=ax1[0].transAxes, fontsize=18)

        if subtitle is not None:
            ax1[0].set_title(subtitle, size=14)

        if infotext is not None:
            plt.text(-0.11, -0.11, infotext, ha='left', va='center',
                     transform=ax1[0].transAxes, fontsize=6)

        ax1[0].set_ylabel('Depth', fontsize=12.0)
        ax1[0].set_xlabel('Length along well', fontsize=12)

        ax2.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, left=False, labelbottom=False,
                        labeltop=False, labelright=False, labelleft=False)

        ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, left=False, labelbottom=False,
                        labeltop=False, labelright=False, labelleft=False)

        # need these also, a bug in functions above?
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax3.xaxis.set_major_formatter(plt.NullFormatter())
        ax3.yaxis.set_major_formatter(plt.NullFormatter())

        self._ax1 = ax1
        self._ax2 = ax2
        self._ax3 = ax3

    def plot_well(self, zonelogname='ZONELOG'):
        """Input an XTGeo Well object and plot it."""
        ax = self._ax1[self._surfaceplot_count - 1]
        wo = self._well

        # reduce the well data by Pandas operations
        df = wo.dataframe
        wo.dataframe = df[df['Z_TVDSS'] > self._zmin]

        # Create a relative XYLENGTH vector (0.0 where well starts)
        wo.create_relative_hlen()

        df = wo.dataframe
        if df.empty:
            self._showok = False
            return

        # get the well trajectory (numpies) as copy
        zv = df['Z_TVDSS'].values.copy()
        hv = df['R_HLEN'].values.copy()
        self._plot_well_traj(df, ax, zv, hv)

        haszone = False
        if zonelogname in df.columns:
            haszone = True

        if haszone:
            self._plot_well_zlog(df, ax, zv, hv, zonelogname)

    def _plot_well_traj(self, df, ax, zv, hv):
        """Plot the trajectory as a black line"""

        zv_copy = ma.masked_where(zv < self._zmin, zv)
        hv_copy = ma.masked_where(zv < self._zmin, hv)

        ax.plot(hv_copy, zv_copy, linewidth=6,
                c='black')

    def _plot_well_zlog(self, df, ax, zv, hv, zonelogname):
        """Plot the zone log as colored segments."""
        zo = df[zonelogname].values
        zomin = 0
        zomax = 0
        try:
            zomin = int(df[zonelogname].min())
            zomax = int(df[zonelogname].max())
        except ValueError:
            self._showok = False
            return

        logger.info('ZONELOG min - max is {} - {}'.format(zomin, zomax))

        zshift = 0
        if self._zonelogshift != 0:
            zshift = self._zonelogshift

        # let the part with ZONELOG have a colour
        ctable = self.get_colormap_as_table()
        for zone in range(zomin, zomax + 1):

            # the minus one since zone no 1 use color entry no 0
            if (zone + zshift - 1) < 0:
                color = (0.9, 0.9, 0.9)
            else:
                color = ctable[zone + zshift - 1]

            zv_copy = ma.masked_where(zo != zone, zv)
            hv_copy = ma.masked_where(zo != zone, hv)

            logger.debug('Zone is {}, color no is {}'.
                         format(zone, zone + zshift - 1))
            ax.plot(hv_copy, zv_copy, linewidth=4, c=color)

    def plot_surfaces(self, fill=False, surfaces=None, surfacenames=None,
                      colormap=None, linewidth=1.0, legendtitle=None,
                      fancyline=False):
        """Input a surface list (ordered from top to base) , and plot them."""

        ax1 = self._ax1
        self._surfaceplot_count += 1

        bba = (1, 1.09)
        nc = self._surfaceplot_count
        if nc > 1:
            ax1.append(ax1[0].twinx())
            bba = (1 + (nc - 1) * 0.14, 1.09)

        ax = self._ax1[self._surfaceplot_count - 1]

        # either use surfaces from __init__, or override with surfaces
        # speciefied here

        if surfaces is None:
            surfaces = self._surfaces
            surfacenames = self._surfacenames

        if legendtitle is None:
            legendtitle = self._legendtitle

        if colormap is None:
            colormap = self._colormap
        else:
            self.define_colormap(colormap)

        nlen = len(surfaces)

        # legend
        slegend = []
        if surfacenames is None:
            for i in range(nlen):
                slegend.append('Surf {}'.format(i))

        else:
            # do a check
            if len(surfacenames) != nlen:
                msg = ('Wrong number of entries in surfacenames! '
                       'Number of names is {} while number of files '
                       'is {}'.format(len(surfacenames), nlen))
                logger.critical(msg)
                raise SystemExit(msg)
            else:
                slegend = surfacenames

        if self._colormap.N < nlen:
            msg = 'Too few colors in color table vs number of surfaces'
            raise SystemExit(msg)

        # need to resample the surface along the well trajectory
        # create a sampled fence from well path, and include extension
        # This fence is numpy vector [[XYZ ...]]
        wfence = self._well.get_fence_polyline(sampling=20, extend=5,
                                               tvdmin=self._zmin)

        self._wfence = wfence

        # sample the horizon to the fence:
        colortable = self.get_colormap_as_table()
        for i in range(nlen):
            logger.info(i)
            if not fill:
                hfence = surfaces[i].get_fence(wfence)
                if fancyline:
                    xcol = 'white'
                    c = colortable[i]
                    if c[0] + c[1] + c[2] > 1.5:
                        xcol = 'black'
                    ax.plot(hfence[:, 3], hfence[:, 2],
                            linewidth=1.2 * linewidth,
                            c=xcol)
                ax.plot(hfence[:, 3], hfence[:, 2], linewidth=linewidth,
                        c=colortable[i], label=slegend[i])
                if fancyline:
                    ax.plot(hfence[:, 3], hfence[:, 2],
                            linewidth=0.3 * linewidth,
                            c=xcol)
            else:
                # need copy() .. why?? found by debugging...
                hfence1 = surfaces[i].get_fence(wfence).copy()
                x1 = hfence1[:, 3]
                y1 = hfence1[:, 2]
                if i < (nlen - 1):
                    hfence2 = surfaces[i + 1].get_fence(wfence).copy()
                    y2 = hfence2[:, 2]
                else:
                    y2 = y1.copy()

                ax.plot(x1, y1, linewidth=0.1 * linewidth, c='black')
                ax.fill_between(x1, y1, y2,
                                facecolor=colortable[i],
                                label=slegend[i])

        # invert min,max to invert the Y axis
        ax.set_ylim([self._zmax, self._zmin])

        ax.legend(loc='upper left', bbox_to_anchor=bba,
                  prop={'size': 7}, title=legendtitle)

        if self._surfaceplot_count > 1:
            ax.set_yticklabels([])

        ax.tick_params(axis='y', direction='in')
        self._ax1 = ax1

    def plot_wellmap(self):
        """
        Plot well location map as local view
        """
        ax = self._ax2
        if self._wfence is not None:

            xwellarray = self._well.dataframe['X_UTME'].values
            ywellarray = self._well.dataframe['Y_UTMN'].values

            ax.plot(xwellarray, ywellarray,
                    linewidth=4, c='cyan')

            ax.plot(self._wfence[:, 0], self._wfence[:, 1],
                    linewidth=1, c='black')
            ax.annotate('A', xy=(self._wfence[0, 0], self._wfence[0, 1]),
                        fontsize=8)
            ax.annotate('B', xy=(self._wfence[-1, 0], self._wfence[-1, 1]),
                        fontsize=8)
            ax.set_aspect('equal', 'datalim')

    def plot_map(self):
        """Plot well location map as an overall view (with field outline)."""

        ax = self._ax3
        if self._outline is not None and self._wfence is not None:

            xp = self._outline.dataframe['X_UTME'].values
            yp = self._outline.dataframe['Y_UTMN'].values
            ip = self._outline.dataframe['POLY_ID'].values

            ax.plot(self._wfence[:, 0], self._wfence[:, 1],
                    linewidth=3, c='red')

            for i in range(int(ip.min()), int(ip.max()) + 1):
                xpc = xp.copy()[ip == i]
                ypc = yp.copy()[ip == i]
                if len(xpc) > 1:
                    ax.plot(xpc, ypc, linewidth=0.3, c='black')

            ax.set_aspect('equal', 'datalim')

    def show(self):
        """
        Call to matplotlib.pyplot show().

        Returns:
            True of plotting is done; otherwise False
        """
        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            logger.info('Calling plt show method...')
            plt.show()
            return True
        else:
            logger.warning('Nothing to plot (well outside Z range?)')
            return False

    def savefig(self, filename, fformat='png'):
        """Call to matplotlib.pyplot savefig().

        Args:
            filename: Root name to export to
            fformat: Either 'png' or 'svg'

        Returns:
            True of plotting is done; otherwise False
        """
        logger.info('FORMAT is {}'.format(fformat))

        if self._tight:
            self._fig.tight_layout()

        if self._showok:
            plt.savefig(filename, format=fformat)
            plt.close(self._fig)
            return True
        else:
            logger.warning('Nothing to plot (well outside Z range?)')
            return False
