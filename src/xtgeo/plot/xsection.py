"""Module for fast XSection plots of wells/surfaces, using matplotlib."""

from __future__ import print_function

from collections import OrderedDict

import numpy.ma as ma
import matplotlib.pyplot as plt

from xtgeo.common import XTGeoDialog
from xtgeo.plot import BasePlot

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class XSection(BasePlot):
    """Class for plotting a cross-section of a well.

    Args:
        zmin (float): Upper level of the plot (top Y axis).
        zmax (float): Lower level of the plot (bottom Y axis).
        well (Well): XTGeo well object.
        surfaces (list): List of XTGeo RegularSurface objects
        surfacenames (list): List of surface names (str)for legend
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
        self._legendtitle = 'Zones'
        self._legendsize = 7

        if colormap is None:
            self._colormap = plt.cm.viridis
        else:
            self.define_colormap(colormap)

        self._colormap_facies = self.define_any_colormap('xtgeo')
        self._colormap_facies_dict = {idx: idx for idx in range(100)}

        self._colormap_perf = self.define_any_colormap('xtgeo')
        self._colormap_perf_dict = {idx: idx for idx in range(100)}

        logger.info('Ran __init__ ...')
        logger.info('Colormap is {}'.format(self._colormap))

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def pagesize(self):
        """Returns page size."""
        return self._pagesize

    @property
    def colormap_facies(self):
        """Set or get the facies colormap"""
        return self._colormap_facies

    @colormap_facies.setter
    def colormap_facies(self, cmap):
        self._colormap_facies = self.define_any_colormap(cmap)

    @property
    def colormap_perf(self):
        """Set or get the perforations colormap"""
        return self._colormap_perf

    @colormap_perf.setter
    def colormap_perf(self, cmap):
        self._colormap_perf = self.define_any_colormap(cmap)

    @property
    def colormap_facies_dict(self):
        """Set or get the facies colormap actual dict table"""
        return self._colormap_facies_dict

    @colormap_facies_dict.setter
    def colormap_facies_dict(self, xdict):
        if not isinstance(xdict, dict):
            raise ValueError('Input is not a dict')

        # if not all(isinstance(item, int) for item in list(xdict.values)):
        #     raise ValueError('Dict values is a list, but some elems are '
        #                      'not ints!')

        self._colormap_facies_dict = xdict

    @property
    def colormap_perf_dict(self):
        """Set or get the perf colormap actual dict table"""
        return self._colormap_perf_dict

    @colormap_perf_dict.setter
    def colormap_perf_dict(self, xdict):
        if not isinstance(xdict, dict):
            raise ValueError('Input is not a dict')

        # if not all(isinstance(item, int) for item in list(xdict.values)):
        #     raise ValueError('Dict values is a list, but some elems are '
        #                      'not ints!')

        self._colormap_perf_dict = xdict

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
        ax1 = OrderedDict()

        ax1['main'] = plt.subplot2grid((20, 28), (0, 0), rowspan=20,
                                       colspan=23)

        ax2 = plt.subplot2grid((20, 28), (10, 23), rowspan=5,
                               colspan=5)
        ax3 = plt.subplot2grid((20, 28), (15, 23), rowspan=5,
                               colspan=5)
        # indicate A to B
        plt.text(0.02, 0.98, 'A', ha='left', va='top',
                 transform=ax1['main'].transAxes, fontsize=8)
        plt.text(0.98, 0.98, 'B', ha='right', va='top',
                 transform=ax1['main'].transAxes, fontsize=8)

        # title her:
        if title is not None:
            plt.text(0.5, 1.09, title, ha='center', va='center',
                     transform=ax1['main'].transAxes, fontsize=18)

        if subtitle is not None:
            ax1['main'].set_title(subtitle, size=14)

        if infotext is not None:
            plt.text(-0.11, -0.11, infotext, ha='left', va='center',
                     transform=ax1['main'].transAxes, fontsize=6)

        ax1['main'].set_ylabel('Depth', fontsize=12.0)
        ax1['main'].set_xlabel('Length along well', fontsize=12)

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

    def plot_well(self, zonelogname='ZONELOG', facieslogname=None,
                  perflogname=None):
        """Input an XTGeo Well object and plot it."""
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

        # plot the perflog, if any, first
        if perflogname:
            ax, bba = self._currentax(axisname='perf')
            self._plot_well_perflog(df, ax, bba, zv, hv, perflogname)

        # plot the facies, if any, behind the trajectory; ie. first or second
        if facieslogname:
            ax, bba = self._currentax(axisname='facies')
            self._plot_well_faclog(df, ax, bba, zv, hv, facieslogname)

        axx, bbxa = self._currentax(axisname='well')

        self._plot_well_traj(df, axx, zv, hv)

        if zonelogname:
            self._plot_well_zlog(df, axx, zv, hv, zonelogname)

    def _plot_well_traj(self, df, ax, zv, hv):
        """Plot the trajectory as a black line"""

        zv_copy = ma.masked_where(zv < self._zmin, zv)
        hv_copy = ma.masked_where(zv < self._zmin, hv)

        ax.plot(hv_copy, zv_copy, linewidth=6,
                c='black')

    def _plot_well_zlog(self, df, ax, zv, hv, zonelogname):
        """Plot the zone log as colored segments."""

        if zonelogname not in df.columns:
            return

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

    def _plot_well_faclog(self, df, ax, bba, zv, hv, facieslogname,
                          facieslist=None):
        """Plot the facies log as colored segments.

        Args:
            df (dataframe): The Well dataframe.
            ax (axes): The ax plot object.
            zv (ndarray): The numpy Z TVD array.
            hv (ndarray): The numpy Length  array.
            facieslogname (str): name of the facies log.
            facieslist (list): List of values to be plotted as facies
        """

        if facieslogname not in df.columns:
            return

        cmap = self.colormap_facies
        ctable = self.get_any_colormap_as_table(cmap)
        idx = self.colormap_facies_dict

        if facieslist is None:
            facieslist = list(idx.keys())

        fa = df[facieslogname].values

        # let the part with FACIESLOG have a colour
        for facies in facieslist:

            color = ctable[idx[facies]]

            zv_copy = ma.masked_where(fa != facies, zv)
            hv_copy = ma.masked_where(fa != facies, hv)

            fname = self._well.get_logrecord_codename(facieslogname, facies)

            ax.plot(hv_copy, zv_copy, linewidth=10, c=color,
                    label=str(fname))

        self._drawlegend(ax, bba, title='Facies')

    def _plot_well_perflog(self, df, ax, bba, zv, hv, perflogname,
                           perflist=None):
        """Plot the perforation log as colored segments.

        Args:
            df (dataframe): The Well dataframe.
            ax (axes): The ax plot object.
            zv (ndarray): The numpy Z TVD array.
            hv (ndarray): The numpy Length  array.
            perflogname (str): name of the perforation log.
            perflist (list): List of values to be plotted as PERF
        """

        if perflogname not in df.columns:
            return

        cmap = self.colormap_perf
        ctable = self.get_any_colormap_as_table(cmap)
        idx = self.colormap_perf_dict

        if perflist is None:
            perflist = list(idx.keys())

        prf = df[perflogname].values

        # let the part with ZONELOG have a colour
        for perf in perflist:

            color = ctable[idx[perf]]

            zv_copy = ma.masked_where(perf != prf, zv)
            hv_copy = ma.masked_where(perf != prf, hv)

            ax.plot(hv_copy, zv_copy, linewidth=15, c=color, label='PERF')

        self._drawlegend(ax, bba, title='Perforations')

    def _drawlegend(self, ax, bba, title=None):

        leg = ax.legend(loc='upper left', bbox_to_anchor=bba,
                        prop={'size': self._legendsize}, title=title,
                        handlelength=2)

        for myleg in leg.get_lines():
            myleg.set_linewidth(5)


    def _currentax(self, axisname='main'):
        """Keep track of current axis; is needed as one new legend need one
        new axis.
        """
        # for multiple legends, bba is dynamic
        bbapos = {
            'main': (1.22, 1.12, 1, 0),
            'contacts': (1.01, 1.12),
            'facies': (1.01, 1.00),
            'perf': (1.01, 0.7)
        }

        ax1 = self._ax1

        if axisname != 'main':
            ax1[axisname] = self._ax1['main'].twinx()

            # invert min,max to invert the Y axis
            ax1[axisname].set_ylim([self._zmax, self._zmin])

            ax1[axisname].set_yticklabels([])
            ax1[axisname].tick_params(axis='y', direction='in')

        ax = self._ax1[axisname]

        if axisname in bbapos:
            bba = bbapos[axisname]
        else:
            bba = (1.22, 0.5)

        return ax, bba

    def plot_surfaces(self, fill=False, surfaces=None, surfacenames=None,
                      colormap=None, onecolor=None, linewidth=1.0, legend=True,
                      legendtitle=None, fancyline=False, axisname='main'):
        """Input a surface list (ordered from top to base) , and plot them."""

        ax, bba = self._currentax(axisname=axisname)
        # either use surfaces from __init__, or override with surfaces
        # speciefied here

        if surfaces is None:
            surfaces = self._surfaces
            surfacenames = self._surfacenames

        surfacenames = [surf.name for surf in surfaces]

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
            usecolor = colortable[i]
            if onecolor:
                usecolor = onecolor
            if not fill:
                hfence = surfaces[i].get_fence(wfence)
                if fancyline:
                    xcol = 'white'
                    c = usecolor
                    if c[0] + c[1] + c[2] > 1.5:
                        xcol = 'black'
                    ax.plot(hfence[:, 3], hfence[:, 2],
                            linewidth=1.2 * linewidth,
                            c=xcol)
                ax.plot(hfence[:, 3], hfence[:, 2], linewidth=linewidth,
                        c=usecolor, label=slegend[i])
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

        if legend:
            self._drawlegend(ax, bba, title=legendtitle)

        if axisname != 'main':
            ax.set_yticklabels([])

        ax.tick_params(axis='y', direction='in')

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
