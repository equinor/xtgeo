"""Module for fast XSection plots of wells/surfaces etc, using matplotlib."""

from __future__ import print_function

from collections import OrderedDict

import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from xtgeo.common import XTGeoDialog
from xtgeo.xyz import Polygons

from .baseplot import BasePlot

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class XSection(BasePlot):
    """Class for plotting a cross-section of a well.

    Args:
        zmin (float): Upper level of the plot (top Y axis).
        zmax (float): Lower level of the plot (bottom Y axis).
        well (Well): XTGeo well object.
        surfaces (list): List of XTGeo RegularSurface objects
        surfacenames (list): List of surface names (str) for legend
        cube (Cube): A XTGeo Cube instance
        colormap (str): Name of colormap, e.g. 'Set1'. Default is 'xtgeo'
        outline (obj): XTGeo Polygons object

    """

    def __init__(
        self,
        zmin=0,
        zmax=9999,
        well=None,
        surfaces=None,
        colormap=None,
        zonelogshift=0,
        surfacenames=None,
        cube=None,
        outline=None,
    ):

        super(XSection, self).__init__()

        self._zmin = zmin
        self._zmax = zmax
        self._well = well
        self._surfaces = surfaces
        self._surfacenames = surfacenames
        self._cube = cube
        self._zonelogshift = zonelogshift
        self._outline = outline

        self._pagesize = "A4"
        self._wfence = None
        self._legendtitle = "Zones"
        self._legendsize = 5

        self._ax1 = None
        self._ax2 = None
        self._ax3 = None

        self._colormap_cube = None
        self._colorlegend_cube = False

        if colormap is None:
            self._colormap = plt.cm.get_cmap("viridis")
        else:
            self.define_colormap(colormap)

        self._colormap_facies = self.define_any_colormap("xtgeo")
        self._colormap_facies_dict = {idx: idx for idx in range(100)}

        self._colormap_perf = self.define_any_colormap("xtgeo")
        self._colormap_perf_dict = {idx: idx for idx in range(100)}

        logger.info("Ran __init__ ...")
        logger.info("Colormap is %s", self._colormap)

    # ==================================================================================
    # Properties
    # Notice additonal props in base class
    # ==================================================================================
    @property
    def pagesize(self):
        """Returns page size."""
        return self._pagesize

    @property
    def legendsize(self):
        """Returns or set the legend size"""
        return self._legendsize

    @legendsize.setter
    def legendsize(self, lsize):
        """Returns or set the legend size"""
        self._legendsize = lsize

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
            raise ValueError("Input is not a dict")

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
            raise ValueError("Input is not a dict")

        # if not all(isinstance(item, int) for item in list(xdict.values)):
        #     raise ValueError('Dict values is a list, but some elems are '
        #                      'not ints!')

        self._colormap_perf_dict = xdict

    # ==================================================================================
    # Functions methods (public)
    # ==================================================================================

    def canvas(self, title=None, subtitle=None, infotext=None, figscaling=1.0):
        """Prepare the canvas to plot on, with title and subtitle.

        Args:
            title (str, optional): Title of plot.
            subtitle (str, optional): Sub title of plot.
            infotext (str, optional): Text to be written as info string.
            figscaling (str, optional): Figure scaling, default is 1.0


        """

        # overriding the base class canvas

        plt.rcParams["axes.xmargin"] = 0  # fill the plot margins

        # self._fig, (ax1, ax2) = plt.subplots(2, figsize=(11.69, 8.27))
        self._fig, __ = plt.subplots(figsize=(11.69 * figscaling, 8.27 * figscaling))
        ax1 = OrderedDict()

        ax1["main"] = plt.subplot2grid((20, 28), (0, 0), rowspan=20, colspan=23)

        ax2 = plt.subplot2grid((20, 28), (10, 23), rowspan=5, colspan=5)
        ax3 = plt.subplot2grid((20, 28), (15, 23), rowspan=5, colspan=5)
        # indicate A to B
        plt.text(
            0.02,
            0.98,
            "A",
            ha="left",
            va="top",
            transform=ax1["main"].transAxes,
            fontsize=8,
        )
        plt.text(
            0.98,
            0.98,
            "B",
            ha="right",
            va="top",
            transform=ax1["main"].transAxes,
            fontsize=8,
        )

        # title here:
        if title is not None:
            plt.text(
                0.5,
                1.09,
                title,
                ha="center",
                va="center",
                transform=ax1["main"].transAxes,
                fontsize=18,
            )

        if subtitle is not None:
            ax1["main"].set_title(subtitle, size=14)

        if infotext is not None:
            plt.text(
                -0.11,
                -0.11,
                infotext,
                ha="left",
                va="center",
                transform=ax1["main"].transAxes,
                fontsize=6,
            )

        ax1["main"].set_ylabel("Depth", fontsize=12.0)
        ax1["main"].set_xlabel("Length along well", fontsize=12)

        ax2.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            right=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False,
        )

        ax3.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            right=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False,
        )

        # need these also, a bug in functions above?
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax3.xaxis.set_major_formatter(plt.NullFormatter())
        ax3.yaxis.set_major_formatter(plt.NullFormatter())

        self._ax1 = ax1
        self._ax2 = ax2
        self._ax3 = ax3

    def plot_well(
        self,
        zonelogname="ZONELOG",
        facieslogname=None,
        perflogname=None,
        wellcrossings=None,
    ):
        """Input an XTGeo Well object and plot it."""
        wo = self._well

        # reduce the well data by Pandas operations
        dfr = wo.dataframe
        wo.dataframe = dfr[dfr["Z_TVDSS"] > self._zmin]

        # Create a relative XYLENGTH vector (0.0 where well starts)
        wo.create_relative_hlen()

        dfr = wo.dataframe
        if dfr.empty:
            self._showok = False
            return

        # get the well trajectory (numpies) as copy
        zv = dfr["Z_TVDSS"].values.copy()
        hv = dfr["R_HLEN"].values.copy()

        # plot the perflog, if any, first
        if perflogname:
            ax, bba = self._currentax(axisname="perf")
            self._plot_well_perflog(dfr, ax, bba, zv, hv, perflogname)

        # plot the facies, if any, behind the trajectory; ie. first or second
        if facieslogname:
            ax, bba = self._currentax(axisname="facies")
            self._plot_well_faclog(dfr, ax, bba, zv, hv, facieslogname)

        axx, _bbxa = self._currentax(axisname="well")
        self._plot_well_traj(axx, zv, hv)

        if zonelogname:
            self._plot_well_zlog(dfr, axx, zv, hv, zonelogname)

        if wellcrossings is not None and wellcrossings.empty:
            wellcrossings = None

        if wellcrossings is not None:
            self._plot_well_crossings(dfr, axx, wellcrossings)

    def _plot_well_traj(self, ax, zv, hv):
        """Plot the trajectory as a black line"""

        zv_copy = ma.masked_where(zv < self._zmin, zv)
        hv_copy = ma.masked_where(zv < self._zmin, hv)

        ax.plot(hv_copy, zv_copy, linewidth=6, c="black")

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

        logger.info("ZONELOG min - max is %s - %s", zomin, zomax)

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

            logger.debug("Zone is %s, color no is %s", zone, zone + zshift - 1)

            ax.plot(hv_copy, zv_copy, linewidth=4, c=color, solid_capstyle="butt")

    def _plot_well_faclog(self, df, ax, bba, zv, hv, facieslogname, facieslist=None):
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

        frecord = self._well.get_logrecord(facieslogname)
        frecord = {val: fname for val, fname in frecord.items() if val >= 0}

        if facieslist is None:
            facieslist = list(frecord.keys())

        fa = df[facieslogname].values

        for fcc in frecord:

            if isinstance(idx[fcc], str):
                color = idx[fcc]
            else:
                color = ctable[idx[fcc]]

            zv_copy = ma.masked_where(fa != fcc, zv)
            hv_copy = ma.masked_where(fa != fcc, hv)

            _myline, = ax.plot(
                hv_copy,
                zv_copy,
                linewidth=9,
                c=color,
                label=frecord[fcc],
                solid_capstyle="butt",
            )

        self._drawlegend(ax, bba, title="Facies")

    def _plot_well_perflog(self, df, ax, bba, zv, hv, perflogname, perflist=None):
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

        precord = self._well.get_logrecord(perflogname)
        precord = {val: pname for val, pname in precord.items() if val >= 0}

        idx = self.colormap_perf_dict

        if perflist is None:
            perflist = list(precord.keys())

        prf = df[perflogname].values

        # let the part with ZONELOG have a colour
        for perf in perflist:

            if isinstance(idx[perf], str):
                color = idx[perf]
            else:
                color = ctable[idx[perf]]

            zv_copy = ma.masked_where(perf != prf, zv)
            hv_copy = ma.masked_where(perf != prf, hv)

            ax.plot(
                hv_copy,
                zv_copy,
                linewidth=15,
                c=color,
                label=precord[perf],
                solid_capstyle="butt",
            )

        self._drawlegend(ax, bba, title="Perforations")

    @staticmethod
    def _plot_well_crossings(dfr, ax, wcross):
        """Plot well crossing based on dataframe (wcross)

        The well crossing coordinates are identified for this well,
        and then it is looking for the closest coordinate. Given this
        coordinate, a position is chosen.

        The pandas dataframe wcross shall have the following columns:

        * Name of crossing wells named CWELL
        * Coordinate X named X_UTME
        * Coordinate Y named Y_UTMN
        * Coordinate Z named Z_TVDSS

        Args:
            dfr: Well dataframe
            ax: current axis
            wcross: A pandas dataframe with precomputed well crossings
        """

        placings = {
            0: (40, 40),
            1: (40, -20),
            2: (-30, 30),
            3: (30, 20),
            4: (-40, 30),
            5: (-20, 40),
        }

        for index, row in wcross.iterrows():
            xcoord = row.X_UTME
            ycoord = row.Y_UTMN

            dfrc = dfr.copy()

            dfrc["DLEN"] = pow(
                pow(dfrc.X_UTME - xcoord, 2) + pow(dfrc.Y_UTMN - ycoord, 2), 0.5
            )

            minindx = dfrc.DLEN.idxmin()

            ax.scatter(
                dfrc.R_HLEN[minindx],
                row.Z_TVDSS,
                marker="o",
                color="black",
                s=70,
                zorder=100,
            )
            ax.scatter(
                dfrc.R_HLEN[minindx],
                row.Z_TVDSS,
                marker="o",
                color="orange",
                s=38,
                zorder=102,
            )

            modulo = index % 5

            ax.annotate(
                row.CWELL,
                size=6,
                xy=(dfrc.R_HLEN[minindx], row.Z_TVDSS),
                xytext=placings[modulo],
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=90"
                ),
                color="black",
            )

    def _drawlegend(self, ax, bba, title=None):

        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=bba,
            prop={"size": self._legendsize},
            title=title,
            handlelength=2,
        )

        for myleg in leg.get_lines():
            myleg.set_linewidth(5)

    def _currentax(self, axisname="main"):
        """Keep track of current axis; is needed as one new legend need one
        new axis.
        """
        # for multiple legends, bba is dynamic
        bbapos = {
            "main": (1.22, 1.12, 1, 0),
            "contacts": (1.01, 1.12),
            "second": (1.22, 0.50),
            "facies": (1.01, 1.00),
            "perf": (1.22, 0.45),
        }

        ax1 = self._ax1

        if axisname != "main":
            ax1[axisname] = self._ax1["main"].twinx()

            # invert min,max to invert the Y axis
            ax1[axisname].set_ylim([self._zmax, self._zmin])

            ax1[axisname].set_yticklabels([])
            ax1[axisname].tick_params(axis="y", direction="in")

        ax = self._ax1[axisname]

        bba = bbapos.get(axisname, (1.22, 0.5))

        return ax, bba

    def plot_cube(
        self,
        colormap="seismic",
        vmin=None,
        vmax=None,
        alpha=0.7,
        interpolation="gaussian",
        sampling="nearest",
    ):
        """Plot a cube backdrop.

        Args:
            colormap (ColorMap): Name of color map (default 'seismic')
            vmin (float): Minimum value in plot.
            vmax (float); Maximum value in plot
            alpha (float): Alpah blending number beween 0 and 1.
            interpolation (str): Interpolation for plotting, cf. matplotlib
                documentation on this. Also gaussianN is allowed, where
                N = 1..9.
            sampling (str): 'nearest' (default) or 'trilinear'

        Raises:
            ValueError: No cube is loaded

        """
        if self._cube is None:
            raise ValueError("Ask for plot cube, but noe cube is loaded")

        ax, _bba = self._currentax(axisname="main")

        if self._wfence is None:
            wfence = self._well.get_fence_polyline(
                sampling=20, extend=5, tvdmin=self._zmin
            )
            if wfence is False:
                return
            self._wfence = wfence

        zinc = self._cube.zinc / 2.0

        zvv = self._cube.get_randomline(
            self._wfence,
            zmin=self._zmin,
            zmax=self._zmax,
            zincrement=zinc,
            sampling=sampling,
        )

        h1, h2, v1, v2, arr = zvv

        # if vmin is not None or vmax is not None:
        #     arr = np.clip(arr, vmin, vmax)

        logger.info("Number of masked elems: %s", ma.count_masked(arr))

        if self._colormap_cube is None:
            if colormap is None:
                colormap = "seismic"
            self._colormap_cube = self.define_any_colormap(colormap)

        if "gaussian" in interpolation:  # allow gaussian3 etc
            nnv = interpolation[-1]
            try:
                nnv = int(nnv)
                arr = gaussian_filter(arr, nnv)
                interpolation = "none"
            except ValueError:
                interpolation = "gaussian"

        img = ax.imshow(
            arr,
            cmap=self._colormap_cube,
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
            extent=(h1, h2, v2, v1),
            aspect="auto",
            alpha=alpha,
        )

        logger.info("Actual VMIN and VMAX: %s", img.get_clim())
        # steer this?
        if self._colorlegend_cube:
            self._fig.colorbar(img, ax=ax)

    def plot_surfaces(
        self,
        fill=False,
        surfaces=None,
        surfacenames=None,
        colormap=None,
        onecolor=None,
        linewidth=1.0,
        legend=True,
        legendtitle=None,
        fancyline=False,
        axisname="main",
        gridlines=False,
    ):  # pylint: disable=too-many-branches, too-many-statements

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
                slegend.append("Surf {}".format(i))

        else:
            # do a check
            if len(surfacenames) != nlen:
                msg = (
                    "Wrong number of entries in surfacenames! "
                    "Number of names is {} while number of files "
                    "is {}".format(len(surfacenames), nlen)
                )
                logger.critical(msg)
                raise SystemExit(msg)

            slegend = surfacenames

        if self._colormap.N < nlen:
            msg = "Too few colors in color table vs number of surfaces"
            raise SystemExit(msg)

        # need to resample the surface along the well trajectory
        # create a sampled fence from well path, and include extension
        # This fence is numpy vector [[XYZ ...]]
        if self._wfence is None:
            wfence = self._well.get_fence_polyline(
                sampling=20, extend=5, tvdmin=self._zmin
            )
            if wfence is False:
                return

            self._wfence = wfence

        if self._wfence is False:
            return

        # sample the horizon to the fence:
        colortable = self.get_colormap_as_table()
        for i in range(nlen):
            usecolor = colortable[i]
            if onecolor:
                usecolor = onecolor
            if not fill:
                hfence = surfaces[i].get_fence(self._wfence)
                if fancyline:
                    xcol = "white"
                    cxx = usecolor
                    if cxx[0] + cxx[1] + cxx[2] > 1.5:
                        xcol = "black"
                    ax.plot(
                        hfence[:, 3], hfence[:, 2], linewidth=1.2 * linewidth, c=xcol
                    )
                ax.plot(
                    hfence[:, 3],
                    hfence[:, 2],
                    linewidth=linewidth,
                    c=usecolor,
                    label=slegend[i],
                )
                if fancyline:
                    ax.plot(
                        hfence[:, 3], hfence[:, 2], linewidth=0.3 * linewidth, c=xcol
                    )
            else:
                # need copy() .. why?? found by debugging...
                hfence1 = surfaces[i].get_fence(self._wfence).copy()
                x1 = hfence1[:, 3]
                y1 = hfence1[:, 2]
                if i < (nlen - 1):
                    hfence2 = surfaces[i + 1].get_fence(self._wfence).copy()
                    y2 = hfence2[:, 2]
                else:
                    y2 = y1.copy()

                ax.plot(x1, y1, linewidth=0.1 * linewidth, c="black")
                ax.fill_between(x1, y1, y2, facecolor=colortable[i], label=slegend[i])

        # invert min,max to invert the Y axis
        ax.set_ylim([self._zmax, self._zmin])

        if legend:
            self._drawlegend(ax, bba, title=legendtitle)

        if axisname != "main":
            ax.set_yticklabels([])

        ax.tick_params(axis="y", direction="in")

        if axisname == "main" and gridlines:
            ax.grid(color="grey", linewidth=0.2)

    def plot_wellmap(self, otherwells=None, expand=1):
        """Plot well map as local view, optionally with nearby wells.

        Args:
            otherwells (list of Polygons): List of surrounding wells to plot,
                these wells are repr as Polygons instances, one per well.
            expand (float): Plot axis expand factor (default is 1); larger
                values may be used if other wells are plotted.


        """
        ax = self._ax2

        if self._wfence is not None:

            xwellarray = self._well.dataframe["X_UTME"].values
            ywellarray = self._well.dataframe["Y_UTMN"].values

            ax.plot(xwellarray, ywellarray, linewidth=4, c="cyan")

            ax.plot(self._wfence[:, 0], self._wfence[:, 1], linewidth=1, c="black")
            ax.annotate("A", xy=(self._wfence[0, 0], self._wfence[0, 1]), fontsize=8)
            ax.annotate("B", xy=(self._wfence[-1, 0], self._wfence[-1, 1]), fontsize=8)
            ax.set_aspect("equal", "datalim")

            left, right = ax.get_xlim()
            xdiff = right - left
            bottom, top = ax.get_ylim()
            ydiff = top - bottom

            ax.set_xlim(left - (expand - 1.0) * xdiff, right + (expand - 1.0) * xdiff)
            ax.set_ylim(bottom - (expand - 1.0) * ydiff, top + (expand - 1.0) * ydiff)
        if otherwells:
            for poly in otherwells:
                if not isinstance(poly, Polygons):
                    xtg.warn(
                        "<otherw> not a Polygons instance, but "
                        "a {}".format(type(poly))
                    )
                    continue
                if poly.name == self._well.xwellname:
                    continue
                xwp = poly.dataframe[poly.xname].values
                ywp = poly.dataframe[poly.yname].values
                ax.plot(xwp, ywp, linewidth=1, c="grey")
                ax.annotate(poly.name, xy=(xwp[-1], ywp[-1]), color="grey", size=5)

    def plot_map(self):
        """Plot well location map as an overall view (with field outline)."""

        if not self._outline:
            return

        ax = self._ax3
        if self._wfence is not None:

            xp = self._outline.dataframe["X_UTME"].values
            yp = self._outline.dataframe["Y_UTMN"].values
            ip = self._outline.dataframe["POLY_ID"].values

            ax.plot(self._wfence[:, 0], self._wfence[:, 1], linewidth=3, c="red")

            for i in range(int(ip.min()), int(ip.max()) + 1):
                xpc = xp.copy()[ip == i]
                ypc = yp.copy()[ip == i]
                if len(xpc) > 1:
                    ax.plot(xpc, ypc, linewidth=0.3, c="black")

            ax.set_aspect("equal", "datalim")
