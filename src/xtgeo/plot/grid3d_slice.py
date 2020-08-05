"""Module for 3D Grid slice plots, using matplotlib."""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from xtgeo.common import XTGeoDialog
from xtgeo.plot.baseplot import BasePlot

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Grid3DSlice(BasePlot):
    """Class for plotting a row, a column, or a layer, using matplotlib."""

    def __init__(self):
        """The __init__ (constructor) method for a Grid3DSlice object."""

        super(Grid3DSlice, self).__init__()

        self._wells = None
        self._surface = None
        self._tight = False

        self._wfence = None
        self._legendtitle = "Map"

        self._colormap = "rainbow"
        self._linecolor = "black"
        self._clist = None
        self._prop = None
        self._grid = None
        self._geomlist = None
        self._index = 1
        self._actnum = None
        self._window = None
        self._active = True
        self._minvalue = None
        self._maxvalue = None

    # ==================================================================================
    # Functions methods (public)
    # Beware general methods in base class also!
    # ==================================================================================

    def plot_gridslice(
        self,
        grid,
        prop=None,
        mode="layer",
        minvalue=None,
        maxvalue=None,
        colormap=None,
        linecolor="black",
        index=1,
        window=None,
        activeonly=True,
    ):

        """Plot a row slice, column slice or layer slice of a grid.

        Args:
            grid (Grid): The XTGeo grid object
            prop (GridProperty, optional): The XTGeo grid property object
            mode (str): Choose between 'column', 'row', 'layer' (default)
            minvalue (float): Minimum level color scale (default: from data)
            maxvalue (float): Maximum level color scale (default: from data)
            index (int): Index to plot e.g layer number if layer slice (first=1)
            colormap: Color map to use for cells, e.g. 'rainbow' or an rmscol file
            linecolor (str or tuple): Color of grid lines (black/white/grey
                or a tuple with 4 numbers on valid matplotlib format)

        """

        self._index = index
        if colormap is not None:
            self._colormap = colormap

        self._linecolor = linecolor
        if not isinstance(linecolor, tuple) and linecolor not in (
            "black",
            "grey",
            "white",
        ):
            raise ValueError("Value of linecolor is invalid")

        self._clist = grid.get_xyz_corners()  # get XYZ for each corner, 24 arrays
        self._grid = grid
        self._prop = prop
        self._window = window

        if self._geomlist is None:
            # returns (xori, yori, zori, xmin, xmax, ymin, ymax, zmin, zmax,...)
            self._geomlist = grid.get_geometrics(allcells=True, cellcenter=False)

        self._active = activeonly

        self._minvalue = minvalue
        self._maxvalue = maxvalue

        if mode == "column":
            pass  # self._plot_row()
        elif mode == "row":
            pass  # self._plot_row()
        else:
            self._plot_layer()

    # def _plot_row(self):

    #     geomlist = self._geomlist

    #     if self._window is None:
    #         xmin = geomlist[3] - 0.05 * (abs(geomlist[4] - geomlist[3]))
    #         xmax = geomlist[4] + 0.05 * (abs(geomlist[4] - geomlist[3]))
    #         zmin = geomlist[7] - 0.05 * (abs(geomlist[8] - geomlist[7]))
    #         zmax = geomlist[8] + 0.05 * (abs(geomlist[8] - geomlist[7]))
    #     else:
    #         xmin, xmax, zmin, zmax = self._window

    #     # now some numpy operations, numbering is intended
    #     clist = self._clist
    #     xz0 = np.column_stack((clist[0].values1d, clist[2].values1d))
    #     xz1 = np.column_stack((clist[3].values1d, clist[5].values1d))
    #     xz2 = np.column_stack((clist[15].values1d, clist[17].values1d))
    #     xz3 = np.column_stack((clist[12].values1d, clist[14].values1d))

    #     xyc = np.column_stack((xz0, xz1, xz2, xz3))
    #     xyc = xyc.reshape(self._grid.nlay, self._grid.ncol * self._grid.nrow, 4, 2)

    #     patches = []

    #     for pos in range(self._grid.nrow * self._grid.nlay):
    #         nppol = xyc[self._index - 1, pos, :, :]
    #         if nppol.mean() > 0.0:
    #             polygon = Polygon(nppol, True)
    #             patches.append(polygon)

    #     black = (0, 0, 0, 1)
    #     patchcoll = PatchCollection(patches, edgecolors=(black,), cmap=self.colormap)

    #     # patchcoll.set_array(np.array(pvalues))

    #     # patchcoll.set_clim([minvalue, maxvalue])

    #     im = self._ax.add_collection(patchcoll)
    #     self._ax.set_xlim((xmin, xmax))
    #     self._ax.set_ylim((zmin, zmax))
    #     self._ax.invert_yaxis()
    #     self._fig.colorbar(im)

    #     # plt.gca().set_aspect("equal", adjustable="box")

    def _plot_layer(self):

        xyc, ibn = self._grid.get_layer_slice(self._index, activeonly=self._active)

        xval = xyc[:, :, 0]
        yval = xyc[:, :, 1]

        xvmin = xval.min()
        xvmax = xval.max()
        yvmin = yval.min()
        yvmax = yval.max()

        if self._window is None:
            xmin = xvmin - 0.05 * (abs(xvmax - xvmin))
            xmax = xvmax + 0.05 * (abs(xvmax - xvmin))
            ymin = yvmin - 0.05 * (abs(yvmax - yvmin))
            ymax = yvmax + 0.05 * (abs(yvmax - yvmin))
        else:
            xmin, xmax, ymin, ymax = self._window

        patches = []

        for pos in range(len(ibn)):
            nppol = xyc[pos, :, :]
            if nppol.mean() > 0.0:
                polygon = Polygon(nppol, True)
                patches.append(polygon)

        patchcoll = PatchCollection(
            patches, edgecolors=(self._linecolor,), cmap=self.colormap
        )

        if self._prop:
            pvalues = self._prop.values
            pvalues = pvalues[:, :, self._index - 1]
            pvalues = pvalues[~pvalues.mask]

            patchcoll.set_array(pvalues)

            pmin = self._minvalue
            if self._minvalue is None:
                pmin = pvalues.min()

            pmax = self._minvalue
            if self._maxvalue is None:
                pmax = pvalues.max()

            patchcoll.set_clim([pmin, pmax])

        im = self._ax.add_collection(patchcoll)
        self._ax.set_xlim((xmin, xmax))
        self._ax.set_ylim((ymin, ymax))
        self._fig.colorbar(im)

        plt.gca().set_aspect("equal", adjustable="box")
