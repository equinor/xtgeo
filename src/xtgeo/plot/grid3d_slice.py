"""Module for 3D Grid slice plots, using matplotlib."""

from __future__ import print_function, division, absolute_import

# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection

# import numpy as np

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
        self._clist = None
        self._prop = None
        self._grid = None
        self._geomlist = None
        self._index = 1
        self._actnum = None
        self._window = None

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
        index=1,
        window=None,
    ):

        """Plot a row slice, column slice or layer slice of a grid.

        Args:
            grid (Grid): The XTGeo grid object
            prop (GridProperty, optional): The XTGeo grid property object
            mode (str): Choose between 'column', 'row', 'layer' (default)
            minvalue (float): Minimum level color scale (default: from data)
            maxvalue (float): Maximum level color scale (default: from data)
            index: Index to plot e.g layer number if layer slice (first=1)
            colormap: Color map to use, e.g. 'rainbow' or an rmscol file

        """

        raise NotImplementedError("In prep")

    #     self._index = index
    #     if colormap is not None:
    #         self._colormap = colormap

    #     self._clist = grid.get_xyz_corners()  # get XYZ for each corner, 24 arrays
    #     self._actnum = grid.get_actnum()
    #     self._grid = grid
    #     self._prop = prop
    #     self._window = window

    #     if self._geomlist is None:
    #         # returns (xori, yori, zori, xmin, xmax, ymin, ymax, zmin, zmax,...)
    #         self._geomlist = grid.get_geometrics(allcells=True, cellcenter=False)

    #     if mode == "column":
    #         self._plot_row()
    #     elif mode == "row":
    #         self._plot_row()
    #     else:
    #         self._plot_layer()

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

    # def _plot_layer(self):

    #     geomlist = self._geomlist

    #     if self._window is None:
    #         xmin = geomlist[3] - 0.05 * (abs(geomlist[4] - geomlist[3]))
    #         xmax = geomlist[4] + 0.05 * (abs(geomlist[4] - geomlist[3]))
    #         ymin = geomlist[5] - 0.05 * (abs(geomlist[6] - geomlist[5]))
    #         ymax = geomlist[6] + 0.05 * (abs(geomlist[6] - geomlist[5]))
    #     else:
    #         xmin, xmax, ymin, ymax = self._window

    #     # now some numpy operations, numbering is intended
    #     clist = self._clist
    #     xy0 = np.column_stack((clist[0].values1d, clist[1].values1d))
    #     xy1 = np.column_stack((clist[3].values1d, clist[4].values1d))
    #     xy2 = np.column_stack((clist[9].values1d, clist[10].values1d))
    #     xy3 = np.column_stack((clist[6].values1d, clist[7].values1d))

    #     xyc = np.column_stack((xy0, xy1, xy2, xy3))
    #     xyc = xyc.reshape(self._grid.nlay, self._grid.ncol * self._grid.nrow, 4, 2)

    #     patches = []

    #     for pos in range(self._grid.nrow * self._grid.ncol):
    #         nppol = xyc[self._index - 1, pos, :, :]
    #         print(nppol)
    #         if nppol.mean() > 0.0:
    #             polygon = Polygon(nppol, True)
    #             patches.append(polygon)

    #     print("Number is {}".format(len(patches)))
    #     black = (0, 0, 0, 1)
    #     patchcoll = PatchCollection(patches, edgecolors=(black,), cmap=self.colormap)

    #     # patchcoll.set_array(np.array(pvalues))

    #     # patchcoll.set_clim([minvalue, maxvalue])

    #     im = self._ax.add_collection(patchcoll)
    #     self._ax.set_xlim((xmin, xmax))
    #     self._ax.set_ylim((ymin, ymax))
    #     self._fig.colorbar(im)

    #     plt.gca().set_aspect("equal", adjustable="box")
