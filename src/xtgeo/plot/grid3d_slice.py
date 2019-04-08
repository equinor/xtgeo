"""Module for 3D Grid slice plots, using matplotlib."""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np

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

    # ==================================================================================
    # Functions methods (public)
    # Beware general methods in base class also!
    # ==================================================================================

    def plot_gridslice(
        self,
        grid,
        prop,
        mode="layer",
        minvalue=None,
        maxvalue=None,
        colormap=None,
        index=1,
        window=None,
    ):  # pylint: disable=too-many-locals

        """Input a a slice of a 3D grid and plot it.

        Under construction, has bugs; only layer is supported now!

        Args:
            grid: The XTGeo grid object
            prop: The XTGeo grid property object (must belong to grid).
            mode (str): Choose between 'column', 'row', 'layer' (default)
            minvalue (float): Minimum level color scale (default: from data)
            maxvalue (float): Maximum level color scale (default: from data)
            index: Index to plot e.g layer number if layer slice (first=1)
            colormap: Color map to use, e.g. 'rainbow' or an rmscol file

        """
        logger.info("Mode %s is not active", mode)
        if colormap is not None:
            self.colormap = colormap
        else:
            self.colormap = "rainbow"

        clist = grid.get_xyz_corners()
        actnum = grid.get_actnum()

        for cli in clist:
            # mark the inactive cells
            cli.values[actnum.values == 0] = -999.0

        pvalues = prop.values[:, :, index - 1].flatten(order="K")

        # how to remove the masked elements (lol):
        pvalues = pvalues[~pvalues.mask]

        logger.debug(pvalues)
        print(pvalues.shape, grid.ncol * grid.nrow)

        geomlist = grid.get_geometrics(allcells=True, cellcenter=False)

        if window is None:
            xmin = geomlist[3] - 0.05 * (abs(geomlist[4] - geomlist[3]))
            xmax = geomlist[4] + 0.05 * (abs(geomlist[4] - geomlist[3]))
            ymin = geomlist[5] - 0.05 * (abs(geomlist[6] - geomlist[5]))
            ymax = geomlist[6] + 0.05 * (abs(geomlist[6] - geomlist[5]))
        else:
            xmin, xmax, ymin, ymax = window

        # now some numpy operations, numbering is intended
        xy0 = np.column_stack((clist[0].values1d, clist[1].values1d))
        xy1 = np.column_stack((clist[3].values1d, clist[4].values1d))
        xy2 = np.column_stack((clist[9].values1d, clist[10].values1d))
        xy3 = np.column_stack((clist[6].values1d, clist[7].values1d))

        xyc = np.column_stack((xy0, xy1, xy2, xy3))
        xyc = xyc.reshape(grid.nlay, grid.ncol * grid.nrow, 4, 2)

        patches = []

        for pos in range(grid.ncol * grid.nrow):
            nppol = xyc[index - 1, pos, :, :]
            if nppol.mean() > 0.0:
                polygon = Polygon(nppol, True)
                patches.append(polygon)

        print(pvalues.shape, len(patches))

        black = (0, 0, 0, 1)
        patchcoll = PatchCollection(patches, edgecolors=(black,), cmap=self.colormap)

        patchcoll.set_array(np.array(pvalues))

        patchcoll.set_clim([minvalue, maxvalue])

        im = self._ax.add_collection(patchcoll)
        self._ax.set_xlim((xmin, xmax))
        self._ax.set_ylim((ymin, ymax))
        self._fig.colorbar(im)

        plt.gca().set_aspect("equal", adjustable="box")
