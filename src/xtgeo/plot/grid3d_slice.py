"""Module for 3D Grid slice plots, using matplotlib."""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import logging
import numpy as np

from xtgeo.common import XTGeoDialog
from xtgeo.plot.baseplot import BasePlot


class Grid3DSlice(BasePlot):
    """Class for plotting a row, acolumn, or a layer, using matplotlib."""

    def __init__(self):
        """The __init__ (constructor) method for a Grid3DSlice object."""

        super(Grid3DSlice, self).__init__()

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

    def plot_gridslice(self, prop, mode='layer',
                       minvalue=None, maxvalue=None,
                       colortable=None, index=1, window=None):

        """Input a a slice of a 3D grid and plot it.

        Under construction; only layer is supported now!

        Args:
            prop: The XTGeo grid property object (needs link to the grid)
            mode (str): Choose between 'column', 'row', 'layer' (default)
            minvalue (float): Minimum level color scale (default: from data)
            maxvalue (float): Maximum level color scale (default: from data)
            index: Index to plot e.g layer number if layer slice (first=1)
            colortable: Color table to use, e.g. 'rainbow' or an rmscol file

        """

        if prop.grid is None:
            raise RuntimeError('Need to connect property to a grid for plots')

        if colortable is not None:
            self.set_colortable(colortable)
        else:
            self.set_colortable('rainbow')

        grid = prop.grid

        clist = grid.get_xyz_corners()
        actnum = grid.get_actnum()

        for i in range(len(clist)):
            # mark the inactive cells
            clist[i].values[actnum.values == 0] = -999.0

        pvalues = prop.values3d[:, :, index - 1].flatten(order='F')

        # how to remove the masked elements (lol):
        pvalues = pvalues[~pvalues.mask]

        self.logger.debug(pvalues)
        print(pvalues.shape, grid.ncol * grid.nrow)

        geomlist = grid.get_geometrics(allcells=True, cellcenter=False)

        if window is None:
            xmin = geomlist[3] - 0.05 * (abs(geomlist[4] - geomlist[3]))
            xmax = geomlist[4] + 0.05 * (abs(geomlist[4] - geomlist[3]))
            ymin = geomlist[5] - 0.05 * (abs(geomlist[6] - geomlist[5]))
            ymax = geomlist[6] + 0.05 * (abs(geomlist[6] - geomlist[5]))
        else:
            xmin, xmax, ymin, ymax = window

        # now some numpy operations
        xy0 = np.column_stack((clist[0].values, clist[1].values))
        xy1 = np.column_stack((clist[3].values, clist[4].values))
        xy2 = np.column_stack((clist[9].values, clist[10].values))  # intended!
        xy3 = np.column_stack((clist[6].values, clist[7].values))

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
        patchcoll = PatchCollection(patches, edgecolors=(black,),
                                    cmap=self.colormap)

        patchcoll.set_array(np.array(pvalues))

        patchcoll.set_clim([minvalue, maxvalue])

        im = self._ax.add_collection(patchcoll)
        self._ax.set_xlim((xmin, xmax))
        self._ax.set_ylim((ymin, ymax))
        self._fig.colorbar(im)

        plt.gca().set_aspect('equal', adjustable='box')
