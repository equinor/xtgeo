"""Module for 3D Grid slice plots, using matplotlib."""

from xtgeo.common import null_logger
from xtgeo.plot.baseplot import BasePlot

logger = null_logger(__name__)


class Grid3DSlice(BasePlot):
    """Class for plotting a row, a column, or a layer, using matplotlib."""

    def __init__(self):
        """Construct an instance for a Grid3DSlice object."""
        super().__init__()

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
            activeonly (bool): If only use active cells
            window (str): Some window

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

        import matplotlib as mpl

        for pos in range(len(ibn)):
            nppol = xyc[pos, :, :]
            if nppol.mean() > 0.0:
                polygon = mpl.patches.Polygon(nppol)
                patches.append(polygon)

        patchcoll = mpl.collections.PatchCollection(
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

            pmax = self._maxvalue
            if self._maxvalue is None:
                pmax = pvalues.max()

            patchcoll.set_clim([pmin, pmax])

        im = self._ax.add_collection(patchcoll)
        self._ax.set_xlim((xmin, xmax))
        self._ax.set_ylim((ymin, ymax))
        self._fig.colorbar(im)

        import matplotlib.pyplot as plt

        plt.gca().set_aspect("equal", adjustable="box")
