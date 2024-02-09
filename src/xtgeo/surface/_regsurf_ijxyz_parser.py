"""IJXYZ files (OW/DSG) parsing.

This format lacks explicit information such as origin, xinc, yinc, rotation etc; i.e.
all these setting must be computed from the input.

Missing inline and xline values are assumed to be undefined. Examples:


========================================================================================
815	1210	777574.220000	6736507.760000	1010.000000
816	1210	777561.894910	6736521.890000	1010.000000
817	1210	777549.569820	6736536.020000	1010.000000
818	1210	777537.244731	6736550.150000	1010.000000
819	1210	777524.919641	6736564.280000	1010.000000
820	1210	777512.594551	6736578.410000	1010.000000
821	1210	777500.269461	6736592.540000	1010.000000
822	1210	777487.944371	6736606.670000	1010.000000
823	1210	777475.619281	6736620.800000	1010.000000
...
846	1211	777201.561266	6736954.005898	1010.000000
847	1211	777189.236176	6736968.135898	1010.000000
848	1211	777176.911086	6736982.265898	1010.000000
849	1211	777164.585996	6736996.395898	1010.000000
850	1211	777152.260906	6737010.525898	1010.000000
851	1211	777139.935817	6737024.655898	1010.000000
...

Here inline is fastest and increasing and every inline and xline is applied

========================================================================================

24	3	774906.625000	6736030.500000	1741.107300
26	3	774946.017310	6736023.554073	1730.530884
28	3	774985.409620	6736016.608146	1719.947876
30	3	775024.801930	6736009.662219	1709.229370
32	3	775064.194240	6736002.716292	1698.470337
34	3	775103.586551	6735995.770364	1687.320435
...
272	3	779791.271455	6735169.205039	1404.342041
274	3	779830.663765	6735162.259112	1413.154175
24	6	774915.307409	6736079.740388	1727.455078
26	6	774954.699719	6736072.794461	1719.031128
28	6	774994.092029	6736065.848533	1710.606201
30	6	775033.484339	6736058.902606	1702.161255
32	6	775072.876649	6736051.956679	1693.707153
34	6	775112.268959	6736045.010752	1685.164673

Here every second inline and every third xline is applied

========================================================================================

 1312  2015   655905.938642   6627044.343360   1671.420800
 1312  2010   655858.839186   6627003.259513   1669.091400
 1302  2028   656151.649337   6627009.862811   1680.437500
 1312  2011   655868.259077   6627011.476282   1669.444000
 1302  2029   656161.069228   6627018.079581   1681.000000
 1302  2026   656132.809555   6626993.429272   1679.312500
 1302  2027   656142.229446   6627001.646042   1679.875000
 1302  2024   656113.969773   6626976.995733   1678.187500
 1302  2025   656123.389664   6626985.212503   1678.750000

This seems to lack obvious patterns (mostly due to many undefied cells?). Hence we
need to detect minimum spacing and min/max of both ilines and xlines, unless a template
is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from numpy.ma import MaskedArray

from xtgeo.common.calc import find_flip, xyori_from_ij
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.cube.cube1 import Cube

    from .regular_surface import RegularSurface

logger = null_logger(__name__)


@dataclass
class SurfaceIJXYZ:
    """Temporary class for parsing IJXYZ settings.

    Attributes:
        xcoords_in: Input X coordinates.
        ycoords_in: Input Y coordinates.
        values_in: Input values.
        ilines_in: Input inline values.
        xlines_in: Input crossline values.
        template: Geometrical template.
    """

    xcoords_in: MaskedArray
    ycoords_in: MaskedArray
    values_in: MaskedArray
    ilines_in: np.ndarray
    xlines_in: np.ndarray
    template: Optional[RegularSurface | Cube] = None

    ncol: int = field(default=1, init=False)
    nrow: int = field(default=1, init=False)
    xori: float = field(default=0.0, init=False)
    yori: float = field(default=0.0, init=False)
    xinc: float = field(default=1.0, init=False)
    yinc: float = field(default=1.0, init=False)
    yflip: Literal[-1, 1] = field(default=1, init=False)
    xcoords: MaskedArray = field(default_factory=MaskedArray, init=False)
    ycoords: MaskedArray = field(default_factory=MaskedArray, init=False)
    values: MaskedArray = field(default_factory=MaskedArray, init=False)
    ilines: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)
    xlines: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)

    def __post_init__(self) -> None:
        self._parse_arrays()
        if self.template:
            self._map_on_template()
        else:
            self._calculate_geometrics()

    def _parse_arrays(self) -> None:
        """Parse input and arrange basic data and arrays."""

        logger.debug("Parse ijxyz arrays")
        # both ilines and xlines may jump e.g. every second; detect minimum jump:
        inline_mindiff = int(np.min(np.diff(np.unique(np.sort(self.ilines_in)))))
        xline_mindiff = int(np.min(np.diff(np.unique(np.sort(self.xlines_in)))))

        i_index = np.divide(self.ilines_in, inline_mindiff)
        i_index = (i_index - i_index.min()).astype("int32")

        j_index = np.divide(self.xlines_in, xline_mindiff)
        j_index = (j_index - j_index.min()).astype("int32")

        self.ncol = int(i_index.max() + 1)
        self.nrow = int(j_index.max() + 1)

        self.ilines = np.array(
            range(
                self.ilines_in.min(),
                self.ilines_in.max() + inline_mindiff,
                inline_mindiff,
            )
        ).astype("int32")

        self.xlines = np.array(
            range(
                self.xlines_in.min(),
                self.xlines_in.max() + xline_mindiff,
                xline_mindiff,
            )
        ).astype("int32")

        shape = (self.ncol, self.nrow)

        xvalues = np.full(shape, np.nan)
        xvalues[i_index, j_index] = self.xcoords_in
        self.xcoords = np.ma.masked_invalid(xvalues)

        yvalues = np.full(shape, np.nan)
        yvalues[i_index, j_index] = self.ycoords_in
        self.ycoords = np.ma.masked_invalid(yvalues)

        zvalues = np.full(shape, np.nan)
        zvalues[i_index, j_index] = self.values_in
        self.values = np.ma.masked_invalid(zvalues)

    def _map_on_template(self) -> None:
        """An existing RegularSurface or Cube forms the geometrical template."""
        logger.debug("Parse ijxyz with template")

        # TODO: Remove these when moved to xtgeo.io
        from xtgeo.cube.cube1 import Cube

        from .regular_surface import RegularSurface

        if not isinstance(self.template, (RegularSurface, Cube)):
            raise ValueError(
                "The provided template is not of type RegularSurface or Cube"
            )

        ilines_orig = self.ilines.copy()
        xlines_orig = self.xlines.copy()

        il_step = int(np.diff(np.sort(ilines_orig)).min())
        xl_step = int(np.diff(np.sort(xlines_orig)).min())

        # now ovveride with settings from template
        for attr in (
            "ncol",
            "nrow",
            "xori",
            "yori",
            "xinc",
            "yinc",
            "rotation",
            "yflip",
            "ilines",
            "xlines",
        ):
            setattr(self, attr, getattr(self.template, attr))

        i_start = abs(int((self.ilines.min() - ilines_orig.min()) / il_step))
        j_start = abs(int((self.xlines.min() - xlines_orig.min()) / xl_step))

        shape = (self.ncol, self.nrow)

        zvalues = np.full(shape, np.nan)
        zvalues[
            i_start : i_start + self.values.shape[0],
            j_start : j_start + self.values.shape[1],
        ] = np.ma.filled(self.values, fill_value=np.nan)

        self.values = np.ma.masked_invalid(zvalues)

    def _calculate_geometrics(self) -> None:
        """Compute the geometrics such as rotation, flip origin and increments."""

        logger.debug("Compute ijxyz geometrics from arrays")
        # along_east and along_north here refers to the case when rotation is 0.0
        dx_along_east = np.diff(self.xcoords, axis=0)
        dy_along_east = np.diff(self.ycoords, axis=0)
        dx_along_north = np.diff(self.xcoords, axis=1)
        dy_along_north = np.diff(self.ycoords, axis=1)

        self.yflip = find_flip(
            (dx_along_east.mean(), dy_along_east.mean(), 0.0),
            (dx_along_north.mean(), dy_along_north.mean(), 0.0),
        )

        self.rotation = float(
            np.degrees(np.arctan2(dy_along_east, dx_along_east)).mean()
        )

        distances_xinc = np.sqrt(dx_along_east**2 + dy_along_east**2)
        distances_yinc = np.sqrt(dx_along_north**2 + dy_along_north**2)
        self.xinc = float(distances_xinc.mean())
        self.yinc = float(distances_yinc.mean())

        cols, rows = np.ma.where(~self.xcoords.mask)  # get active indices

        first_active_col = int(cols[0])
        first_active_row = int(rows[0])

        self.xori, self.yori = xyori_from_ij(
            first_active_col,
            first_active_row,
            self.xcoords[first_active_col, first_active_row],
            self.ycoords[first_active_col, first_active_row],
            self.xinc,
            self.yinc,
            self.ncol,
            self.nrow,
            self.yflip,
            float(self.rotation),
        )


def parse_ijxyz(mfile, template=None) -> tuple:
    """Read IJXYZ data file and parse the content."""

    if mfile.memstream:
        mfile.file.seek(0)

    data = np.loadtxt(mfile.file, comments=["@", "#", "EOB"])

    inline = data[:, 0].astype("int32")
    xline = data[:, 1].astype("int32")
    x_arr = data[:, 2].astype("float64")
    y_arr = data[:, 3].astype("float64")
    z_arr = data[:, 4].astype("float64")

    return SurfaceIJXYZ(x_arr, y_arr, z_arr, inline, xline, template=template)
