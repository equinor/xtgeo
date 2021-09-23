from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from np.typing import ArrayLike


class CornerPointGeometry(ABC):
    """
    A CornerPointGeometry describes cells layed out in three dimensions by
    giving 1) linear corner point lines with a non-zero projection in z
    direction and 2) set of z-values on each line describing the location of
    cell corners. The cells are hexagons formed by the corners on 4 adjacent
    corner lines.

    The corner lines are usually layed out in rows and columns, so one references
    the corner line by a tuple of indecies (i,j) giving the row (i) and column (j)
    number of that line.

    The corners are likewise usually layed out in layers: there are 8 corners
    for each (i,j,k) index (i=row, j=column, k=layer). The reason for there being
    8 values is due to there being a cell in 8 directions from that position (planar
    directions north-west, north-east, south-west, south-east and above and below given
    corner).

    """

    @abstractmethod
    @property
    def number_of_rows(self) -> np.integer:
        ...

    @abstractmethod
    @property
    def number_of_columns(self) -> np.integer:
        ...

    @abstractmethod
    @property
    def number_of_layers(self) -> np.integer:
        ...

    def dimensions(self):
        return self.number_of_rows, self.number_of_columns, self.number_of_layers

    @abstractmethod
    @property
    def coordinates(self) -> ArrayLike[np.number]:
        """
        An array with shape (nx, ny, 2, 3) giving the
        upper and lower corner line points.
        """
        ...

    def planar_function(self, i: np.integer, j: np.integer) -> Callable:
        """
        Returns:
            The planar function of the (i,j)th corner point line. Ie. returns
            x, y coordinates of the line for a given z value:

            # (x,y,z) is a point along the i=0, j=1 corner line
            x, y = cpg.planar_function(0,1)(z)

        """
        lower_point = self.coordinates[i, j, 0, :]
        upper_point = self.coordinates[i, j, 1, :]
        diff = upper_point - lower_point

        lower_z = lower_point[2]
        lower_plane = lower_point[(0, 1)]
        xy_slope = diff[(0, 1)] / diff[2]

        return lambda z: (z - lower_z) * xy_slope + lower_plane

    @abstractmethod
    def layer_heights(
        self, i: np.integer, j: np.integer
    ) -> ArrayLike[(np.integer, np.integer, np.integer, 4, 2), np.number]:
        """
        Returns:
            Array of k heights (z-values) of the intersection between the kth layer
            and the (i,j)th corner line. First dimension
            is in the order (nw, ne, sw, se), the second in lower then upper.
        """
        ...
