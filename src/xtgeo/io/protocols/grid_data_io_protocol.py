from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

# https://typing.python.org/en/latest/spec/protocol.html
# https://numpy.org/doc/stable/reference/typing.html

# Leave types to indicate possible future use
T = TypeVar("T")
Vertex2DType = tuple[float, float]
Vertex3DType = tuple[float, float, float]
TriangleType = tuple[int, int, int]
HexagonType = tuple[int, int, int, int, int, int]
# Variable number of vertices, e.g. for Voronoi grids
# PolygonType = tuple[int, ...]

VertexT = TypeVar("VertexT", Vertex2DType, Vertex3DType, covariant=True)
CellT = TypeVar("CellT", TriangleType, HexagonType, covariant=True)


class GridDataIOProtocol(Protocol):
    """
    Protocol for grid data IO operations in any dimension.
    It specifies a common API which can be applied by any class that handles grid data,
    enabling e.g. bulk handling of such classes.
    """

    @property
    def get_vertices(self) -> npt.NDArray[np.float64]:
        """
        Return the vertices of the grid.
        """
        # Not possible to explicitly specify shape of numpy array in numpy typing
        # See https://numpy.org/doc/stable/reference/typing.html

    @property
    def get_cells(self) -> npt.NDArray[np.int64]:
        """
        Return the cells of the grid.
        Each cell is defined by a set of vertex indices.
        """
