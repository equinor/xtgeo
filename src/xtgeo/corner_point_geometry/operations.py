from typing import Callable, List

import numpy as np
from enums import Enum, auto, unique
from numpy.typing import ArrayLike, NDArray

from .corner_point_geometry import CornerPointGeometry


@unique
class Handedness(Enum):
    LEFT = auto()
    Right = auto()


def estimate_handedness(cpg: CornerPointGeometry) -> Handedness:
    """
    Estimates the handedness of a corner point geometry
    """
    ...


@unique
class LayerDesign(Enum):
    PROPORTIONAL = auto()
    TOPCONFORM = auto()
    BASECONFORM = auto()
    MIXED = auto()


def estimate_design(
    cpg: CornerPointGeometry, layers: ArrayLike[np.integer]
) -> LayerDesign:
    """
    Estimates the design of a layer, returns None if no design can
    be determined.
    """
    ...


def cell_heights(
    cpg: CornerPointGeometry, i: np.integer, j: np.integer, k: np.integer
) -> ArrayLike[(4, 2), np.number]:
    """
    Returns:
        Array of the 8 corner heights for the i,j,kth cell
    """
    ...


def cell_volumes(
    planar_functions: ArrayLike[4, Callable],
    corner_heights: ArrayLike[(4, 2), np.number],
) -> NDArray[np.floating]:
    """Calculates the bulk volume of one cell"""
    ...


def pillar_heights(
    cpg: CornerPointGeometry, i: np.integer, j: np.integer, k: np.integer
) -> ArrayLike[(np.integer, 4, 2), np.number]:
    """
    Returns:
        Array of the 8 corner heights for the i,jth pillar.
    """
    ...


def pillar_planar_functions(
    cpg: CornerPointGeometry, i: np.integer, j: np.integer
) -> List[Callable]:
    """
    Returns:
        The four planar functions for the i,jth pillar.
    """
    ...


def pillar_volumes(
    planar_functions: ArrayLike[4, Callable],
    corner_heights: ArrayLike[(np.integer, np.integer, np.integer), np.number],
) -> NDArray[np.floating]:
    """Calculates the bulk volume of the cells in a pillar."""
    ...


def grid_volumes(
    cpg: CornerPointGeometry,
) -> NDArray[(np.integer, np.integer, np.integer), np.floating]:
    """Calculates the bulk volume of all cells in a corner point geometry."""
    ...
