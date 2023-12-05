"""
Tests for roxar grids with a repeat-sections. The mocked example
has the following setup:

The dimensions of the geometry is (2,1,4), and cell corners
are at whole numbers. There is one reverse fault going
from i=1, j=1 to i=2,j=2. The below picture shows the
geometry, and the numbering is to identify
each cell for later use.


!~~~~~~~!!~~~~~~~!
!  1    !!  2    !
!~~~~~~~!!~~~~~~~!
!~~~~~~~!
!  3    !
!~~~~~~~!
         !~~~~~~~!
         !  4    !
         !~~~~~~~!
!~~~~~~~!!~~~~~~~!
!  5    !!  6    !
!~~~~~~~!!~~~~~~~!

The grid is given a different indexing scheme in order to save space.
This is signaled by the Grid3D.has_dual_index_system flag and the alternative
indexing system is available through the simbox_indexer property. The
indexing scheme rearranges the cells as follows:

!~~~~~~~!!~~~~~~~!!~~~~~~~!
!  1    !!  2    !!  4    !
!~~~~~~~!!~~~~~~~!!~~~~~~~!
!~~~~~~~!!~~~~~~~!!~~~~~~~!
!  3    !!  5    !!  6    !
!~~~~~~~!!~~~~~~~!!~~~~~~~!

We set up two zones describing the two layers (disregarding the fault):

>> rsg_grid = project.grid_models["repeat_sections_grid"].get_grid()
>> rsg_grid.grid_indexer.zonation
{0: [range(0,1), range(2,3)], 1: [range(1,2), range(3,4)]}
>> rsg_grid.simbox_indexer.zonation
{0: [range(0,1)], 1: [range(1,2)]}

As mentioned, the dual index is described by the has_dual_index_system
property and signals that simbox_indexer and grid_indexer are different.

>> print(rsg_grid.has_dual_index_system)
True
>> print(rsg_grid.grid_indexer.dimensions)
(2,1,4)
>> print(rsg_grid.simbox_indexer.dimensions)
(3,1,2)
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xtgeo.grid3d.grid


class MockGeometry:
    """
    The Geometry object returned by rsg_grid.get_geometry().
    It uses the indexing scheme by rsg_grid.grid_indexer.
    """

    def get_defined_cells(self):
        return np.ones((2, 1, 4), dtype=bool)

    def get_pillar_data(self, i, j):
        """
        Corners of cells are on whole numbers.
        """
        top_of_pillar = (i, j, 0)
        base_of_pillar = (i, j, 4)
        hight_of_corners = np.ones((4, 5))
        for i in range(3):
            hight_of_corners[:, i] = i
        return top_of_pillar, base_of_pillar, hight_of_corners


def rsg_grid_mock():
    mock_grid = MagicMock(spec=["grid_indexer", "get_geometry", "simbox_indexer"])
    mock_grid.grid_indexer.dimensions = (2, 1, 4)
    mock_grid.get_geometry.return_value = MockGeometry()

    mock_grid.has_dual_index_system = True
    mock_grid.simbox_indexer.dimensions = (3, 1, 2)
    mock_grid.simbox_indexer.is_simbox_type = True

    mock_grid.zone_names = ["0", "1"]

    mock_grid.grid_indexer.zonation = {
        0: [range(1), range(2, 3)],
        1: [range(1, 2), range(3, 4)],
    }
    mock_grid.simbox_indexer.zonation = {
        0: [range(1)],
        1: [range(1, 2)],
    }

    return mock_grid


def test_simbox_index():
    with patch("xtgeo.grid3d._grid_roxapi.RoxUtils") as mock_rox_utils:
        mock_grid_getter = MagicMock()
        mock_grid_getter.get_grid.return_value = rsg_grid_mock()
        mock_rox_utils.return_value.project.grid_models = {
            "repeat_sections_grid": mock_grid_getter
        }
        mock_rox_utils.version_required.return_value = True

        with pytest.warns(UserWarning, match="dual index system"):
            xtgeo.grid3d.grid.grid_from_roxar("project", "repeat_sections_grid")
