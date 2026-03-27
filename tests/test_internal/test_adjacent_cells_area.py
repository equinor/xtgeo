"""Tests for adjacent_cells_overlap_area and helpers.

Covers:
- get_cell_face: extracts the correct 4 corners for each of the 6 face labels.
- adjacent_cells_overlap_area (FaceDirection overload): conforming IJK neighbours in
  a box grid where the shared face is a unit square.
- adjacent_cells_overlap_area (CellFaceLabel overload): correct for a smaller cell
  fully contained within a larger cell's face, as occurs in nested hybrid grids where
  the touching cells are not IJK-neighbours.
- Partial overlap: two neighbour cells whose faces share only part of their area
  (faulted / shifted cell).
- Non-adjacent cells: faces that are parallel but far apart must return 0 when
  max_normal_gap is supplied.
- Zero overlap: faces that are completely non-overlapping in XY.
"""

import math

import numpy as np
import pytest
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.grid3d import (  # type: ignore
    CellCorners,
    CellFaceLabel,
    FaceDirection,
    FaceOverlapResult,
    adjacent_cells_overlap_area,
    face_overlap_result,
    get_cell_face,
)

import xtgeo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell(
    x0: float,
    y0: float,
    z_top: float,
    x1: float,
    y1: float,
    z_bot: float,
) -> CellCorners:
    """Build an axis-aligned box CellCorners.

    The flat 24-element array layout expected by CellCorners is:
        upper_sw (x,y,z), upper_se (x,y,z), upper_nw (x,y,z), upper_ne (x,y,z),
        lower_sw (x,y,z), lower_se (x,y,z), lower_nw (x,y,z), lower_ne (x,y,z)

    In the right-handed C++ convention used here, Z increases upward, so *upper*
    means shallower (higher Z value) and *lower* means deeper (lower Z value).
    z_top > z_bot is therefore required.
    """
    corners = np.array(
        [
            x0,
            y0,
            z_top,  # upper_sw
            x1,
            y0,
            z_top,  # upper_se
            x0,
            y1,
            z_top,  # upper_nw
            x1,
            y1,
            z_top,  # upper_ne
            x0,
            y0,
            z_bot,  # lower_sw
            x1,
            y0,
            z_bot,  # lower_se
            x0,
            y1,
            z_bot,  # lower_nw
            x1,
            y1,
            z_bot,  # lower_ne
        ],
        dtype=np.float64,
    )
    return CellCorners(corners)


# Unit cube at origin: x∈[0,1], y∈[0,1], z∈[0,1] (top=1, bot=0)
UNIT = _make_cell(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)

# Neighbour directly to the East: x∈[1,2], same y and z
EAST = _make_cell(1.0, 0.0, 1.0, 2.0, 1.0, 0.0)

# Neighbour directly to the North: y∈[1,2], same x and z
NORTH = _make_cell(0.0, 1.0, 1.0, 1.0, 2.0, 0.0)

# Neighbour directly below (deeper): z∈[-1, 0]
BELOW = _make_cell(0.0, 0.0, 0.0, 1.0, 1.0, -1.0)


# ---------------------------------------------------------------------------
# get_cell_face
# ---------------------------------------------------------------------------


class TestGetCellFace:
    """Verify that get_cell_face extracts the right 4 corners."""

    def _face_xy_bbox(self, cell: CellCorners, label: CellFaceLabel):
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) of a face."""
        pts = get_cell_face(cell, label)
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def test_top_face_z(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.Top)
        assert zmn == pytest.approx(1.0) and zmx == pytest.approx(1.0)
        assert xmn == pytest.approx(0.0) and xmx == pytest.approx(1.0)

    def test_bottom_face_z(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.Bottom)
        assert zmn == pytest.approx(0.0) and zmx == pytest.approx(0.0)

    def test_east_face_x(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.East)
        assert xmn == pytest.approx(1.0) and xmx == pytest.approx(1.0)

    def test_west_face_x(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.West)
        assert xmn == pytest.approx(0.0) and xmx == pytest.approx(0.0)

    def test_north_face_y(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.North)
        assert ymn == pytest.approx(1.0) and ymx == pytest.approx(1.0)

    def test_south_face_y(self):
        xmn, xmx, ymn, ymx, zmn, zmx = self._face_xy_bbox(UNIT, CellFaceLabel.South)
        assert ymn == pytest.approx(0.0) and ymx == pytest.approx(0.0)

    def test_face_has_four_points(self):
        for label in CellFaceLabel.__members__.values():
            assert len(get_cell_face(UNIT, label)) == 4


# ---------------------------------------------------------------------------
# FaceDirection overload (conforming IJK neighbours)
# ---------------------------------------------------------------------------


class TestFaceDirectionOverload:
    """Unit cube neighbours — every shared face is a 1×1 unit square → area=1."""

    def test_i_direction(self):
        area = adjacent_cells_overlap_area(UNIT, EAST, FaceDirection.I)
        assert area == pytest.approx(1.0)

    def test_j_direction(self):
        area = adjacent_cells_overlap_area(UNIT, NORTH, FaceDirection.J)
        assert area == pytest.approx(1.0)

    def test_k_direction(self):
        area = adjacent_cells_overlap_area(UNIT, BELOW, FaceDirection.K)
        assert area == pytest.approx(1.0)

    def test_2x2_face_area(self):
        """2×1 tall cells sharing an East/West face → face area = 1×2 = 2."""
        left = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 0.0)
        right = _make_cell(1.0, 0.0, 2.0, 2.0, 1.0, 0.0)
        area = adjacent_cells_overlap_area(left, right, FaceDirection.I)
        assert area == pytest.approx(2.0)

    def test_k_direction_rectangular(self):
        """3×2 base face → area = 6."""
        top_cell = _make_cell(0.0, 0.0, 2.0, 3.0, 2.0, 1.0)
        bot_cell = _make_cell(0.0, 0.0, 1.0, 3.0, 2.0, 0.0)
        area = adjacent_cells_overlap_area(top_cell, bot_cell, FaceDirection.K)
        assert area == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# CellFaceLabel overload — nested hybrid grid scenario
# ---------------------------------------------------------------------------


class TestFaceLabelOverload:
    """Non-IJK-neighbour cases: the two cells touch in 3D but are far apart in IJK."""

    def test_conforming_identical_to_direction_overload(self):
        """Face-label and direction overloads must agree for a conforming neighbour."""
        fl = adjacent_cells_overlap_area(
            UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West
        )
        fd = adjacent_cells_overlap_area(UNIT, EAST, FaceDirection.I)
        assert fl == pytest.approx(fd)

    def test_small_cell_inside_large_face(self):
        """Smaller cell (0.25×0.25) whose top face is completely inside the bottom face
        of a larger cell (1×1) — overlap must equal the small cell's face area."""
        large = _make_cell(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        small = _make_cell(0.25, 0.25, 0.0, 0.75, 0.75, -1.0)
        area = adjacent_cells_overlap_area(
            large, CellFaceLabel.Bottom, small, CellFaceLabel.Top
        )
        assert area == pytest.approx(0.25, rel=1e-6)

    def test_large_cell_touches_small_from_east(self):
        """Large cell (2×1) East face vs. small cell (1×0.5) West face that sits in
        the middle of the large face — overlap = 1×0.5 = 0.5."""
        large = _make_cell(0.0, 0.0, 1.0, 1.0, 2.0, 0.0)
        small = _make_cell(1.0, 0.5, 1.0, 2.0, 1.0, 0.0)
        area = adjacent_cells_overlap_area(
            large, CellFaceLabel.East, small, CellFaceLabel.West
        )
        assert area == pytest.approx(0.5, rel=1e-6)

    def test_same_face_both_sides(self):
        """Two cells stacked along K: large bottom face vs. small top face."""
        large = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        small = _make_cell(0.1, 0.1, 1.0, 0.9, 0.9, 0.0)
        area = adjacent_cells_overlap_area(
            large, CellFaceLabel.Bottom, small, CellFaceLabel.Top
        )
        assert area == pytest.approx(0.64, rel=1e-5)


# ---------------------------------------------------------------------------
# Partial overlap (faulted / shifted neighbours)
# ---------------------------------------------------------------------------


class TestPartialOverlap:
    """Cells that share part of a face (fault offset or Z shift)."""

    def test_half_overlap_east_west(self):
        """UNIT is shifted +0.5 in Y relative to EAST-type neighbour → 50 % overlap."""
        left = _make_cell(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        right = _make_cell(1.0, 0.5, 1.0, 2.0, 1.5, 0.0)
        area = adjacent_cells_overlap_area(
            left, CellFaceLabel.East, right, CellFaceLabel.West
        )
        assert area == pytest.approx(0.5, rel=1e-5)

    def test_quarter_overlap_k_face(self):
        """Top cell shifted +0.5 in X and +0.5 in Y → 0.25 overlap on a 1×1 face."""
        top_cell = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        bot_cell = _make_cell(0.5, 0.5, 1.0, 1.5, 1.5, 0.0)
        area = adjacent_cells_overlap_area(
            top_cell, CellFaceLabel.Bottom, bot_cell, CellFaceLabel.Top
        )
        assert area == pytest.approx(0.25, rel=1e-5)

    def test_no_overlap_fully_separated_xy(self):
        """Cells that touch in Z but are completely separate in XY → 0."""
        top_cell = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        bot_cell = _make_cell(5.0, 5.0, 1.0, 6.0, 6.0, 0.0)
        area = adjacent_cells_overlap_area(
            top_cell, CellFaceLabel.Bottom, bot_cell, CellFaceLabel.Top
        )
        assert area == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Non-adjacency guard (max_normal_gap)
# ---------------------------------------------------------------------------


class TestMaxNormalGap:
    """Faces that are parallel but separated along their normal.

    Without the guard the Sutherland-Hodgman projection would still return a
    non-zero area because projection collapses the normal direction. With the
    guard the function must return 0.
    """

    def test_far_apart_k_faces_no_guard_nonzero(self):
        """With the guard explicitly disabled (max_normal_gap=inf), two perfectly
        aligned parallel K-faces far apart in Z produce a false-positive non-zero
        area — they project on top of each other after normal-direction collapse."""
        top_cell = _make_cell(0.0, 0.0, 10.0, 1.0, 1.0, 9.0)
        far_below = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        # Explicitly disable the guard to expose the false-positive
        area_no_guard = adjacent_cells_overlap_area(
            top_cell,
            CellFaceLabel.Bottom,
            far_below,
            CellFaceLabel.Top,
            max_normal_gap=float("inf"),
        )
        assert area_no_guard > 0.0  # proves the false-positive exists when guard is off

    def test_far_apart_k_faces_with_gap_guard_zero(self):
        """With a tight gap guard, the same pair must return 0."""
        top_cell = _make_cell(0.0, 0.0, 10.0, 1.0, 1.0, 9.0)
        far_below = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        # centroids are at z=9 and z=2 → gap along Z-normal ≈ 7
        area = adjacent_cells_overlap_area(
            top_cell,
            CellFaceLabel.Bottom,
            far_below,
            CellFaceLabel.Top,
            max_normal_gap=1.0,
        )
        assert area == pytest.approx(0.0, abs=1e-10)

    def test_truly_adjacent_k_faces_pass_guard(self):
        """Genuinely adjacent faces (gap ≈ 0) must not be killed by the guard."""
        top_cell = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        bot_cell = _make_cell(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        area = adjacent_cells_overlap_area(
            top_cell,
            CellFaceLabel.Bottom,
            bot_cell,
            CellFaceLabel.Top,
            max_normal_gap=1.0,
        )
        assert area == pytest.approx(1.0)

    def test_faulted_cells_within_tolerance(self):
        """Faulted cells can have a small normal gap; guard at a generous tolerance
        must still return the correct overlap area."""
        left = _make_cell(0.0, 0.0, 1.2, 1.0, 1.0, 0.2)  # shifted up 0.2
        right = _make_cell(1.0, 0.0, 1.0, 2.0, 1.0, 0.0)
        # East face of left: x=1, y∈[0,1], z∈[0.2,1.2]
        # West face of right: x=1, y∈[0,1], z∈[0.0,1.0]
        # centroid gap along x-normal = 0 (both at x=1)
        area = adjacent_cells_overlap_area(
            left,
            CellFaceLabel.East,
            right,
            CellFaceLabel.West,
            max_normal_gap=0.5,
        )
        # Overlap in Z: [0.2, 1.0] → height 0.8, width 1 → area 0.8
        assert area == pytest.approx(0.8, rel=1e-5)

    def test_direction_overload_respects_gap_guard(self):
        """FaceDirection overload must also honour max_normal_gap."""
        top_cell = _make_cell(0.0, 0.0, 10.0, 1.0, 1.0, 9.0)
        far_below = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        area = adjacent_cells_overlap_area(
            top_cell, far_below, FaceDirection.K, max_normal_gap=1.0
        )
        assert area == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Box-grid integration: use a real xtgeo grid
# ---------------------------------------------------------------------------


class TestBoxGridIntegration:
    """Sanity-check against a real xtgeo box grid where geometry is known."""

    @pytest.fixture(scope="class")
    def box_grid_cpp(self):
        grid = xtgeo.create_box_grid((4, 4, 4))
        return _internal.grid3d.Grid(grid)

    def test_i_neighbour_area_in_box_grid(self, box_grid_cpp):
        """For a 1×1×1 box grid, every I-adjacent pair shares a 1×1 face."""
        c1 = box_grid_cpp.get_cell_corners_from_ijk(1, 1, 1)
        c2 = box_grid_cpp.get_cell_corners_from_ijk(2, 1, 1)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.I)
        assert area == pytest.approx(1.0)

    def test_j_neighbour_area_in_box_grid(self, box_grid_cpp):
        c1 = box_grid_cpp.get_cell_corners_from_ijk(1, 1, 1)
        c2 = box_grid_cpp.get_cell_corners_from_ijk(1, 2, 1)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.J)
        assert area == pytest.approx(1.0)

    def test_k_neighbour_area_in_box_grid(self, box_grid_cpp):
        c1 = box_grid_cpp.get_cell_corners_from_ijk(1, 1, 1)
        c2 = box_grid_cpp.get_cell_corners_from_ijk(1, 1, 2)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.K)
        assert area == pytest.approx(1.0)

    def test_non_neighbour_far_apart_returns_zero(self, box_grid_cpp):
        """Two cells in the same column but far apart in K must return 0 when the
        gap guard is tighter than their separation."""
        c1 = box_grid_cpp.get_cell_corners_from_ijk(0, 0, 0)
        c3 = box_grid_cpp.get_cell_corners_from_ijk(0, 0, 3)
        area = adjacent_cells_overlap_area(
            c1, CellFaceLabel.Bottom, c3, CellFaceLabel.Top, max_normal_gap=0.5
        )
        assert area == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# FaceOverlapResult — area, normal, and TPFA half-distances
# ---------------------------------------------------------------------------


class TestFaceOverlapResult:
    """Tests for face_overlap_result, which returns the data needed for TPFA.

    TPFA transmissibility:
        HT_i = k_i * area / d_i
        T    = HT1 * HT2 / (HT1 + HT2)
    """

    def test_area_matches_overlap_area(self):
        """area field must equal adjacent_cells_overlap_area for the same inputs."""
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        expected = adjacent_cells_overlap_area(
            UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West
        )
        assert result.area == pytest.approx(expected)

    def test_returns_face_overlap_result_type(self):
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        assert isinstance(result, FaceOverlapResult)

    def test_normal_is_unit_length(self):
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        length = math.sqrt(result.normal.x**2 + result.normal.y**2 + result.normal.z**2)
        assert length == pytest.approx(1.0)

    def test_normal_aligned_with_x_axis_for_east_west_face(self):
        """East/West face has a normal in the ±X direction."""
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        assert abs(result.normal.x) == pytest.approx(1.0, abs=1e-10)
        assert result.normal.y == pytest.approx(0.0, abs=1e-10)
        assert result.normal.z == pytest.approx(0.0, abs=1e-10)

    def test_normal_aligned_with_z_axis_for_k_face(self):
        """Top/Bottom face has a normal in the ±Z direction."""
        result = face_overlap_result(
            UNIT, CellFaceLabel.Bottom, BELOW, CellFaceLabel.Top
        )
        assert abs(result.normal.z) == pytest.approx(1.0, abs=1e-10)
        assert result.normal.x == pytest.approx(0.0, abs=1e-10)
        assert result.normal.y == pytest.approx(0.0, abs=1e-10)

    def test_d1_unit_cube_east_face(self):
        """UNIT centre is at x=0.5; East face centroid is at x=1; d1 = |1-0.5| = 0.5."""
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        assert result.d1 == pytest.approx(0.5)

    def test_d2_unit_cube_east_face(self):
        """EAST centre is at x=1.5; West face centroid is at x=1; d2 = |1-1.5| = 0.5."""
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        assert result.d2 == pytest.approx(0.5)

    def test_d1_d2_unit_cube_k_face(self):
        """UNIT centre z=0.5, Bottom face z=0 → d1=0.5; BELOW centre z=-0.5 → d2=0.5."""
        result = face_overlap_result(
            UNIT, CellFaceLabel.Bottom, BELOW, CellFaceLabel.Top
        )
        assert result.d1 == pytest.approx(0.5)
        assert result.d2 == pytest.approx(0.5)

    def test_no_overlap_returns_all_zero(self):
        """Completely non-overlapping faces → area == d1 == d2 == 0."""
        far = _make_cell(5.0, 5.0, 1.0, 6.0, 6.0, 0.0)
        result = face_overlap_result(UNIT, CellFaceLabel.Bottom, far, CellFaceLabel.Top)
        assert result.area == pytest.approx(0.0, abs=1e-10)
        assert result.d1 == pytest.approx(0.0, abs=1e-10)
        assert result.d2 == pytest.approx(0.0, abs=1e-10)

    def test_gap_guard_returns_all_zero(self):
        """Gap-guard rejection → area == d1 == d2 == 0."""
        top_cell = _make_cell(0.0, 0.0, 10.0, 1.0, 1.0, 9.0)
        far_below = _make_cell(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        result = face_overlap_result(
            top_cell,
            CellFaceLabel.Bottom,
            far_below,
            CellFaceLabel.Top,
            max_normal_gap=1.0,
        )
        assert result.area == pytest.approx(0.0, abs=1e-10)
        assert result.d1 == pytest.approx(0.0, abs=1e-10)
        assert result.d2 == pytest.approx(0.0, abs=1e-10)

    def test_tpfa_symmetric_unit_cubes(self):
        """Two identical unit cubes sharing an East/West face.
        k=1, A=1, d1=d2=0.5 → HT1=HT2=2 → T = 2*2/(2+2) = 1.
        """
        result = face_overlap_result(UNIT, CellFaceLabel.East, EAST, CellFaceLabel.West)
        k = 1.0
        ht1 = k * result.area / result.d1
        ht2 = k * result.area / result.d2
        trans = ht1 * ht2 / (ht1 + ht2)
        assert trans == pytest.approx(1.0)

    def test_tpfa_asymmetric_cells(self):
        """Narrow cell (width 0.5) next to wide cell (width 2.0).
        d1=0.25, d2=1.0, A=1 (shared 1×1 face), k=1.
        T = k*A / (d1+d2) = 1/1.25 = 0.8.
        """
        narrow = _make_cell(0.0, 0.0, 1.0, 0.5, 1.0, 0.0)  # width 0.5 in X
        wide = _make_cell(0.5, 0.0, 1.0, 2.5, 1.0, 0.0)  # width 2.0 in X
        result = face_overlap_result(
            narrow, CellFaceLabel.East, wide, CellFaceLabel.West
        )
        assert result.d1 == pytest.approx(0.25)
        assert result.d2 == pytest.approx(1.0)
        k = 1.0
        ht1 = k * result.area / result.d1
        ht2 = k * result.area / result.d2
        trans = ht1 * ht2 / (ht1 + ht2)
        # T = A * k / (d1 + d2) = 1.0 / 1.25 = 0.8
        assert trans == pytest.approx(0.8, rel=1e-5)


class TestNestedHybrid:
    """Use a box grid and created a refined nested hybrid grid."""

    @pytest.fixture(scope="class")
    def nested_hybrid_grid_cpp(self):
        grid = xtgeo.create_box_grid((4, 4, 4))
        refined = grid.copy()

        # let two og the cells in the middle of the box grid be refined into 2×2×2
        # subcells, creating a nested hybrid scenario where the small cells are
        # not IJK neighbours of the large cell but still share part of its face
        refined.crop((2, 2), (2, 3), (1, 4))  # crop out the middle 2×2×2 block
        refined.refine(3, 3, 4)

        act = grid.get_actnum()
        act.values[1, 1:3, :] = 0  # inactivate the cropped
        grid.set_actnum(act)

        # merged = grid.copy()
        merged = xtgeo.grid_merge(grid, refined)

        return _internal.grid3d.Grid(merged)

    def test_i_neighbour_area_in_nested_hybrid_grid(self, nested_hybrid_grid_cpp):
        """For a 1×1×1 box grid, every I-adjacent pair shares a 1×1 face."""
        c1 = nested_hybrid_grid_cpp.get_cell_corners_from_ijk(0, 1, 0)
        c2 = nested_hybrid_grid_cpp.get_cell_corners_from_ijk(5, 0, 0)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.I)
        assert area == pytest.approx(1.0 / (3 * 4))

        c2 = nested_hybrid_grid_cpp.get_cell_corners_from_ijk(5, 2, 0)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.I)
        assert area == pytest.approx(1.0 / (3 * 4))

    def test_non_i_neighbour_area_in_nested_hybrid_grid(self, nested_hybrid_grid_cpp):
        """For a 1×1×1 box grid, every I-adjacent pair shares a 1×1 face."""
        c1 = nested_hybrid_grid_cpp.get_cell_corners_from_ijk(0, 1, 0)
        c2 = nested_hybrid_grid_cpp.get_cell_corners_from_ijk(6, 1, 0)
        area = adjacent_cells_overlap_area(c1, c2, FaceDirection.I, max_normal_gap=0.01)
        assert area == pytest.approx(0.0)
