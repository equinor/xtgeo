"""Tests for ``Well.get_cell_intersections``.

Uses a regular box grid with known geometry, and a set of hand-crafted
trajectories whose expected entry/exit points are known analytically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xtgeo

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def box_grid():
    """A 5 x 4 x 3 box grid with known geometry.

    - Origin (0, 0, 1000)
    - dx = 100, dy = 100, dz = 50
    - No rotation, axis aligned -> easy analytical answers
    """
    return xtgeo.create_box_grid(
        (5, 4, 3),
        increment=(100.0, 100.0, 50.0),
        origin=(0.0, 0.0, 1000.0),
        rotation=0.0,
    )


def _make_well(xv, yv, zv, mdv, name="W1"):
    """Build a real xtgeo.Well from coordinate arrays."""
    df = pd.DataFrame(
        {
            "X_UTME": xv,
            "Y_UTMN": yv,
            "Z_TVDSS": zv,
            "M_MDEPTH": mdv,
        }
    )
    return xtgeo.Well(
        rkb=0.0,
        xpos=float(xv[0]),
        ypos=float(yv[0]),
        wname=name,
        df=df,
        mdlogname="M_MDEPTH",
    )


def _vertical_well(x, y, z_top, z_bot, name="W1"):
    z = np.array([z_top, z_bot])
    md = z.copy()
    return _make_well(np.array([x, x]), np.array([y, y]), z, md, name=name)


# ---------------------------------------------------------------------------
# DataFrame / validation tests
# ---------------------------------------------------------------------------


def test_returns_dataframe_with_expected_columns(box_grid):
    well = _vertical_well(250.0, 150.0, 900.0, 1200.0)
    df = well.get_cell_intersections(box_grid, zerobased=True)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "I",
        "J",
        "K",
        "ENTRY_EASTING",
        "ENTRY_NORTHING",
        "ENTRY_TVD",
        "ENTRY_MD",
        "EXIT_EASTING",
        "EXIT_NORTHING",
        "EXIT_TVD",
        "EXIT_MD",
        "LENGTH_MD",
    ]
    assert len(df) == 3
    np.testing.assert_array_equal(df["I"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["J"].to_numpy(), [1, 1, 1])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["LENGTH_MD"], [50.0, 50.0, 50.0], atol=1e-2)


def test_default_indices_are_one_based(box_grid):
    """Default zerobased=False returns 1-based I/J/K, consistent with
    other XTGeo grid methods like Grid.get_ijk()."""
    well = _vertical_well(250.0, 150.0, 900.0, 1200.0)
    df = well.get_cell_intersections(box_grid)

    np.testing.assert_array_equal(df["I"].to_numpy(), [3, 3, 3])
    np.testing.assert_array_equal(df["J"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["K"].to_numpy(), [1, 2, 3])


def test_invalid_sampling_step_raises(box_grid):
    well = _vertical_well(250.0, 150.0, 900.0, 1200.0)
    with pytest.raises(ValueError, match="sampling_step"):
        well.get_cell_intersections(box_grid, sampling_step=0.0)


def test_negative_refine_iters_raises(box_grid):
    well = _vertical_well(250.0, 150.0, 900.0, 1200.0)
    with pytest.raises(ValueError, match="refine_iters"):
        well.get_cell_intersections(box_grid, refine_iters=-1)


def test_well_without_md_log_raises(box_grid):
    df = pd.DataFrame(
        {
            "X_UTME": [250.0, 250.0],
            "Y_UTMN": [150.0, 150.0],
            "Z_TVDSS": [900.0, 1200.0],
        }
    )
    well = xtgeo.Well(rkb=0.0, xpos=250.0, ypos=150.0, wname="W", df=df)
    assert well.mdlogname is None
    with pytest.raises(ValueError, match="MD log"):
        well.get_cell_intersections(box_grid)


# ---------------------------------------------------------------------------
# NaN / undefined-row handling
# ---------------------------------------------------------------------------


def test_all_nan_rows_returns_empty(box_grid):
    """A well where every coordinate row is NaN should return empty."""
    nan = float("nan")
    well = _make_well(
        np.array([nan, nan]),
        np.array([nan, nan]),
        np.array([nan, nan]),
        np.array([nan, nan]),
    )
    df = well.get_cell_intersections(box_grid)
    assert len(df) == 0


def test_nan_rows_are_filtered(box_grid):
    """NaN rows in the trajectory should be transparently removed and the
    remaining finite points processed as a single trajectory."""
    nan = float("nan")
    # Vertical well at (250,150) with NaN rows interspersed
    xv = np.array([nan, 250.0, 250.0, nan])
    yv = np.array([nan, 150.0, 150.0, nan])
    zv = np.array([nan, 900.0, 1200.0, nan])
    mdv = np.array([nan, 900.0, 1200.0, nan])

    well = _make_well(xv, yv, zv, mdv, name="NAN_WELL")
    df = well.get_cell_intersections(box_grid, zerobased=True)

    assert len(df) == 3
    np.testing.assert_array_equal(df["I"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["J"].to_numpy(), [1, 1, 1])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])


def test_single_finite_sample_between_nans_skipped(box_grid):
    """A single finite sample surrounded by NaN rows cannot form a segment
    (needs >= 2 points) and should return an empty DataFrame."""
    nan = float("nan")
    xv = np.array([nan, 250.0, nan])
    yv = np.array([nan, 150.0, nan])
    zv = np.array([nan, 1050.0, nan])
    mdv = np.array([nan, 1050.0, nan])

    well = _make_well(xv, yv, zv, mdv, name="SINGLE_PT")
    df = well.get_cell_intersections(box_grid)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Vertical well tests
# ---------------------------------------------------------------------------


def test_vertical_well_through_three_layers(box_grid):
    """A vertical well drilled at the centre of cell (i=2, j=1) should
    enter the grid at z=1000, then cross every layer interface at the
    expected depths and exit at z=1150."""
    md_vals = np.array([900.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0])
    z_vals = md_vals.copy()
    x_vals = np.full_like(md_vals, 250.0)
    y_vals = np.full_like(md_vals, 150.0)

    well = _make_well(x_vals, y_vals, z_vals, md_vals)
    df = well.get_cell_intersections(
        box_grid, sampling_step=2.0, refine_iters=25, zerobased=True
    )

    assert len(df) == 3
    np.testing.assert_array_equal(df["I"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["J"].to_numpy(), [1, 1, 1])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])

    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=0.5)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=0.5)
    np.testing.assert_allclose(df["ENTRY_MD"], [1000.0, 1050.0, 1100.0], atol=0.5)
    np.testing.assert_allclose(df["EXIT_MD"], [1050.0, 1100.0, 1150.0], atol=0.5)

    np.testing.assert_allclose(df["ENTRY_EASTING"], 250.0)
    np.testing.assert_allclose(df["EXIT_EASTING"], 250.0)
    np.testing.assert_allclose(df["ENTRY_NORTHING"], 150.0)
    np.testing.assert_allclose(df["EXIT_NORTHING"], 150.0)


@pytest.mark.parametrize(
    "i,j",
    [(0, 0), (2, 1), (4, 3), (1, 2), (3, 0)],
)
def test_vertical_well_two_samples_each_column(box_grid, i, j):
    """Coarse-resolution vertical well: ONLY 2 trajectory samples covering
    the full grid depth. The hybrid ray-tracer must still report all 3
    K-layers with the correct (i, j) and exact entry/exit depths.

    Cell (i, j) centre is ((i + 0.5)*100, (j + 0.5)*100).
    """
    cx = (i + 0.5) * 100.0
    cy = (j + 0.5) * 100.0
    md = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([cx, cx]),
        np.array([cy, cy]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(box_grid, sampling_step=10.0, zerobased=True)

    assert len(df) == 3, f"expected 3 records, got {len(df)}"
    np.testing.assert_array_equal(df["I"].to_numpy(), [i, i, i])
    np.testing.assert_array_equal(df["J"].to_numpy(), [j, j, j])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_MD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_MD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_EASTING"], cx, atol=1e-6)
    np.testing.assert_allclose(df["EXIT_EASTING"], cx, atol=1e-6)
    np.testing.assert_allclose(df["ENTRY_NORTHING"], cy, atol=1e-6)
    np.testing.assert_allclose(df["EXIT_NORTHING"], cy, atol=1e-6)


def test_vertical_well_starts_inside_grid_two_samples(box_grid):
    """Coarse vertical well that starts INSIDE the grid (in K=1) and goes
    below. Only two samples. Should produce 2 records (k=1 and k=2),
    with the entry of k=1 being the well start (not a cell face)."""
    md = np.array([1075.0, 1200.0])  # start at z=1075 (middle of K=1)
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    np.testing.assert_array_equal(df["K"].to_numpy(), [1, 2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1075.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_MD"], [1075.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_MD"], [1100.0, 1150.0], atol=1e-2)


def test_vertical_well_ends_inside_grid_two_samples(box_grid):
    """Coarse vertical well that enters from above and STOPS inside K=1
    (does not exit the grid)."""
    md = np.array([900.0, 1080.0])  # ends at z=1080 inside K=1
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0], atol=1e-2)
    # Exit of last record is the trajectory end, not a cell face
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1080.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_MD"], [1000.0, 1050.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_MD"], [1050.0, 1080.0], atol=1e-2)


def test_vertical_well_md_offset_from_z(box_grid):
    """Vertical well where MD is offset from TVDSS (e.g. RKB shift):
    MD = Z + 200. Two samples only. Verify MD reporting handles the
    offset correctly while X, Y, Z are still cell-correct."""
    z = np.array([900.0, 1200.0])
    md = z + 200.0  # MD shifted by RKB-like offset
    well = _make_well(np.array([350.0, 350.0]), np.array([250.0, 250.0]), z, md)
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    np.testing.assert_array_equal(df["I"].to_numpy(), [3, 3, 3])
    np.testing.assert_array_equal(df["J"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    # MD = Z + 200 along the trajectory
    np.testing.assert_allclose(df["ENTRY_MD"], [1200.0, 1250.0, 1300.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_MD"], [1250.0, 1300.0, 1350.0], atol=1e-2)


def test_vertical_well_on_corner_pillar_two_samples(box_grid):
    """Vertical well drilled exactly along the pillar at (x=200, y=200) —
    a shared corner of cells (1,1), (1,2), (2,1), (2,2). The algorithm
    must still produce 3 records (one per K-layer) and pick a single,
    consistent (i, j) column. Two samples only."""
    md = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([200.0, 200.0]),
        np.array([200.0, 200.0]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    assert len(df) == 3
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    # Same (i, j) for all 3 layers — algorithm must be self-consistent
    assert len(set(df["I"].tolist())) == 1
    assert len(set(df["J"].tolist())) == 1
    # And the chosen column must be one of the four meeting at the pillar
    assert df["I"].iloc[0] in (1, 2)
    assert df["J"].iloc[0] in (1, 2)
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)


# ---------------------------------------------------------------------------
# Horizontal / diagonal / deviated well tests
# ---------------------------------------------------------------------------


def test_horizontal_well_crosses_columns(box_grid):
    """A horizontal well at y=150, z=1025 crosses cells (0..4, 1, 0)
    along X (i.e. enters at i=0 and exits at i=4)."""
    x_vals = np.array([-50.0, 550.0])
    y_vals = np.array([150.0, 150.0])
    z_vals = np.array([1025.0, 1025.0])
    md_vals = np.array([0.0, 600.0])

    well = _make_well(x_vals, y_vals, z_vals, md_vals)
    df = well.get_cell_intersections(
        box_grid, sampling_step=1.0, refine_iters=25, zerobased=True
    )

    assert len(df) == 5
    np.testing.assert_array_equal(df["I"].to_numpy(), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(df["J"].to_numpy(), [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 0, 0, 0, 0])

    np.testing.assert_allclose(
        df["ENTRY_EASTING"], [0.0, 100.0, 200.0, 300.0, 400.0], atol=0.5
    )
    np.testing.assert_allclose(
        df["EXIT_EASTING"], [100.0, 200.0, 300.0, 400.0, 500.0], atol=0.5
    )

    # MD == X - x_start because the well is horizontal and starts at x=-50
    expected_entry_md = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
    expected_exit_md = np.array([150.0, 250.0, 350.0, 450.0, 550.0])
    np.testing.assert_allclose(df["ENTRY_MD"], expected_entry_md, atol=0.5)
    np.testing.assert_allclose(df["EXIT_MD"], expected_exit_md, atol=0.5)


def test_diagonal_well_crosses_i_and_j(box_grid):
    """A horizontal diagonal well at z=1025 going from corner (0,0) to
    corner (5,4) of the grid.  It must cross cells in both the I and J
    directions, producing more intersections than a straight horizontal
    traversal along a single row or column."""
    x = np.array([-50.0, 550.0])
    y = np.array([-37.5, 412.5])  # slope dy/dx = 450/600 = 0.75
    z = np.array([1025.0, 1025.0])
    md = np.array([0.0, np.hypot(600.0, 450.0)])

    well = _make_well(x, y, z, md)
    df = well.get_cell_intersections(
        box_grid, sampling_step=1.0, refine_iters=25, zerobased=True
    )

    # Must have more than 5 cells (pure row or column gives exactly 5 or 4).
    assert len(df) >= 7
    # All in K=0 (z=1025 is inside 1000-1050)
    np.testing.assert_array_equal(df["K"].to_numpy(), np.zeros(len(df), dtype=np.int32))
    # i values span 0..4 and j values span 0..3 (whole grid)
    assert df["I"].min() == 0 and df["I"].max() == 4
    assert df["J"].min() == 0 and df["J"].max() == 3
    # Entry MD < Exit MD for every record
    assert np.all(df["ENTRY_MD"].to_numpy() < df["EXIT_MD"].to_numpy() + 1e-6)


def test_md_is_consistent_with_segment_geometry(box_grid):
    """For a deviated well the MD reported at entry/exit must equal
    the linear interpolation of MD along the segment — so total MD increase
    over a single cell traversal must match the segment-internal arithmetic."""
    x = np.linspace(50.0, 450.0, 21)
    y = np.linspace(50.0, 350.0, 21)
    z = np.linspace(1010.0, 1140.0, 21)
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    md = np.concatenate([[0.0], np.cumsum(seg)])

    well = _make_well(x, y, z, md)
    df = well.get_cell_intersections(
        box_grid, sampling_step=1.0, refine_iters=30, zerobased=True
    )

    assert len(df) >= 1
    entry_md = df["ENTRY_MD"].to_numpy()
    exit_md = df["EXIT_MD"].to_numpy()
    # Entry MD must be < Exit MD for every record
    assert np.all(entry_md < exit_md + 1e-6)
    # MDs must be monotonically non-decreasing across consecutive records
    assert np.all(np.diff(entry_md) >= -1e-6)
    # Recover (X, Y, Z) at the reported MDs by interpolating the original log
    for arr_md, col_x, col_y, col_z in [
        (entry_md, "ENTRY_EASTING", "ENTRY_NORTHING", "ENTRY_TVD"),
        (exit_md, "EXIT_EASTING", "EXIT_NORTHING", "EXIT_TVD"),
    ]:
        np.testing.assert_allclose(
            np.interp(arr_md, md, x), df[col_x].to_numpy(), atol=0.5
        )
        np.testing.assert_allclose(
            np.interp(arr_md, md, y), df[col_y].to_numpy(), atol=0.5
        )
        np.testing.assert_allclose(
            np.interp(arr_md, md, z), df[col_z].to_numpy(), atol=0.5
        )


# ---------------------------------------------------------------------------
# Edge / degenerate cases
# ---------------------------------------------------------------------------


def test_well_entirely_outside_grid_returns_empty(box_grid):
    """A well entirely outside the grid bounding box -> no intersections."""
    x = np.array([-1000.0, -900.0])
    y = np.array([-1000.0, -900.0])
    z = np.array([1100.0, 1110.0])
    md = np.array([0.0, 100.0])
    well = _make_well(x, y, z, md)
    df = well.get_cell_intersections(box_grid, sampling_step=10.0, zerobased=True)
    assert len(df) == 0


def test_short_trajectory_returns_empty(box_grid):
    """Trajectory with a single sample yields no intersections."""
    df_well = pd.DataFrame(
        {
            "X_UTME": [250.0],
            "Y_UTMN": [150.0],
            "Z_TVDSS": [1025.0],
            "M_MDEPTH": [0.0],
        }
    )
    well = xtgeo.Well(
        rkb=0.0, xpos=250.0, ypos=150.0, wname="W", df=df_well, mdlogname="M_MDEPTH"
    )
    df = well.get_cell_intersections(box_grid, zerobased=True)
    assert len(df) == 0


def test_well_reentries_grid(box_grid):
    """A trajectory that enters the grid, exits through the top, and
    re-enters from above — exercising fallback-path recovery after
    the ray-tracing fast path loses the current cell."""
    pts = np.array(
        [
            [-50.0, 150.0, 1025.0],
            [175.0, 150.0, 1025.0],
            [250.0, 150.0, 900.0],
            [325.0, 150.0, 1025.0],
            [550.0, 150.0, 1025.0],
        ]
    )
    segs = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    md = np.concatenate([[0.0], np.cumsum(segs)])

    well = _make_well(pts[:, 0], pts[:, 1], pts[:, 2], md)
    df = well.get_cell_intersections(
        box_grid, sampling_step=1.0, refine_iters=25, zerobased=True
    )

    i_vals = df["I"].tolist()
    assert 0 in i_vals, "should hit column i=0"
    assert 1 in i_vals, "should hit column i=1"
    assert 3 in i_vals or 4 in i_vals, "should re-enter and hit columns i=3 or i=4"
    assert len(df) >= 4
    # All at K=0
    np.testing.assert_array_equal(df["K"].to_numpy(), np.zeros(len(df), dtype=np.int32))
    # MDs must be monotonically non-decreasing
    assert np.all(np.diff(df["ENTRY_MD"].to_numpy()) >= -1e-6)


def test_zero_length_segment(box_grid):
    """A trajectory with a duplicated point (zero-length segment) must
    not crash or produce garbage."""
    z = np.array([900.0, 1050.0, 1050.0, 1200.0])
    md = z.copy()
    x = np.full_like(z, 250.0)
    y = np.full_like(z, 150.0)

    well = _make_well(x, y, z, md)
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1.0)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1.0)


def test_well_along_cell_face_boundary(box_grid):
    """Vertical well drilled exactly on the J-face boundary at y=100.
    This is a degenerate configuration; the algorithm must still produce
    3 records (one per K-layer) with a single consistent j value."""
    md = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([100.0, 100.0]),  # exactly on j=0/j=1 boundary
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(box_grid, sampling_step=5.0, zerobased=True)

    assert len(df) == 3
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    assert len(set(df["J"].tolist())) == 1
    assert df["J"].iloc[0] in (0, 1)
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)


# ---------------------------------------------------------------------------
# Ray-tracing / precision tests
# ---------------------------------------------------------------------------


def test_ray_tracing_precision_on_box_grid(box_grid):
    """The hybrid algorithm uses analytic ray-tracing on convex (planar-faced)
    cells. On an axis-aligned box grid this should give near machine-precision
    entry/exit coordinates."""
    md_vals = np.array([900.0, 1200.0])
    z_vals = md_vals.copy()
    x_vals = np.full_like(md_vals, 150.0)
    y_vals = np.full_like(md_vals, 250.0)

    well = _make_well(x_vals, y_vals, z_vals, md_vals)
    df = well.get_cell_intersections(
        box_grid, sampling_step=5.0, refine_iters=10, zerobased=True
    )

    assert len(df) == 3
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_MD"], [1000.0, 1050.0, 1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_MD"], [1050.0, 1100.0, 1150.0], atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_EASTING"], 150.0, atol=1e-6)
    np.testing.assert_allclose(df["ENTRY_NORTHING"], 250.0, atol=1e-6)


def test_refine_iters_zero(box_grid):
    """With refine_iters=0 (no bisection) the algorithm still produces
    correct cell indices, though boundary positions may be less precise."""
    md = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(
        box_grid, sampling_step=5.0, refine_iters=0, zerobased=True
    )

    np.testing.assert_array_equal(df["I"].to_numpy(), [2, 2, 2])
    np.testing.assert_array_equal(df["J"].to_numpy(), [1, 1, 1])
    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 1, 2])
    # Positions should be in the right ballpark (wider tolerance)
    np.testing.assert_allclose(df["ENTRY_TVD"], [1000.0, 1050.0, 1100.0], atol=5.0)
    np.testing.assert_allclose(df["EXIT_TVD"], [1050.0, 1100.0, 1150.0], atol=5.0)


# ---------------------------------------------------------------------------
# active_only tests
# ---------------------------------------------------------------------------


def test_active_only_skips_inactive_cells(box_grid):
    """When `active_only=True`, cells with ACTNUM=0 must NOT be reported."""
    grid_for_test = box_grid.copy()
    grid_for_test._actnumsv[2, 1, 1] = 0

    md_vals = np.array([1000.0, 1150.0])
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md_vals.copy(),
        md_vals,
    )

    df_all = well.get_cell_intersections(
        grid_for_test, sampling_step=2.0, active_only=False, zerobased=True
    )
    df_act = well.get_cell_intersections(
        grid_for_test, sampling_step=2.0, active_only=True, zerobased=True
    )

    # All 3 layers when including inactive cells
    np.testing.assert_array_equal(df_all["K"].to_numpy(), [0, 1, 2])
    # Only k=0 and k=2 when active_only=True
    np.testing.assert_array_equal(df_act["K"].to_numpy(), [0, 2])


def test_active_only_md_continuity_through_inactive_layer(box_grid):
    """When `active_only=True` and a middle layer is inactive, only the active
    cells are reported. The reported MDs must remain monotonic, but there is an
    unreported interval through the inactive block, so exit_md before it is less
    than entry_md after it."""
    grid_for_test = box_grid.copy()
    grid_for_test._actnumsv[2, 1, 1] = 0

    md_vals = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md_vals.copy(),
        md_vals,
    )
    df = well.get_cell_intersections(
        grid_for_test, sampling_step=5.0, active_only=True, zerobased=True
    )

    np.testing.assert_array_equal(df["K"].to_numpy(), [0, 2])
    np.testing.assert_allclose(df["EXIT_MD"].iloc[0], 1050.0, atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_MD"].iloc[1], 1100.0, atol=1e-2)
    np.testing.assert_allclose(df["ENTRY_EASTING"], 250.0, atol=1e-6)
    np.testing.assert_allclose(df["ENTRY_NORTHING"], 150.0, atol=1e-6)


def test_multiple_consecutive_inactive_layers(box_grid):
    """When two consecutive K-layers are inactive and `active_only=True`,
    the ride-through must chain across both before finding the next
    active cell."""
    grid_for_test = box_grid.copy()
    grid_for_test._actnumsv[2, 1, 0] = 0
    grid_for_test._actnumsv[2, 1, 1] = 0

    md = np.array([900.0, 1200.0])
    well = _make_well(
        np.array([250.0, 250.0]),
        np.array([150.0, 150.0]),
        md.copy(),
        md,
    )
    df = well.get_cell_intersections(
        grid_for_test, sampling_step=5.0, active_only=True, zerobased=True
    )

    # Only the bottom layer k=2 is active
    np.testing.assert_array_equal(df["K"].to_numpy(), [2])
    np.testing.assert_allclose(df["ENTRY_TVD"], [1100.0], atol=1e-2)
    np.testing.assert_allclose(df["EXIT_TVD"], [1150.0], atol=1e-2)
