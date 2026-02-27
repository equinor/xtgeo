"""Comprehensive tests for grid_merge functionality."""

import pathlib
import warnings

import numpy as np
import pytest

import xtgeo
from xtgeo.common.log import functimer, null_logger

logger = null_logger(__name__)

EMERALD = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")


def test_basic_merge():
    """Test basic merging of two grids."""
    g1 = xtgeo.create_box_grid((3, 2, 2))
    g2 = xtgeo.create_box_grid((4, 5, 2))

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.ncol == 8  # 3 + 4 + 1
    assert merged.nrow == 5  # max(2, 5)
    assert merged.nlay == 2

    assert merged.nactive == g1.nactive + g2.nactive


def test_default_placement():
    """Test merging with default placement (1-cell gap)."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((2, 2, 1))

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.ncol == 2 + 1 + 2  # g1.ncol + 1 + g2.ncol
    assert merged.nrow == 2  # max(g1.nrow, g2.nrow)
    assert merged.nactive == g1.nactive + g2.nactive


def test_geometry_preservation():
    """Test that grid geometries are preserved after merging."""
    g1 = xtgeo.create_box_grid(
        dimension=(2, 2, 1),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )
    g2 = xtgeo.create_box_grid(
        dimension=(2, 2, 1),
        origin=(500.0, 500.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )

    g1._set_xtgformat2()
    g2._set_xtgformat2()
    g1_coordsv = g1._coordsv.copy()
    g2_coordsv = g2._coordsv.copy()

    merged = xtgeo.grid_merge(g1, g2)
    merged._set_xtgformat2()

    # Check that g1's coordinates are preserved at (0, 0)
    # g1 has shape (2+1, 2+1, 6) = (3, 3, 6)
    np.testing.assert_array_almost_equal(
        merged._coordsv[0:3, 0:3, :],
        g1_coordsv,
        decimal=5,
        err_msg="Grid1 coordinates not preserved",
    )

    # Check that g2's coordinates are preserved at (3, 0)
    # g2 has shape (2+1, 2+1, 6) = (3, 3, 6)
    np.testing.assert_array_almost_equal(
        merged._coordsv[3:6, 0:3, :],
        g2_coordsv,
        decimal=5,
        err_msg="Grid2 coordinates not preserved",
    )


def test_handedness_consistency():
    """Test that grids with different handedness are properly handled."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((2, 2, 1))

    g1_handedness = g1.ijk_handedness
    g2_handedness = g2.ijk_handedness

    # If they're the same, reverse one to test the fix
    if g1_handedness == g2_handedness:
        g2.reverse_row_axis()

    g2_handedness_before = g2.ijk_handedness

    assert g1.ijk_handedness != g2_handedness_before, (
        "Grids should have different handedness for this test"
    )

    merged = xtgeo.grid_merge(g1, g2)

    # Verify the merged grid has the same handedness as g1
    assert merged.ijk_handedness == g1_handedness, (
        f"Merged grid should have g1's handedness ({g1_handedness})"
    )

    # Verify that the original g2 was NOT mutated (should still have
    # different handedness)
    assert g2.ijk_handedness == g2_handedness_before, (
        "Original grid2 should not be mutated by grid_merge"
    )


def test_actnum_values():
    """Test that actnum values are correct after merging."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((3, 3, 1))

    g1._set_xtgformat2()
    g2._set_xtgformat2()

    g1._actnumsv[0, 0, 0] = 0
    g2._actnumsv[1, 1, 0] = 0

    original_g1_active = np.sum(g1._actnumsv)
    original_g2_active = np.sum(g2._actnumsv)

    merged = xtgeo.grid_merge(g1, g2)
    merged._set_xtgformat2()

    # Check total active cells
    assert merged.nactive == original_g1_active + original_g2_active

    # Check that g1's inactive cell is preserved
    assert merged._actnumsv[0, 0, 0] == 0

    # Check that g2's inactive cell is preserved at offset (2+1, 0)
    assert merged._actnumsv[3 + 1, 1, 0] == 0


def test_gap_cells_inactive():
    """Test that cells in gaps between grids are inactive."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((2, 2, 1))

    merged = xtgeo.grid_merge(g1, g2)
    merged._set_xtgformat2()

    # With default 1-cell gap, cell at i=2 should be inactive
    assert merged._actnumsv[2, 0, 0] == 0, "Gap cell should be inactive"

    # But cells in g1 (i=0,1) and g2 (i=3,4) should be active
    assert merged._actnumsv[0, 0, 0] == 1, "G1 cell should be active"
    assert merged._actnumsv[1, 0, 0] == 1, "G1 cell should be active"
    assert merged._actnumsv[3, 0, 0] == 1, "G2 cell should be active"
    assert merged._actnumsv[4, 0, 0] == 1, "G2 cell should be active"


# ============================================================================
# Different Layer Count Tests
# ============================================================================


def test_merge_different_layers():
    """Test merging grids with different numbers of layers."""
    g1 = xtgeo.create_box_grid((10, 7, 3), origin=(0, 0, 0), increment=(100, 100, 10))
    g2 = xtgeo.create_box_grid((2, 2, 6), origin=(100, 100, 0), increment=(60, 60, 30))

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.nlay == 6, f"Expected 6 layers, got {merged.nlay}"
    assert merged.ncol == g1.ncol + g2.ncol + 1, (
        f"Expected {g1.ncol + g2.ncol + 1} cols, got {merged.ncol}"
    )
    assert merged.nrow == max(g1.nrow, g2.nrow), (
        f"Expected {max(g1.nrow, g2.nrow)} rows, got {merged.nrow}"
    )

    # Check active cells
    # g1 has 10*7*3 = 210 active cells in first 3 layers
    # g2 has 2*2*6 = 24 active cells in all 6 layers
    # Total expected: 210 + 24 = 234
    expected_active = g1.nactive + g2.nactive
    assert merged.nactive == expected_active

    # Check that g1's extended layers (3, 4, 5) are inactive
    g1_extended_layers = merged._actnumsv[0 : g1.ncol, 0 : g1.nrow, 3:6]
    assert np.all(g1_extended_layers == 0), "G1's extended layers should be inactive"

    # Check that g1's original layers (0, 1, 2) are active
    g1_original_layers = merged._actnumsv[0 : g1.ncol, 0 : g1.nrow, 0:3]
    assert np.all(g1_original_layers == 1), "G1's original layers should be active"


def test_merge_g1_more_layers():
    """Test when grid1 has more layers than grid2."""
    g1 = xtgeo.create_box_grid((3, 3, 5))
    g2 = xtgeo.create_box_grid((2, 2, 2))

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.nlay == 5, f"Expected 5 layers, got {merged.nlay}"

    # g2 should have extended layers 2, 3, 4 inactive
    merged._set_xtgformat2()
    g2_offset = g1.ncol + 1
    g2_extended_layers = merged._actnumsv[
        g2_offset : g2_offset + g2.ncol, 0 : g2.nrow, 2:5
    ]
    assert np.all(g2_extended_layers == 0), "G2's extended layers should be inactive"

    # g2's original layers should be active
    g2_original_layers = merged._actnumsv[
        g2_offset : g2_offset + g2.ncol, 0 : g2.nrow, 0:2
    ]
    assert np.all(g2_original_layers == 1), "G2's original layers should be active"


def test_layer_extension_geometry():
    """Test that the extended layers have correct geometry."""
    g1 = xtgeo.create_box_grid(
        dimension=(2, 2, 2),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )
    g2 = xtgeo.create_box_grid(
        dimension=(2, 2, 4),
        origin=(500.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )

    merged = xtgeo.grid_merge(g1, g2)
    merged._set_xtgformat2()

    # Check that g1's bottom is extended
    # Original g1 bottom is at layer 2 (z-index 2)
    # Layer 1-2 transition should be ~10.0 units
    g1_layer1_bottom = merged._zcornsv[0:3, 0:3, 2, :]
    g1_layer2_bottom = merged._zcornsv[0:3, 0:3, 3, :]

    # The extended layer should maintain the same thickness
    thickness_diff = g1_layer2_bottom - g1_layer1_bottom
    expected_thickness = 10.0

    # Allow small floating point differences
    assert np.allclose(thickness_diff, expected_thickness, atol=0.1), (
        f"Extended layer thickness should be ~{expected_thickness}"
    )


# ============================================================================
# Refined Grid Tests
# ============================================================================


def test_merge_refined_grid(tmp_path):
    """Test merging with a refined cropped grid."""
    g1 = xtgeo.create_box_grid((10, 7, 3), origin=(0, 0, 0), increment=(100, 100, 10))

    g2 = g1.copy()
    g2.crop(colcrop=(2, 3), rowcrop=(3, 4), laycrop=(1, 1))
    g2.refine(4, 4, 3)

    merged = xtgeo.grid_merge(g1, g2)

    # Verify dimensions
    expected_nlay = max(g1.nlay, g2.nlay)
    assert merged.nlay == expected_nlay, (
        f"Expected {expected_nlay} layers, got {merged.nlay}"
    )

    test_file = tmp_path / "merged_refined.grdecl"
    merged.to_file(str(test_file), fformat="grdecl")

    merged_read = xtgeo.grid_from_file(str(test_file), fformat="grdecl")
    assert merged_read.dimensions == merged.dimensions


def test_gap_pillar_geometry(tmp_path):
    """Test that gap pillars have valid geometry."""
    g1 = xtgeo.create_box_grid((2, 2, 1), origin=(0, 0, 1000), increment=(100, 100, 10))
    g2 = xtgeo.create_box_grid(
        (2, 2, 1), origin=(500, 500, 1000), increment=(100, 100, 10)
    )

    merged = xtgeo.grid_merge(g1, g2)

    merged._set_xtgformat2()

    # Gap cells should be inactive
    gap_cells = merged._actnumsv[2:5, 2:5, :]
    assert gap_cells.sum() == 0, "Gap cells should be inactive"


# ============================================================================
# RMS API Compatibility Tests
# ============================================================================


def test_merged_grid_rms_api_no_warnings():
    """Test that merged grids convert to RMS API without warnings."""
    g1 = xtgeo.create_box_grid(dimension=(5, 5, 2))
    g2 = xtgeo.create_box_grid(dimension=(3, 3, 2))

    merged = xtgeo.grid_merge(g1, g2)

    grid_cpp = merged._get_grid_cpp()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid_cpp.convert_xtgeo_to_rmsapi()

        # Check for "Equal Z coordinates" warning
        equal_z_warnings = [
            warning
            for warning in w
            if "Equal Z coordinates detected" in str(warning.message)
        ]

        assert len(equal_z_warnings) == 0, (
            f"Merged grid produced {len(equal_z_warnings)} 'Equal Z coordinates' "
            "warning(s) during RMS API conversion"
        )


def test_refined_merged_grid_rms_api():
    """Test RMS API conversion with refined grid merge (user's workflow)."""
    g1 = xtgeo.create_box_grid(dimension=(10, 7, 3))
    g2 = g1.copy()
    g2.crop((1, 1), (2, 2), (1, 1))
    g2.refine(4, 4, 3)

    merged = xtgeo.grid_merge(g1, g2)

    grid_cpp = merged._get_grid_cpp()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid_cpp.convert_xtgeo_to_rmsapi()

        equal_z_warnings = [
            warning
            for warning in w
            if "Equal Z coordinates detected" in str(warning.message)
        ]

        assert len(equal_z_warnings) == 0, (
            "Refined merged grid produced 'Equal Z coordinates' warning"
        )


def test_different_layers_rms_api():
    """Test RMS API conversion with different layer counts."""
    g1 = xtgeo.create_box_grid(dimension=(5, 5, 3))
    g2 = xtgeo.create_box_grid(dimension=(3, 3, 2))

    merged = xtgeo.grid_merge(g1, g2)

    grid_cpp = merged._get_grid_cpp()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid_cpp.convert_xtgeo_to_rmsapi()

        equal_z_warnings = [
            warning
            for warning in w
            if "Equal Z coordinates detected" in str(warning.message)
        ]

        assert len(equal_z_warnings) == 0, (
            "Different layer counts produced 'Equal Z coordinates' warning"
        )


# ============================================================================
# Property Merging Tests
# ============================================================================


def test_merge_with_continuous_properties():
    """Test merging grids with continuous properties."""
    g1 = xtgeo.create_box_grid((3, 3, 2))
    g2 = xtgeo.create_box_grid((2, 2, 2))

    poro1 = xtgeo.GridProperty(g1, name="PORO", values=np.full(g1.dimensions, 0.25))
    g1.append_prop(poro1)

    poro2 = xtgeo.GridProperty(g2, name="PORO", values=np.full(g2.dimensions, 0.30))
    g2.append_prop(poro2)

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.get_prop_by_name("PORO") is not None

    merged_poro = merged.get_prop_by_name("PORO")
    assert np.allclose(merged_poro.values[0:3, 0:3, :], 0.25)

    assert np.allclose(merged_poro.values[4:6, 0:2, :], 0.30)

    gap_val = merged_poro.values[3, 0, 0]
    if not np.ma.is_masked(gap_val):
        assert np.isclose(gap_val, 0.0)


def test_merge_with_discrete_properties_same_codes():
    """Test merging grids with discrete properties having same codes."""
    g1 = xtgeo.create_box_grid((3, 3, 2))
    g2 = xtgeo.create_box_grid((2, 2, 2))

    zone1 = xtgeo.GridProperty(
        g1,
        name="ZONE",
        discrete=True,
        values=np.full(g1.dimensions, 1, dtype=np.int32),
        codes={1: "Upper", 2: "Lower"},
    )
    g1.append_prop(zone1)

    zone2 = xtgeo.GridProperty(
        g2,
        name="ZONE",
        discrete=True,
        values=np.full(g2.dimensions, 2, dtype=np.int32),
        codes={1: "Upper", 2: "Lower"},
    )
    g2.append_prop(zone2)

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.get_prop_by_name("ZONE") is not None

    merged_zone = merged.get_prop_by_name("ZONE")
    assert merged_zone.isdiscrete is True

    assert merged_zone.codes == {1: "Upper", 2: "Lower"}

    assert np.all(merged_zone.values[0:3, 0:3, :] == 1)
    assert np.all(merged_zone.values[4:6, 0:2, :] == 2)


def test_merge_with_discrete_properties_different_codes():
    """Test merging grids with discrete properties having different codes."""
    g1 = xtgeo.create_box_grid((3, 3, 2))
    g2 = xtgeo.create_box_grid((2, 2, 2))

    zone1 = xtgeo.GridProperty(
        g1,
        name="ZONE",
        discrete=True,
        values=np.full(g1.dimensions, 1, dtype=np.int32),
        codes={1: "Upper", 2: "Lower"},
    )
    g1.append_prop(zone1)

    zone2 = xtgeo.GridProperty(
        g2,
        name="ZONE",
        discrete=True,
        values=np.full(g2.dimensions, 1, dtype=np.int32),
        codes={1: "Top", 2: "Bottom"},  # Different codes!
    )
    g2.append_prop(zone2)

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.get_prop_by_name("ZONE") is not None
    assert merged.get_prop_by_name("ZONE_2") is not None

    zone_g1 = merged.get_prop_by_name("ZONE")
    assert zone_g1.codes == {1: "Upper", 2: "Lower"}

    zone_g2 = merged.get_prop_by_name("ZONE_2")
    assert zone_g2.codes == {1: "Top", 2: "Bottom"}


def test_merge_with_multiple_properties():
    """Test merging grids with multiple different properties."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((2, 2, 1))

    poro1 = xtgeo.GridProperty(g1, name="PORO", values=np.full(g1.dimensions, 0.2))
    perm1 = xtgeo.GridProperty(g1, name="PERM", values=np.full(g1.dimensions, 100.0))
    g1.append_prop(poro1)
    g1.append_prop(perm1)

    poro2 = xtgeo.GridProperty(g2, name="PORO", values=np.full(g2.dimensions, 0.3))
    swat2 = xtgeo.GridProperty(g2, name="SWAT", values=np.full(g2.dimensions, 0.5))
    g2.append_prop(poro2)
    g2.append_prop(swat2)

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.get_prop_by_name("PORO") is not None
    assert merged.get_prop_by_name("PERM") is not None
    assert merged.get_prop_by_name("SWAT") is not None

    # PERM should only have values from g1
    perm_merged = merged.get_prop_by_name("PERM")
    assert np.allclose(perm_merged.values[0:2, 0:2, :], 100.0)
    # g2 region is zero (may be masked if inactive)
    g2_perm = perm_merged.values[3:5, 0:2, :]
    if isinstance(g2_perm, np.ma.MaskedArray):
        assert np.allclose(g2_perm[~g2_perm.mask], 0.0)
    else:
        assert np.allclose(g2_perm, 0.0)

    # SWAT should only have values from g2
    swat_merged = merged.get_prop_by_name("SWAT")
    # g1 region is zero (may be masked if inactive)
    g1_swat = swat_merged.values[0:2, 0:2, :]
    if isinstance(g1_swat, np.ma.MaskedArray):
        assert np.allclose(g1_swat[~g1_swat.mask], 0.0)
    else:
        assert np.allclose(g1_swat, 0.0)
    assert np.allclose(swat_merged.values[3:5, 0:2, :], 0.5)


def test_merge_no_properties():
    """Test merging grids without any properties."""
    g1 = xtgeo.create_box_grid((2, 2, 1))
    g2 = xtgeo.create_box_grid((2, 2, 1))

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.props is None or len(merged.props) == 0


def test_merge_with_rotated_grid():
    """Test merging a grid with a rotated copy - expose coordinate line issues."""
    g1 = xtgeo.create_box_grid((3, 3, 2))

    g2 = g1.copy()
    g2.translate_coordinates(
        translate=(10.0, 10.0, 0.0),
        add_rotation=120.0,
    )

    xyz1 = g1.get_xyz(asmasked=False)
    xyz2 = g2.get_xyz(asmasked=False)

    assert not np.allclose(xyz1[0].values, xyz2[0].values)
    assert not np.allclose(xyz1[1].values, xyz2[1].values)

    g1_coordsv_orig = g1._coordsv.copy()
    g2_coordsv_orig = g2._coordsv.copy()

    merged = xtgeo.grid_merge(g1, g2)

    assert merged.nactive == g1.nactive + g2.nactive
    assert merged.dimensions.ncol == 7

    merged_xyz = merged.get_xyz(asmasked=False)
    assert np.all(np.isfinite(merged_xyz[0].values[~merged_xyz[0].values.mask]))
    assert np.all(np.isfinite(merged_xyz[1].values[~merged_xyz[1].values.mask]))
    assert np.all(np.isfinite(merged_xyz[2].values[~merged_xyz[2].values.mask]))

    merged_g1_coords = merged._coordsv[0:4, 0:4, :]
    assert np.all(np.isfinite(merged_g1_coords))

    merged_g2_coords = merged._coordsv[4:8, 0:4, :]
    assert np.all(np.isfinite(merged_g2_coords))

    assert np.allclose(merged._coordsv[0:4, 0:4, :], g1_coordsv_orig, rtol=1e-10)

    assert np.allclose(merged._coordsv[4:8, 0:4, :], g2_coordsv_orig, rtol=1e-10)

    assert np.allclose(merged_xyz[0].values[0:3, 0:3, :], xyz1[0].values, rtol=1e-5)

    assert np.allclose(merged_xyz[0].values[4:7, 0:3, :], xyz2[0].values, rtol=1e-5)


def test_merge_with_real_world_coordinates():
    """Test merging grids with real-world UTM coordinates.

    This test exposes a bug where gap pillars with large coordinates
    become zero because _fix_zero_pillars only sets Z, leaving X,Y at zero.
    """
    g1 = xtgeo.create_box_grid(
        dimension=(3, 3, 2),
        origin=(500000.0, 6000000.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )

    g2 = xtgeo.create_box_grid(
        dimension=(3, 3, 2),
        origin=(500500.0, 6000000.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )

    merged = xtgeo.grid_merge(g1, g2)

    merged._set_xtgformat2()

    x_top = merged._coordsv[:, :, 0]
    x_bot = merged._coordsv[:, :, 3]

    # All X coordinates should be around 500000
    assert np.all(x_top > 400000), (
        f"Found X_top values < 400000: {x_top[x_top < 400000]}"
    )
    assert np.all(x_bot > 400000), (
        f"Found X_bot values < 400000: {x_bot[x_bot < 400000]}"
    )

    y_top = merged._coordsv[:, :, 1]
    y_bot = merged._coordsv[:, :, 4]

    assert np.all(y_top > 5900000), (
        f"Found Y_top values < 5900000: {y_top[y_top < 5900000]}"
    )
    assert np.all(y_bot > 5900000), (
        f"Found Y_bot values < 5900000: {y_bot[y_bot < 5900000]}"
    )

    # Verify the gap pillars (column 3) have reasonable coordinates
    gap_pillars = merged._coordsv[3, :, :]

    gap_x_top = gap_pillars[:, 0]
    gap_x_bot = gap_pillars[:, 3]

    assert np.all(gap_x_top > 500200), f"Gap pillar X_top too small: {gap_x_top}"
    assert np.all(gap_x_top < 500600), f"Gap pillar X_top too large: {gap_x_top}"
    assert np.all(gap_x_bot > 500200), f"Gap pillar X_bot too small: {gap_x_bot}"
    assert np.all(gap_x_bot < 500600), f"Gap pillar X_bot too large: {gap_x_bot}"


@pytest.mark.bigtest
def test_grid_merge_speed():
    """Show the speed of a very large case, enable with logging output in pytest."""
    g1 = xtgeo.create_box_grid(
        dimension=(300, 400, 58),
        origin=(500000.0, 6000000.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )

    g2 = xtgeo.create_box_grid(
        dimension=(200, 600, 99),
        origin=(501500.0, 6000000.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )
    logger.debug(
        "Created box grids... million active cells %s and %s",
        g1.nactive / 1e6,
        g2.nactive / 1e6,
    )

    @functimer(comment="Merging big grids")
    def check_speed():
        return xtgeo.grid_merge(g1, g2)

    merged = check_speed()
    assert merged.dimensions == (
        g1.ncol + g2.ncol + 1,
        max(g1.nrow, g2.nrow),
        max(g1.nlay, g2.nlay),
    )


def test_grid_merge_emerald(testdata_path):
    """Use the Emerald case"""
    g1 = xtgeo.grid_from_file(testdata_path / EMERALD)
    g2 = g1.copy()

    g2.translate_coordinates((10000, 10000, 0), add_rotation=120)

    logger.debug(
        "Emerald case, million active cells %s and %s",
        g1.nactive / 1e6,
        g2.nactive / 1e6,
    )

    @functimer(comment="Merging big grids")
    def check_speed():
        return xtgeo.grid_merge(g1, g2)

    merged = check_speed()

    assert merged._coordsv.min() > 1000.0
    assert merged.dimensions == (
        g1.ncol + g2.ncol + 1,
        max(g1.nrow, g2.nrow),
        max(g1.nlay, g2.nlay),
    )


def test_merge_with_properties_crop_and_refine():
    """Test merging with a grid that has properties, then crop and refine."""
    # Create original grid with properties
    g1 = xtgeo.create_box_grid(
        (10, 8, 4), origin=(0, 0, 1000), increment=(100, 100, 10)
    )

    # Add continuous property
    poro1 = xtgeo.GridProperty(
        g1,
        name="PORO",
        values=np.full(g1.dimensions, 0.25),
    )
    g1.append_prop(poro1)
    print(g1._props.props)

    # Add discrete property
    zone1 = xtgeo.GridProperty(
        g1,
        name="ZONE",
        discrete=True,
        values=np.full(g1.dimensions, 1, dtype=np.int32),
        codes={1: "Upper", 2: "Middle", 3: "Lower"},
    )
    g1.append_prop(zone1)

    # Make a copy, crop, and refine
    g2 = g1.copy()
    g2.crop(colcrop=(3, 5), rowcrop=(2, 4), laycrop=(1, 2), props="all")
    g2.refine(2, 2, 2)

    # Verify g2 has the expected dimensions after crop and refine
    # After crop: (3-3+1)*(2-2+1)*(2-1+1) = 3*3*2
    # After refine by 2,2,2: 6*6*4
    assert g2.ncol == 6, f"Expected g2.ncol=6, got {g2.ncol}"
    assert g2.nrow == 6, f"Expected g2.nrow=6, got {g2.nrow}"
    assert g2.nlay == 4, f"Expected g2.nlay=4, got {g2.nlay}"

    # Verify properties exist in g2
    assert g2.get_prop_by_name("PORO") is not None
    assert g2.get_prop_by_name("ZONE") is not None

    # Merge the original grid with the refined cropped grid
    merged = xtgeo.grid_merge(g1, g2)

    # Verify merged dimensions
    expected_ncol = g1.ncol + g2.ncol + 1  # 10 + 6 + 1 = 17
    expected_nrow = max(g1.nrow, g2.nrow)  # max(8, 6) = 8
    expected_nlay = max(g1.nlay, g2.nlay)  # max(4, 4) = 4

    assert merged.ncol == expected_ncol, (
        f"Expected ncol={expected_ncol}, got {merged.ncol}"
    )
    assert merged.nrow == expected_nrow, (
        f"Expected nrow={expected_nrow}, got {merged.nrow}"
    )
    assert merged.nlay == expected_nlay, (
        f"Expected nlay={expected_nlay}, got {merged.nlay}"
    )

    # Verify properties exist in merged grid
    assert merged.get_prop_by_name("PORO") is not None, "PORO missing"
    assert merged.get_prop_by_name("ZONE") is not None, "ZONE missing"

    merged_poro = merged.get_prop_by_name("PORO")
    merged_zone = merged.get_prop_by_name("ZONE")

    # Verify property types
    assert merged_poro.isdiscrete is False
    assert merged_zone.isdiscrete is True
    assert merged_zone.codes == {1: "Upper", 2: "Middle", 3: "Lower"}

    # Verify property values in g1 region (original grid area)
    assert np.allclose(merged_poro.values[0 : g1.ncol, 0 : g1.nrow, :], 0.25)
    assert np.all(merged_zone.values[0 : g1.ncol, 0 : g1.nrow, :] == 1)

    # Verify property values in g2 region (refined cropped grid area)
    # g2 starts at column g1.ncol + 1 (after the gap)
    g2_col_start = g1.ncol + 1
    g2_col_end = g2_col_start + g2.ncol
    g2_values = merged_poro.values[g2_col_start:g2_col_end, 0 : g2.nrow, :]
    assert np.allclose(g2_values, 0.25), "PORO values in g2 region should be 0.25"

    # Verify active cell count
    expected_active = g1.nactive + g2.nactive
    assert merged.nactive == expected_active, (
        f"Expected {expected_active} active cells, got {merged.nactive}"
    )
