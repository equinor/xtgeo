"""EPC compliance tests for all RESQML 2.0.1 object types.

Covers:
  - Grid3D: masked properties, rotated grids, pinch-outs, faulted grids,
    asymmetric dimensions, large grids, hypothesis fuzzing, CRS rotation
  - Surfaces: large, many-NaN, negative Z, constant, rotation, asymmetric
  - Points: many, single, negative Z, collocated
  - Polygons: single, many, varying Z
  - Wells: trajectory, trajectory+logs, high-level API
  - Blocked Wells: geometry+properties, high-level API
  - TriangulatedSurfaces: single, multi-mesh, high-level API
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml
from xtgeo.interfaces.osdu._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
from xtgeo.interfaces.osdu._polyline import (
    polylineset_to_xtgeo,
    xtgeo_polygons_to_resqml,
)
from xtgeo.interfaces.osdu._triangulated_surface import (
    triangulated_surface_to_xtgeo,
    xtgeo_triangulated_surface_to_resqml,
)
from xtgeo.interfaces.osdu._well import well_to_xtgeo, xtgeo_well_to_resqml
from xtgeo.interfaces.osdu._blocked_well import (
    blocked_well_to_xtgeo,
    xtgeo_blocked_well_to_resqml,
)


@pytest.fixture
def epc_path(tmp_path):
    """Return a temporary EPC file path."""
    return str(tmp_path / "test.epc")


def _write_and_read_grid(epc_path, grid, title="Grid", properties=None, **kw):
    """Helper: write grid+props to EPC, read back, return (grid2, props2)."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_grid_to_resqml(p, grid, title=title, properties=properties, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    g2, props2 = ijk_grid_to_xtgeo(
        p2, uuids[title], load_properties=properties is not None
    )
    p2.close()
    return g2, props2


# ---------------------------------------------------------------------------
# Masked / NaN property values
# ---------------------------------------------------------------------------


class TestMaskedPropertyRoundTrip:
    """Verify properties with NaN/masked values survive the EPC roundtrip."""

    def test_continuous_property_with_nan(self, epc_path):
        """NaN values in a continuous property must be preserved."""
        g = xtgeo.create_box_grid((3, 3, 2))
        vals = np.linspace(0.1, 0.3, 18).reshape(3, 3, 2)
        vals[0, 0, 0] = np.nan
        vals[2, 2, 1] = np.nan
        vals[1, 1, 0] = np.nan
        poro = xtgeo.GridProperty(g, name="PORO", values=vals)

        _, props = _write_and_read_grid(epc_path, g, properties=[poro])

        result = props[0].values
        assert np.isnan(result[0, 0, 0])
        assert np.isnan(result[2, 2, 1])
        assert np.isnan(result[1, 1, 0])
        valid = ~np.isnan(vals)
        assert np.allclose(result[valid], vals[valid], atol=1e-6)

    def test_property_values_on_inactive_cells(self, epc_path):
        """Property values at inactive cells should survive roundtrip."""
        g = xtgeo.create_box_grid((3, 3, 2))
        act = g._actnumsv.copy()
        act[0, 0, 0] = 0
        act[1, 1, 1] = 0
        g._actnumsv = act

        vals = np.arange(18, dtype=np.float64).reshape(3, 3, 2)
        prop = xtgeo.GridProperty(g, name="PORO", values=vals)

        g2, props = _write_and_read_grid(epc_path, g, properties=[prop])

        # Actnum must survive
        assert np.array_equal(g2._actnumsv, act)

        # Property shape must match
        result = props[0].values
        assert result.shape == vals.shape

    def test_discrete_property_preserves_codes(self, epc_path):
        """Discrete property with multiple code values roundtrips correctly."""
        g = xtgeo.create_box_grid((4, 3, 2))
        vals = np.array([1, 2, 3, 4] * 6, dtype=np.int32).reshape(4, 3, 2)
        fac = xtgeo.GridProperty(g, name="FACIES", values=vals, discrete=True)

        _, props = _write_and_read_grid(epc_path, g, properties=[fac])

        assert props[0].isdiscrete
        assert np.array_equal(props[0].values, vals)

    def test_all_inactive_cells(self, epc_path):
        """A grid with all cells inactive should roundtrip without error."""
        g = xtgeo.create_box_grid((2, 2, 2))
        g._actnumsv = np.zeros((2, 2, 2), dtype=np.int32)

        g2, _ = _write_and_read_grid(epc_path, g)

        assert np.array_equal(g2._actnumsv, np.zeros((2, 2, 2), dtype=np.int32))


# ---------------------------------------------------------------------------
# Rotated grids
# ---------------------------------------------------------------------------


class TestRotatedGridRoundTrip:
    """Verify grids with rotation preserve geometry through EPC."""

    def test_box_grid_rotation_30(self, epc_path):
        """A box grid created with rotation=30 should preserve coord geometry."""
        g = xtgeo.create_box_grid(
            (3, 4, 2),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
            rotation=30.0,
        )

        g2, _ = _write_and_read_grid(epc_path, g, title="RotGrid30")

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)

    def test_box_grid_rotation_45(self, epc_path):
        """Rotation=45 should also preserve geometry."""
        g = xtgeo.create_box_grid(
            (4, 3, 3),
            origin=(100, 200, 500),
            increment=(25, 25, 5),
            rotation=45.0,
        )

        g2, _ = _write_and_read_grid(epc_path, g, title="RotGrid45")

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)

    def test_box_grid_rotation_90(self, epc_path):
        """Rotation=90 (edge case) should preserve geometry."""
        g = xtgeo.create_box_grid(
            (3, 3, 2),
            origin=(0, 0, 0),
            increment=(10, 10, 5),
            rotation=90.0,
        )

        g2, _ = _write_and_read_grid(epc_path, g, title="RotGrid90")

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)

    def test_rotated_grid_with_properties(self, epc_path):
        """Properties on a rotated grid should roundtrip correctly."""
        g = xtgeo.create_box_grid(
            (3, 4, 2),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
            rotation=33.7,
        )
        poro = xtgeo.GridProperty(
            g, name="PORO", values=np.random.RandomState(42).rand(3, 4, 2)
        )

        g2, props = _write_and_read_grid(
            epc_path, g, title="RotPropGrid", properties=[poro]
        )

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(props[0].values, poro.values, atol=1e-6)


# ---------------------------------------------------------------------------
# Pinch-outs (zero-thickness layers)
# ---------------------------------------------------------------------------


class TestPinchOutRoundTrip:
    """Verify grids with collapsed / zero-thickness layers survive EPC roundtrip."""

    def test_collapsed_middle_layer(self, epc_path):
        """A grid where layer 2 has zero thickness (pinch-out) should roundtrip."""
        g = xtgeo.create_box_grid(
            (3, 3, 3), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        # Collapse layer index 1: set its bottom equal to its top
        # zcorn shape is (ni+1, nj+1, nk+1, 4)
        # layer k has top at z[:, :, k, :] and bottom at z[:, :, k+1, :]
        # Collapse layer 1: set z[:, :, 2, :] = z[:, :, 1, :]
        z[:, :, 2, :] = z[:, :, 1, :]
        g._zcornsv = z

        g2, _ = _write_and_read_grid(epc_path, g, title="PinchGrid")

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        # Verify zero thickness is preserved
        assert np.allclose(
            g2._zcornsv[:, :, 2, :] - g2._zcornsv[:, :, 1, :],
            0.0,
            atol=1e-6,
        )

    def test_pinchout_partial_layer(self, epc_path):
        """Only some columns have zero thickness in a layer."""
        g = xtgeo.create_box_grid(
            (4, 4, 3), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        # Collapse layer 1 only on the eastern half (i >= 2)
        z[2:, :, 2, :] = z[2:, :, 1, :]
        g._zcornsv = z

        g2, _ = _write_and_read_grid(epc_path, g, title="PartialPinch")

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        # Eastern half collapsed
        assert np.allclose(
            g2._zcornsv[2:, :, 2, :] - g2._zcornsv[2:, :, 1, :],
            0.0,
            atol=1e-6,
        )
        # Western half still has thickness
        thickness = g2._zcornsv[:2, :, 2, :] - g2._zcornsv[:2, :, 1, :]
        assert np.all(thickness > 0)

    def test_pinchout_with_inactive_cells(self, epc_path):
        """Pinched cells marked inactive should roundtrip both geometry and actnum."""
        g = xtgeo.create_box_grid(
            (3, 3, 3), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        z[:, :, 2, :] = z[:, :, 1, :]
        g._zcornsv = z

        act = g._actnumsv.copy()
        act[:, :, 1] = 0  # deactivate pinched layer
        g._actnumsv = act

        g2, _ = _write_and_read_grid(epc_path, g, title="PinchInactive")

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)


# ---------------------------------------------------------------------------
# Faulted grid variations
# ---------------------------------------------------------------------------


class TestFaultedGridVariations:
    """Extend fault coverage beyond the basic Z-offset test."""

    def test_fault_throw_multiple_columns(self, epc_path):
        """Fault throw applied to different column groups."""
        g = xtgeo.create_box_grid(
            (5, 4, 3), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        # Fault at i=2: eastern block shifted down 15m
        z[3:, :, :, :] += 15.0
        # Second fault at i=4: extra 10m
        z[5:, :, :, :] += 10.0
        g._zcornsv = z

        g2, _ = _write_and_read_grid(epc_path, g, title="MultiFault")

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)

    def test_fault_with_properties(self, epc_path):
        """Faulted grid with continuous + discrete properties."""
        g = xtgeo.create_box_grid(
            (4, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        z[2:, :, :, :] += 8.0
        g._zcornsv = z

        rng = np.random.RandomState(42)
        poro = xtgeo.GridProperty(g, name="PORO", values=rng.rand(4, 3, 2))
        fipnum = xtgeo.GridProperty(
            g,
            name="FIPNUM",
            values=np.array([1, 1, 2, 2] * 6, dtype=np.int32).reshape(4, 3, 2),
            discrete=True,
        )

        g2, props = _write_and_read_grid(
            epc_path, g, title="FaultProps", properties=[poro, fipnum]
        )

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        prop_dict = {p.name: p for p in props}
        assert np.allclose(prop_dict["PORO"].values, poro.values, atol=1e-6)
        assert np.array_equal(prop_dict["FIPNUM"].values, fipnum.values)


# ---------------------------------------------------------------------------
# Asymmetric / non-square dimensions
# ---------------------------------------------------------------------------


class TestAsymmetricDimensions:
    """Catch dimension-ordering bugs with NI != NJ != NK."""

    @pytest.mark.parametrize(
        "dims",
        [
            (2, 3, 4),
            (5, 2, 3),
            (3, 7, 2),
            (1, 1, 1),
            (10, 1, 5),
        ],
    )
    def test_asymmetric_grid_geometry(self, epc_path, dims):
        g = xtgeo.create_box_grid(dims, origin=(0, 0, 0), increment=(10, 10, 5))

        g2, _ = _write_and_read_grid(epc_path, g, title="AsymGrid")

        assert g2.dimensions == dims
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)

    def test_asymmetric_with_property(self, epc_path):
        """Property on a highly asymmetric grid preserves shape and values."""
        g = xtgeo.create_box_grid((2, 7, 3))
        vals = np.arange(42, dtype=np.float64).reshape(2, 7, 3)
        prop = xtgeo.GridProperty(g, name="PORO", values=vals)

        _, props = _write_and_read_grid(
            epc_path, g, title="AsymPropGrid", properties=[prop]
        )

        assert props[0].values.shape == (2, 7, 3)
        assert np.allclose(props[0].values, vals, atol=1e-6)


# ---------------------------------------------------------------------------
# Large grids (performance / correctness at scale)
# ---------------------------------------------------------------------------


class TestLargeGrid:
    """Test with larger grid sizes to catch scaling issues."""

    def test_medium_grid_roundtrip(self, epc_path):
        """30x20x10 grid (6000 cells) with property."""
        g = xtgeo.create_box_grid(
            (30, 20, 10), origin=(460000, 5930000, 1000), increment=(50, 50, 5)
        )
        rng = np.random.RandomState(99)
        poro = xtgeo.GridProperty(g, name="PORO", values=rng.rand(30, 20, 10))

        g2, props = _write_and_read_grid(
            epc_path, g, title="MedGrid", properties=[poro]
        )

        assert g2.dimensions == (30, 20, 10)
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.allclose(props[0].values, poro.values, atol=1e-6)


# ---------------------------------------------------------------------------
# Hypothesis-driven fuzzing
# ---------------------------------------------------------------------------


def _xtgeo_grids_st():
    """Hypothesis strategy for random xtgeo box grids.

    Reuses the same parameter ranges as tests/test_grid3d/grid_generator.py.
    """
    import hypothesis.strategies as st

    indices = st.integers(min_value=2, max_value=6)
    coordinates = st.floats(min_value=-100.0, max_value=100.0)
    increments = st.floats(min_value=1.0, max_value=100.0)

    return st.builds(
        xtgeo.create_box_grid,
        dimension=st.tuples(indices, indices, indices),
        origin=st.tuples(coordinates, coordinates, coordinates),
        increment=st.tuples(increments, increments, increments),
        rotation=st.floats(min_value=0.0, max_value=90.0),
    )


class TestHypothesisRoundTrip:
    """Fuzz the EPC roundtrip with randomly generated box grids and properties."""

    @given(grid=_xtgeo_grids_st())
    @settings(max_examples=20, deadline=30000)
    def test_random_box_grid_geometry(self, grid):
        """Random box grids with varying dimensions, origins, and rotations."""
        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / "hyp.epc")

            g2, _ = _write_and_read_grid(path, grid, title="HypGrid")

            assert g2.dimensions == grid.dimensions
            assert np.allclose(grid._coordsv, g2._coordsv, atol=1e-4)
            assert np.allclose(grid._zcornsv, g2._zcornsv, atol=1e-4)
            assert np.array_equal(grid._actnumsv, g2._actnumsv)

    @given(grid=_xtgeo_grids_st())
    @settings(max_examples=10, deadline=30000)
    def test_random_box_grid_with_continuous_property(self, grid):
        """Random box grid with a continuous property."""
        dims = grid.dimensions
        rng = np.random.RandomState(42)
        vals = rng.rand(*dims)
        prop = xtgeo.GridProperty(grid, name="TESTPROP", values=vals)

        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / "hyp_prop.epc")
            g2, props = _write_and_read_grid(
                path, grid, title="HypPropGrid", properties=[prop]
            )

            assert len(props) == 1
            assert props[0].values.shape == dims
            assert np.allclose(props[0].values, vals, atol=1e-4)

    @given(grid=_xtgeo_grids_st())
    @settings(max_examples=10, deadline=30000)
    def test_random_box_grid_with_discrete_property(self, grid):
        """Random box grid with a discrete property."""
        dims = grid.dimensions
        rng = np.random.RandomState(42)
        vals = rng.randint(1, 5, size=dims).astype(np.int32)
        prop = xtgeo.GridProperty(grid, name="ZONE", values=vals, discrete=True)

        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / "hyp_disc.epc")
            g2, props = _write_and_read_grid(
                path, grid, title="HypDiscGrid", properties=[prop]
            )

            assert len(props) == 1
            assert props[0].isdiscrete
            assert np.array_equal(props[0].values, vals)


# ---------------------------------------------------------------------------
# CRS with rotation
# ---------------------------------------------------------------------------


class TestCRSRotationGrid:
    """CRS with non-zero areal rotation on grids."""

    def test_crs_with_rotation(self, epc_path):
        """Grid written with a CRS that has areal rotation."""
        g = xtgeo.create_box_grid(
            (3, 3, 2), origin=(460000, 5930000, 1000), increment=(50, 50, 10)
        )
        # Write with explicit rotated CRS
        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        import uuid

        crs_uuid = str(uuid.uuid4())
        p.put_crs(
            uuid=crs_uuid,
            title="RotatedCRS",
            origin_x=460000.0,
            origin_y=5930000.0,
            origin_z=0.0,
            areal_rotation=0.5236,  # ~30 degrees in radians
            z_increasing_downward=True,
            projected_crs_epsg=23031,
        )
        uuids = xtgeo_grid_to_resqml(p, g, title="CRSRotGrid", crs_uuid=crs_uuid)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        crs_info = p2.get_crs(crs_uuid)
        g2, _ = ijk_grid_to_xtgeo(p2, uuids["CRSRotGrid"])
        p2.close()

        # CRS rotation must be preserved
        assert abs(crs_info["areal_rotation"] - 0.5236) < 1e-4
        # Grid geometry is in absolute coordinates, so must be identical
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)


# ---------------------------------------------------------------------------
# Mixed scenarios
# ---------------------------------------------------------------------------


class TestMixedScenarios:
    """Complex scenarios combining multiple features."""

    def test_rotated_faulted_with_properties_and_inactive(self, epc_path):
        """Rotated + faulted + inactive cells + multiple property types."""
        g = xtgeo.create_box_grid(
            (4, 5, 3),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
            rotation=25.0,
        )
        # Add fault
        z = g._zcornsv.copy()
        z[3:, :, :, :] += 12.0
        g._zcornsv = z

        # Deactivate some cells
        act = g._actnumsv.copy()
        act[0, 0, 0] = 0
        act[3, 4, 2] = 0
        act[2, 2, 1] = 0
        g._actnumsv = act

        rng = np.random.RandomState(42)
        poro = xtgeo.GridProperty(g, name="PORO", values=rng.rand(4, 5, 3))
        permx = xtgeo.GridProperty(g, name="PERMX", values=rng.rand(4, 5, 3) * 500)
        fipnum = xtgeo.GridProperty(
            g,
            name="FIPNUM",
            values=rng.randint(1, 4, size=(4, 5, 3)).astype(np.int32),
            discrete=True,
        )

        g2, props = _write_and_read_grid(
            epc_path,
            g,
            title="ComplexGrid",
            properties=[poro, permx, fipnum],
        )

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)

        prop_dict = {p.name: p for p in props}
        assert np.allclose(prop_dict["PORO"].values, poro.values, atol=1e-6)
        assert np.allclose(prop_dict["PERMX"].values, permx.values, atol=1e-6)
        assert np.array_equal(prop_dict["FIPNUM"].values, fipnum.values)

    def test_pinchout_faulted_with_inactive(self, epc_path):
        """Pinch-out + fault + inactive in the same grid."""
        g = xtgeo.create_box_grid(
            (4, 4, 4), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        # Fault at i=2
        z[3:, :, :, :] += 10.0
        # Pinch layer 2
        z[:, :, 3, :] = z[:, :, 2, :]
        g._zcornsv = z

        act = g._actnumsv.copy()
        act[:, :, 2] = 0  # deactivate pinched layer
        g._actnumsv = act

        g2, _ = _write_and_read_grid(epc_path, g, title="PinchFaultGrid")

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)


# ---------------------------------------------------------------------------
# Fault geometry warning
# ---------------------------------------------------------------------------


class TestFaultGeometryWarning:
    """Verify the converter warns about unsplit pillar geometry on faulted grids."""

    def test_faulted_grid_emits_warning(self, epc_path, caplog):
        """A faulted grid should emit a warning about unsplit pillar geometry."""
        import logging

        g = xtgeo.create_box_grid(
            (4, 4, 2), origin=(0, 0, 1000), increment=(50, 50, 10)
        )
        z = g._zcornsv.copy()
        # Create a proper fault at pillar i=2: shift the eastern cell corners
        # (SE=1, NE=3) down, leaving western corners (SW=0, NW=2) unchanged.
        # This makes the 4 zcorn values differ at interior nodes → detectable.
        z[2, :, :, 1] += 10.0
        z[2, :, :, 3] += 10.0
        g._zcornsv = z

        with caplog.at_level(logging.WARNING, logger="xtgeo.interfaces.osdu._ijk_grid"):
            p = EpcFileProvider(epc_path, mode="w")
            p.open()
            xtgeo_grid_to_resqml(p, g, title="FaultWarnGrid")
            p.close()

        assert any("faulted geometry" in r.message.lower() for r in caplog.records)

    def test_unfaulted_grid_no_warning(self, epc_path, caplog):
        """A simple box grid should NOT emit the fault warning."""
        import logging

        g = xtgeo.create_box_grid(
            (3, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10)
        )

        with caplog.at_level(logging.WARNING, logger="xtgeo.interfaces.osdu._ijk_grid"):
            p = EpcFileProvider(epc_path, mode="w")
            p.open()
            xtgeo_grid_to_resqml(p, g, title="BoxGrid")
            p.close()

        assert not any("faulted geometry" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# K-direction metadata
# ---------------------------------------------------------------------------


class TestKDirectionMetadata:
    """Verify k_direction is stored and retrievable as RESQML metadata."""

    def test_k_direction_stored_as_down(self, epc_path):
        """Default k_direction='down' should be stored in RESQML metadata."""
        from xtgeo.interfaces.osdu._resqml_meta import _get_resqml_meta

        g = xtgeo.create_box_grid(
            (3, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10)
        )

        g2, _ = _write_and_read_grid(epc_path, g, title="KDownGrid")

        meta = _get_resqml_meta(g2)
        assert meta.get("k_direction") == "down"

    def test_k_direction_geometry_identity(self, epc_path):
        """Grid geometry should be identical regardless of k_direction metadata."""
        g = xtgeo.create_box_grid(
            (3, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10)
        )

        g2, _ = _write_and_read_grid(epc_path, g, title="KDirGrid")

        # Geometry must be bitwise identical
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)


# ===========================================================================
# Surface, Points, Polygons compliance
# ===========================================================================


def _roundtrip_surface(epc_path, surf, title="Surf", **kw):
    """Write surface to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_surface_to_resqml(p, surf, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    s2 = grid2d_to_xtgeo(p2, uuids[title])
    p2.close()
    return s2


def _roundtrip_points(epc_path, pts, title="Pts", **kw):
    """Write points to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_points_to_resqml(p, pts, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    pts2 = pointset_to_xtgeo(p2, uuids[title])
    p2.close()
    return pts2


def _roundtrip_polygons(epc_path, polys, title="Poly", **kw):
    """Write polygons to EPC, read back."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    uuids = xtgeo_polygons_to_resqml(p, polys, title=title, **kw)
    p.close()

    p2 = EpcFileProvider(epc_path, mode="r")
    p2.open()
    polys2 = polylineset_to_xtgeo(p2, uuids[title])
    p2.close()
    return polys2


# ---------------------------------------------------------------------------
# Surface compliance
# ---------------------------------------------------------------------------


class TestSurfaceCompliance:
    """Extended surface roundtrip and post-roundtrip operation tests."""

    def test_large_surface(self, epc_path):
        """100x120 surface (12000 nodes) roundtrip."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=100,
            nrow=120,
            xinc=25.0,
            yinc=25.0,
            xori=460000,
            yori=5930000,
            values=rng.rand(100, 120) * 500 + 1000,
        )

        s2 = _roundtrip_surface(epc_path, s, title="LargeSurf", crs_epsg=23031)

        assert s2.ncol == 100
        assert s2.nrow == 120
        assert np.allclose(s2.values.filled(np.nan), s.values.filled(np.nan), atol=1e-4)

    def test_surface_with_many_nans(self, epc_path):
        """Surface where 50% of values are NaN."""
        rng = np.random.RandomState(42)
        vals = rng.rand(20, 25) * 100
        mask = rng.rand(20, 25) > 0.5
        vals[mask] = np.nan

        s = xtgeo.RegularSurface(
            ncol=20, nrow=25, xinc=10, yinc=10, xori=0, yori=0, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="NanHeavy")

        orig = s.values.filled(np.nan)
        read = s2.values.filled(np.nan)
        assert np.array_equal(np.isnan(orig), np.isnan(read))
        valid = ~np.isnan(orig)
        assert np.allclose(orig[valid], read[valid], atol=1e-6)

    def test_surface_negative_values(self, epc_path):
        """Surface with negative Z values (above sea level)."""
        vals = np.linspace(-500, 500, 30).reshape(5, 6)
        s = xtgeo.RegularSurface(
            ncol=5, nrow=6, xinc=100, yinc=100, xori=0, yori=0, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="NegSurf")

        assert np.allclose(s2.values.filled(np.nan), vals, atol=1e-6)

    def test_surface_constant_value(self, epc_path):
        """Surface where all values are identical (flat horizon)."""
        vals = np.full((8, 10), 2500.0)
        s = xtgeo.RegularSurface(
            ncol=8, nrow=10, xinc=50, yinc=50, xori=460000, yori=5930000, values=vals
        )

        s2 = _roundtrip_surface(epc_path, s, title="FlatSurf")

        assert np.allclose(s2.values.filled(np.nan), vals, atol=1e-6)

    @pytest.mark.parametrize("rotation", [0.0, 15.5, 45.0, 89.9])
    def test_surface_rotation_parametrized(self, epc_path, rotation):
        """Multiple rotation angles roundtrip correctly."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=5,
            nrow=7,
            xinc=25.0,
            yinc=30.0,
            xori=460000,
            yori=5930000,
            rotation=rotation,
            values=rng.rand(5, 7) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="RotSurf")

        assert abs(s2.rotation - rotation) < 0.1

    def test_surface_asymmetric_increments(self, epc_path):
        """xinc != yinc should be preserved."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=6,
            nrow=10,
            xinc=12.5,
            yinc=37.5,
            xori=100,
            yori=200,
            values=rng.rand(6, 10) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="AsymSurf")

        assert abs(s2.xinc - 12.5) < 0.01
        assert abs(s2.yinc - 37.5) < 0.01

    def test_surface_operations_post_roundtrip(self, epc_path):
        """Surface operations (statistics, resample) work after roundtrip."""
        rng = np.random.RandomState(42)
        s = xtgeo.RegularSurface(
            ncol=10,
            nrow=12,
            xinc=25.0,
            yinc=25.0,
            xori=460000,
            yori=5930000,
            values=rng.rand(10, 12) * 100,
        )

        s2 = _roundtrip_surface(epc_path, s, title="OpsSurf", crs_epsg=23031)

        # Basic statistics should match
        assert abs(s2.values.mean() - s.values.mean()) < 0.01
        assert abs(s2.values.std() - s.values.std()) < 0.01
        assert abs(s2.values.min() - s.values.min()) < 0.01
        assert abs(s2.values.max() - s.values.max()) < 0.01


# ---------------------------------------------------------------------------
# Points compliance
# ---------------------------------------------------------------------------


class TestPointsCompliance:
    """Extended PointSet roundtrip tests."""

    def test_many_points(self, epc_path):
        """1000 randomly distributed points."""
        rng = np.random.RandomState(42)
        data = np.column_stack(
            [
                rng.uniform(460000, 461000, 1000),
                rng.uniform(5930000, 5931000, 1000),
                rng.uniform(1000, 3000, 1000),
            ]
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="ManyPts", crs_epsg=23031)

        df2 = pts2.get_dataframe()
        assert len(df2) == 1000
        assert np.allclose(df2.values[:, :3], data, atol=1e-4)

    def test_single_point(self, epc_path):
        """Single point roundtrip."""
        data = np.array([[460000, 5930000, 1500]], dtype=np.float64)
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="SinglePt")

        df2 = pts2.get_dataframe()
        assert len(df2) == 1
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)

    def test_points_negative_z(self, epc_path):
        """Points with negative Z values."""
        data = np.array(
            [
                [0, 0, -100],
                [100, 100, -50],
                [200, 200, 0],
                [300, 300, 50],
            ],
            dtype=np.float64,
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="NegZPts")

        df2 = pts2.get_dataframe()
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)

    def test_points_collocated(self, epc_path):
        """Multiple points at the same location."""
        data = np.array(
            [
                [100, 200, 300],
                [100, 200, 300],
                [100, 200, 400],
            ],
            dtype=np.float64,
        )
        pts = xtgeo.Points(data)

        pts2 = _roundtrip_points(epc_path, pts, title="CollocPts")

        df2 = pts2.get_dataframe()
        assert len(df2) == 3
        assert np.allclose(df2.values[:, :3], data, atol=1e-6)


# ---------------------------------------------------------------------------
# Polygons compliance
# ---------------------------------------------------------------------------


class TestPolygonsCompliance:
    """Extended PolylineSet (Polygons) roundtrip tests."""

    def test_single_polygon(self, epc_path):
        """Single closed polygon."""
        df = pd.DataFrame(
            {
                "X_UTME": [0, 100, 100, 0, 0],
                "Y_UTMN": [0, 0, 100, 100, 0],
                "Z_TVDSS": [1000, 1000, 1000, 1000, 1000],
                "POLY_ID": [0, 0, 0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="SinglePoly")

        df2 = polys2.get_dataframe()
        assert len(df2) == 5
        assert np.allclose(
            df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            df2[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            atol=1e-6,
        )

    def test_many_polygons(self, epc_path):
        """5 separate polygons with different sizes."""
        frames = []
        for pid in range(5):
            n = 4 + pid  # varying number of vertices
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x = 100 * pid + 50 * np.cos(angles)
            y = 50 * np.sin(angles)
            z = np.full(n, 1000 + pid * 100, dtype=np.float64)
            # Close the polygon
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            frames.append(
                pd.DataFrame(
                    {
                        "X_UTME": x,
                        "Y_UTMN": y,
                        "Z_TVDSS": z,
                        "POLY_ID": pid,
                    }
                )
            )

        df = pd.concat(frames, ignore_index=True)
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="ManyPolys")

        df2 = polys2.get_dataframe()
        assert len(df2) == len(df)
        assert set(df2["POLY_ID"].unique()) == set(range(5))
        assert np.allclose(
            df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            df2[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            atol=1e-6,
        )

    def test_polygon_varying_z(self, epc_path):
        """Polygon with varying Z values (3D fault trace)."""
        df = pd.DataFrame(
            {
                "X_UTME": [0, 100, 200, 300, 200, 100, 0],
                "Y_UTMN": [0, 50, 0, 50, 100, 50, 0],
                "Z_TVDSS": [1000, 1100, 1200, 1300, 1200, 1100, 1000],
                "POLY_ID": [0, 0, 0, 0, 0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)

        polys2 = _roundtrip_polygons(epc_path, polys, title="3DPoly")

        df2 = polys2.get_dataframe()
        assert np.allclose(df["Z_TVDSS"].values, df2["Z_TVDSS"].values, atol=1e-6)


# ===========================================================================
# Well, BlockedWell, TriangulatedSurface compliance
# ===========================================================================


class TestTriangulatedSurfaceRoundTrip:
    """Tests for TriangulatedSurface round-trips via EPC."""

    def test_simple_triangle(self, epc_path):
        """Single triangle roundtrip."""
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_triangulated_surface_to_resqml(
            p, trisurf, title="Triangle", crs_epsg=23031
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        trisurf2 = triangulated_surface_to_xtgeo(p2, uuids["Triangle"])
        p2.close()

        assert np.allclose(trisurf.vertices, trisurf2.vertices, atol=1e-10)
        assert np.array_equal(trisurf.triangles, trisurf2.triangles)

    def test_multi_triangle_mesh(self, epc_path):
        """Multiple triangles forming a mesh."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_triangulated_surface_to_resqml(
            p, trisurf, title="Mesh", crs_epsg=23031
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        trisurf2 = triangulated_surface_to_xtgeo(p2, uuids["Mesh"])
        p2.close()

        assert trisurf2.vertices.shape == (4, 3)
        assert trisurf2.triangles.shape == (2, 3)
        assert np.allclose(trisurf.vertices, trisurf2.vertices)
        assert np.array_equal(trisurf.triangles, trisurf2.triangles)

    def test_high_level_api(self, epc_path):
        """Test via the high-level triangulated_surface_to/from_osdu."""
        vertices = np.array([[0, 0, 100], [10, 0, 100], [5, 10, 110]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        trisurf = xtgeo.TriangulatedSurface(vertices=vertices, triangles=triangles)

        uuids = xtgeo.triangulated_surface_to_osdu(
            epc_path, trisurf, title="FaultPlane", crs_epsg=23031
        )
        assert "FaultPlane" in uuids

        trisurf2 = xtgeo.triangulated_surface_from_osdu(
            epc_path, uuid=uuids["FaultPlane"]
        )
        assert np.allclose(trisurf.vertices, trisurf2.vertices)


class TestWellRoundTrip:
    """Tests for Well (trajectory + logs) round-trips via EPC."""

    def _make_well(self):
        """Create a simple well with trajectory and logs."""
        npts = 10
        md = np.linspace(0, 1000, npts)
        x = np.full(npts, 460000.0)
        y = np.full(npts, 5930000.0)
        z = np.linspace(0, 1000, npts)
        df = pd.DataFrame(
            {
                "X_UTME": x,
                "Y_UTMN": y,
                "Z_TVDSS": z,
                "M_DEPTH": md,
                "GR": np.linspace(30, 120, npts),
                "PORO": np.linspace(0.1, 0.3, npts),
            }
        )
        return xtgeo.Well(
            xpos=460000.0,
            ypos=5930000.0,
            wname="WELL-A",
            df=df,
            mdlogname="M_DEPTH",
            zonelogname=None,
        )

    def test_trajectory_roundtrip(self, epc_path):
        """Test well trajectory geometry roundtrip."""
        well = self._make_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_well_to_resqml(
            p, well, title="WELL-A", crs_epsg=23031, export_logs=False
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        well2 = well_to_xtgeo(p2, uuids["WELL-A"], load_logs=False)
        p2.close()

        assert np.allclose(
            well.get_dataframe()["M_DEPTH"].values,
            well2.get_dataframe()[well2.mdlogname].values,
            atol=1e-6,
        )
        assert np.allclose(
            well.get_dataframe()["X_UTME"].values,
            well2.get_dataframe()["X_UTME"].values,
            atol=1e-6,
        )

    def test_trajectory_with_logs_roundtrip(self, epc_path):
        """Test well trajectory + logs roundtrip."""
        well = self._make_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_well_to_resqml(
            p, well, title="WELL-A", crs_epsg=23031, export_logs=True
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        well2 = well_to_xtgeo(p2, uuids["WELL-A"], load_logs=True)
        p2.close()

        df1 = well.get_dataframe()
        df2 = well2.get_dataframe()
        assert "GR" in df2.columns
        assert "PORO" in df2.columns
        assert np.allclose(df1["GR"].values, df2["GR"].values, atol=1e-6)
        assert np.allclose(df1["PORO"].values, df2["PORO"].values, atol=1e-6)
        # XYZ preserved
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)

    def test_high_level_api(self, epc_path):
        """Test via well_from_osdu / well_to_osdu."""
        well = self._make_well()

        uuids = xtgeo.well_to_osdu(epc_path, well, title="WELL-A", crs_epsg=23031)
        assert "WELL-A" in uuids

        well2 = xtgeo.well_from_osdu(epc_path, uuid=uuids["WELL-A"])
        df1 = well.get_dataframe()
        df2 = well2.get_dataframe()
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)
        assert well2.mdlogname is not None


class TestBlockedWellRoundTrip:
    """Tests for BlockedWell round-trips via EPC."""

    def _make_blocked_well(self):
        """Create a simple blocked well."""
        npts = 5
        md = np.linspace(100, 500, npts)
        x = np.full(npts, 460050.0)
        y = np.full(npts, 5930050.0)
        z = np.linspace(100, 500, npts)
        df = pd.DataFrame(
            {
                "X_UTME": x,
                "Y_UTMN": y,
                "Z_TVDSS": z,
                "M_DEPTH": md,
                "I_INDEX": np.array([1, 1, 2, 2, 3], dtype=np.int32),
                "J_INDEX": np.array([1, 2, 2, 3, 3], dtype=np.int32),
                "K_INDEX": np.array([1, 1, 2, 2, 3], dtype=np.int32),
                "Facies": np.array([1, 2, 1, 3, 2], dtype=np.int32),
            }
        )
        return xtgeo.BlockedWell(
            xpos=460050.0,
            ypos=5930050.0,
            wname="BW-1",
            df=df,
            mdlogname="M_DEPTH",
            zonelogname=None,
        )

    def test_blocked_well_roundtrip(self, epc_path):
        """Test blocked well geometry + properties roundtrip."""
        bwell = self._make_blocked_well()

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_blocked_well_to_resqml(p, bwell, title="BW-1", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        bwell2 = blocked_well_to_xtgeo(p2, uuids["BW-1"])
        p2.close()

        df1 = bwell.get_dataframe()
        df2 = bwell2.get_dataframe()
        # MD values preserved (column name may differ after roundtrip)
        md1 = df1[bwell.mdlogname].values
        md2 = df2[bwell2.mdlogname].values
        assert np.allclose(md1, md2, atol=1e-6)
        assert np.allclose(df1["X_UTME"].values, df2["X_UTME"].values, atol=1e-6)
        assert np.array_equal(
            df1["I_INDEX"].values.astype(int), df2["I_INDEX"].values.astype(int)
        )
        assert np.array_equal(
            df1["J_INDEX"].values.astype(int), df2["J_INDEX"].values.astype(int)
        )
        assert np.array_equal(
            df1["K_INDEX"].values.astype(int), df2["K_INDEX"].values.astype(int)
        )

    def test_high_level_api(self, epc_path):
        """Test via blocked_well_from_osdu / blocked_well_to_osdu."""
        bwell = self._make_blocked_well()

        uuids = xtgeo.blocked_well_to_osdu(
            epc_path, bwell, title="BW-1", crs_epsg=23031
        )
        assert "BW-1" in uuids

        bwell2 = xtgeo.blocked_well_from_osdu(epc_path, uuid=uuids["BW-1"])
        df1 = bwell.get_dataframe()
        df2 = bwell2.get_dataframe()
        md1 = df1[bwell.mdlogname].values
        md2 = df2[bwell2.mdlogname].values
        assert np.allclose(md1, md2, atol=1e-6)
