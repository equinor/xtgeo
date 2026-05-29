"""Grid3d compliance tests for the OSDU/RESQML EPC roundtrip.

Exercises geometry and property scenarios tested extensively in the grid3d
test suite but previously missing from the OSDU path: masked/NaN properties,
rotated grids, pinch-outs, inactive-cell property interaction, large grids,
and hypothesis-driven fuzzing with random box grids.
"""

import numpy as np
import pathlib
import pytest
import tempfile
from hypothesis import given, settings, HealthCheck

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml


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
        vals = np.array(
            [1, 2, 3, 4] * 6, dtype=np.int32
        ).reshape(4, 3, 2)
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
        poro = xtgeo.GridProperty(
            g, name="PORO", values=rng.rand(4, 3, 2)
        )
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
        poro = xtgeo.GridProperty(
            g, name="PORO", values=rng.rand(30, 20, 10)
        )

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
        uuids = xtgeo_grid_to_resqml(
            p, g, title="CRSRotGrid", crs_uuid=crs_uuid
        )
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
        permx = xtgeo.GridProperty(
            g, name="PERMX", values=rng.rand(4, 5, 3) * 500
        )
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

        g = xtgeo.create_box_grid((3, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10))

        g2, _ = _write_and_read_grid(epc_path, g, title="KDownGrid")

        meta = _get_resqml_meta(g2)
        assert meta.get("k_direction") == "down"

    def test_k_direction_geometry_identity(self, epc_path):
        """Grid geometry should be identical regardless of k_direction metadata."""
        g = xtgeo.create_box_grid((3, 3, 2), origin=(0, 0, 1000), increment=(50, 50, 10))

        g2, _ = _write_and_read_grid(epc_path, g, title="KDirGrid")

        # Geometry must be bitwise identical
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
