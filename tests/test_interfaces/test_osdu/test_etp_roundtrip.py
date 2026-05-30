"""ETP dataspace-level roundtrip integration tests.

Tests the full flow: create objects → write to dataspace → read back → compare.
Also tests dataspace copy: read source → write to new dataspace → compare.

Requires a running local RDDMS at ws://localhost:9002.
"""

import contextlib

import numpy as np
import pytest

import xtgeo

pytestmark = pytest.mark.requires_rddms
from xtgeo.interfaces.osdu import (
    CrsSnapshot,
    DataspaceSnapshot,
    EtpConnectionConfig,
    EtpProvider,
    GridSnapshot,
    PointSetSnapshot,
    PolylineSetSnapshot,
    PropertySnapshot,
    SurfaceSnapshot,
    compare_snapshots,
    read_dataspace,
    write_dataspace,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def etp_config():
    """Base ETP config for local RDDMS."""
    import uuid as _uuid

    ds_path = f"xtgeo/test_rt_{_uuid.uuid4().hex[:8]}"
    return EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace=f"eml:///dataspace('{ds_path}')",
    ), ds_path


@pytest.fixture
def provider(etp_config):
    """ETP provider with a fresh test dataspace."""
    cfg, ds_path = etp_config
    p = EtpProvider(cfg)
    try:
        p.open()
    except Exception:
        pytest.skip("Local RDDMS not available at ws://localhost:9002")

    # Create test dataspace
    with contextlib.suppress(Exception):
        p.put_dataspace(ds_path)

    yield p
    with contextlib.suppress(Exception):
        p.delete_dataspace(ds_path)
    p.close()


@pytest.fixture
def fresh_provider(etp_config):
    """Provider factory that creates a new provider for a given dataspace."""

    def _make(dataspace_path: str):
        cfg = EtpConnectionConfig(
            url="ws://localhost:9002",
            dataspace=f"eml:///dataspace('{dataspace_path}')",
        )
        p = EtpProvider(cfg)
        try:
            p.open()
        except Exception:
            pytest.skip("Local RDDMS not available at ws://localhost:9002")
        return p

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_snapshot() -> DataspaceSnapshot:
    """Build a representative DataspaceSnapshot with all object types."""
    import uuid

    crs_uuid = str(uuid.uuid4())
    grid_uuid = str(uuid.uuid4())

    crs = CrsSnapshot(
        uuid=crs_uuid,
        title="TestCRS_EPSG23031",
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        areal_rotation=0.0,
        z_increasing_downward=True,
        projected_crs_epsg=23031,
    )

    # Non-square grid (4x5x3) to catch dimension ordering bugs
    ni, nj, nk = 4, 5, 3
    g = xtgeo.create_box_grid(
        (ni, nj, nk),
        origin=(460000.0, 5930000.0, 1000.0),
        increment=(50.0, 50.0, 10.0),
    )

    # Add fault throw on column 2
    z = g._zcornsv.copy()
    z[2:, :, :, :] += 5.0
    g._zcornsv = z

    # Inactive cells
    act = g._actnumsv.copy()
    act[0, 0, 0] = 0
    act[3, 4, 2] = 0
    g._actnumsv = act

    # Continuous property (porosity)
    np.random.seed(42)
    poro_vals = np.random.uniform(0.05, 0.35, size=ni * nj * nk).astype(np.float64)
    poro_vals = poro_vals.reshape((ni, nj, nk))

    # Discrete property (facies)
    facies_vals = np.random.randint(1, 5, size=ni * nj * nk).astype(np.int32)
    facies_vals = facies_vals.reshape((ni, nj, nk))

    grid = GridSnapshot(
        uuid=grid_uuid,
        title="TestGrid_Faulted",
        resqml_type="resqml20.IjkGridRepresentation",
        crs_uuid=crs_uuid,
        ni=ni,
        nj=nj,
        nk=nk,
        coord=g._coordsv.flatten().astype(np.float64),
        zcorn=g._zcornsv.flatten().astype(np.float32),
        actnum=g._actnumsv.flatten().astype(np.int32),
        k_direction="down",
        properties=[
            PropertySnapshot(
                uuid=str(uuid.uuid4()),
                title="PORO",
                resqml_type="resqml20.ContinuousProperty",
                property_kind="porosity",
                indexable_element="cells",
                supporting_representation_uuid=grid_uuid,
                is_discrete=False,
                uom="v/v",
                values=poro_vals.flatten().astype(np.float64),
            ),
            PropertySnapshot(
                uuid=str(uuid.uuid4()),
                title="FACIES",
                resqml_type="resqml20.DiscreteProperty",
                property_kind="facies",
                indexable_element="cells",
                supporting_representation_uuid=grid_uuid,
                is_discrete=True,
                values=facies_vals.flatten().astype(np.int32),
            ),
        ],
    )

    # Rotated surface
    import math

    rotation_rad = math.radians(15.0)
    surf_vals = np.random.uniform(900.0, 1100.0, size=10 * 12).astype(np.float64)
    surf_vals[5] = np.nan  # inject NaN
    surface = SurfaceSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestSurface_Rotated",
        resqml_type="resqml20.Grid2dRepresentation",
        crs_uuid=crs_uuid,
        ni=10,
        nj=12,
        origin_x=460000.0,
        origin_y=5930000.0,
        di=25.0,
        dj=30.0,
        rotation=rotation_rad,
        values=surf_vals.reshape((12, 10)),
    )

    # PointSet
    pts = np.array(
        [
            [460000.0, 5930000.0, 1000.0],
            [460100.0, 5930100.0, 1010.0],
            [460200.0, 5930050.0, 1020.0],
            [460050.0, 5930200.0, 1005.0],
        ],
        dtype=np.float64,
    )
    pointset = PointSetSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestPoints",
        resqml_type="resqml20.PointSetRepresentation",
        crs_uuid=crs_uuid,
        points=pts,
    )

    # PolylineSet (two polygons, one closed)
    poly1 = np.array(
        [
            [460000.0, 5930000.0, 0.0],
            [460100.0, 5930000.0, 0.0],
            [460100.0, 5930100.0, 0.0],
            [460000.0, 5930100.0, 0.0],
        ],
        dtype=np.float64,
    )
    poly2 = np.array(
        [
            [460200.0, 5930200.0, 0.0],
            [460300.0, 5930200.0, 0.0],
            [460250.0, 5930300.0, 0.0],
        ],
        dtype=np.float64,
    )
    polylineset = PolylineSetSnapshot(
        uuid=str(uuid.uuid4()),
        title="TestPolygons",
        resqml_type="resqml20.PolylineSetRepresentation",
        crs_uuid=crs_uuid,
        polylines=[poly1, poly2],
        closed=[True, False],
    )

    return DataspaceSnapshot(
        grids=[grid],
        surfaces=[surface],
        pointsets=[pointset],
        polylinesets=[polylineset],
        crs_list=[crs],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataspaceRoundTrip:
    """Write a full dataset to ETP, read back, and compare bitwise."""

    def test_write_read_compare(self, provider):
        """Full roundtrip: build snapshot → write → read back → compare."""
        original = _build_test_snapshot()

        # Write
        write_dataspace(provider, original)

        # Read back
        readback = read_dataspace(provider)

        # Compare
        diffs = compare_snapshots(original, readback, atol=1e-6)
        if diffs:
            msg = "\n".join(
                f"  {d.object_type}/{d.title}.{d.field}: {d.detail}" for d in diffs
            )
            pytest.fail(f"Roundtrip differences:\n{msg}")

    def test_grid_geometry_exact(self, provider):
        """Verify grid geometry survives ETP roundtrip with full precision."""
        original = _build_test_snapshot()
        write_dataspace(provider, original)
        readback = read_dataspace(provider)

        assert len(readback.grids) >= 1
        g_orig = original.grids[0]
        g_read = next(g for g in readback.grids if g.title == g_orig.title)

        assert g_read.ni == g_orig.ni
        assert g_read.nj == g_orig.nj
        assert g_read.nk == g_orig.nk

        # Coord: float64 → exact
        np.testing.assert_allclose(
            g_read.coord.astype(np.float64),
            g_orig.coord.astype(np.float64),
            atol=1e-10,
            err_msg="Pillar coordinates differ",
        )

        # Zcorn: float32 → exact within float32 precision
        np.testing.assert_allclose(
            g_read.zcorn.astype(np.float32),
            g_orig.zcorn.astype(np.float32),
            atol=1e-6,
            err_msg="Z-corners differ",
        )

        # Actnum: exact integer match
        np.testing.assert_array_equal(
            g_read.actnum.flatten().astype(np.int32),
            g_orig.actnum.flatten().astype(np.int32),
        )

    def test_properties_exact(self, provider):
        """Verify grid properties survive roundtrip with full precision."""
        original = _build_test_snapshot()
        write_dataspace(provider, original)
        readback = read_dataspace(provider)

        g_read = next(g for g in readback.grids if g.title == "TestGrid_Faulted")
        props_by_name = {p.title: p for p in g_read.properties}

        # Continuous - float64 exact
        assert "PORO" in props_by_name
        poro = props_by_name["PORO"]
        assert not poro.is_discrete
        np.testing.assert_allclose(
            poro.values.astype(np.float64),
            original.grids[0].properties[0].values.astype(np.float64),
            atol=1e-10,
        )

        # Discrete - int32 exact
        assert "FACIES" in props_by_name
        facies = props_by_name["FACIES"]
        assert facies.is_discrete
        np.testing.assert_array_equal(
            facies.values.astype(np.int32),
            original.grids[0].properties[1].values.astype(np.int32),
        )

    def test_surface_rotation_preserved(self, provider):
        """Verify surface rotation survives ETP roundtrip."""
        import math

        original = _build_test_snapshot()
        write_dataspace(provider, original)
        readback = read_dataspace(provider)

        s_read = next(s for s in readback.surfaces if s.title == "TestSurface_Rotated")
        s_orig = original.surfaces[0]

        assert s_read.ni == s_orig.ni
        assert s_read.nj == s_orig.nj
        assert abs(s_read.origin_x - s_orig.origin_x) < 1e-6
        assert abs(s_read.origin_y - s_orig.origin_y) < 1e-6
        assert abs(s_read.di - s_orig.di) < 1e-6
        assert abs(s_read.dj - s_orig.dj) < 1e-6
        assert abs(s_read.rotation - s_orig.rotation) < 1e-6, (
            f"Rotation mismatch: {math.degrees(s_read.rotation):.4f}° "
            f"vs {math.degrees(s_orig.rotation):.4f}°"
        )

    def test_crs_metadata_preserved(self, provider):
        """Verify CRS EPSG and metadata survive roundtrip."""
        original = _build_test_snapshot()
        write_dataspace(provider, original)
        readback = read_dataspace(provider)

        assert len(readback.crs_list) >= 1
        crs_read = next(c for c in readback.crs_list if c.title == "TestCRS_EPSG23031")
        crs_orig = original.crs_list[0]

        assert crs_read.projected_crs_epsg == crs_orig.projected_crs_epsg
        assert crs_read.z_increasing_downward == crs_orig.z_increasing_downward
        assert abs(crs_read.areal_rotation - crs_orig.areal_rotation) < 1e-10


class TestDataspaceCopy:
    """Copy entire dataspace to a new one and verify equivalence."""

    def test_copy_dataspace(self, fresh_provider):
        """Read from source ds → write to target ds → compare both."""
        source_path = "xtgeo/test_copy_src"
        target_path = "xtgeo/test_copy_dst"

        # Setup: create source dataspace and write test data
        src = fresh_provider(source_path)
        with contextlib.suppress(Exception):
            src.put_dataspace(source_path)

        original = _build_test_snapshot()
        write_dataspace(src, original)

        # Read everything from source
        snap_source = read_dataspace(src)
        src.close()

        # Create target and write
        tgt = fresh_provider(target_path)
        with contextlib.suppress(Exception):
            tgt.put_dataspace(target_path)

        write_dataspace(tgt, snap_source)

        # Read back from target
        snap_target = read_dataspace(tgt)
        tgt.close()

        # Compare
        diffs = compare_snapshots(snap_source, snap_target, atol=1e-6)
        if diffs:
            msg = "\n".join(
                f"  {d.object_type}/{d.title}.{d.field}: {d.detail}" for d in diffs
            )
            pytest.fail(f"Dataspace copy differences:\n{msg}")

        # Cleanup
        cleanup = fresh_provider(source_path)
        try:
            cleanup.delete_dataspace(source_path)
            cleanup.delete_dataspace(target_path)
        except Exception:
            pass
        cleanup.close()
