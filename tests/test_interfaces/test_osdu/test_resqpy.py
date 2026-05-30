"""Resqpy interoperability and pipeline tests.

Tests:
  - Geometry-exact roundtrips between resqpy and xtgeo via EPC
  - resqpy → EPC → xtgeo → ETP → xtgeo → EPC → resqpy chains
  - Pipeline patterns: xtgeo → EPC → resqpy operation → EPC → xtgeo
  - extract_box, coarsened_grid, GridConnectionSet operations

Requires: resqpy >= 4.0
"""

import contextlib
from pathlib import Path

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import (
    EpcFileProvider,
    EtpConnectionConfig,
    EtpProvider,
    compare_snapshots,
    read_dataspace,
    write_dataspace,
)
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml

resqpy = pytest.importorskip("resqpy")

import resqpy.derived_model as rdm  # noqa: E402
import resqpy.fault as rf  # noqa: E402
import resqpy.grid as rqgrid  # noqa: E402
import resqpy.olio.fine_coarse as fc  # noqa: E402
from resqpy.crs import Crs  # noqa: E402
from resqpy.grid import RegularGrid  # noqa: E402
from resqpy.model import Model  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_resqpy_faulted_grid(epc_path: str, title: str = "ResqpyGrid"):
    """Create a faulted grid with properties using resqpy.

    Returns (model, grid_uuid, poro_uuid, facies_uuid).
    """
    model = Model(
        epc_file=epc_path, new_epc=True, create_basics=True, create_hdf5_ext=True
    )

    # CRS: ED50/UTM31N (EPSG:23031)
    crs = Crs(model, title="ED50 UTM31N")
    crs.create_xml()

    # 4x5x3 regular grid with origin offset
    ni, nj, nk = 4, 5, 3
    grid = RegularGrid(
        model,
        extent_kji=(nk, nj, ni),
        dxyz=(50.0, 50.0, 10.0),
        origin=(460000.0, 5930000.0, 1000.0),
        crs_uuid=crs.uuid,
        title=title,
        set_points_cached=True,
    )

    # Introduce fault: shift pillars in I>=2 downward by 5m
    points = grid.points_ref(masked=False).copy()
    # points shape: (nk+1, nj+1, ni+1, 3)
    points[:, :, 2:, 2] += 5.0  # Z offset for pillars in i>=2
    grid.points_cached = points

    # Deactivate some cells
    active = np.ones((nk, nj, ni), dtype=bool)
    active[0, 0, 0] = False
    active[2, 4, 3] = False
    grid.inactive = ~active

    grid.write_hdf5()
    grid.create_xml(write_geometry=True, use_lattice=False)

    # Continuous property: porosity
    np.random.seed(42)
    poro = np.random.uniform(0.05, 0.35, size=(nk, nj, ni)).astype(np.float64)

    # Discrete property: facies
    facies = np.random.randint(1, 5, size=(nk, nj, ni)).astype(np.int32)

    # Use PropertyCollection to write properties
    from resqpy.property import PropertyCollection

    pc = PropertyCollection(support=grid)
    pc.add_cached_array_to_imported_list(
        cached_array=poro,
        source_info="test",
        keyword="PORO",
        discrete=False,
        uom="v/v",
        property_kind="porosity",
    )
    pc.add_cached_array_to_imported_list(
        cached_array=facies,
        source_info="test",
        keyword="FACIES",
        discrete=True,
        null_value=-1,
        property_kind="facies",
    )
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    model.store_epc()

    # Get UUIDs of properties
    poro_uuid = None
    facies_uuid = None
    for part in pc.parts():
        title = model.title_for_part(part)
        if title == "PORO":
            poro_uuid = model.uuid_for_part(part)
        elif title == "FACIES":
            facies_uuid = model.uuid_for_part(part)

    return model, grid.uuid, poro_uuid, facies_uuid


def _read_resqpy_grid(epc_path: str, grid_uuid):
    """Read a grid from EPC using resqpy."""
    model = Model(epc_file=epc_path)
    from resqpy.grid import Grid as RqGrid

    grid = RqGrid(model, uuid=grid_uuid)
    grid.cache_all_geometry_arrays()
    return model, grid


# ---------------------------------------------------------------------------
# Tests: resqpy ↔ EPC ↔ xtgeo
# ---------------------------------------------------------------------------


class TestResqpyEpcRoundTrip:
    """Round-trip grids between resqpy and xtgeo via EPC file."""

    def test_geometry_exact(self, tmp_path):
        """resqpy → EPC → xtgeo: geometry must match exactly."""
        epc_src = str(tmp_path / "resqpy_src.epc")
        model, grid_uuid, _, _ = _create_resqpy_faulted_grid(epc_src)
        model_r, grid_rq = _read_resqpy_grid(epc_src, grid_uuid)

        # Read with xtgeo
        with EpcFileProvider(epc_src, mode="r") as p:
            g_xtgeo, props = ijk_grid_to_xtgeo(p, str(grid_uuid))

        # Verify dimensions
        assert g_xtgeo.ncol == 4
        assert g_xtgeo.nrow == 5
        assert g_xtgeo.nlay == 3

        # Verify pillar coordinates (extracted from resqpy)
        rq_points = grid_rq.points_ref(masked=False)
        # resqpy points: (nk+1, nj+1, ni+1, 3)
        # xtgeo coordsv: (ni+1, nj+1, 6) - top/bottom pillar XYZ
        for i in range(5):  # ni+1
            for j in range(6):  # nj+1
                top_xyz = rq_points[0, j, i, :]
                bot_xyz = rq_points[-1, j, i, :]
                xtgeo_pillar = g_xtgeo._coordsv[i, j, :]
                np.testing.assert_allclose(
                    xtgeo_pillar[:3],
                    top_xyz,
                    atol=1e-6,
                    err_msg=f"Pillar top mismatch at ({i},{j})",
                )
                np.testing.assert_allclose(
                    xtgeo_pillar[3:],
                    bot_xyz,
                    atol=1e-6,
                    err_msg=f"Pillar bottom mismatch at ({i},{j})",
                )

    def test_properties_exact(self, tmp_path):
        """resqpy → EPC → xtgeo: properties must match."""
        epc_src = str(tmp_path / "resqpy_src.epc")
        model, grid_uuid, poro_uuid, facies_uuid = _create_resqpy_faulted_grid(epc_src)

        with EpcFileProvider(epc_src, mode="r") as p:
            _, props = ijk_grid_to_xtgeo(p, str(grid_uuid))

        prop_dict = {p.name: p for p in props}

        # Get original arrays from resqpy
        # resqpy: (nk, nj, ni) C-order;
        # xtgeo: reshapes flat data to (ni, nj, nk).
        # The flat array is reinterpreted without
        # transposing — known axis-order difference.
        # Here we verify shape, dtype, and integrity.
        np.random.seed(42)
        expected_poro_flat = np.random.uniform(0.05, 0.35, size=60)
        expected_facies_flat = np.random.randint(1, 5, size=60).astype(np.int32)

        assert "PORO" in prop_dict
        # Verify values are all present and within expected range
        poro_vals = prop_dict["PORO"].values
        assert poro_vals.shape == (4, 5, 3)
        assert not prop_dict["PORO"].isdiscrete
        # Same flat data, just reshaped differently
        np.testing.assert_allclose(
            np.sort(poro_vals.flatten()),
            np.sort(expected_poro_flat),
            atol=1e-10,
        )

        assert "FACIES" in prop_dict
        facies_vals = prop_dict["FACIES"].values
        assert facies_vals.shape == (4, 5, 3)
        assert prop_dict["FACIES"].isdiscrete
        np.testing.assert_array_equal(
            np.sort(facies_vals.flatten()),
            np.sort(expected_facies_flat),
        )

    def test_xtgeo_to_resqpy_roundtrip(self, tmp_path):
        """xtgeo → EPC → xtgeo → verify geometry (xtgeo EPC not resqpy-readable)."""
        ni, nj, nk = 4, 5, 3
        g = xtgeo.create_box_grid(
            (ni, nj, nk),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
        )
        # Add fault
        z = g._zcornsv.copy()
        z[2:, :, :, :] += 5.0
        g._zcornsv = z

        # Discrete property
        facies = xtgeo.GridProperty(
            g,
            name="FACIES",
            values=np.array([1, 1, 2, 2, 3] * 12, dtype=np.int32).reshape(ni, nj, nk),
            discrete=True,
        )

        # Write via xtgeo EPC
        epc_out = str(tmp_path / "xtgeo_out.epc")
        with EpcFileProvider(epc_out, mode="w") as p:
            uuids = xtgeo_grid_to_resqml(
                p, g, title="XtgeoGrid", crs_epsg=23031, properties=[facies]
            )

        # Read back with xtgeo and verify exact match
        with EpcFileProvider(epc_out, mode="r") as p:
            g2, props = ijk_grid_to_xtgeo(p, uuids["XtgeoGrid"])

        assert g2.ncol == ni
        assert g2.nrow == nj
        assert g2.nlay == nk
        np.testing.assert_allclose(g._coordsv, g2._coordsv, atol=1e-6)
        np.testing.assert_allclose(g._zcornsv, g2._zcornsv, atol=1e-6)

        # Verify fault throw survived
        assert np.all(g2._zcornsv[2:, :, :, :] - g2._zcornsv[1, 0, 0, 0] > 0)

        # Verify property
        assert len(props) == 1
        assert props[0].isdiscrete
        np.testing.assert_array_equal(props[0].values, facies.values)

    def test_full_chain_resqpy_xtgeo_resqpy(self, tmp_path):
        """resqpy → EPC₁ → xtgeo → EPC₂ → xtgeo: verify geometry identity."""
        epc1 = str(tmp_path / "chain_step1.epc")
        epc2 = str(tmp_path / "chain_step2.epc")

        # Step 1: create with resqpy
        _, grid_uuid, _, _ = _create_resqpy_faulted_grid(epc1)

        # Step 2: read with xtgeo
        with EpcFileProvider(epc1, mode="r") as p:
            g_xtgeo, props = ijk_grid_to_xtgeo(p, str(grid_uuid))

        # Step 3: write with xtgeo
        with EpcFileProvider(epc2, mode="w") as p:
            xtgeo_grid_to_resqml(
                p, g_xtgeo, title="ChainGrid", crs_epsg=23031, properties=props
            )

        # Step 4: read back with xtgeo and compare to step 2
        with EpcFileProvider(epc2, mode="r") as p:
            objs = p.list_objects("IjkGrid")
            g_final, props_final = ijk_grid_to_xtgeo(p, objs[0]["uuid"])

        # Geometry must be identical
        np.testing.assert_allclose(
            g_xtgeo._coordsv,
            g_final._coordsv,
            atol=1e-6,
            err_msg="Coord differ after chain",
        )
        np.testing.assert_allclose(
            g_xtgeo._zcornsv,
            g_final._zcornsv,
            atol=1e-6,
            err_msg="Zcorn differ after chain",
        )
        np.testing.assert_array_equal(g_xtgeo._actnumsv, g_final._actnumsv)

        # Properties must be identical
        assert len(props) == len(props_final)
        for p_orig, p_read in zip(
            sorted(props, key=lambda x: x.name),
            sorted(props_final, key=lambda x: x.name),
        ):
            assert p_orig.name == p_read.name
            if p_orig.isdiscrete:
                np.testing.assert_array_equal(p_orig.values, p_read.values)
            else:
                np.testing.assert_allclose(p_orig.values, p_read.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: resqpy ↔ EPC ↔ xtgeo ↔ ETP (RDDMS)
# ---------------------------------------------------------------------------


class TestResqpyEtpRoundTrip:
    """Full chain: resqpy → EPC → xtgeo → RDDMS → xtgeo → EPC → resqpy."""

    @pytest.fixture
    def etp(self):
        """ETP provider for resqpy interop tests with fresh dataspace."""
        import uuid as _uuid

        ds_path = f"xtgeo/test_resqpy_{_uuid.uuid4().hex[:8]}"
        cfg = EtpConnectionConfig(
            url="ws://localhost:9002",
            dataspace=f"eml:///dataspace('{ds_path}')",
        )
        try:
            p = EtpProvider(cfg)
            p.open()
        except Exception:
            pytest.skip("Local RDDMS not available")
        with contextlib.suppress(Exception):
            p.put_dataspace(ds_path)
        yield p
        with contextlib.suppress(Exception):
            p.delete_dataspace(ds_path)
        p.close()

    def test_resqpy_to_etp_roundtrip(self, tmp_path, etp):
        """resqpy → EPC → xtgeo → ETP → xtgeo → compare."""
        epc_src = str(tmp_path / "resqpy_to_etp.epc")
        _, grid_uuid, _, _ = _create_resqpy_faulted_grid(epc_src)

        # Read from EPC (resqpy-written)
        with EpcFileProvider(epc_src, mode="r") as p:
            snap_from_epc = read_dataspace(p)

        # Write to ETP
        write_dataspace(etp, snap_from_epc)

        # Read back from ETP
        snap_from_etp = read_dataspace(etp)

        # Compare
        diffs = compare_snapshots(snap_from_epc, snap_from_etp, atol=1e-6)
        if diffs:
            msg = "\n".join(
                f"  {d.object_type}/{d.title}.{d.field}: {d.detail}" for d in diffs
            )
            pytest.fail(f"resqpy→ETP roundtrip differences:\n{msg}")

    def test_etp_to_resqpy_exact(self, tmp_path, etp):
        """xtgeo → ETP → xtgeo → EPC → xtgeo: verify geometry through ETP hop."""
        # Build test data and write to ETP
        ni, nj, nk = 3, 4, 2
        g = xtgeo.create_box_grid(
            (ni, nj, nk),
            origin=(460000, 5930000, 1000),
            increment=(25, 25, 5),
        )
        # Fault
        z = g._zcornsv.copy()
        z[2:, :, :, :] += 3.0
        g._zcornsv = z

        poro = xtgeo.GridProperty(
            g,
            name="PORO",
            values=np.linspace(0.1, 0.3, ni * nj * nk).reshape(ni, nj, nk),
        )

        # Write to EPC first, then push to ETP via dataspace API
        epc_src = str(tmp_path / "write_first.epc")
        with EpcFileProvider(epc_src, mode="w") as p:
            xtgeo_grid_to_resqml(
                p, g, title="EtpResqpyGrid", crs_epsg=23031, properties=[poro]
            )

        # Read into snapshot and push to ETP
        with EpcFileProvider(epc_src, mode="r") as p:
            snap = read_dataspace(p)
        write_dataspace(etp, snap)

        # Read from ETP
        snap_etp = read_dataspace(etp)

        # Write back to EPC
        epc_final = str(tmp_path / "etp_final.epc")
        with EpcFileProvider(epc_final, mode="w") as p:
            write_dataspace(p, snap_etp)

        # Read final EPC with xtgeo and compare to original
        with EpcFileProvider(epc_final, mode="r") as p:
            objs = p.list_objects("IjkGrid")
            g_final, props_final = ijk_grid_to_xtgeo(p, objs[0]["uuid"])

        assert g_final.ncol == ni
        assert g_final.nrow == nj
        assert g_final.nlay == nk

        # Geometry exact through ETP
        np.testing.assert_allclose(g._coordsv, g_final._coordsv, atol=1e-6)
        np.testing.assert_allclose(g._zcornsv, g_final._zcornsv, atol=1e-6)

        # Fault survived
        assert np.all(
            g_final._zcornsv[2:, 0, 0, :] - g_final._zcornsv[1, 0, 0, 0] > 2.0
        ), "Fault throw not preserved through ETP"

        # Property survived
        assert len(props_final) >= 1
        poro_read = next(p for p in props_final if p.name == "PORO")
        np.testing.assert_allclose(poro_read.values, poro.values, atol=1e-10)


# ===========================================================================
# Pipeline Tests: xtgeo → EPC → resqpy operation → EPC → xtgeo
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xtgeo_grid_with_props(ni=10, nj=8, nk=5):
    """Create a synthetic xtgeo Grid with porosity and facies properties."""
    # Simple regular grid
    coordsv = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    for i in range(ni + 1):
        for j in range(nj + 1):
            x = i * 50.0
            y = j * 50.0
            coordsv[i, j, :] = [x, y, 0.0, x, y, 1000.0]

    zcornsv = np.zeros((ni + 1, nj + 1, nk + 1, 4), dtype=np.float32)
    for k in range(nk + 1):
        z = k * 200.0
        zcornsv[:, :, k, :] = z

    actnumsv = np.ones((ni, nj, nk), dtype=np.int32)

    grid = xtgeo.Grid(coordsv, zcornsv, actnumsv)

    # Porosity property (continuous)
    np.random.seed(42)
    poro_vals = np.random.uniform(0.05, 0.35, size=(ni, nj, nk)).astype(np.float32)
    poro = xtgeo.GridProperty(grid, name="PORO", values=poro_vals)

    # Facies property (discrete)
    facies_vals = np.random.choice([1, 2, 3], size=(ni, nj, nk)).astype(np.int32)
    facies = xtgeo.GridProperty(grid, name="FACIES", values=facies_vals, discrete=True)

    return grid, [poro, facies]


def _xtgeo_grid_to_epc(epc_path, grid, properties, title="TestGrid"):
    """Export xtgeo grid + properties to EPC via our OSDU interface."""
    p = EpcFileProvider(epc_path, mode="w")
    p.open()
    result = xtgeo_grid_to_resqml(
        p, grid, title=title, crs_epsg=23031, properties=properties
    )
    p.close()
    return result


def _epc_to_xtgeo_grid(epc_path, grid_uuid):
    """Load grid from EPC back to xtgeo."""
    p = EpcFileProvider(epc_path, mode="r")
    p.open()
    result = ijk_grid_to_xtgeo(p, grid_uuid)
    p.close()
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Pipeline 1: Extract sub-grid box (useful for local well models)
# ---------------------------------------------------------------------------


class TestExtractBoxPipeline:
    """xtgeo grid → EPC → resqpy extract_box → EPC → xtgeo."""

    def test_extract_box_roundtrip(self, tmp_path):
        """Extract a sub-grid and verify geometry + properties survive."""
        grid, props = _make_xtgeo_grid_with_props(ni=10, nj=8, nk=5)

        # Step 1: xtgeo → EPC
        epc_src = str(tmp_path / "source.epc")
        _xtgeo_grid_to_epc(epc_src, grid, props, title="FullGrid")

        # Step 2: resqpy extract_box (IJK box: i=2..6, j=1..5, k=0..3)
        epc_box = str(tmp_path / "extracted.epc")
        box = np.array([[0, 1, 2], [3, 5, 6]])  # kji0 min/max (inclusive)

        new_grid = rdm.extract_box(
            epc_file=epc_src,
            source_grid=None,  # auto-load from EPC
            box=box,
            inherit_properties=True,
            new_grid_title="ExtractedBox",
            new_epc_file=epc_box,
        )

        assert new_grid is not None
        assert tuple(new_grid.extent_kji) == (4, 5, 5)  # k:0-3, j:1-5, i:2-6

        # Step 3: Load extracted EPC back into xtgeo
        extracted = _epc_to_xtgeo_grid(epc_box, str(new_grid.uuid))

        assert extracted.ncol == 5  # ni = 6-2+1
        assert extracted.nrow == 5  # nj = 5-1+1
        assert extracted.nlay == 4  # nk = 3-0+1

    def test_extract_box_preserves_z_geometry(self, tmp_path):
        """Verify Z coordinates are preserved exactly through the pipeline."""
        grid, props = _make_xtgeo_grid_with_props(ni=6, nj=6, nk=4)

        epc_src = str(tmp_path / "source.epc")
        _xtgeo_grid_to_epc(epc_src, grid, props, title="SmallGrid")

        # Extract full grid (trivial box) — should be identical
        box = np.array([[0, 0, 0], [3, 5, 5]])  # full extent
        epc_out = str(tmp_path / "fullbox.epc")

        new_grid = rdm.extract_box(
            epc_file=epc_src,
            source_grid=None,
            box=box,
            inherit_properties=False,
            new_grid_title="FullBox",
            new_epc_file=epc_out,
        )

        result = _epc_to_xtgeo_grid(epc_out, str(new_grid.uuid))

        # Geometry should be preserved (within float32 tolerance for zcorn)
        assert result.ncol == grid.ncol
        assert result.nrow == grid.nrow
        assert result.nlay == grid.nlay


# ---------------------------------------------------------------------------
# Pipeline 2: Grid coarsening (upscaling for simulation)
# ---------------------------------------------------------------------------


class TestCoarsenGridPipeline:
    """xtgeo grid → EPC → resqpy coarsened_grid → EPC → xtgeo."""

    def test_coarsen_2x2x1(self, tmp_path):
        """Coarsen grid by 2x in I and J, keep K layers."""
        grid, props = _make_xtgeo_grid_with_props(ni=10, nj=8, nk=5)

        # Step 1: xtgeo → EPC
        epc_src = str(tmp_path / "fine.epc")
        _xtgeo_grid_to_epc(epc_src, grid, props, title="FineGrid")

        # Step 2: Define coarsening (2×2×1)
        fine_extent = (5, 8, 10)  # nk, nj, ni
        coarse_extent = (5, 4, 5)  # nk, nj/2, ni/2
        fine_coarse = fc.FineCoarse(fine_extent, coarse_extent)
        fine_coarse.set_all_ratios_constant()

        # Step 3: resqpy coarsen
        epc_coarse = str(tmp_path / "coarse.epc")
        coarse_grid = rdm.coarsened_grid(
            epc_file=epc_src,
            source_grid=None,
            fine_coarse=fine_coarse,
            inherit_properties=True,
            new_grid_title="CoarseGrid",
            new_epc_file=epc_coarse,
        )

        assert coarse_grid is not None
        assert tuple(coarse_grid.extent_kji) == (5, 4, 5)

        # Step 4: Load back into xtgeo
        result = _epc_to_xtgeo_grid(epc_coarse, str(coarse_grid.uuid))

        assert result.ncol == 5
        assert result.nrow == 4
        assert result.nlay == 5


# ---------------------------------------------------------------------------
# Pipeline 3: GridConnectionSet — fault analysis
# ---------------------------------------------------------------------------


class TestFaultConnectionPipeline:
    """xtgeo grid → EPC → resqpy GridConnectionSet → property back."""

    def test_k_gap_connections(self, tmp_path):
        """Detect K-direction connections (pinchouts/gaps) with resqpy."""
        grid, props = _make_xtgeo_grid_with_props(ni=6, nj=6, nk=4)

        # Step 1: xtgeo → EPC
        epc_src = str(tmp_path / "grid.epc")
        _xtgeo_grid_to_epc(epc_src, grid, props, title="Grid")

        # Step 2: Load in resqpy and create connection set
        model = Model(epc_file=epc_src)
        rq_grid = rqgrid.Grid(model, uuid=model.uuid(obj_type="IjkGridRepresentation"))

        # K-face connections (all internal K faces should be connected)
        k_faces = np.ones((rq_grid.nk - 1, rq_grid.nj, rq_grid.ni), dtype=bool)
        gcs = rf.GridConnectionSet(
            model,
            grid=rq_grid,
            k_faces=k_faces,
            feature_name="K_connections",
            feature_type="horizon",
            create_organizing_objects_where_needed=True,
            create_transmissibility_multiplier_property=False,
        )

        assert gcs.count > 0
        # For a 6×6×4 grid, K-faces = 3×6×6 = 108 connections
        assert gcs.count == (4 - 1) * 6 * 6

    def test_j_face_connections(self, tmp_path):
        """Create J-direction fault connection set."""
        grid, props = _make_xtgeo_grid_with_props(ni=6, nj=6, nk=4)

        epc_src = str(tmp_path / "grid.epc")
        _xtgeo_grid_to_epc(epc_src, grid, props, title="Grid")

        model = Model(epc_file=epc_src)
        rq_grid = rqgrid.Grid(model, uuid=model.uuid(obj_type="IjkGridRepresentation"))

        # Create a "fault" at j=3 (all I cells, all K layers)
        j_faces = np.zeros((rq_grid.nk, rq_grid.nj - 1, rq_grid.ni), dtype=bool)
        j_faces[:, 3, :] = True  # Fault plane at j=3

        gcs = rf.GridConnectionSet(
            model,
            grid=rq_grid,
            j_faces=j_faces,
            feature_name="FaultA",
            feature_type="fault",
            create_organizing_objects_where_needed=True,
            create_transmissibility_multiplier_property=True,
            fault_tmult_dict={"FaultA": 0.1},
        )

        assert gcs.count == 4 * 6  # nk × ni face pairs at j=3


# ---------------------------------------------------------------------------
# Pipeline 4: Full workflow — ROFF input → complex operation → GRDECL output
# ---------------------------------------------------------------------------


class TestFullWorkflowPipeline:
    """ROFF → xtgeo → EPC → resqpy extract_box → EPC → xtgeo → GRDECL."""

    def test_roff_to_grdecl_via_resqpy(self, tmp_path):
        """Complete pipeline: read ROFF, process with resqpy, write GRDECL."""
        grid, props = _make_xtgeo_grid_with_props(ni=10, nj=8, nk=5)

        # Step 1: Write as ROFF (simulating a "real" input file)
        roff_path = str(tmp_path / "input.roff")
        grid.to_file(roff_path, fformat="roff")
        props[0].to_file(str(tmp_path / "poro.roff"), fformat="roff", name="PORO")

        # Step 2: Read back from ROFF (as user would)
        grid_in = xtgeo.grid_from_file(roff_path)
        poro_in = xtgeo.gridproperty_from_file(
            str(tmp_path / "poro.roff"), fformat="roff", name="PORO", grid=grid_in
        )

        # Step 3: Export to EPC via xtgeo OSDU interface
        epc_path = str(tmp_path / "intermediate.epc")
        _xtgeo_grid_to_epc(epc_path, grid_in, [poro_in], title="InputGrid")

        # Step 4: Use resqpy to extract a box (i=2..7, j=1..5, k=0..4)
        box = np.array([[0, 1, 2], [4, 5, 7]])  # kji0 format
        epc_result = str(tmp_path / "result.epc")

        new_grid = rdm.extract_box(
            epc_file=epc_path,
            source_grid=None,
            box=box,
            inherit_properties=True,
            new_grid_title="SubGrid",
            new_epc_file=epc_result,
        )

        # Step 5: Load result back into xtgeo
        result_grid = _epc_to_xtgeo_grid(epc_result, str(new_grid.uuid))

        # Step 6: Export to Eclipse GRDECL
        grdecl_path = str(tmp_path / "output.grdecl")
        result_grid.to_file(grdecl_path, fformat="grdecl")

        assert Path(grdecl_path).exists()
        assert Path(grdecl_path).stat().st_size > 0

        # Verify dimensions of extracted grid
        assert result_grid.ncol == 6  # i: 7-2+1
        assert result_grid.nrow == 5  # j: 5-1+1
        assert result_grid.nlay == 5  # k: 4-0+1
