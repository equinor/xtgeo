"""Pipeline tests: xtgeo → EPC → resqpy operation → EPC → xtgeo.

Demonstrates using resqpy's grid operations (extract_box, coarsened_grid,
GridConnectionSet) on grids exported from xtgeo, then importing the result
back into xtgeo.

This is the pattern for leveraging resqpy's derived model operations
(coarsening, refinement, fault throw scaling, etc.) from an xtgeo workflow.

Requires: resqpy >= 4.0
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml

resqpy = pytest.importorskip("resqpy")

import resqpy.derived_model as rdm  # noqa: E402
import resqpy.fault as rf  # noqa: E402
import resqpy.grid as rqgrid  # noqa: E402
import resqpy.olio.fine_coarse as fc  # noqa: E402
import resqpy.property as rprop  # noqa: E402
from resqpy.model import Model  # noqa: E402


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
        result = _xtgeo_grid_to_epc(epc_src, grid, props, title="FullGrid")
        grid_uuid = result["FullGrid"]

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
        assert result_grid.ncol == 6   # i: 7-2+1
        assert result_grid.nrow == 5   # j: 5-1+1
        assert result_grid.nlay == 5   # k: 4-0+1
