# -*- coding: utf-8 -*-
"""Tests for the high-level user API (_api.py).

Tests cover:
- list_osdu_objects / search_osdu discovery
- grid_from_osdu / grid_to_osdu roundtrip
- surface_from_osdu / surface_to_osdu roundtrip
- points_from_osdu / points_to_osdu roundtrip
- polygons_from_osdu / polygons_to_osdu roundtrip
- Name-based search resolution
- Error handling (not found, bad input)

Uses EPC file backend (no network dependency).
"""

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import (
    grid_from_osdu,
    grid_to_osdu,
    list_osdu_objects,
    points_from_osdu,
    points_to_osdu,
    polygons_from_osdu,
    polygons_to_osdu,
    search_osdu,
    surface_from_osdu,
    surface_to_osdu,
)


@pytest.fixture
def tmp_epc(tmp_path):
    """Return a temporary EPC file path."""
    return str(tmp_path / "test.epc")


class TestDiscovery:
    """Test list and search functions."""

    def test_list_objects_empty_epc(self, tmp_epc):
        """EPC with only CRS should return that object."""
        # Create an EPC with just a grid to have at least one object
        grid = xtgeo.create_box_grid((2, 2, 2))
        grid_to_osdu(tmp_epc, grid, title="MiniGrid", crs_epsg=23031)
        objects = list_osdu_objects(tmp_epc)
        assert isinstance(objects, list)
        assert len(objects) >= 1

    def test_list_objects_with_type_filter(self, tmp_epc):
        """Filter by type returns only matching objects."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        grid_to_osdu(tmp_epc, grid, title="TestGrid", crs_epsg=23031)

        grids = list_osdu_objects(tmp_epc, object_type="grid")
        surfaces = list_osdu_objects(tmp_epc, object_type="surface")

        assert len(grids) >= 1
        assert all("IjkGrid" in obj.get("type", "") for obj in grids)
        assert len(surfaces) == 0

    def test_search_by_name(self, tmp_epc):
        """Search by name pattern."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        grid_to_osdu(tmp_epc, grid, title="DrogonGrid", crs_epsg=23031)

        results = search_osdu(tmp_epc, name="*Drogon*")
        assert len(results) >= 1
        assert any("DrogonGrid" in r.get("title", "") for r in results)

        results = search_osdu(tmp_epc, name="*NoMatch*")
        assert len(results) == 0

    def test_search_by_name_and_type(self, tmp_epc):
        """Combined name + type filter."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.full((3, 3, 2), 0.2))
        grid_to_osdu(tmp_epc, grid, title="TestGrid", properties=[poro], crs_epsg=23031)

        # Search for property by name
        results = search_osdu(tmp_epc, name="PORO", object_type="property")
        assert len(results) >= 1


class TestGridRoundtrip:
    """Test grid read/write via the high-level API."""

    def test_basic_grid(self, tmp_epc):
        """Write and read back a basic grid."""
        grid = xtgeo.create_box_grid((5, 4, 3))
        grid_to_osdu(tmp_epc, grid, title="BoxGrid", crs_epsg=23031)

        grid2, props = grid_from_osdu(tmp_epc, name="BoxGrid")
        assert grid2.ncol == 5
        assert grid2.nrow == 4
        assert grid2.nlay == 3

    def test_grid_with_properties(self, tmp_epc):
        """Write and read grid with properties."""
        grid = xtgeo.create_box_grid((4, 4, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(4, 4, 2))
        permx = xtgeo.GridProperty(
            grid, name="PERMX", values=np.random.rand(4, 4, 2) * 100
        )

        grid_to_osdu(
            tmp_epc, grid, title="PropGrid", properties=[poro, permx], crs_epsg=23031
        )

        grid2, props2 = grid_from_osdu(tmp_epc, name="PropGrid")
        assert len(props2) >= 2
        names = {p.name for p in props2}
        assert "PORO" in names
        assert "PERMX" in names

    def test_grid_no_properties(self, tmp_epc):
        """Read grid without loading properties."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.ones((3, 3, 2)))
        grid_to_osdu(tmp_epc, grid, title="G", properties=[poro], crs_epsg=23031)

        grid2, props2 = grid_from_osdu(tmp_epc, name="G", load_properties=False)
        assert grid2.ncol == 3
        assert len(props2) == 0

    def test_grid_geometry_exact(self, tmp_epc):
        """Geometry is preserved exactly."""
        grid = xtgeo.create_box_grid((3, 4, 5))
        orig_coordsv = grid._coordsv.copy()
        orig_zcornsv = grid._zcornsv.copy()
        orig_actnumsv = grid._actnumsv.copy()

        grid_to_osdu(tmp_epc, grid, title="Exact", crs_epsg=23031)
        grid2, _ = grid_from_osdu(tmp_epc, name="Exact")

        np.testing.assert_array_equal(grid2._coordsv, orig_coordsv)
        np.testing.assert_array_equal(grid2._zcornsv, orig_zcornsv)
        np.testing.assert_array_equal(grid2._actnumsv, orig_actnumsv)


class TestSurfaceRoundtrip:
    """Test surface read/write via the high-level API."""

    def test_basic_surface(self, tmp_epc):
        """Write and read back a surface."""
        vals = np.arange(20 * 30, dtype=np.float64).reshape(20, 30) * 0.5
        surf = xtgeo.RegularSurface(
            ncol=20,
            nrow=30,
            xinc=25.0,
            yinc=25.0,
            xori=100.0,
            yori=200.0,
            values=vals,
        )
        surface_to_osdu(tmp_epc, surf, title="TestSurf", crs_epsg=23031)

        surf2 = surface_from_osdu(tmp_epc, name="TestSurf")
        assert surf2.ncol == 20
        assert surf2.nrow == 30
        np.testing.assert_allclose(surf2.values, vals, atol=1e-10)


class TestPointsRoundtrip:
    """Test points read/write via the high-level API."""

    def test_basic_points(self, tmp_epc):
        """Write and read back points."""
        import pandas as pd

        pts = xtgeo.Points(
            pd.DataFrame(
                {
                    "X_UTME": [1.0, 2.0, 3.0],
                    "Y_UTMN": [4.0, 5.0, 6.0],
                    "Z_TVDSS": [7.0, 8.0, 9.0],
                }
            )
        )
        points_to_osdu(tmp_epc, pts, title="TestPts", crs_epsg=23031)

        pts2 = points_from_osdu(tmp_epc, name="TestPts")
        np.testing.assert_allclose(
            pts2.get_dataframe()["X_UTME"].values, [1.0, 2.0, 3.0]
        )


class TestPolygonsRoundtrip:
    """Test polygons read/write via the high-level API."""

    def test_basic_polygons(self, tmp_epc):
        """Write and read back polygons."""
        import pandas as pd

        polys = xtgeo.Polygons(
            pd.DataFrame(
                {
                    "X_UTME": [0.0, 1.0, 1.0, 0.0, 5.0, 6.0, 6.0, 5.0],
                    "Y_UTMN": [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0],
                    "Z_TVDSS": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    "POLY_ID": [0, 0, 0, 0, 1, 1, 1, 1],
                }
            )
        )
        polygons_to_osdu(tmp_epc, polys, title="TestPolys", crs_epsg=23031)

        polys2 = polygons_from_osdu(tmp_epc, name="TestPolys")
        df2 = polys2.get_dataframe()
        assert len(df2) == 8


class TestErrorHandling:
    """Test error cases."""

    def test_no_name_or_uuid(self, tmp_epc):
        """Should raise ValueError when neither name nor uuid given."""
        grid = xtgeo.create_box_grid((2, 2, 2))
        grid_to_osdu(tmp_epc, grid, title="G", crs_epsg=23031)

        with pytest.raises(ValueError, match="Either 'uuid' or 'name'"):
            grid_from_osdu(tmp_epc)

    def test_name_not_found(self, tmp_epc):
        """Should raise ValueError when name doesn't match."""
        grid = xtgeo.create_box_grid((2, 2, 2))
        grid_to_osdu(tmp_epc, grid, title="G", crs_epsg=23031)

        with pytest.raises(ValueError, match="No grid found"):
            grid_from_osdu(tmp_epc, name="NonExistent")

    def test_unsupported_file_format(self):
        """Should raise ValueError for non-EPC files."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            grid_from_osdu("/tmp/model.roff", name="X")

    def test_bad_source_type(self):
        """Should raise TypeError for invalid source."""
        with pytest.raises(TypeError, match="Expected OsduSession or file path"):
            grid_from_osdu(12345, name="X")


# Helper
def _make_crs_xml(uuid, epsg):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<eml:LocalDepth3dCrs xmlns:eml="http://www.energistics.org/energyml/data/commonv2"
    uuid="{uuid}" schemaVersion="2.0">
    <eml:Citation>
        <eml:Title>EPSG:{epsg}</eml:Title>
    </eml:Citation>
    <ProjectedCrs>
        <EpsgCode>{epsg}</EpsgCode>
    </ProjectedCrs>
    <VerticalCrs>
        <EpsgCode>5714</EpsgCode>
    </VerticalCrs>
</eml:LocalDepth3dCrs>"""
