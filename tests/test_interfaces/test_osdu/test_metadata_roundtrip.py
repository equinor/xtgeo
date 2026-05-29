# -*- coding: utf-8 -*-
"""Tests for RESQML metadata preservation through xtgeo objects.

Verifies that:
- Reading from OSDU/EPC attaches RESQML metadata (UUID, CRS, property kind)
- Writing back reuses those UUIDs when available
- Objects without native metadata (Points, Polygons) still carry RESQML info
"""

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import (
    grid_from_osdu,
    grid_to_osdu,
    points_from_osdu,
    points_to_osdu,
    polygons_from_osdu,
    polygons_to_osdu,
    surface_from_osdu,
    surface_to_osdu,
)
from xtgeo.interfaces.osdu._resqml_meta import _get_resqml_meta, _set_resqml_meta


@pytest.fixture
def tmp_epc(tmp_path):
    return str(tmp_path / "test.epc")


class TestGridMetadataPreservation:
    """Grid + GridProperty metadata roundtrip."""

    def test_grid_has_resqml_metadata(self, tmp_epc):
        """After reading from EPC, grid carries RESQML metadata."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        grid_to_osdu(tmp_epc, grid, title="TestGrid", crs_epsg=23031)

        grid2, _ = grid_from_osdu(tmp_epc, name="TestGrid")
        meta = _get_resqml_meta(grid2)

        assert meta["uuid"]  # non-empty UUID
        assert meta["schema_version"] == "2.0.1"
        assert meta["object_type"] == "IjkGridRepresentation"
        assert meta["crs_uuid"]  # non-empty CRS UUID

    def test_grid_property_has_resqml_metadata(self, tmp_epc):
        """Grid properties carry their RESQML identity."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.full((3, 3, 2), 0.25))
        grid_to_osdu(tmp_epc, grid, title="G", properties=[poro], crs_epsg=23031)

        _, props = grid_from_osdu(tmp_epc, name="G")
        assert len(props) >= 1
        prop_meta = _get_resqml_meta(props[0])
        assert prop_meta["uuid"]
        assert prop_meta["property_kind"] == "Porosity"
        assert prop_meta["uom"] == "fraction"
        assert prop_meta["osdu_reference"] is not None

    def test_grid_uuid_preserved_on_rewrite(self, tmp_epc, tmp_path):
        """Writing a grid back preserves the original UUID if no explicit uuid given."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        grid_to_osdu(tmp_epc, grid, title="G", crs_epsg=23031)

        grid2, _ = grid_from_osdu(tmp_epc, name="G")
        original_uuid = _get_resqml_meta(grid2)["uuid"]

        # Write to a new file — UUID should be preserved from metadata
        epc2 = str(tmp_path / "out.epc")
        result = grid_to_osdu(epc2, grid2, title="G2", crs_epsg=23031)
        assert result["G2"] == original_uuid

    def test_property_uuid_preserved_on_rewrite(self, tmp_epc, tmp_path):
        """Property UUIDs are preserved through roundtrip."""
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.ones((3, 3, 2)) * 0.3)
        grid_to_osdu(tmp_epc, grid, title="G", properties=[poro], crs_epsg=23031)

        grid2, props2 = grid_from_osdu(tmp_epc, name="G")
        orig_prop_uuid = _get_resqml_meta(props2[0])["uuid"]

        epc2 = str(tmp_path / "out.epc")
        result = grid_to_osdu(
            epc2, grid2, title="G2", properties=props2, crs_epsg=23031
        )
        assert result["PORO"] == orig_prop_uuid


class TestSurfaceMetadataPreservation:
    """Surface metadata roundtrip."""

    def test_surface_has_resqml_metadata(self, tmp_epc):
        """Surface carries RESQML metadata after read."""
        vals = np.arange(100, dtype=np.float64).reshape(10, 10)
        surf = xtgeo.RegularSurface(ncol=10, nrow=10, xinc=25, yinc=25, values=vals)
        surface_to_osdu(tmp_epc, surf, title="TestSurf", crs_epsg=23031)

        surf2 = surface_from_osdu(tmp_epc, name="TestSurf")
        meta = _get_resqml_meta(surf2)

        assert meta["uuid"]
        assert meta["object_type"] == "Grid2dRepresentation"
        assert meta["crs_uuid"]

    def test_surface_uuid_preserved(self, tmp_epc, tmp_path):
        """Surface UUID roundtrips."""
        vals = np.zeros((5, 5))
        surf = xtgeo.RegularSurface(ncol=5, nrow=5, xinc=10, yinc=10, values=vals)
        surface_to_osdu(tmp_epc, surf, title="S", crs_epsg=23031)

        surf2 = surface_from_osdu(tmp_epc, name="S")
        orig_uuid = _get_resqml_meta(surf2)["uuid"]

        epc2 = str(tmp_path / "out.epc")
        result = surface_to_osdu(epc2, surf2, title="S2", crs_epsg=23031)
        assert result["S2"] == orig_uuid


class TestPointsMetadataPreservation:
    """Points metadata (uses _resqml_meta fallback since no native metadata)."""

    def test_points_have_resqml_metadata(self, tmp_epc):
        """Points carry RESQML metadata via fallback attribute."""
        import pandas as pd

        pts = xtgeo.Points(
            pd.DataFrame(
                {"X_UTME": [1.0, 2.0], "Y_UTMN": [3.0, 4.0], "Z_TVDSS": [5.0, 6.0]}
            )
        )
        points_to_osdu(tmp_epc, pts, title="Pts", crs_epsg=23031)

        pts2 = points_from_osdu(tmp_epc, name="Pts")
        meta = _get_resqml_meta(pts2)

        assert meta["uuid"]
        assert meta["object_type"] == "PointSetRepresentation"

    def test_points_uuid_preserved(self, tmp_epc, tmp_path):
        """Points UUID roundtrips."""
        import pandas as pd

        pts = xtgeo.Points(
            pd.DataFrame({"X_UTME": [1.0], "Y_UTMN": [2.0], "Z_TVDSS": [3.0]})
        )
        points_to_osdu(tmp_epc, pts, title="P", crs_epsg=23031)

        pts2 = points_from_osdu(tmp_epc, name="P")
        orig_uuid = _get_resqml_meta(pts2)["uuid"]

        epc2 = str(tmp_path / "out.epc")
        result = points_to_osdu(epc2, pts2, title="P2", crs_epsg=23031)
        assert result["P2"] == orig_uuid


class TestPolygonsMetadataPreservation:
    """Polygons metadata."""

    def test_polygons_have_resqml_metadata(self, tmp_epc):
        """Polygons carry RESQML metadata."""
        import pandas as pd

        polys = xtgeo.Polygons(
            pd.DataFrame(
                {
                    "X_UTME": [0, 1, 1, 0],
                    "Y_UTMN": [0, 0, 1, 1],
                    "Z_TVDSS": [0, 0, 0, 0],
                    "POLY_ID": [0, 0, 0, 0],
                }
            )
        )
        polygons_to_osdu(tmp_epc, polys, title="Poly", crs_epsg=23031)

        polys2 = polygons_from_osdu(tmp_epc, name="Poly")
        meta = _get_resqml_meta(polys2)

        assert meta["uuid"]
        assert meta["object_type"] == "PolylineSetRepresentation"

    def test_polygons_uuid_preserved(self, tmp_epc, tmp_path):
        """Polygons UUID roundtrips."""
        import pandas as pd

        polys = xtgeo.Polygons(
            pd.DataFrame(
                {
                    "X_UTME": [0, 1, 1, 0],
                    "Y_UTMN": [0, 0, 1, 1],
                    "Z_TVDSS": [0, 0, 0, 0],
                    "POLY_ID": [0, 0, 0, 0],
                }
            )
        )
        polygons_to_osdu(tmp_epc, polys, title="Q", crs_epsg=23031)

        polys2 = polygons_from_osdu(tmp_epc, name="Q")
        orig_uuid = _get_resqml_meta(polys2)["uuid"]

        epc2 = str(tmp_path / "out.epc")
        result = polygons_to_osdu(epc2, polys2, title="Q2", crs_epsg=23031)
        assert result["Q2"] == orig_uuid


class TestMetadataManualAttachment:
    """Users can manually set RESQML metadata before writing."""

    def test_manual_uuid_on_grid(self, tmp_epc):
        """User can set a custom UUID via metadata before export."""
        grid = xtgeo.create_box_grid((2, 2, 2))
        custom_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        _set_resqml_meta(grid, {"uuid": custom_uuid, "crs_uuid": None})

        result = grid_to_osdu(tmp_epc, grid, title="Custom", crs_epsg=23031)
        assert result["Custom"] == custom_uuid

    def test_manual_uuid_on_points(self, tmp_epc):
        """User can set UUID on objects without native metadata."""
        import pandas as pd

        pts = xtgeo.Points(
            pd.DataFrame({"X_UTME": [1.0], "Y_UTMN": [2.0], "Z_TVDSS": [3.0]})
        )
        custom_uuid = "11111111-2222-3333-4444-555555555555"
        _set_resqml_meta(pts, {"uuid": custom_uuid})

        result = points_to_osdu(tmp_epc, pts, title="ManualPts", crs_epsg=23031)
        assert result["ManualPts"] == custom_uuid
