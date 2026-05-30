# -*- coding: utf-8 -*-
"""Tests targeting offline coverage gaps in the OSDU interface modules.

Covers: _api.py, _metadata.py, _properties.py, _crs.py, _epc_provider.py,
        _ijk_grid.py, _polyline.py, _grid2d.py, _pointset.py edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import (
    EpcFileProvider,
)
from xtgeo.interfaces.osdu._crs import LocalDepth3dCrs
from xtgeo.interfaces.osdu._metadata import (
    OsduPropertyMapping,
    OsduWorkProductMetadata,
    ecl_keyword_to_osdu,
    list_supported_properties,
    osdu_name_to_ecl_keyword,
    osdu_reference_to_mapping,
    resolve_property_mapping,
)

# ---------------------------------------------------------------------------
# _metadata.py coverage
# ---------------------------------------------------------------------------


class TestResolvePropertyMapping:
    """Cover resolve_property_mapping branches."""

    def test_resolve_by_title_porosity_maps_to_poro(self):
        m = resolve_property_mapping(title="POROSITY")
        assert m is not None
        assert m.ecl_keyword == "PORO"

    def test_resolve_by_title_net_gross_synonym_maps_to_ntg(self):
        m = resolve_property_mapping(title="NET/GROSS")
        assert m is not None
        assert m.ecl_keyword == "NTG"

    def test_resolve_by_title_sw_alias_maps_to_swat(self):
        m = resolve_property_mapping(title="SW")
        assert m is not None
        assert m.ecl_keyword == "SWAT"

    def test_resolve_by_title_klogh_alias_maps_to_permx(self):
        m = resolve_property_mapping(title="KLOGH")
        assert m is not None
        assert m.ecl_keyword == "PERMX"

    def test_resolve_by_title_facies_code_alias_is_discrete(self):
        m = resolve_property_mapping(title="FACIES_CODE")
        assert m is not None
        assert m.ecl_keyword == "FACIES"
        assert m.is_discrete

    def test_resolve_by_title_returns_none_for_unrecognized_name(self):
        m = resolve_property_mapping(title="UNKNOWN_XYZ")
        assert m is None

    def test_resolve_by_kind_porosity_maps_to_poro(self):
        m = resolve_property_mapping(property_kind="porosity")
        assert m is not None
        assert m.ecl_keyword == "PORO"

    def test_resolve_by_kind_permeability_rock_with_i_facet_maps_to_permx(self):
        m = resolve_property_mapping(
            property_kind="permeability rock", facet_direction="I"
        )
        assert m is not None
        assert m.ecl_keyword == "PERMX"

    def test_resolve_by_kind_permeability_rock_with_j_facet_maps_to_permy(self):
        m = resolve_property_mapping(
            property_kind="permeability rock", facet_direction="J"
        )
        assert m is not None
        assert m.ecl_keyword == "PERMY"

    def test_resolve_by_kind_permeability_with_k_facet_maps_to_permz(self):
        m = resolve_property_mapping(property_kind="permeability", facet_direction="K")
        assert m is not None
        assert m.ecl_keyword == "PERMZ"

    def test_resolve_by_kind_transmissibility_with_x_facet_maps_to_tranx(self):
        m = resolve_property_mapping(
            property_kind="transmissibility", facet_direction="X"
        )
        assert m is not None
        assert m.ecl_keyword == "TRANX"

    def test_resolve_by_kind_transmissibility_with_y_facet_maps_to_trany(self):
        m = resolve_property_mapping(
            property_kind="transmissibility", facet_direction="Y"
        )
        assert m is not None
        assert m.ecl_keyword == "TRANY"

    def test_resolve_by_kind_transmissibility_with_z_facet_maps_to_tranz(self):
        m = resolve_property_mapping(
            property_kind="transmissibility", facet_direction="Z"
        )
        assert m is not None
        assert m.ecl_keyword == "TRANZ"

    def test_resolve_by_kind_water_saturation_maps_to_swat(self):
        m = resolve_property_mapping(property_kind="water saturation")
        assert m is not None
        assert m.ecl_keyword == "SWAT"

    def test_resolve_by_kind_cell_thickness_maps_to_dz(self):
        m = resolve_property_mapping(property_kind="cell thickness")
        assert m is not None
        assert m.ecl_keyword == "DZ"

    def test_resolve_by_kind_active_maps_to_actnum_discrete(self):
        m = resolve_property_mapping(property_kind="active")
        assert m is not None
        assert m.ecl_keyword == "ACTNUM"
        assert m.is_discrete

    def test_resolve_by_kind_region_maps_to_fipnum(self):
        m = resolve_property_mapping(property_kind="region")
        assert m is not None
        assert m.ecl_keyword == "FIPNUM"

    def test_resolve_by_kind_rock_type_maps_to_rocknum(self):
        m = resolve_property_mapping(property_kind="rock type")
        assert m is not None
        assert m.ecl_keyword == "ROCKNUM"

    def test_resolve_by_kind_returns_none_for_unrecognized_kind(self):
        m = resolve_property_mapping(property_kind="completely_unknown_kind")
        assert m is None

    def test_resolve_returns_none_when_no_title_or_kind_given(self):
        m = resolve_property_mapping()
        assert m is None

    def test_resolve_by_title_strips_whitespace_before_matching(self):
        m = resolve_property_mapping(title="  PORO  ")
        assert m is not None


class TestEclKeywordToOsdu:
    def test_poro_maps_to_osdu_porosity(self):
        m = ecl_keyword_to_osdu("PORO")
        assert m is not None
        assert m.osdu_name == "Porosity"

    def test_swat_maps_to_osdu_water_saturation(self):
        m = ecl_keyword_to_osdu("SWAT")
        assert m is not None
        assert m.osdu_name == "Water Saturation"

    def test_returns_none_for_unrecognized_keyword(self):
        m = ecl_keyword_to_osdu("UNKNOWN_KW")
        assert m is None

    def test_matches_lowercase_keyword(self):
        m = ecl_keyword_to_osdu("poro")
        assert m is not None

    def test_strips_whitespace_from_keyword(self):
        m = ecl_keyword_to_osdu("  PERMX  ")
        assert m is not None
        assert m.ecl_keyword == "PERMX"


class TestOsduReferenceToMapping:
    def test_exact_osdu_reference_resolves_to_poro(self):
        m = ecl_keyword_to_osdu("PORO")
        assert m is not None
        m2 = osdu_reference_to_mapping(m.osdu_reference)
        assert m2 is not None
        assert m2.ecl_keyword == "PORO"

    def test_camel_case_reference_resolves_to_swat(self):
        m = osdu_reference_to_mapping(
            "osdu:reference-data--PropertyNameType:WaterSaturation:1.0.0"
        )
        assert m is not None
        assert m.ecl_keyword == "SWAT"

    def test_returns_none_for_empty_or_null_reference(self):
        assert osdu_reference_to_mapping("") is None
        assert osdu_reference_to_mapping(None) is None

    def test_returns_none_for_unknown_property_reference(self):
        m = osdu_reference_to_mapping("osdu:reference-data--PropertyNameType:Xyz:1.0.0")
        assert m is None


class TestOsduNameToEclKeyword:
    def test_porosity_maps_to_poro(self):
        assert osdu_name_to_ecl_keyword("Porosity") == "PORO"

    def test_water_saturation_maps_to_swat(self):
        assert osdu_name_to_ecl_keyword("Water Saturation") == "SWAT"

    def test_permeability_x_maps_to_permx(self):
        assert osdu_name_to_ecl_keyword("Permeability X") == "PERMX"

    def test_matches_lowercase_osdu_name(self):
        assert osdu_name_to_ecl_keyword("porosity") == "PORO"

    def test_returns_none_for_unrecognized_name(self):
        assert osdu_name_to_ecl_keyword("Unknown Property") is None

    def test_returns_none_for_empty_or_null_name(self):
        assert osdu_name_to_ecl_keyword("") is None
        assert osdu_name_to_ecl_keyword(None) is None


class TestListSupportedProperties:
    def test_returns_at_least_30_supported_mappings(self):
        props = list_supported_properties()
        assert isinstance(props, list)
        assert len(props) >= 30  # We have ~40 mappings

    def test_every_entry_has_ecl_keyword_and_osdu_name(self):
        for m in list_supported_properties():
            assert isinstance(m, OsduPropertyMapping)
            assert m.ecl_keyword
            assert m.osdu_name


class TestOsduWorkProductMetadata:
    def test_initializes_with_empty_legal_and_acl_defaults(self):
        meta = OsduWorkProductMetadata(uuid="test-uuid", name="TestObj")
        assert meta.legal_tags == []
        assert meta.acl_viewers == []
        assert meta.acl_owners == []
        assert meta.ancestry_inputs == []

    def test_to_osdu_record_populates_legal_and_acl_fields(self):
        meta = OsduWorkProductMetadata(
            uuid="test-uuid",
            kind="osdu:wks:work-product-component--ResqmlIjkGridRepresentation:1.0.0",
            name="TestGrid",
            description="A test grid",
            legal_tags=["my-legal-tag"],
            acl_owners=["owners@test"],
            acl_viewers=["viewers@test"],
            data_partition="test-partition",
        )
        record = meta.to_osdu_record()
        assert record["id"] == "test-uuid"
        assert record["kind"].startswith("osdu:wks:")
        assert "legal" in record
        assert record["legal"]["legaltags"] == ["my-legal-tag"]
        assert record["acl"]["owners"] == ["owners@test"]
        assert record["acl"]["viewers"] == ["viewers@test"]


# ---------------------------------------------------------------------------
# _crs.py coverage
# ---------------------------------------------------------------------------


class TestLocalDepth3dCrs:
    def test_converts_radians_to_degrees(self):
        crs = LocalDepth3dCrs(areal_rotation=math.pi / 4)
        assert abs(crs.rotation_degrees - 45.0) < 1e-10

    def test_compute_mapaxes_at_zero_rotation(self):
        crs = LocalDepth3dCrs(origin_x=100.0, origin_y=200.0, areal_rotation=0.0)
        p1, p2, p3 = crs.compute_mapaxes()
        assert p1 == (100.0, 200.0)
        assert abs(p2[0] - 101.0) < 1e-10
        assert abs(p3[1] - 201.0) < 1e-10

    def test_local_to_global_translates_without_rotation(self):
        crs = LocalDepth3dCrs(origin_x=100.0, origin_y=200.0, origin_z=0.0)
        gx, gy, gz = crs.local_to_global(10.0, 20.0, 30.0)
        assert abs(gx - 110.0) < 1e-10
        assert abs(gy - 220.0) < 1e-10
        assert abs(gz - 30.0) < 1e-10

    def test_local_to_global_rotates_90_degrees(self):
        crs = LocalDepth3dCrs(origin_x=0.0, origin_y=0.0, areal_rotation=math.pi / 2)
        gx, gy, gz = crs.local_to_global(1.0, 0.0, 0.0)
        assert abs(gx - 0.0) < 1e-10
        assert abs(gy - 1.0) < 1e-10

    def test_local_global_roundtrip_with_rotation_preserves_coordinates(self):
        crs = LocalDepth3dCrs(
            origin_x=500.0,
            origin_y=600.0,
            origin_z=100.0,
            areal_rotation=0.3,
        )
        lx, ly, lz = 10.0, 20.0, 30.0
        gx, gy, gz = crs.local_to_global(lx, ly, lz)
        lx2, ly2, lz2 = crs.global_to_local(gx, gy, gz)
        assert abs(lx2 - lx) < 1e-10
        assert abs(ly2 - ly) < 1e-10
        assert abs(lz2 - lz) < 1e-10

    def test_z_increasing_upward_negates_z_in_transform(self):
        crs = LocalDepth3dCrs(origin_z=0.0, z_increasing_downward=False)
        _, _, gz = crs.local_to_global(0, 0, 10.0)
        assert abs(gz - (-10.0)) < 1e-10
        _, _, lz = crs.global_to_local(0, 0, -10.0)
        assert abs(lz - 10.0) < 1e-10

    def test_xml_roundtrip_preserves_all_fields(self):
        crs = LocalDepth3dCrs(
            title="Test CRS",
            origin_x=100.0,
            origin_y=200.0,
            origin_z=50.0,
            areal_rotation=0.5,
            projected_crs_epsg=23031,
            vertical_crs_epsg=5714,
            z_increasing_downward=True,
        )
        xml = crs.to_xml()
        crs2 = LocalDepth3dCrs.from_xml(xml)
        assert crs2.title == "Test CRS"
        assert abs(crs2.origin_x - 100.0) < 1e-10
        assert abs(crs2.origin_y - 200.0) < 1e-10
        assert abs(crs2.origin_z - 50.0) < 1e-10
        assert abs(crs2.areal_rotation - 0.5) < 1e-10
        assert crs2.projected_crs_epsg == 23031
        assert crs2.vertical_crs_epsg == 5714
        assert crs2.z_increasing_downward is True

    def test_xml_roundtrip_with_no_vertical_crs_keeps_none(self):
        crs = LocalDepth3dCrs(projected_crs_epsg=23031, vertical_crs_epsg=None)
        xml = crs.to_xml()
        crs2 = LocalDepth3dCrs.from_xml(xml)
        assert crs2.projected_crs_epsg == 23031
        assert crs2.vertical_crs_epsg is None

    def test_from_xml_with_only_uuid_uses_zero_defaults(self):
        """XML with no optional elements."""
        from lxml import etree

        from xtgeo.interfaces.osdu._resqml_enums import NS_RESQML20, RESQML_NS_MAP

        root = etree.Element(
            f"{{{NS_RESQML20}}}LocalDepth3dCrs",
            nsmap=RESQML_NS_MAP,
        )
        root.set("uuid", "test-uuid")
        crs = LocalDepth3dCrs.from_xml(root)
        assert crs.uuid == "test-uuid"
        assert crs.origin_x == 0.0
        assert crs.areal_rotation == 0.0
        assert crs.z_increasing_downward is True


# ---------------------------------------------------------------------------
# _api.py coverage — EPC-based paths
# ---------------------------------------------------------------------------


class TestApiSearchOsdu:
    """Test search_osdu with EPC file paths."""

    def test_wildcard_name_matches_grid_title(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="Drogon_v1", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="*Drogon*")
        assert any(r["title"] == "Drogon_v1" for r in results)

    def test_search_by_uuid_returns_single_result(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        uuids = xtgeo.grid_to_osdu(epc, grid, title="GridA", crs_epsg=23031)
        grid_uuid = uuids["GridA"]

        results = xtgeo.search_osdu(epc, uuid=grid_uuid)
        assert len(results) == 1
        assert results[0]["uuid"] == grid_uuid

    def test_search_by_type_filters_grids_from_crs(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="GridB", crs_epsg=23031)

        grids = xtgeo.search_osdu(epc, object_type="grid")
        assert len(grids) >= 1
        # CRS should not appear in grid-type search
        crs_results = xtgeo.search_osdu(epc, object_type="crs")
        for r in crs_results:
            assert "IjkGrid" not in r.get("type", "")

    def test_search_returns_empty_list_when_name_not_found(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="GridC", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="NonExistent*")
        assert results == []


class TestApiImportOsdu:
    """Test import_osdu with EPC files."""

    def test_import_grid_with_property_returns_tuple(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((4, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.ones((4, 3, 2)) * 0.3)
        xtgeo.grid_to_osdu(
            epc, grid, title="ImpGrid", properties=[poro], crs_epsg=23031
        )

        results = xtgeo.search_osdu(epc, name="ImpGrid", object_type="grid")
        result = results[0]
        imported = xtgeo.import_osdu(epc, result)
        assert isinstance(imported, tuple)
        grid2, props2 = imported
        assert grid2.ncol == 4

    def test_import_surface_returns_regular_surface(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        surf = xtgeo.RegularSurface(
            ncol=10, nrow=10, xinc=25, yinc=25, values=np.zeros((10, 10))
        )
        xtgeo.surface_to_osdu(epc, surf, title="ImpSurf", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpSurf", object_type="surface")
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.RegularSurface)

    def test_import_points_returns_points_object(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        pts = xtgeo.Points(values=np.array([[1, 2, 3], [4, 5, 6]]))
        xtgeo.points_to_osdu(epc, pts, title="ImpPts", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpPts", object_type="points")
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.Points)

    def test_import_polygons_returns_polygons_object(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        import pandas as pd

        df = pd.DataFrame(
            {
                "X_UTME": [0, 1, 2],
                "Y_UTMN": [0, 1, 0],
                "Z_TVDSS": [0, 0, 0],
                "POLY_ID": [0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)
        xtgeo.polygons_to_osdu(epc, polys, title="ImpPoly", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpPoly", object_type="polygons")
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.Polygons)

    def test_import_well_returns_well_object(self, tmp_path):
        import pandas as pd

        epc = str(tmp_path / "test.epc")
        df = pd.DataFrame(
            {
                "X_UTME": [460000.0, 460010.0, 460020.0],
                "Y_UTMN": [5930000.0, 5930000.0, 5930000.0],
                "Z_TVDSS": [0.0, 500.0, 1000.0],
                "M_DEPTH": [0.0, 500.0, 1000.0],
            }
        )
        well = xtgeo.Well(
            xpos=460000, ypos=5930000, wname="ImpW", df=df, mdlogname="M_DEPTH"
        )
        xtgeo.well_to_osdu(epc, well, title="ImpW", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpW", object_type="well")
        assert len(results) >= 1
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.Well)

    def test_import_blocked_well_returns_blocked_well_object(self, tmp_path):
        import pandas as pd

        epc = str(tmp_path / "test.epc")
        df = pd.DataFrame(
            {
                "X_UTME": [460000.0, 460010.0, 460020.0],
                "Y_UTMN": [5930000.0, 5930000.0, 5930000.0],
                "Z_TVDSS": [100.0, 200.0, 300.0],
                "M_DEPTH": [100.0, 200.0, 300.0],
                "I_INDEX": np.array([1, 2, 3], dtype=np.int32),
                "J_INDEX": np.array([1, 1, 2], dtype=np.int32),
                "K_INDEX": np.array([1, 1, 1], dtype=np.int32),
            }
        )
        bwell = xtgeo.BlockedWell(
            xpos=460000, ypos=5930000, wname="ImpBW", df=df, mdlogname="M_DEPTH"
        )
        xtgeo.blocked_well_to_osdu(epc, bwell, title="ImpBW", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpBW", object_type="blocked_well")
        assert len(results) >= 1
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.BlockedWell)

    def test_import_triangulated_surface_returns_trisurf(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        trisurf = xtgeo.TriangulatedSurface(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        xtgeo.triangulated_surface_to_osdu(epc, trisurf, title="ImpTri", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="ImpTri", object_type="trisurface")
        assert len(results) >= 1
        imported = xtgeo.import_osdu(epc, results[0])
        assert isinstance(imported, xtgeo.TriangulatedSurface)

    def test_import_raises_valueerror_for_unknown_object_type(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot import"):
            xtgeo.import_osdu(
                str(tmp_path / "dummy.epc"),
                {"uuid": "fake", "type": "UnknownType"},
            )

    def test_import_raises_valueerror_when_uuid_missing(self, tmp_path):
        with pytest.raises(ValueError, match="uuid"):
            xtgeo.import_osdu(str(tmp_path / "dummy.epc"), {"type": "grid"})


class TestApiFormatTable:
    """Test _format_table helper."""

    def test_format_table_returns_placeholder_for_empty_list(self):
        from xtgeo.interfaces.osdu._api import _format_table

        assert _format_table([]) == "(no objects found)"

    def test_format_table_includes_dataspace_column_when_present(self):
        from xtgeo.interfaces.osdu._api import _format_table

        results = [
            {
                "type": "resqml20.IjkGridRepresentation",
                "uuid": "abcdef123456",
                "title": "TestGrid",
                "dataspace": "team/proj",
            }
        ]
        table = _format_table(results)
        assert "TestGrid" in table
        assert "team/proj" in table
        assert "IjkGridRepresentation" in table

    def test_format_table_truncates_long_uuids_with_ellipsis(self):
        from xtgeo.interfaces.osdu._api import _format_table

        results = [
            {"type": "resqml20.Grid2dRepresentation", "uuid": "x" * 20, "title": "Surf"}
        ]
        table = _format_table(results)
        assert "Surf" in table
        assert "…" in table  # UUID truncation


class TestApiQueryOsdu:
    """Test query_osdu (single-dataspace query via EPC)."""

    def test_query_osdu_wraps_search_and_returns_matching_grids(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="QGrid", crs_epsg=23031)

        # query_osdu wraps search_osdu, should work with EPC
        results = xtgeo.query_osdu(epc, name="QGrid", object_type="grid")
        assert len(results) >= 1


class TestApiDeepQueryOsduEpc:
    """Test deep_query_osdu against EPC files.

    EPC provider doesn't support discover.
    """

    def test_deep_query_raises_attribute_error_on_epc_provider(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.ones((3, 3, 2)) * 0.2)
        xtgeo.grid_to_osdu(epc, grid, title="DQGrid", properties=[poro], crs_epsg=23031)

        with pytest.raises(AttributeError):
            xtgeo.deep_query_osdu(epc, depth=0)


class TestApiEdgeCases:
    """Edge cases in _api.py."""

    def test_open_provider_raises_for_non_epc_file_extension(self):
        from xtgeo.interfaces.osdu._api import _open_provider

        with pytest.raises((ValueError, TypeError)):
            _open_provider("/tmp/not_an_epc.txt")

    def test_grid_from_osdu_raises_when_neither_name_nor_uuid_given(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="G", crs_epsg=23031)

        with pytest.raises(ValueError, match="uuid.*name"):
            xtgeo.grid_from_osdu(epc)


# ---------------------------------------------------------------------------
# _properties.py coverage
# ---------------------------------------------------------------------------


class TestPropertyReadWrite:
    """Test read_grid_properties and write_grid_property via EPC."""

    def test_discrete_fipnum_values_preserved_through_epc_roundtrip(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((4, 3, 2))
        fipnum = xtgeo.GridProperty(
            grid,
            name="FIPNUM",
            discrete=True,
            values=np.array([[[1, 2], [3, 1], [2, 3]]] * 4, dtype=np.int32),
        )
        xtgeo.grid_to_osdu(
            epc, grid, title="DProp", properties=[fipnum], crs_epsg=23031
        )

        grid2, props2 = xtgeo.grid_from_osdu(epc, name="DProp")
        fip = next(p for p in props2 if p.name == "FIPNUM")
        assert fip.isdiscrete
        assert np.array_equal(fip.values, fipnum.values)

    def test_three_continuous_properties_all_roundtrip_through_epc(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.full((3, 3, 2), 0.25))
        permx = xtgeo.GridProperty(grid, name="PERMX", values=np.full((3, 3, 2), 100.0))
        swat = xtgeo.GridProperty(grid, name="SWAT", values=np.full((3, 3, 2), 0.7))

        xtgeo.grid_to_osdu(
            epc,
            grid,
            title="MultiP",
            properties=[poro, permx, swat],
            crs_epsg=23031,
        )
        _, props2 = xtgeo.grid_from_osdu(epc, name="MultiP")
        names = sorted(p.name for p in props2)
        assert "PORO" in names
        assert "PERMX" in names
        assert "SWAT" in names

    def test_custom_property_name_preserved_through_epc_roundtrip(self, tmp_path):
        """Property with a non-standard name should roundtrip with original name."""
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        custom = xtgeo.GridProperty(
            grid, name="MY_CUSTOM_PROP", values=np.random.rand(3, 3, 2)
        )
        xtgeo.grid_to_osdu(
            epc, grid, title="CustP", properties=[custom], crs_epsg=23031
        )

        _, props2 = xtgeo.grid_from_osdu(epc, name="CustP")
        assert any(p.name == "MY_CUSTOM_PROP" for p in props2)

    def test_write_then_read_continuous_property_via_provider_api(self, tmp_path):
        """Exercise read_grid_properties/write_grid_property directly."""
        from xtgeo.interfaces.osdu._properties import (
            read_grid_properties,
            write_grid_property,
        )

        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 4, 2))
        xtgeo.grid_to_osdu(epc, grid, title="PropDirect", crs_epsg=23031)

        # Get the grid UUID
        results = xtgeo.search_osdu(epc, name="PropDirect", object_type="grid")
        grid_uuid = results[0]["uuid"]

        # Write a property directly
        provider = EpcFileProvider(epc, mode="a")
        provider.open()
        try:
            poro = xtgeo.GridProperty(
                grid, name="PORO", values=np.full((3, 4, 2), 0.15)
            )
            prop_uuid = write_grid_property(provider, poro, grid_uuid)
            assert prop_uuid  # Should return a UUID
        finally:
            provider.close()

        # Read it back directly
        provider = EpcFileProvider(epc, mode="r")
        provider.open()
        try:
            props = read_grid_properties(provider, grid_uuid, ni=3, nj=4, nk=2)
            assert len(props) >= 1
            poro_read = next((p for p in props if p.name == "PORO"), None)
            assert poro_read is not None
            assert np.allclose(poro_read.values, 0.15)
        finally:
            provider.close()

    def test_write_then_read_discrete_property_via_provider_api(self, tmp_path):
        """Exercise write_grid_property with discrete property."""
        from xtgeo.interfaces.osdu._properties import (
            read_grid_properties,
            write_grid_property,
        )

        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="DiscDirect", crs_epsg=23031)

        results = xtgeo.search_osdu(epc, name="DiscDirect", object_type="grid")
        grid_uuid = results[0]["uuid"]

        provider = EpcFileProvider(epc, mode="a")
        provider.open()
        try:
            satnum = xtgeo.GridProperty(
                grid,
                name="SATNUM",
                discrete=True,
                values=np.ones((3, 3, 2), dtype=np.int32) * 2,
            )
            write_grid_property(provider, satnum, grid_uuid)
        finally:
            provider.close()

        provider = EpcFileProvider(epc, mode="r")
        provider.open()
        try:
            props = read_grid_properties(provider, grid_uuid, ni=3, nj=3, nk=2)
            sat = next((p for p in props if p.name == "SATNUM"), None)
            assert sat is not None
            assert sat.isdiscrete
        finally:
            provider.close()


# ---------------------------------------------------------------------------
# _epc_provider.py coverage — edge cases
# ---------------------------------------------------------------------------


class TestEpcProviderEdgeCases:
    def test_list_objects_filters_by_resqml_type(self, tmp_path):
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        xtgeo.grid_to_osdu(epc, grid, title="LO_Grid", crs_epsg=23031)

        provider = EpcFileProvider(epc, mode="r")
        provider.open()
        try:
            all_objs = provider.list_objects()
            grid_objs = provider.list_objects("IjkGrid")
            crs_objs = provider.list_objects("LocalDepth3dCrs")

            assert len(grid_objs) >= 1
            assert len(crs_objs) >= 1
            assert len(all_objs) >= len(grid_objs) + len(crs_objs)
        finally:
            provider.close()

    def test_open_nonexistent_epc_file_raises(self, tmp_path):
        epc = str(tmp_path / "nonexistent.epc")
        provider = EpcFileProvider(epc, mode="r")
        with pytest.raises(Exception):
            provider.open()

    def test_rotated_surface_geometry_preserved_through_epc(self, tmp_path):
        """Exercise EPC surface write path with put_grid2d/get_grid2d_geometry."""
        epc = str(tmp_path / "surf.epc")
        surf = xtgeo.RegularSurface(
            ncol=5,
            nrow=8,
            xinc=50,
            yinc=25,
            rotation=15.0,
            values=np.random.rand(5, 8),
        )
        xtgeo.surface_to_osdu(epc, surf, title="MySurf", crs_epsg=23031)

        surf2 = xtgeo.surface_from_osdu(epc, name="MySurf")
        assert surf2.ncol == 5
        assert surf2.nrow == 8
        assert abs(surf2.xinc - 50.0) < 1e-10
        assert abs(surf2.rotation - 15.0) < 0.01

    def test_three_points_roundtrip_through_epc(self, tmp_path):
        """Exercise EPC pointset write path."""
        epc = str(tmp_path / "pts.epc")
        pts = xtgeo.Points(
            values=np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])
        )
        xtgeo.points_to_osdu(epc, pts, title="MyPts", crs_epsg=23031)

        pts2 = xtgeo.points_from_osdu(epc, name="MyPts")
        assert len(pts2.get_dataframe()) == 3


# ---------------------------------------------------------------------------
# _polyline.py coverage — closed polygon detection
# ---------------------------------------------------------------------------


class TestPolylineClosedDetection:
    """Test that closed polygons (first == last point) are handled."""

    def test_closed_polygon_with_repeated_endpoint_roundtrips(self, tmp_path):
        import pandas as pd

        epc = str(tmp_path / "poly.epc")
        # Closed polygon: last point == first point
        df = pd.DataFrame(
            {
                "X_UTME": [0, 10, 10, 0, 0],
                "Y_UTMN": [0, 0, 10, 10, 0],
                "Z_TVDSS": [0, 0, 0, 0, 0],
                "POLY_ID": [0, 0, 0, 0, 0],
            }
        )
        polys = xtgeo.Polygons(df)
        xtgeo.polygons_to_osdu(epc, polys, title="ClosedPoly", crs_epsg=23031)

        polys2 = xtgeo.polygons_from_osdu(epc, name="ClosedPoly")
        df2 = polys2.get_dataframe()
        assert len(df2) >= 4  # May or may not re-close


class TestPolylineMultiPoly:
    """Test multiple polylines with POLY_ID column."""

    def test_two_polylines_with_different_poly_ids_roundtrip(self, tmp_path):
        import pandas as pd

        epc = str(tmp_path / "mpoly.epc")
        df = pd.DataFrame(
            {
                "X_UTME": [0, 1, 2, 10, 11, 12, 13],
                "Y_UTMN": [0, 1, 0, 10, 11, 10, 12],
                "Z_TVDSS": [0, 0, 0, 5, 5, 5, 5],
                "POLY_ID": [0, 0, 0, 1, 1, 1, 1],
            }
        )
        polys = xtgeo.Polygons(df)
        xtgeo.polygons_to_osdu(epc, polys, title="MultiPoly", crs_epsg=23031)

        polys2 = xtgeo.polygons_from_osdu(epc, name="MultiPoly")
        df2 = polys2.get_dataframe()
        # Should have points from both polylines
        assert len(df2) >= 7


# ---------------------------------------------------------------------------
# _ijk_grid.py and _grid2d.py edge cases
# ---------------------------------------------------------------------------


class TestIjkGridEdgeCases:
    def test_all_inactive_cells_preserved_through_epc(self, tmp_path):
        """Grid where all cells are inactive."""
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        actnum = xtgeo.GridProperty(
            grid,
            name="ACTNUM",
            discrete=True,
            values=np.zeros((3, 3, 2), dtype=np.int32),
        )
        grid.set_actnum(actnum)

        xtgeo.grid_to_osdu(epc, grid, title="AllInactive", crs_epsg=23031)
        grid2, _ = xtgeo.grid_from_osdu(epc, name="AllInactive")
        assert np.all(grid2.get_actnum().values == 0)

    def test_grid_import_with_load_properties_false_returns_empty_list(self, tmp_path):
        """Requesting geometry only."""
        epc = str(tmp_path / "test.epc")
        grid = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(grid, name="PORO", values=np.ones((3, 3, 2)) * 0.2)
        xtgeo.grid_to_osdu(epc, grid, title="NoProp", properties=[poro], crs_epsg=23031)

        grid2, props2 = xtgeo.grid_from_osdu(epc, name="NoProp", load_properties=False)
        assert grid2.ncol == 3
        assert props2 == []


class TestGrid2dEdgeCases:
    def test_single_cell_surface_roundtrips_value_through_epc(self, tmp_path):
        """1x1 surface."""
        epc = str(tmp_path / "test.epc")
        surf = xtgeo.RegularSurface(
            ncol=1, nrow=1, xinc=100, yinc=100, values=np.array([[42.0]])
        )
        xtgeo.surface_to_osdu(epc, surf, title="Tiny", crs_epsg=23031)
        surf2 = xtgeo.surface_from_osdu(epc, name="Tiny")
        assert surf2.ncol == 1
        assert abs(surf2.values[0, 0] - 42.0) < 1e-10
