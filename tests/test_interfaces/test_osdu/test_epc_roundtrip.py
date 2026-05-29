"""Comprehensive EPC round-trip tests for the OSDU/RESQML 2.0.1 interface.

Tests geometry fidelity (including faults, split nodes), property types
(discrete/continuous), rotation preservation, CRS mapping, and all supported
object types: IJK Grid, Grid2D (Surface), PointSet, and PolylineSet.
"""

import numpy as np
import pandas as pd
import pytest

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml
from xtgeo.interfaces.osdu._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
from xtgeo.interfaces.osdu._polyline import (
    polylineset_to_xtgeo,
    xtgeo_polygons_to_resqml,
)


@pytest.fixture
def epc_path(tmp_path):
    """Return a temporary EPC file path."""
    return str(tmp_path / "test.epc")


class TestIJKGridRoundTrip:
    """Tests for IJK Grid geometry and property round-trips via EPC."""

    def test_box_grid_geometry(self, epc_path):
        g = xtgeo.create_box_grid(
            (4, 5, 3), origin=(460000, 5930000, 1000), increment=(50, 50, 10)
        )
        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(p, g, title="BoxGrid", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        g2, _ = ijk_grid_to_xtgeo(p2, uuids["BoxGrid"])
        p2.close()

        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)
        assert np.array_equal(g._actnumsv, g2._actnumsv)

    def test_faulted_grid_geometry(self, epc_path):
        g = xtgeo.create_box_grid(
            (3, 3, 2), origin=(460000, 5930000, 1000), increment=(50, 50, 10)
        )
        # Add fault throw
        z = g._zcornsv.copy()
        z[2:, :, :, :] += 5.0
        g._zcornsv = z

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(p, g, title="FaultGrid", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        g2, _ = ijk_grid_to_xtgeo(p2, uuids["FaultGrid"])
        p2.close()

        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)

    def test_inactive_cells(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), origin=(0, 0, 0), increment=(1, 1, 1))
        act = g._actnumsv.copy()
        act[0, 0, 0] = 0
        act[2, 2, 1] = 0
        act[1, 1, 0] = 0
        g._actnumsv = act

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(p, g, title="InactiveGrid")
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        g2, _ = ijk_grid_to_xtgeo(p2, uuids["InactiveGrid"])
        p2.close()

        assert np.array_equal(g._actnumsv, g2._actnumsv)

    def test_continuous_property(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2))
        poro = xtgeo.GridProperty(
            g, name="PORO", values=np.linspace(0.1, 0.3, 18).reshape(3, 3, 2)
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(p, g, title="Grid", properties=[poro])
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        _, props = ijk_grid_to_xtgeo(p2, uuids["Grid"], load_properties=True)
        p2.close()

        assert len(props) == 1
        prop = props[0]
        assert prop.name == "PORO"
        assert not prop.isdiscrete
        assert np.allclose(prop.values, poro.values, atol=1e-6)

    def test_discrete_property(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2))
        fipnum = xtgeo.GridProperty(
            g,
            name="FIPNUM",
            values=np.array([1] * 9 + [2] * 9, dtype=np.int32).reshape(3, 3, 2),
            discrete=True,
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(p, g, title="Grid", properties=[fipnum])
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        _, props = ijk_grid_to_xtgeo(p2, uuids["Grid"], load_properties=True)
        p2.close()

        assert len(props) == 1
        prop = props[0]
        assert prop.name == "FIPNUM"
        assert prop.isdiscrete
        assert np.array_equal(prop.values, fipnum.values)

    def test_multiple_properties(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2))
        np.random.seed(42)
        poro = xtgeo.GridProperty(
            g, name="PORO", values=np.random.rand(18).reshape(3, 3, 2)
        )
        permx = xtgeo.GridProperty(
            g, name="PERMX", values=np.random.rand(18).reshape(3, 3, 2) * 500
        )
        fipnum = xtgeo.GridProperty(
            g,
            name="FIPNUM",
            values=np.array([1] * 9 + [2] * 9, dtype=np.int32).reshape(3, 3, 2),
            discrete=True,
        )
        satnum = xtgeo.GridProperty(
            g,
            name="SATNUM",
            values=np.array([1, 1, 2, 2] * 4 + [3, 3], dtype=np.int32).reshape(3, 3, 2),
            discrete=True,
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_grid_to_resqml(
            p, g, title="Grid", properties=[poro, permx, fipnum, satnum]
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        _, props = ijk_grid_to_xtgeo(p2, uuids["Grid"], load_properties=True)
        p2.close()

        assert len(props) == 4
        prop_dict = {p.name: p for p in props}
        assert np.allclose(prop_dict["PORO"].values, poro.values, atol=1e-6)
        assert np.allclose(prop_dict["PERMX"].values, permx.values, atol=1e-6)
        assert np.array_equal(prop_dict["FIPNUM"].values, fipnum.values)
        assert np.array_equal(prop_dict["SATNUM"].values, satnum.values)
        assert prop_dict["FIPNUM"].isdiscrete
        assert prop_dict["SATNUM"].isdiscrete
        assert not prop_dict["PORO"].isdiscrete
        assert not prop_dict["PERMX"].isdiscrete


class TestSurfaceRoundTrip:
    """Tests for Grid2D (Surface) round-trips via EPC."""

    def test_basic_surface(self, epc_path):
        np.random.seed(42)
        s = xtgeo.RegularSurface(
            ncol=10,
            nrow=12,
            xinc=25.0,
            yinc=30.0,
            xori=460000,
            yori=5930000,
            rotation=0.0,
            values=np.random.rand(10, 12) * 100,
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_surface_to_resqml(p, s, title="TopSurf", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        s2 = grid2d_to_xtgeo(p2, uuids["TopSurf"])
        p2.close()

        assert s2.ncol == s.ncol
        assert s2.nrow == s.nrow
        assert abs(s2.xinc - s.xinc) < 0.01
        assert abs(s2.yinc - s.yinc) < 0.01
        assert abs(s2.xori - s.xori) < 0.01
        assert abs(s2.yori - s.yori) < 0.01
        assert np.allclose(s2.values.filled(np.nan), s.values.filled(np.nan), atol=1e-6)

    def test_surface_rotation(self, epc_path):
        np.random.seed(42)
        s = xtgeo.RegularSurface(
            ncol=5,
            nrow=7,
            xinc=25.0,
            yinc=30.0,
            xori=460000,
            yori=5930000,
            rotation=33.7,
            values=np.random.rand(5, 7) * 100,
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_surface_to_resqml(p, s, title="RotSurf", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        s2 = grid2d_to_xtgeo(p2, uuids["RotSurf"])
        p2.close()

        assert abs(s2.rotation - s.rotation) < 0.1

    def test_surface_nan_values(self, epc_path):
        np.random.seed(42)
        vals = np.random.rand(5, 7) * 100
        vals[0, 0] = np.nan
        vals[2, 3] = np.nan
        vals[4, 6] = np.nan
        s = xtgeo.RegularSurface(
            ncol=5,
            nrow=7,
            xinc=25.0,
            yinc=30.0,
            xori=460000,
            yori=5930000,
            rotation=0.0,
            values=vals,
        )

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_surface_to_resqml(p, s, title="NanSurf")
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        s2 = grid2d_to_xtgeo(p2, uuids["NanSurf"])
        p2.close()

        orig = s.values.filled(np.nan)
        read = s2.values.filled(np.nan)
        assert np.array_equal(np.isnan(orig), np.isnan(read))
        valid = ~np.isnan(orig)
        assert np.allclose(orig[valid], read[valid], atol=1e-6)


class TestPointSetRoundTrip:
    """Tests for PointSet round-trips via EPC."""

    def test_basic_points(self, epc_path):
        pts_data = np.array(
            [
                [460000, 5930000, 1500],
                [460100, 5930100, 1600],
                [460200, 5930200, 1700],
            ],
            dtype=np.float64,
        )
        pts = xtgeo.Points(pts_data)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_points_to_resqml(p, pts, title="Tops", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        pts2 = pointset_to_xtgeo(p2, uuids["Tops"])
        p2.close()

        assert np.allclose(pts2.get_dataframe().values[:, :3], pts_data, atol=1e-6)


class TestPolylineSetRoundTrip:
    """Tests for PolylineSet (Polygons) round-trips via EPC."""

    def test_multiple_polygons(self, epc_path):
        poly_df = pd.DataFrame(
            {
                "X_UTME": [0, 1, 1, 0, 0, 2, 3, 3, 2, 2],
                "Y_UTMN": [0, 0, 1, 1, 0, 2, 2, 3, 3, 2],
                "Z_TVDSS": [100] * 5 + [200] * 5,
                "POLY_ID": [0] * 5 + [1] * 5,
            }
        )
        polys = xtgeo.Polygons(poly_df)

        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        uuids = xtgeo_polygons_to_resqml(p, polys, title="FaultPoly", crs_epsg=23031)
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        polys2 = polylineset_to_xtgeo(p2, uuids["FaultPoly"])
        p2.close()

        df1 = polys.get_dataframe()
        df2 = polys2.get_dataframe()
        assert len(df1) == len(df2)
        assert np.allclose(
            df1[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            df2[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values,
            atol=1e-6,
        )
        assert np.array_equal(df1["POLY_ID"].values, df2["POLY_ID"].values)


class TestCRSRoundTrip:
    """Tests for CRS mapping preservation via EPC."""

    def test_crs_epsg_preserved(self, epc_path):
        p = EpcFileProvider(epc_path, mode="w")
        p.open()
        import uuid

        crs_uuid = str(uuid.uuid4())
        p.put_crs(
            uuid=crs_uuid,
            title="ED50-UTM31",
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            areal_rotation=0.0,
            z_increasing_downward=True,
            projected_crs_epsg=23031,
        )
        p.close()

        p2 = EpcFileProvider(epc_path, mode="r")
        p2.open()
        crs_info = p2.get_crs(crs_uuid)
        p2.close()

        assert crs_info["projected_crs_epsg"] == 23031
        assert crs_info["z_increasing_downward"] is True
        assert crs_info["areal_rotation"] == 0.0
