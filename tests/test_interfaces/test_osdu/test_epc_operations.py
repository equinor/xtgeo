"""Post-roundtrip grid3d operations tests.

Verifies that grids imported from EPC/RESQML work correctly with downstream
xtgeo operations (bulk volume, dx/dy/dz, crop, get_dataframe, etc.).
This catches subtle format-induced differences in internal array layout or
precision that raw array comparison would miss.
"""

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import EpcFileProvider
from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml


@pytest.fixture
def epc_path(tmp_path):
    return str(tmp_path / "test.epc")


def _roundtrip_grid(epc_path, grid, title="Grid", properties=None, **kw):
    """Write grid to EPC, read it back."""
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


class TestBulkVolume:
    """get_bulk_volume() on roundtripped grids."""

    def test_box_grid_bulk_volume(self, epc_path):
        g = xtgeo.create_box_grid((4, 5, 3), increment=(50, 50, 10))
        g2, _ = _roundtrip_grid(epc_path, g)

        bv1 = g.get_bulk_volume()
        bv2 = g2.get_bulk_volume()

        assert np.allclose(bv1.values, bv2.values, rtol=1e-4)

    def test_rotated_grid_bulk_volume(self, epc_path):
        g = xtgeo.create_box_grid((3, 4, 2), increment=(50, 50, 10), rotation=30.0)
        g2, _ = _roundtrip_grid(epc_path, g, title="RotBV")

        bv1 = g.get_bulk_volume()
        bv2 = g2.get_bulk_volume()

        assert np.allclose(bv1.values, bv2.values, rtol=1e-4)

    def test_inactive_cells_bulk_volume(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), increment=(10, 10, 5))
        act = g._actnumsv.copy()
        act[0, 0, 0] = 0
        act[2, 2, 1] = 0
        g._actnumsv = act

        g2, _ = _roundtrip_grid(epc_path, g, title="InactBV")

        bv1 = g.get_bulk_volume()
        bv2 = g2.get_bulk_volume()

        assert np.allclose(bv1.values, bv2.values, rtol=1e-4)


class TestCellDimensions:
    """get_dx/dy/dz on roundtripped grids."""

    def test_dx_dy_dz_box(self, epc_path):
        g = xtgeo.create_box_grid((3, 4, 2), increment=(25, 30, 10))
        g2, _ = _roundtrip_grid(epc_path, g, title="DxDyDz")

        for method in ("get_dx", "get_dy", "get_dz"):
            v1 = getattr(g, method)()
            v2 = getattr(g2, method)()
            assert np.allclose(v1.values, v2.values, rtol=1e-4), f"{method} mismatch"

    def test_dx_dy_dz_rotated(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), increment=(50, 50, 10), rotation=45.0)
        g2, _ = _roundtrip_grid(epc_path, g, title="DxDyDzRot")

        for method in ("get_dx", "get_dy", "get_dz"):
            v1 = getattr(g, method)()
            v2 = getattr(g2, method)()
            assert np.allclose(v1.values, v2.values, rtol=1e-4), f"{method} mismatch"


class TestGetXYZ:
    """get_xyz() and get_xyz_cell_corners() on roundtripped grids."""

    def test_get_xyz(self, epc_path):
        g = xtgeo.create_box_grid((3, 4, 2), increment=(50, 50, 10))
        g2, _ = _roundtrip_grid(epc_path, g, title="XYZ")

        x1, y1, z1 = g.get_xyz()
        x2, y2, z2 = g2.get_xyz()

        assert np.allclose(x1.values, x2.values, atol=1e-4)
        assert np.allclose(y1.values, y2.values, atol=1e-4)
        assert np.allclose(z1.values, z2.values, atol=1e-4)

    def test_get_xyz_cell_corners(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), increment=(10, 10, 5))
        g2, _ = _roundtrip_grid(epc_path, g, title="Corners")

        c1 = g.get_xyz_cell_corners(ijk=(1, 1, 1))
        c2 = g2.get_xyz_cell_corners(ijk=(1, 1, 1))

        assert c1 is not None and c2 is not None
        assert np.allclose(c1, c2, atol=1e-4)

    def test_get_xyz_cell_corners_all_cells(self, epc_path):
        """Verify all active cell corners match."""
        g = xtgeo.create_box_grid((3, 3, 2), increment=(10, 10, 5))
        g2, _ = _roundtrip_grid(epc_path, g, title="AllCorners")

        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(1, 3):
                    c1 = g.get_xyz_cell_corners(ijk=(i, j, k))
                    c2 = g2.get_xyz_cell_corners(ijk=(i, j, k))
                    assert c1 is not None and c2 is not None
                    assert np.allclose(c1, c2, atol=1e-4), f"Mismatch at ({i},{j},{k})"


class TestGetDataframe:
    """get_dataframe() on roundtripped grids."""

    def test_dataframe_values(self, epc_path):
        g = xtgeo.create_box_grid(
            (3, 4, 2), origin=(460000, 5930000, 1000), increment=(50, 50, 10)
        )
        g2, _ = _roundtrip_grid(epc_path, g, title="DF")

        df1 = g.get_dataframe()
        df2 = g2.get_dataframe()

        assert len(df1) == len(df2)
        for col in ("IX", "JY", "KZ"):
            assert np.array_equal(df1[col].values, df2[col].values), f"{col} mismatch"
        for col in ("X_UTME", "Y_UTMN", "Z_TVDSS"):
            assert np.allclose(df1[col].values, df2[col].values, atol=0.1), (
                f"{col} mismatch"
            )

    def test_dataframe_with_inactive(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), increment=(10, 10, 5))
        act = g._actnumsv.copy()
        act[0, 0, 0] = 0
        act[1, 1, 1] = 0
        g._actnumsv = act

        g2, _ = _roundtrip_grid(epc_path, g, title="DFInact")

        df1 = g.get_dataframe(activeonly=True)
        df2 = g2.get_dataframe(activeonly=True)

        assert len(df1) == len(df2)


class TestGetGeometrics:
    """get_geometrics() on roundtripped grids."""

    def test_geometrics_box(self, epc_path):
        g = xtgeo.create_box_grid(
            (4, 5, 3), origin=(460000, 5930000, 1000), increment=(50, 50, 10)
        )
        g2, _ = _roundtrip_grid(epc_path, g, title="Geom")

        geo1 = g.get_geometrics(return_dict=True)
        geo2 = g2.get_geometrics(return_dict=True)

        for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            assert abs(geo1[key] - geo2[key]) < 0.1, f"{key} mismatch"

    def test_geometrics_rotated(self, epc_path):
        g = xtgeo.create_box_grid(
            (3, 3, 2),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
            rotation=33.0,
        )
        g2, _ = _roundtrip_grid(epc_path, g, title="GeomRot")

        geo1 = g.get_geometrics(return_dict=True)
        geo2 = g2.get_geometrics(return_dict=True)

        for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            assert abs(geo1[key] - geo2[key]) < 0.1, f"{key} mismatch"


class TestCrop:
    """crop() on roundtripped grids."""

    def test_crop_basic(self, epc_path):
        g = xtgeo.create_box_grid((5, 6, 4), increment=(10, 10, 5))
        g2, _ = _roundtrip_grid(epc_path, g, title="Crop")

        g2.crop((2, 4), (2, 5), (1, 3))

        assert g2.ncol == 3
        assert g2.nrow == 4
        assert g2.nlay == 3

    def test_crop_with_property(self, epc_path):
        g = xtgeo.create_box_grid((5, 6, 4), increment=(10, 10, 5))
        rng = np.random.RandomState(42)
        poro = xtgeo.GridProperty(g, name="PORO", values=rng.rand(5, 6, 4))

        g2, props = _roundtrip_grid(epc_path, g, title="CropProp", properties=[poro])

        prop2 = props[0]
        g2.crop((2, 4), (2, 5), (1, 3), props=[prop2])

        assert g2.ncol == 3
        assert g2.nrow == 4
        assert g2.nlay == 3
        assert prop2.values.shape == (3, 4, 3)
        # Cropped property values should match the original sub-region
        expected = poro.values[1:4, 1:5, 0:3]
        assert np.allclose(prop2.values, expected, atol=1e-6)


class TestGridQuality:
    """get_gridquality_properties() on roundtripped grids."""

    def test_quality_box_grid(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 2), increment=(50, 50, 10))
        g2, _ = _roundtrip_grid(epc_path, g, title="Quality")

        qp1 = g.get_gridquality_properties()
        qp2 = g2.get_gridquality_properties()

        for propname in ("minangle_topbase", "maxangle_topbase", "collapsed"):
            p1 = qp1.get_prop_by_name(propname)
            p2 = qp2.get_prop_by_name(propname)
            assert np.allclose(p1.values.filled(0), p2.values.filled(0), atol=1e-3), (
                f"{propname} mismatch"
            )


class TestReduceToOneLayer:
    """reduce_to_one_layer() on roundtripped grids."""

    def test_reduce_box(self, epc_path):
        g = xtgeo.create_box_grid((3, 3, 5), increment=(10, 10, 5))
        g2, _ = _roundtrip_grid(epc_path, g, title="Reduce")

        g.reduce_to_one_layer()
        g2.reduce_to_one_layer()

        assert g.nlay == 1
        assert g2.nlay == 1
        assert np.allclose(g._coordsv, g2._coordsv, atol=1e-6)
        assert np.allclose(g._zcornsv, g2._zcornsv, atol=1e-6)


class TestSurfaceFromGrid:
    """xtgeo.surface_from_grid3d() on roundtripped grids."""

    def test_top_surface_from_box_grid(self, epc_path):
        g = xtgeo.create_box_grid(
            (5, 6, 3),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
        )
        g2, _ = _roundtrip_grid(epc_path, g, title="SurfGrid")

        s1 = xtgeo.surface_from_grid3d(g)
        s2 = xtgeo.surface_from_grid3d(g2)

        assert s1.ncol == s2.ncol
        assert s1.nrow == s2.nrow
        valid1 = ~np.isnan(s1.values.filled(np.nan))
        valid2 = ~np.isnan(s2.values.filled(np.nan))
        assert np.array_equal(valid1, valid2)
        assert np.allclose(
            s1.values.filled(np.nan)[valid1],
            s2.values.filled(np.nan)[valid2],
            atol=0.5,
        )

    def test_base_surface_from_box_grid(self, epc_path):
        g = xtgeo.create_box_grid(
            (5, 6, 3),
            origin=(460000, 5930000, 1000),
            increment=(50, 50, 10),
        )
        g2, _ = _roundtrip_grid(epc_path, g, title="BaseSurf")

        s1 = xtgeo.surface_from_grid3d(g, where="base")
        s2 = xtgeo.surface_from_grid3d(g2, where="base")

        valid1 = ~np.isnan(s1.values.filled(np.nan))
        valid2 = ~np.isnan(s2.values.filled(np.nan))
        assert np.allclose(
            s1.values.filled(np.nan)[valid1],
            s2.values.filled(np.nan)[valid2],
            atol=0.5,
        )
