"""Check the warning contract of public APIs that report deprecations."""

from __future__ import annotations

import warnings
from functools import partial
from io import BytesIO

import pandas as pd
import pytest

import xtgeo
from xtgeo.common import _telemetry as telemetry


@pytest.fixture
def well_data():
    """Supply a small well with one log."""
    return pd.DataFrame(
        {"X_UTME": [0.0], "Y_UTMN": [0.0], "Z_TVDSS": [0.0], "GR": [1.0]},
    )


@pytest.fixture
def api_objects(well_data):
    """Supply small objects for the deprecated methods."""
    grid = xtgeo.create_box_grid((2, 2, 2))
    return {
        "cube": xtgeo.Cube(2, 2, 2, 1, 1, 1),
        "grid": grid,
        "property": xtgeo.GridProperty(grid),
        "surface": xtgeo.RegularSurface(2, 2, 1, 1),
        "well": xtgeo.Well(df=well_data.copy()),
        "blockedwell": xtgeo.BlockedWell(df=well_data.copy()),
        "points": xtgeo.Points([(0.0, 0.0, 0.0)]),
        "polygons": xtgeo.Polygons([(0.0, 0.0, 0.0, 0)]),
    }


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("cube_from_roxar", ("Cube",)),
        ("grid_from_roxar", ("Grid",)),
        ("gridproperty_from_roxar", ("Grid", "Pressure")),
        ("surface_from_roxar", ("Surface", "Category")),
        ("well_from_roxar", ("Well",)),
        ("blockedwell_from_roxar", ("Grid", "BlockedWell", "Well")),
        ("blockedwells_from_roxar", ("Grid", "BlockedWell")),
        ("points_from_roxar", ("Points", "Category")),
        ("polygons_from_roxar", ("Polygons", "Category")),
    ],
)
def test_legacy_rms_readers_warn_before_opening_a_project(name, args, monkeypatch):
    """Preserve both warning-as-error behavior and the corresponding event."""
    events = []

    def capture(frame, message, category, identifier):
        """Save the warning payload at the telemetry boundary."""
        events.append((frame.f_code.co_name, message, category, identifier))

    class Observer:
        """Provide the deprecation callback without installing monitoring tools."""

        on_deprecation = staticmethod(capture)

    monkeypatch.setattr(telemetry, "_telemetry", Observer())
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(FutureWarning, match=name) as caught:
            getattr(xtgeo, name)("example.rms", *args)
    assert events == [(name, str(caught.value), FutureWarning, None)]


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("cube", ("Cube",)),
        ("grid", ("Grid",)),
        ("property", ("Grid", "Pressure")),
        ("surface", ("Surface", "Category")),
        ("well", ("Well",)),
        ("blockedwell", ("Grid", "BlockedWell", "Well")),
        ("points", ("Points", "Category")),
        ("polygons", ("Polygons", "Category")),
    ],
)
def test_legacy_rms_writers_warn_before_opening_a_project(api_objects, kind, args):
    """Stop deprecated exports before opening RMS when warnings are errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(FutureWarning, match="to_roxar"):
            api_objects[kind].to_roxar("example.rms", *args)


@pytest.mark.parametrize("kind", ["well", "points", "polygons"])
def test_dataframe_warning_preserves_dataframe_access(api_objects, kind):
    """Keep dataframe access and assignment available during deprecation."""
    obj = api_objects[kind]
    with pytest.warns(DeprecationWarning, match="Direct access"):
        data = obj.dataframe
    with pytest.warns(DeprecationWarning, match="Direct access"):
        obj.dataframe = data.copy()
    pd.testing.assert_frame_equal(obj.get_dataframe(), data)


@pytest.mark.parametrize(
    ("operation", "category", "message"),
    [
        (
            lambda: xtgeo.XTGeoDialog.functionlogger(__name__),
            FutureWarning,
            "functionlogger",
        ),
        (
            lambda: xtgeo.RoxUtils("example.rms"),
            FutureWarning,
            "RoxUtils is deprecated",
        ),
        (
            lambda: xtgeo.grid_from_resinsight(None, case_name="Case"),
            FutureWarning,
            "'case_name' argument is deprecated",
        ),
    ],
)
def test_deprecated_entry_points_warn_before_optional_dependencies(
    operation,
    category,
    message,
):
    """Keep warnings available without a live RMS or ResInsight installation."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        with pytest.raises(category, match=message):
            operation()


def test_dialog_deprecation_and_disabled_warning(capsys):
    """Show a deprecation at the caller and respect the runtime warning setting."""
    dialog = xtgeo.XTGeoDialog()
    dialog.show_runtimewarnings(False)
    dialog.warning("disabled")
    assert "disabled" not in capsys.readouterr().out
    with pytest.warns(DeprecationWarning, match="old argument") as caught:
        dialog.warndeprecated("old argument")
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    ("operation", "category", "message"),
    [
        ("grid transmissibilities", FutureWarning, "get_transmissibilities"),
        ("surface mode", FutureWarning, "'mode' argument"),
        ("surface masked", FutureWarning, "'masked' parameter"),
        ("surface export engine", DeprecationWarning, "'engine' keyword"),
        ("surface dataframe", DeprecationWarning, "dataframe"),
        ("cube export engine", UserWarning, "'engine' value"),
    ],
)
def test_deprecated_options_and_methods(
    api_objects, tmp_path, operation, category, message
):
    """Keep the warning-as-error behavior for deprecated grid and surface options."""
    grid = api_objects["grid"]
    surface = api_objects["surface"]
    operations = {
        "grid transmissibilities": partial(grid.get_transmissibilities, 1.0, 1.0, 1.0),
        "surface mode": partial(xtgeo.surface_from_grid3d, grid, mode="depth"),
        "surface masked": partial(xtgeo.RegularSurface, 2, 2, 1, 1, masked=False),
        "surface export engine": partial(surface.to_file, BytesIO(), engine="xtgeo"),
        "surface dataframe": surface.dataframe,
        "cube export engine": partial(
            api_objects["cube"].to_file, tmp_path / "cube.segy", engine="xtgeo"
        ),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        with pytest.raises(category, match=message):
            operations[operation]()


def test_slice_cube_window_warns_about_method_and_algorithm(api_objects, monkeypatch):
    """Warn about both the legacy method and its legacy algorithm."""
    from xtgeo.surface import _regsurf_cube_window

    monkeypatch.setattr(_regsurf_cube_window, "slice_cube_window", lambda *a, **kw: 0)
    with pytest.warns(FutureWarning) as caught:
        result = api_objects["surface"].slice_cube_window(
            api_objects["cube"], algorithm=2
        )
    assert result == 0
    assert len(caught) == 2
    assert "This method is deprecated" in str(caught[0].message)
    assert "Other algorithms than no. 3" in str(caught[1].message)


def test_well_to_hdf_preserves_export_after_warning(api_objects, tmp_path):
    """Write a readable HDF file through the deprecated method."""
    path = tmp_path / "well.hdf"
    well = api_objects["well"]
    with pytest.warns(FutureWarning, match="to_hdf"):
        well.to_hdf(path)
    pd.testing.assert_frame_equal(
        xtgeo.well_from_file(path, fformat="hdf5").get_dataframe(),
        well.get_dataframe(),
    )


def test_legacy_polygon_operation_preserves_point_values():
    """Warn about the old polygon method and change only the enclosed point."""
    points = xtgeo.Points([(1.0, 1.0, 2.0), (3.0, 3.0, 2.0)])
    polygons = xtgeo.Polygons(
        [
            (0.0, 0.0, 0.0, 0),
            (2.0, 0.0, 0.0, 0),
            (2.0, 2.0, 0.0, 0),
            (0.0, 2.0, 0.0, 0),
            (0.0, 0.0, 0.0, 0),
        ]
    )
    with pytest.warns(FutureWarning, match=r"add_inside\(\)"):
        points.add_inside(polygons, 1.0)
    assert points.get_dataframe()[points.zname].tolist() == [3.0, 2.0]
