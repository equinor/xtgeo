"""Exercise XTGeo telemetry through the optional isolated site handler."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from textwrap import dedent
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_SITE_PLUGIN = dedent(
    '''\
    import atexit
    import json
    import logging
    import os
    from pathlib import Path

    folder = Path(os.environ["XTGEO_TEST_TELEMETRY_DIR"])

    class TestHandler(logging.Handler):
        """Simulate an isolated handler whose provider shuts down via atexit."""

        def __init__(self):
            """Store records until the next flush and register provider shutdown."""
            super().__init__()
            self.records = []
            self.provider_closed = False
            atexit.register(self._shutdown_provider)

        def emit(self, record):
            """Stage an event unless the isolated provider already shut down."""
            if self.provider_closed:
                raise RuntimeError("isolated provider shut down before XTGeo flushed")
            self.records.append(record.__dict__)

        def force_flush(self):
            """Write staged records to a local file."""
            if self.provider_closed:
                raise RuntimeError("isolated provider shut down before XTGeo flushed")
            with (folder / "records.jsonl").open("a", encoding="utf-8") as output:
                for record in self.records:
                    output.write(json.dumps(record) + "\\n")
            self.records.clear()

        def _shutdown_provider(self):
            """Simulate OpenTelemetry LoggerProvider's own atexit shutdown."""
            self.provider_closed = True
            with (folder / "provider_shutdown").open("a", encoding="utf-8") as output:
                output.write("shutdown\\n")

    def create_handler():
        """Return the handler for the requested plugin mode."""
        mode = os.environ["XTGEO_TEST_SITE_PLUGIN"]
        if mode == "broken":
            raise ImportError("fmu-logging failed to load")
        if mode == "exiting":
            raise SystemExit(54)
        if mode == "null_handler":
            return logging.NullHandler()
        return TestHandler()
    '''
)

_IMPORT_XTGEO = dedent(
    '''\
    import importlib.metadata
    import os
    from unittest.mock import patch

    def import_xtgeo():
        """Import XTGeo with the test site entry point."""
        entry = importlib.metadata.EntryPoint(
            "fmu-site-configuration",
            "xtgeo_test_site:create_handler",
            "atlas_telemetry",
        )
        entries = [] if os.environ["XTGEO_TEST_SITE_PLUGIN"] == "missing" else [entry]
        with patch.object(importlib.metadata, "entry_points", return_value=entries):
            import xtgeo
        return xtgeo
    '''
)


def _run_script(
    tmp_path: Path,
    script: str,
    *,
    plugin: str = "ready",
    extra_env: dict[str, str] | None = None,
) -> list[dict]:
    """Run XTGeo in a subprocess and return records from the test handler."""
    (tmp_path / "xtgeo_test_site.py").write_text(_SITE_PLUGIN, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(tmp_path), os.environ.get("PYTHONPATH", ""))
        ),
        "XTGEO_TEST_TELEMETRY_DIR": str(tmp_path),
        "XTGEO_TEST_SITE_PLUGIN": plugin,
    }
    env.pop("DISABLE_FMU_AZURE_LOGGING", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_XTGEO + dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "records.jsonl"
    if not output.exists():
        return []
    return [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize(
    "plugin", ["ready", "missing", "broken", "exiting", "null_handler", "disabled"]
)
def test_telemetry_from_normal_import(tmp_path: Path, plugin: str) -> None:
    """Check lazy loading, flush order, plugin failures, and opt-out."""
    extra_env = {"DISABLE_FMU_AZURE_LOGGING": "1"} if plugin == "disabled" else None
    records = _run_script(
        tmp_path,
        """\
        import inspect
        import sys
        import warnings
        from pathlib import Path

        os.environ["RUNRMS_EXEC_MODE"] = "interactive"
        os.environ["_ERT_EXPERIMENT_ID"] = "test-experiment"
        xtgeo = import_xtgeo()
        assert "xtgeo_test_site" not in sys.modules
        from xtgeo.grid3d.grid import grid_from_roxar

        grid = xtgeo.create_box_grid((2, 2, 2))
        plugin_mode = os.environ["XTGEO_TEST_SITE_PLUGIN"]
        if plugin_mode != "ready" and hasattr(sys, "monitoring"):
            assert sys.monitoring.get_tool(4) is None
        assert grid.ncol == 2
        with (
            warnings.catch_warnings(record=True) as caught,
            patch("xtgeo.grid3d.grid.grid_from_rms", return_value=grid),
        ):
            warnings.simplefilter("always")
            line = inspect.currentframe().f_lineno + 1
            assert grid_from_roxar("example.rms", "Grid") is grid
        warning, = caught
        assert warning.category is FutureWarning
        assert (warning.filename, warning.lineno) == ("<string>", line)

        try:
            xtgeo.grid_from_file("missing-telemetry-test-file.EGRID")
        except OSError:
            pass
        else:
            raise AssertionError("the missing file should raise OSError")
        Path(os.environ["XTGEO_TEST_TELEMETRY_DIR"], "host_pid").write_text(
            str(os.getpid()), encoding="utf-8"
        )
        """,
        plugin=plugin,
        extra_env=extra_env,
    )
    if sys.version_info < (3, 12) or plugin != "ready":
        assert not records
        assert not (tmp_path / "provider_shutdown").exists()
        return

    assert sorted((r["event_name"], r["api_name"]) for r in records) == sorted(
        [
            ("api_call", "create_box_grid"),
            ("api_call", "Grid.ncol.get"),
            ("api_call", "grid_from_roxar"),
            ("api_deprecation", "grid_from_roxar"),
            ("api_call", "grid_from_file"),
            ("api_error", "grid_from_file"),
        ]
    )
    assert all(r["app_name"] == "xtgeo" for r in records)
    host_pid = int((tmp_path / "host_pid").read_text(encoding="utf-8"))
    assert all(
        r["process_id"] == host_pid and r["process"] == host_pid for r in records
    )
    assert all(r["caller_module"] == "__main__" for r in records)
    assert all(r["ert_experiment_id"] == "test-experiment" for r in records)
    assert all(r["rms_exec_mode"] == "interactive" for r in records)
    (warning,) = [r for r in records if r["event_name"] == "api_deprecation"]
    assert warning["warning_category"] == "FutureWarning"
    assert warning["rms_project_filename"] == "example.rms"
    (error,) = [r for r in records if r["event_name"] == "api_error"]
    assert error["exception.type"] == "OSError"
    assert "OSError" in error["exception.stacktrace"]
    assert (tmp_path / "provider_shutdown").read_text(encoding="utf-8") == "shutdown\n"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_records_each_api_once_during_a_burst(tmp_path: Path) -> None:
    """Record each API once per process across bursts and subclass sharing."""
    records = _run_script(
        tmp_path,
        """\
        import logging

        xtgeo = import_xtgeo()
        grid = xtgeo.create_box_grid((2, 2, 2))

        logging.disable(logging.WARNING)
        for _ in range(5511):
            assert grid.ncol == 2
        logging.disable(logging.NOTSET)

        points = xtgeo.Points([(0.0, 0.0, 0.0)])
        polygons = xtgeo.Polygons([(0.0, 0.0, 0.0, 0)])
        points.copy()
        points.copy()
        polygons.copy()
        polygons.copy()
        """,
    )
    api_names = [r["api_name"] for r in records if r["event_name"] == "api_call"]
    assert api_names.count("create_box_grid") == 1
    assert api_names.count("Grid.ncol.get") == 1
    assert api_names.count("Points.copy") == 1
    assert api_names.count("Polygons.copy") == 1
    assert len(api_names) == len(set(api_names))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_handles_unhashable_subclasses_aliases_and_context(
    tmp_path: Path,
) -> None:
    """Cover unhashable metaclasses, staticmethod/alias names, and RMS context."""
    records = _run_script(
        tmp_path,
        """\
        import warnings
        from unittest.mock import patch

        import numpy as np
        import pandas as pd

        xtgeo = import_xtgeo()
        from xtgeo.interfaces.rms.rmsapi_utils import RmsApiUtils

        class UnhashableMeta(type):
            __hash__ = None

        class CustomSurface(xtgeo.RegularSurface, metaclass=UnhashableMeta):
            pass

        surf = CustomSurface(ncol=2, nrow=2, xinc=10.0, yinc=10.0)
        assert surf.ncol == 2

        assert xtgeo.Well.get_short_wellname("31/2-A-1") == "A-1"

        utils = object.__new__(RmsApiUtils)
        utils._version = "1.10"
        assert utils.roxversion == "1.10"

        tsurf = xtgeo.TriangulatedSurface(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]),
            triangles=np.array([[0, 1, 2]]),
        )
        assert tsurf.bounding_box.length_z == 2.0

        grid = xtgeo.create_box_grid((2, 2, 2))
        class CustomWell(xtgeo.Well):
            def delete_log(self, name):
                '''Extend the singular method through the inherited plural alias.'''
                return super().delete_logs(name)

        well = CustomWell(
            wname="W1",
            df=pd.DataFrame(
                {"X_UTME": [0.0], "Y_UTMN": [0.0], "Z_TVDSS": [0.0], "GR": [1.0]}
            ),
        )
        well.wname = "W2"
        assert well.wname == "W2"
        assert well.delete_logs("GR") == 1
        assert "GR" not in well.get_dataframe().columns
        with (
            warnings.catch_warnings(record=True) as caught,
            patch("xtgeo.well._well_oper.make_ijk_from_grid"),
        ):
            warnings.simplefilter("always")
            well.make_ijk_from_grid(grid, algorithm=2)
        assert len(caught) == 1
        assert caught[0].category is UserWarning

        prop = xtgeo.GridProperty(grid)
        with patch("xtgeo.grid3d._gridprop_roxapi.export_prop_roxapi"):
            prop.to_rms(
                projectname="example.rms", gridname="Grid", propertyname="Prop"
            )
        """,
    )
    api_names = {r["api_name"] for r in records if r["event_name"] == "api_call"}
    assert "RegularSurface.__init__" in api_names
    assert "Well.get_short_wellname" in api_names
    assert "BlockedWell.get_short_wellname" not in api_names
    assert "Well.wname.get" in api_names
    assert "Well.wname.set" in api_names
    assert "Well.name.get" not in api_names
    assert "Well.delete_logs" in api_names
    assert "Well.delete_log" not in api_names
    assert "RmsApiUtils.roxversion.get" in api_names
    assert "RmsApiUtils.rmsapiversion.get" not in api_names
    assert "BoundingBox3D.length_z.get" in api_names

    (warning,) = [r for r in records if r["event_name"] == "api_deprecation"]
    assert warning["warning_id"] == "Well.make_ijk_from_grid.algorithm"
    assert warning["warning_category"] == "UserWarning"

    (to_rms,) = [
        r
        for r in records
        if r["event_name"] == "api_call" and r["api_name"] == "GridProperty.to_rms"
    ]
    assert to_rms["rms_project_filename"] == "example.rms"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_attributes_shared_code_and_aliases(tmp_path: Path) -> None:
    """Keep unrelated repr calls out of telemetry and resolve each dialog alias."""
    records = _run_script(
        tmp_path,
        """\
        import inspect
        from dataclasses import dataclass
        from pathlib import Path

        import numpy as np

        xtgeo = import_xtgeo()
        from xtgeo.interfaces.rms import RegularSurfaceDataRms

        @dataclass
        class Foreign:
            value: object

        class BadRepr:
            def __repr__(self):
                '''Simulate an exception inside the shared repr wrapper.'''
                raise ValueError("example repr error")

        assert repr(Foreign(1)) == "Foreign(value=1)"
        try:
            repr(Foreign(BadRepr()))
        except ValueError:
            pass
        else:
            raise AssertionError("expected the repr error to propagate")

        data = RegularSurfaceDataRms(
            "surface", 0.0, 0.0, 1, 1, 1.0, 1.0, 0.0, np.ma.array([[1.0]])
        )
        line = inspect.currentframe().f_lineno + 1
        assert repr(data).startswith("RegularSurfaceDataRms(")
        repr(data)
        Path(os.environ["XTGEO_TEST_TELEMETRY_DIR"], "repr_line").write_text(
            str(line), encoding="utf-8"
        )

        dialog = xtgeo.XTGeoDialog()
        dialog.show_runtimewarnings()
        for method in (dialog.speak, dialog.info, dialog.warn, dialog.warning):
            method("example")
            assert dialog._caller == "<module>"
        """,
    )
    (repr_record,) = [
        r for r in records if r["api_name"] == "RegularSurfaceDataRms.__repr__"
    ]
    assert repr_record["event_name"] == "api_call"
    assert repr_record["caller_line"] == int(
        (tmp_path / "repr_line").read_text(encoding="utf-8")
    )
    names = [r["api_name"] for r in records if r["event_name"] == "api_call"]
    for alias in ("speak", "info", "warn", "warning"):
        assert names.count(f"XTGeoDialog.{alias}") == 1


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_retries_cancelled_load_and_ignores_handler_reentry(
    tmp_path: Path,
) -> None:
    """Retry the same API after cancellation and finish a re-entrant delivery."""
    records = _run_script(
        tmp_path,
        """\
        import warnings
        import xtgeo_test_site as site

        xtgeo = import_xtgeo()
        attempts = []

        class ReentrantHandler(site.TestHandler):
            def emit(self, record):
                '''Call the warning helper while delivering a record.'''
                xtgeo.XTGeoDialog.warndeprecated("from handler")
                return super().emit(record)

        def create():
            '''Interrupt only the first attempt to create a handler.'''
            attempts.append(None)
            if len(attempts) == 1:
                raise KeyboardInterrupt
            return ReentrantHandler()

        site.create_handler = create
        try:
            xtgeo.generic_hash("cancel")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("expected the cancellation to propagate")

        with warnings.catch_warnings(record=True) as caught:
            assert xtgeo.generic_hash("retry")
            assert any(str(warning.message) == "from handler" for warning in caught)
        assert len(attempts) == 2
        """,
    )
    assert [(r["event_name"], r["api_name"]) for r in records] == [
        ("api_call", "generic_hash")
    ]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_warning_can_start_handler_without_counting_its_calls(
    tmp_path: Path,
) -> None:
    """Load on a warning and leave handler calls out of the public API inventory."""
    records = _run_script(
        tmp_path,
        """\
        import warnings
        import xtgeo_test_site as site

        xtgeo = import_xtgeo()
        from xtgeo.common._telemetry import warn_deprecated

        class ReentrantHandler(site.TestHandler):
            def emit(self, record):
                '''Use an XTGeo helper during record delivery.'''
                assert xtgeo.generic_hash("handler")
                super().emit(record)

        site.create_handler = ReentrantHandler
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("first event", UserWarning)
        (warning,) = caught
        assert warning.category is UserWarning
        assert str(warning.message) == "first event"
        assert xtgeo.generic_hash("application")
        """,
    )
    deprecation, call = records
    assert deprecation["event_name"] == "api_deprecation"
    assert deprecation["warning_message"] == "first event"
    assert (call["event_name"], call["api_name"], call["caller_function"]) == (
        "api_call",
        "generic_hash",
        "<module>",
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires sys.monitoring")
def test_telemetry_keeps_concurrent_errors_during_load(tmp_path: Path) -> None:
    """Deliver errors from both callers when one caller is loading the handler."""
    records = _run_script(
        tmp_path,
        """\
        from threading import Event, Thread
        import xtgeo_test_site as site

        xtgeo = import_xtgeo()
        loading = Event()
        ready = Event()
        original_factory = site.create_handler

        def create():
            '''Wait for the second caller before returning the handler.'''
            loading.set()
            assert ready.wait(5)
            return original_factory()

        def fail():
            '''Verify that the expected API error still reaches its caller.'''
            try:
                xtgeo.grid_from_file("missing-concurrent-test.EGRID")
            except OSError:
                return
            raise AssertionError("expected the original OSError")

        site.create_handler = create
        worker = Thread(target=fail)
        worker.start()
        assert loading.wait(5)
        try:
            fail()
        finally:
            ready.set()
            worker.join(5)
        assert not worker.is_alive()
        """,
    )
    assert sorted((r["event_name"], r["api_name"]) for r in records) == [
        ("api_call", "grid_from_file"),
        ("api_error", "grid_from_file"),
        ("api_error", "grid_from_file"),
    ]


@pytest.mark.skipif(
    sys.version_info < (3, 12) or not hasattr(os, "fork"),
    reason="Requires sys.monitoring and POSIX fork",
)
def test_telemetry_fork_flushes_worker_and_preserves_host_monitoring(
    tmp_path: Path,
) -> None:
    """Flush via multiprocessing finalizers and keep host DISABLE intact after fork."""
    records = _run_script(
        tmp_path,
        """\
        import multiprocessing.util
        import os
        import sys
        from pathlib import Path

        xtgeo = import_xtgeo()
        host_calls = []

        def host_cb(code, _offset):
            host_calls.append(os.getpid())
            return sys.monitoring.DISABLE

        grid = xtgeo.create_box_grid((2, 2, 2))
        sys.monitoring.use_tool_id(5, "host")
        sys.monitoring.register_callback(5, sys.monitoring.events.PY_START, host_cb)
        sys.monitoring.set_local_events(
            5, xtgeo.Grid.ncol.fget.__code__, sys.monitoring.events.PY_START
        )
        assert grid.ncol == 2
        assert len(host_calls) == 1

        parent_pid = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                import warnings
                import xtgeo_test_site as site

                orig_create = site.create_handler

                def reentrant_create():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        xtgeo.XTGeoDialog.warndeprecated("from detector")
                    return orig_create()

                site.create_handler = reentrant_create
                assert grid.ncol == 2
                # Host callback must remain disabled in the child.
                assert len(host_calls) == 1
                Path(os.environ["XTGEO_TEST_TELEMETRY_DIR"], "child_pid").write_text(
                    str(os.getpid()), encoding="utf-8"
                )
                # Simulate Python 3.12 multiprocessing popen_fork exit path.
                multiprocessing.util._exit_function()
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        Path(os.environ["XTGEO_TEST_TELEMETRY_DIR"], "parent_pid").write_text(
            str(parent_pid), encoding="utf-8"
        )
        """,
    )
    parent_pid = int((tmp_path / "parent_pid").read_text(encoding="utf-8"))
    child_pid = int((tmp_path / "child_pid").read_text(encoding="utf-8"))
    ncol_pids = {
        r["process_id"]
        for r in records
        if r["event_name"] == "api_call" and r["api_name"] == "Grid.ncol.get"
    }
    assert ncol_pids == {parent_pid, child_pid}
    (deprecation,) = [r for r in records if r["event_name"] == "api_deprecation"]
    assert deprecation["process_id"] == child_pid
    assert deprecation["warning_message"] == "from detector"


def test_telemetry_recovers_optional_case_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return base context when optional fields fail and read the repaired file."""
    from xtgeo.common._telemetry import _runtime_context

    pytest.importorskip("yaml")
    metafile = tmp_path / "share" / "metadata" / "fmu_case.yml"
    metafile.parent.mkdir(parents=True)
    metafile.write_text("access: [", encoding="utf-8")
    monkeypatch.setenv("_ERT_RUNPATH", str(tmp_path / "realization-0" / "iter-0"))
    rmsapi = ModuleType("rmsapi")
    rmsapi.__version__ = MagicMock(
        __str__=MagicMock(side_effect=ValueError("bad version"))
    )
    monkeypatch.setitem(sys.modules, "rmsapi", rmsapi)
    frame = inspect.currentframe()
    assert frame is not None
    before = _runtime_context(frame)
    metafile.write_text("access:\n  asset:\n    name: Drogon\n", encoding="utf-8")
    after = _runtime_context(frame)
    assert "asset_name" not in before
    assert after["asset_name"] == "Drogon"
    assert before["process_id"] == after["process_id"] == os.getpid()
