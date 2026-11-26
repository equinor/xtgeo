"""Test callback behavior with ordinary Python frames and a monitoring stub."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from importlib.metadata import EntryPoint
from threading import Event, Thread, current_thread, main_thread
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import pytest

from xtgeo.common import _telemetry as telemetry


@pytest.fixture
def monitor(monkeypatch):
    """Provide monitoring and exit hooks without changing the interpreter's tools."""
    monitor = Mock(DISABLE=object())
    monkeypatch.setattr(
        telemetry,
        "sys",
        SimpleNamespace(
            monitoring=monitor, modules=sys.modules, _getframe=sys._getframe
        ),
    )
    monkeypatch.setattr(telemetry, "atexit", Mock())
    monkeypatch.setattr(telemetry, "_telemetry", None)
    monkeypatch.delenv("DISABLE_FMU_AZURE_LOGGING", raising=False)
    return monitor


class CaptureHandler(logging.Handler):
    """Save delivered telemetry records for inspection."""

    def __init__(self):
        """Start with no records and no flushes."""
        super().__init__()
        self.records = []
        self.flushed = False

    def emit(self, record):
        """Save a delivered record."""
        self.records.append(record)

    def flush(self):
        """Mark the handler as flushed."""
        self.flushed = True


@pytest.fixture
def start_telemetry(monitor):
    """Start telemetry with a recording handler and a supplied API inventory."""
    handler = CaptureHandler()

    def start(apis, factory=None):
        """Supply the requested factory and start the callback adapter."""
        entry = Mock()
        entry.load.return_value = factory or (lambda: handler)
        instance = telemetry._Telemetry(entry, apis)
        instance.start(4)
        return instance, handler

    return start


def _call(instance):
    """Invoke the callback with the frame of the observed call."""
    return instance.on_call(sys._getframe().f_code, 0)


def _invoke(function, *args):
    """Call a function from a frame with a controllable module name."""
    return function(*args)


class SharedApi:
    """Supply one implementation that is used by several receiver classes."""

    def call(self, instance):
        """Pass a shared method call to the monitoring callback."""
        return instance.on_call(sys._getframe().f_code, 0)

    def fail(self, instance, error):
        """Pass a raised exception to the unwind callback."""
        try:
            raise error
        except BaseException as caught:
            instance.on_error(sys._getframe().f_code, 0, caught)


class FirstApi(SharedApi):
    """Supply the first public name for the shared code."""


class SecondApi(SharedApi):
    """Supply the second public name for the shared code."""


def test_shared_call_callback_keeps_other_receivers_enabled(monitor, start_telemetry):
    """Ignore foreign receivers and record each supported shared method once."""
    api = telemetry._API(
        SharedApi.call,
        "FirstApi.call",
        {id(FirstApi): "FirstApi.call", id(SecondApi): "SecondApi.call"},
    )
    instance, handler = start_telemetry({id(api.function.__code__): api})
    internal = FunctionType(_invoke.__code__, {"__name__": "xtgeo._internal"})

    assert internal(FirstApi().call, instance) is None
    assert SharedApi().call(instance) is None
    assert FirstApi().call(instance) is None
    assert FirstApi().call(instance) is None
    assert SecondApi().call(instance) is monitor.DISABLE
    assert [row.api_name for row in handler.records] == [
        "FirstApi.call",
        "SecondApi.call",
    ]


def test_call_callback_rechecks_name_under_lock(monitor, start_telemetry):
    """Keep one record when two callbacks first see the same API name."""
    api = telemetry._API(_call, "call")
    instance, handler = start_telemetry({id(_call.__code__): api})
    observed = Event()
    proceed = Event()

    class PausedLookup(set):
        """Pause the worker after its first unlocked lookup."""

        def __contains__(self, name):
            """Release the stale lookup result after the main thread records the API."""
            result = super().__contains__(name)
            if current_thread() is not main_thread() and not observed.is_set():
                observed.set()
                assert proceed.wait(5)
            return result

    instance._seen_apis = PausedLookup()
    results = []
    worker = Thread(target=lambda: results.append(_call(instance)))
    worker.start()
    try:
        assert observed.wait(5)
        assert _call(instance) is monitor.DISABLE
    finally:
        proceed.set()
        worker.join(5)
    assert results == [monitor.DISABLE]
    assert [row.api_name for row in handler.records] == ["call"]


def test_error_callback_checks_receiver_exception_and_caller(start_telemetry):
    """Record escaping errors and keep internal calls and cancellation out."""
    api = telemetry._API(
        SharedApi.fail, "FirstApi.fail", {id(FirstApi): "FirstApi.fail"}
    )
    instance, handler = start_telemetry({id(api.function.__code__): api})
    internal = FunctionType(_invoke.__code__, {"__name__": "xtgeo._internal"})

    instance.on_error(_invoke.__code__, 0, ValueError("unregistered"))
    internal(FirstApi().fail, instance, ValueError("internal"))
    SharedApi().fail(instance, ValueError("unrelated receiver"))
    FirstApi().fail(instance, KeyboardInterrupt())
    FirstApi().fail(instance, ValueError("external"))

    (record,) = handler.records
    assert (record.event_name, record.api_name, record.levelname) == (
        "api_error",
        "FirstApi.fail",
        "ERROR",
    )
    assert record.error_message == "external"
    assert record.__dict__["exception.stacktrace"].endswith("ValueError: external\n")


def test_startup_limits_repeatable_events_and_keeps_first_api_calls(
    monitor,
    start_telemetry,
    monkeypatch,
):
    """Keep unique API calls when startup reaches its limit for repeated events."""
    monkeypatch.setattr(telemetry, "_MAX_PENDING_RECORDS", 2)
    handler = CaptureHandler()

    def create():
        """Queue an error burst and a distinct usage record during initialization."""
        for index in range(2):
            instance.emit("api_error", sys._getframe(), api_name=f"failure {index}")
        assert _call(instance) is None
        return handler

    api = telemetry._API(_call, "first API")
    instance, _ = start_telemetry({id(_call.__code__): api}, create)
    instance.emit("api_deprecation", sys._getframe(), api_name="old method")
    assert [(r.event_name, r.api_name) for r in handler.records] == [
        ("api_deprecation", "old method"),
        ("api_error", "failure 0"),
        ("api_call", "first API"),
    ]
    assert _call(instance) is monitor.DISABLE


def test_cancelled_startup_discards_buffer_and_retries_api(start_telemetry):
    """Allow the next call to retry after a cancelled startup."""
    handler = CaptureHandler()
    attempts = []

    def create():
        """Cancel the first load after buffering a repeatable event."""
        attempts.append(None)
        if len(attempts) == 1:
            instance.emit("api_error", sys._getframe(), api_name="buffered error")
            raise KeyboardInterrupt
        return handler

    api = telemetry._API(_call, "call")
    instance, _ = start_telemetry({id(_call.__code__): api}, create)
    with pytest.raises(KeyboardInterrupt):
        _call(instance)
    _call(instance)
    assert len(attempts) == 2
    assert [row.api_name for row in handler.records] == ["call"]


def test_failed_plugin_releases_tool_and_preserves_call(monitor, start_telemetry):
    """An unusable handler leaves the API and monitoring slot available."""
    api = telemetry._API(_call, "call")
    instance, handler = start_telemetry({id(_call.__code__): api}, lambda: None)
    assert _call(instance) is None
    assert handler.records == []
    monitor.free_tool_id.assert_called_once_with(4)


def test_shutdown_while_factory_is_running_drops_late_handler(monitor, start_telemetry):
    """A factory that completes after shutdown cannot restart the adapter."""
    handler = CaptureHandler()

    def create():
        """Stop monitoring before returning the handler."""
        instance.shutdown()
        return handler

    instance, _ = start_telemetry({}, create)
    instance.emit("api_call", sys._getframe(), api_name="call")
    assert handler.records == []
    monitor.set_local_events.reset_mock()
    instance.after_fork()
    monitor.set_local_events.assert_not_called()
    instance.emit("api_call", sys._getframe(), api_name="late call")
    assert handler.records == []


def test_shutdown_during_record_creation_rejects_late_record(
    start_telemetry, monkeypatch
):
    """Reject an event if shutdown occurs between collecting and sending it."""
    instance, handler = start_telemetry({})

    def finishing_context(frame):
        """Simulate shutdown while a call is collecting optional context."""
        instance.shutdown()
        return {}

    monkeypatch.setattr(telemetry, "_runtime_context", finishing_context)
    instance.emit("api_call", sys._getframe(), api_name="late call")
    assert handler.records == []


@pytest.mark.parametrize("failure", ["occupied", "claim race", "startup"])
def test_configure_handles_monitoring_slot_failures(monitor, monkeypatch, failure):
    """Handle unavailable slots and release a slot after partial registration."""
    entry = EntryPoint("site", "test:factory", "atlas_telemetry")
    monkeypatch.setattr(telemetry, "entry_points", lambda **kwargs: [entry])
    if failure == "occupied":
        monitor.get_tool.return_value = "host"
    elif failure == "claim race":
        monitor.get_tool.return_value = None
        monitor.use_tool_id.side_effect = ValueError("another tool claimed the slot")
    else:
        monitor.get_tool.side_effect = ["host", None]
        monitor.register_callback.side_effect = [
            RuntimeError("registration failed")
        ] + [None, None]

    telemetry.configure_telemetry()

    assert telemetry._telemetry is None
    if failure == "startup":
        monitor.use_tool_id.assert_called_once_with(3, "xtgeo telemetry")
        monitor.free_tool_id.assert_called_once_with(3)
    else:
        monitor.free_tool_id.assert_not_called()


def test_configure_supports_platform_without_fork(monitor, monkeypatch):
    """Enable telemetry where the platform has no fork hook."""
    entry = EntryPoint("site", "test:factory", "atlas_telemetry")
    monkeypatch.setattr(telemetry, "entry_points", lambda **kwargs: [entry])
    monkeypatch.setattr(telemetry, "os", SimpleNamespace(environ=os.environ))
    monitor.get_tool.return_value = None

    telemetry.configure_telemetry()
    instance = telemetry._telemetry
    assert instance is not None
    telemetry.atexit.register.assert_called_once_with(instance.shutdown)
    instance.shutdown()


def test_deprecation_uses_public_api_outside_context_manager(start_telemetry):
    """Attribute a nested warning to the outer API and identify the inner API."""
    instance, handler = start_telemetry({})
    namespace = {
        "__name__": "xtgeo._test_api",
        "contextmanager": contextmanager,
        "instance": instance,
        "getframe": sys._getframe,
    }
    exec(
        "def outer():\n"
        "    with wrapped():\n"
        "        pass\n"
        "@contextmanager\n"
        "def wrapped():\n"
        "    inner()\n"
        "    yield\n"
        "def inner():\n"
        "    instance.on_deprecation(\n"
        "        getframe(), 'old argument', FutureWarning, 'old')\n",
        namespace,
    )
    for name in ("outer", "inner"):
        function = namespace[name]
        instance.apis[id(function.__code__)] = telemetry._API(function, name)

    namespace["outer"]()
    (record,) = handler.records
    assert (record.api_name, record.warning_id) == ("outer", "inner.old")


def test_deprecation_handles_a_frame_without_a_caller(start_telemetry, monkeypatch):
    """Report an isolated frame with its qualified name and default location."""
    modules = sys.modules.copy()
    modules.pop("multiprocessing.util", None)
    monkeypatch.setattr(telemetry.sys, "modules", modules)
    instance, handler = start_telemetry({})
    namespace = {"__name__": "xtgeo._test_api"}
    exec("def legacy():\n    yield\n", namespace)
    generator = namespace["legacy"]()
    try:
        frame = generator.gi_frame
        assert frame is not None and frame.f_back is None
        instance.on_deprecation(frame, "legacy", FutureWarning, None)
    finally:
        generator.close()
    (record,) = handler.records
    assert record.api_name == record.warning_id == "legacy"
    assert record.lineno == 0
    telemetry.atexit.register.assert_called_once_with(instance.shutdown)


def test_case_context_tries_both_possible_case_roots(tmp_path):
    """Skip an empty metadata file and use the other supported case layout."""
    pytest.importorskip("yaml")
    runpath = tmp_path / "nested" / "realization"
    empty = tmp_path / "share/metadata/fmu_case.yml"
    empty.parent.mkdir(parents=True)
    empty.write_text("fmu: {}\n", encoding="utf-8")
    populated = runpath.parent / "share/metadata/fmu_case.yml"
    populated.parent.mkdir(parents=True)
    populated.write_text("fmu:\n  case:\n    name: Drogon\n", encoding="utf-8")

    assert telemetry._case_context(str(runpath)) == {"fmu_case_name": "Drogon"}


def test_runtime_context_extracts_rms_project_filename(monkeypatch):
    """Collect the filename of an open project and omit unnamed projects."""

    class Project:
        """Stand in for an RMS project opened by the caller."""

        def __init__(self, filename):
            """Store the filename that RMS exposes."""
            self.filename = filename

    def context_for(project):
        """Supply the public API's project argument in the observed frame."""
        return telemetry._runtime_context(sys._getframe())

    monkeypatch.setitem(sys.modules, "rmsapi", SimpleNamespace(Project=Project))
    assert context_for(Project("field.rms"))["rms_project_filename"] == "field.rms"
    assert "rms_project_filename" not in context_for(Project(None))
