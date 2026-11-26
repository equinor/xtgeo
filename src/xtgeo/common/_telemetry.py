"""Send XTGeo events through the optional site logging handler."""

from __future__ import annotations

import atexit
import inspect
import logging
import os
import platform
import sys
import traceback
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from threading import Lock, local
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import EntryPoint
    from types import CodeType, FrameType, FunctionType


_TELEMETRY_ENTRY_POINT_GROUP = "atlas_telemetry"
_DISABLE_ENV_VAR = "DISABLE_FMU_AZURE_LOGGING"
_MAX_PENDING_RECORDS = 2048


_ERT_ENVIRONMENT = {
    "ert_experiment_id": "_ERT_EXPERIMENT_ID",
    "ert_ensemble_id": "_ERT_ENSEMBLE_ID",
    "ert_run_id": "ERT_RUN_ID",
    "ert_simulation_mode": "_ERT_SIMULATION_MODE",
    "ert_iteration_number": "_ERT_ITERATION_NUMBER",
    "ert_realization_number": "_ERT_REALIZATION_NUMBER",
    "ert_runpath": "_ERT_RUNPATH",
    "rms_exec_mode": "RUNRMS_EXEC_MODE",
    "rmsvenv_release": "RMSVENV_RELEASE",
}


def _is_xtgeo(module: object) -> bool:
    """Check whether the module name belongs to XTGeo."""
    return isinstance(module, str) and (
        module == "xtgeo" or module.startswith("xtgeo.")
    )


@dataclass
class _API:
    """Store one Python implementation and its public API names."""

    function: FunctionType
    name: str
    member_names: dict[int, str] = field(default_factory=dict)

    def name_for(self, frame: FrameType) -> str | None:
        """Resolve the public API name from the receiver's class."""
        if not self.member_names:
            return self.name
        receiver = frame.f_locals.get(self.function.__code__.co_varnames[0])
        receiver_type = type(receiver)
        cls = receiver if issubclass(receiver_type, type) else receiver_type
        for base in cls.__mro__:
            if (name := self.member_names.get(id(base))) is not None:
                return name
        # Generated methods can share code with classes outside XTGeo.
        return None


def _class_members(
    cls: type,
) -> Iterable[tuple[str, FunctionType, bool]]:
    """Yield public Python members and indicate which take a receiver."""
    for name, member in inspect.getmembers_static(cls):
        if name.startswith("_") and not (
            name.startswith("__") and name.endswith("__") and name != "__del__"
        ):
            continue
        if inspect.isfunction(member):
            yield name, member, True
        elif isinstance(member, classmethod):
            yield name, member.__func__, True
        elif isinstance(member, staticmethod):
            if name in cls.__dict__:
                yield name, member.__func__, False
        elif isinstance(member, property):
            for action, accessor in (
                ("get", member.fget),
                ("set", member.fset),
                ("delete", member.fdel),
            ):
                if inspect.isfunction(accessor):
                    yield f"{name}.{action}", accessor, True


def _register_class_apis(cls: type, apis: dict[int, _API], classes: set[int]) -> None:
    """Add public Python members of one XTGeo class to the API inventory."""
    cls_id = id(cls)
    if cls_id in classes:
        return
    classes.add(cls_id)
    for name, function, bound in _class_members(cls):
        if not _is_xtgeo(function.__module__):
            continue
        api_name = f"{cls.__name__}.{name}"
        api = apis.setdefault(id(function.__code__), _API(function, api_name))
        if bound:
            api.member_names.setdefault(cls_id, api_name)


def _public_apis() -> dict[int, _API]:
    """Use Python implementations of public exports and their class members."""
    from xtgeo.common_geometry.bounding_box_3d import BoundingBox3D
    from xtgeo.interfaces import resinsight, rms

    apis: dict[int, _API] = {}
    classes: set[int] = set()
    modules = [sys.modules["xtgeo"], resinsight, rms]
    modules.extend(
        module
        for name, module in tuple(sys.modules.items())
        if name.startswith("xtgeo.")
        and all(not part.startswith("_") for part in name.split("."))
        and hasattr(module, "__all__")
        and module not in modules
    )

    for module in modules:
        for name in module.__all__:
            if name.startswith("_"):
                continue
            member = getattr(module, name)
            if not _is_xtgeo(getattr(member, "__module__", "")):
                continue
            if inspect.isfunction(member):
                apis.setdefault(id(member.__code__), _API(member, name))
            elif inspect.isclass(member) and not issubclass(member, BaseException):
                _register_class_apis(member, apis, classes)
    _register_class_apis(BoundingBox3D, apis, classes)
    return apis


@lru_cache(maxsize=1)
def _process_context(pid: int) -> dict[str, str | int]:
    """Return the context shared by events from one process."""
    return {
        "app_name": "xtgeo",
        "app_version": sys.modules["xtgeo"].__version__,
        "process_id": pid,
        "os": platform.system(),
        "architecture": platform.machine(),
    }


@lru_cache(maxsize=64)
def _case_context(runpath: str) -> dict[str, str]:
    """Read FMU case, model, and asset fields for an ERT run path."""
    # fmu-dataio checks both case/realization/iteration and case/realization layouts.
    from importlib import import_module

    with suppress(OSError, RuntimeError):
        for parent in reversed(Path(runpath).resolve().parents[:2]):
            case_file = parent / "share/metadata/fmu_case.yml"
            if not case_file.is_file():
                continue
            with suppress(Exception):
                yaml = import_module("yaml")
                with case_file.open(encoding="utf-8") as stream:
                    metadata = yaml.safe_load(stream)
                fmu = metadata.get("fmu", {})
                case = fmu.get("case", {})
                model = fmu.get("model", {})
                asset = metadata.get("access", {}).get("asset", {})
                properties = {
                    "asset_name": asset.get("name"),
                    "fmu_case_name": case.get("name"),
                    "fmu_case_uuid": case.get("uuid"),
                    "fmu_model_name": model.get("name"),
                    "fmu_model_revision": model.get("revision"),
                }
                context = {
                    key: value
                    for key, value in properties.items()
                    if isinstance(value, str)
                }
                if context:
                    return context
    # Cache successful reads. A later call can pick up a missing or repaired file.
    raise FileNotFoundError(runpath)


def _runtime_context(frame: FrameType) -> dict[str, str | int]:
    """Collect the process, ERT, FMU, and RMS fields for a call."""
    context = _process_context(os.getpid()).copy()
    environment = {
        field: value
        for field, key in _ERT_ENVIRONMENT.items()
        if (value := os.environ.get(key))
    }
    context.update(environment)
    if runpath := environment.get("ert_runpath"):
        with suppress(OSError):
            context.update(_case_context(runpath))

    rmsapi = sys.modules.get("rmsapi")
    with suppress(Exception):
        if version := getattr(rmsapi, "__version__", None):
            context["rms_api_version"] = str(version)
    project_name = next(
        (
            name
            for name in ("project", "project_or_path", "projectname")
            if name in frame.f_code.co_varnames
        ),
        None,
    )
    if project_name is not None:
        project = frame.f_locals.get(project_name)
        with suppress(Exception):
            if isinstance(project, getattr(rmsapi, "Project", ())):
                project = getattr(project, "filename", None)
            if isinstance(project, (str, Path)) and project:
                context["rms_project_filename"] = str(project)
    return context


def _make_record(
    event: str, level: int, fields: dict[str, str | int]
) -> logging.LogRecord:
    """Build a logging record with context collected from the XTGeo call."""
    record = logging.LogRecord(
        name="xtgeo.azure",
        level=level,
        pathname=str(fields.get("caller_file", __file__)),
        lineno=int(fields.get("caller_line", 0)),
        msg=event,
        args=(),
        exc_info=None,
        func=str(fields.get("caller_function", "")),
    )
    record.__dict__.update(fields, event_name=event)
    return record


class _Telemetry:
    """Connect Python monitoring events to the optional site logging handler."""

    def __init__(self, entry_point: EntryPoint, apis: dict[int, _API]) -> None:
        """Store the site entry point and public API inventory."""
        self._entry_point = entry_point
        self.apis = apis
        self._tool_id: int | None = None
        self._lock = Lock()
        self._delivery = local()
        self._handler: logging.Handler | None = None
        self._pending: list[logging.LogRecord] | None = None
        self._seen_apis: set[str] = set()

    def _register_exit_hooks(self) -> None:
        """Flush XTGeo before OpenTelemetry's atexit or multiprocessing exit."""
        atexit.unregister(self.shutdown)
        atexit.register(self.shutdown)
        mp_util = sys.modules.get("multiprocessing.util")
        if mp_util is not None and hasattr(mp_util, "Finalize"):
            with suppress(Exception):
                mp_util.Finalize(None, self.shutdown, exitpriority=10)

    def _load_handler(self, pending: list[logging.LogRecord]) -> None:
        """Load the optional site handler outside locks and drain pending records."""
        loaded: logging.Handler | None = None
        with suppress(Exception, SystemExit):
            candidate = self._entry_point.load()()
            if isinstance(candidate, logging.Handler) and not isinstance(
                candidate, logging.NullHandler
            ):
                self._register_exit_hooks()
                loaded = candidate
        if loaded is None:
            self.shutdown()
            return

        with self._lock:
            if self._pending is not pending or self._tool_id is None:
                return
            self._handler = loaded
            self._pending = None
        for record in pending:
            self._deliver(loaded, record)

    def _deliver(self, handler: logging.Handler, record: logging.LogRecord) -> None:
        """Pass a record to the handler and ignore telemetry from its own calls."""
        self._delivery.active = True
        try:
            with suppress(Exception, SystemExit):
                handler.handle(record)
        finally:
            self._delivery.active = False

    def _emit_record(self, record: logging.LogRecord) -> None:
        """Send one record, loading the handler lazily without holding locks."""
        pending = None
        try:
            with self._lock:
                if self._tool_id is None:
                    return
                handler = self._handler
                if handler is None:
                    if self._pending is not None:
                        # Keep unique API calls and limit repeatable startup events.
                        if (
                            len(self._pending) < _MAX_PENDING_RECORDS
                            or record.msg == "api_call"
                        ):
                            self._pending.append(record)
                        return
                    pending = [record]
                    self._pending = pending
            if handler is not None:
                self._deliver(handler, record)
            else:
                assert pending is not None
                self._load_handler(pending)
        finally:
            if pending is not None:
                with self._lock:
                    if self._pending is pending:
                        # A cancelled load can be retried at the next event.
                        self._pending = None
                        self._seen_apis.difference_update(
                            rec.api_name for rec in pending if rec.msg == "api_call"
                        )

    def after_fork(self) -> None:
        """Reset handler state and re-enable only XTGeo's monitoring slot in a child."""
        self._lock = Lock()
        self._delivery = local()
        self._handler = None
        self._pending = None
        self._seen_apis.clear()
        tool_id = self._tool_id
        if tool_id is not None:
            monitor = sys.monitoring
            with suppress(Exception):
                for api in self.apis.values():
                    code = api.function.__code__
                    monitor.set_local_events(tool_id, code, 0)
                    monitor.set_local_events(tool_id, code, monitor.events.PY_START)

    def start(self, tool_id: int) -> None:
        """Register call and exception callbacks in the selected monitoring slot."""
        self._tool_id = tool_id
        monitor = sys.monitoring
        monitor.register_callback(tool_id, monitor.events.PY_START, self.on_call)
        monitor.register_callback(tool_id, monitor.events.PY_UNWIND, self.on_error)
        for api in self.apis.values():
            monitor.set_local_events(
                tool_id, api.function.__code__, monitor.events.PY_START
            )
        # Python offers unwind events at interpreter scope. Filter by code first.
        monitor.set_events(tool_id, monitor.events.PY_UNWIND)

    def shutdown(self) -> None:
        """Release the monitoring slot and flush the isolated handler."""
        global _telemetry
        tool_id, self._tool_id = self._tool_id, None
        if tool_id is None:
            return
        if _telemetry is self:
            _telemetry = None
        monitor = sys.monitoring
        with suppress(Exception):
            monitor.set_events(tool_id, 0)
            for api in self.apis.values():
                monitor.set_local_events(tool_id, api.function.__code__, 0)
            monitor.register_callback(tool_id, monitor.events.PY_START, None)
            monitor.register_callback(tool_id, monitor.events.PY_UNWIND, None)
            monitor.free_tool_id(tool_id)
        with self._lock:
            handler, self._handler = self._handler, None
            self._pending = None
        if handler is not None:
            with suppress(Exception, SystemExit):
                getattr(handler, "force_flush", handler.flush)()

    def emit(
        self,
        event: str,
        frame: FrameType,
        *,
        level: int = logging.INFO,
        error: BaseException | None = None,
        **attributes: str,
    ) -> None:
        """Send an event with runtime context and optional exception details."""
        # A telemetry error must not affect the Python call or its exception.
        with suppress(Exception):
            if self._tool_id is None or getattr(self._delivery, "active", False):
                return
            caller = frame.f_back
            context = _runtime_context(frame)
            if caller is not None:
                module = caller.f_globals.get("__name__")
                context.update(
                    caller_module=module if isinstance(module, str) else "",
                    caller_function=caller.f_code.co_name,
                    caller_file=caller.f_code.co_filename,
                    caller_line=caller.f_lineno,
                )
            context.update(attributes)
            if error is not None:
                context["exception.type"] = type(error).__qualname__
                context["exception.message"] = attributes["error_message"]
                context["exception.stacktrace"] = "".join(
                    traceback.format_exception(error)
                )
            self._emit_record(_make_record(event, level, context))

    @staticmethod
    def _is_external_call(frame: FrameType) -> bool:
        """Check whether the caller is outside XTGeo or unknown."""
        caller = frame.f_back
        return caller is None or not _is_xtgeo(caller.f_globals.get("__name__", ""))

    def on_call(self, code: CodeType, _offset: int) -> object:
        """Record the first external call to a public API in this process."""
        with suppress(Exception):
            frame = sys._getframe(1)
            if not self._is_external_call(frame) or getattr(
                self._delivery, "active", False
            ):
                return None
            api = self.apis[id(code)]
            api_name = api.name_for(frame)
            if api_name is None:
                return None
            if api_name not in self._seen_apis:
                should_emit = False
                with self._lock:
                    if api_name not in self._seen_apis:
                        self._seen_apis.add(api_name)
                        should_emit = True
                if should_emit:
                    self.emit("api_call", frame, api_name=api_name)
            if self._handler is None or api_name not in self._seen_apis:
                return None
            if not api.member_names or all(
                name in self._seen_apis for name in api.member_names.values()
            ):
                return sys.monitoring.DISABLE
        return None

    def on_error(self, code: CodeType, _offset: int, error: BaseException) -> None:
        """Record an exception that escapes a public API call from outside XTGeo."""
        api = self.apis.get(id(code))
        if api is None or not isinstance(error, Exception):
            return
        with suppress(Exception):
            frame = sys._getframe(1)
            if self._is_external_call(frame) and (api_name := api.name_for(frame)):
                self.emit(
                    "api_error",
                    frame,
                    level=logging.ERROR,
                    error=error,
                    api_name=api_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )

    def on_deprecation(
        self,
        frame: FrameType,
        message: str,
        category: type[Warning],
        identifier: str | None,
    ) -> None:
        """Record the deprecation and identify the associated public API."""
        warning_frame = None
        api_frame = ancestor = frame
        while (
            _is_xtgeo(module := ancestor.f_globals.get("__name__", ""))
            or module == "contextlib"
        ):
            if id(ancestor.f_code) in self.apis:
                api_frame = ancestor
                if warning_frame is None:
                    warning_frame = ancestor
            parent = ancestor.f_back
            if parent is None:
                break
            ancestor = parent
        api = self.apis.get(id(api_frame.f_code))
        api_name = api.name_for(api_frame) if api else None
        warning_id = frame.f_code.co_qualname
        if warning_frame is not None:
            warning_id = (
                self.apis[id(warning_frame.f_code)].name_for(warning_frame)
                or warning_id
            )
        if identifier is not None:
            warning_id = f"{warning_id}.{identifier}"
        self.emit(
            "api_deprecation",
            api_frame,
            level=logging.WARNING,
            api_name=api_name or frame.f_code.co_qualname,
            warning_category=category.__name__,
            warning_message=message,
            warning_id=warning_id,
        )


_telemetry: _Telemetry | None = None


def configure_telemetry() -> None:
    """Enable telemetry when the optional site entry point is available."""
    global _telemetry
    monitor = getattr(sys, "monitoring", None)
    if monitor is None or _telemetry is not None or os.environ.get(_DISABLE_ENV_VAR):
        return

    with suppress(Exception):
        entry = next(iter(entry_points(group=_TELEMETRY_ENTRY_POINT_GROUP)), None)
        tool_id = next(
            (slot for slot in (4, 3) if monitor.get_tool(slot) is None), None
        )
        if entry is None or tool_id is None:
            return
        telemetry = _Telemetry(entry, _public_apis())
        try:
            monitor.use_tool_id(tool_id, "xtgeo telemetry")
        except ValueError:
            return
        try:
            telemetry.start(tool_id)
        except Exception:
            telemetry.shutdown()
            return
        _telemetry = telemetry
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=telemetry.after_fork)
        atexit.register(telemetry.shutdown)


def warn_deprecated(
    message: str,
    category: type[Warning],
    *,
    stacklevel: int = 1,
    identifier: str | None = None,
) -> None:
    """Record deprecated use and preserve Python warning locations and filters."""
    if _telemetry is not None:
        with suppress(Exception):
            _telemetry.on_deprecation(sys._getframe(1), message, category, identifier)
    warnings.warn(message, category, stacklevel=stacklevel + 1)
