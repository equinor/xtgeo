"""Shared base classes for ResInsight data readers/writers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from xtgeo.common.log import null_logger

from ._rips_package import NameConflictPolicy, require_rips
from .rips_utils import RipsApiUtils

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._rips_package import (
        ResInsightInstanceOrPortType,
        RipsCaseType,
        RipsInstanceType,
        RipsPolygonCollectionType,
        RipsProjectType,
        RipsSurfaceCollectionType,
    )

    _ItemT = TypeVar("_ItemT")
    _FolderT = TypeVar("_FolderT", RipsSurfaceCollectionType, RipsPolygonCollectionType)

logger = null_logger(__name__)


def select_by_name(
    items: Iterable[_ItemT],
    name: str,
    find_last: bool = True,
    name_attr: str = "name",
) -> _ItemT | None:
    """Select the item from *items* whose *name_attr* equals *name*.

    Shared lookup helper for the ResInsight readers/writers, which pick a
    named object (case, surface, polygon, folder, ...) from an iterable.
    Returns the last match when *find_last* is ``True`` (default), else the
    first, or ``None`` when nothing matches.
    """
    selected = None
    for item in items:
        if getattr(item, name_attr, None) == name:
            selected = item
            if not find_last:
                break
    return selected


def _resolve_folder(
    root: _FolderT,
    folder_path: str,
    collection_type: type[_FolderT],
    name_attr: str,
    folder_kind: str,
    create: bool = False,
) -> _FolderT:
    """Resolve a folder path using the concrete collection type."""
    folder = root

    for segment in filter(None, folder_path.split("/")):
        # rips uses collection_type as a fallback wrapper class, not a filter;
        # explicitly exclude other container types before matching by name.
        sub = select_by_name(
            (
                child
                for child in folder.children("SubCollections", collection_type)
                if isinstance(child, collection_type)
            ),
            segment,
            find_last=False,
            name_attr=name_attr,
        )
        if sub is None:
            if not create:
                raise RuntimeError(f"Cannot find {folder_kind} folder '{segment}'")

            # rips types add_folder() as PdmNestedCollectionBase, not the concrete
            # collection type; this cast assumes that type without runtime checks.
            sub = cast(
                "_FolderT",
                folder.add_folder(
                    folder_name=segment, on_name_conflict=NameConflictPolicy.FAIL
                ),
            )
        folder = sub

    return folder


def resolve_surface_folder(
    root: RipsSurfaceCollectionType,
    folder_path: str,
    create: bool = False,
) -> RipsSurfaceCollectionType:
    """Resolve a ``/``-separated folder path below a surface collection.

    An empty *folder_path* returns *root*. Missing segments are created when
    *create* is ``True``. Otherwise, a ``RuntimeError`` is raised when a folder
    does not exist.
    """
    rips = require_rips()
    return _resolve_folder(
        root,
        folder_path,
        collection_type=rips.SurfaceCollection,
        name_attr="surface_user_description",
        folder_kind="surface",
        create=create,
    )


def resolve_polygon_folder(
    root: RipsPolygonCollectionType,
    folder_path: str,
    create: bool = False,
) -> RipsPolygonCollectionType:
    """Resolve a ``/``-separated folder path below a polygon collection.

    An empty *folder_path* returns *root*. Missing segments are created when
    *create* is ``True``. Otherwise, a ``RuntimeError`` is raised when a folder
    does not exist.
    """
    rips = require_rips()
    return _resolve_folder(
        root,
        folder_path,
        collection_type=rips.PolygonCollection,
        name_attr="polygon_collection_name",
        folder_kind="polygon",
        create=create,
    )


def validate_case(case: str | RipsCaseType) -> None:
    """Validate a case argument, raising :class:`TypeError` if it is invalid.

    A valid case is either a case name (``str``) or a ``rips`` Case object
    exposing a string ``name`` attribute. This is a cheap, dependency-free guard
    so callers can fail fast before doing expensive work (e.g. extracting grid
    data) or before creating a case with an empty name.

    Args:
        case: A case name (str) or a ``rips`` Case object.

    Raises:
        TypeError: If *case* is neither a string nor an object exposing a string
            ``name`` attribute.
    """
    if isinstance(case, str):
        return
    if not isinstance(getattr(case, "name", None), str):
        raise TypeError(
            "case must be a case name (str) or a rips Case object with a "
            f"'name' attribute, but got {type(case).__name__}"
        )


class _BaseResInsightDataRW:
    """Common init and lookup utilities for ResInsight read/write operations."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        self.instance_or_port = instance_or_port
        self._ripsapi_utils: RipsApiUtils | None = None

    def get_ripsapi_utils(self) -> RipsApiUtils:
        """Create and cache RipsApiUtils."""
        if self._ripsapi_utils is None:
            self._ripsapi_utils = RipsApiUtils(self.instance_or_port)
        return self._ripsapi_utils

    def get_instance(self) -> RipsInstanceType:
        return self.get_ripsapi_utils().instance

    def get_project(self) -> RipsProjectType:
        """Get the active ResInsight project."""
        return self.get_ripsapi_utils().project

    def resolve_case(
        self, case: str | RipsCaseType, find_last: bool = True
    ) -> RipsCaseType | None:
        """Resolve a target case from either a case object or a case name.

        Args:
            case: Either a ``rips`` case object (returned as-is) or the case name
                to look up in the project (see :meth:`get_case`).
            find_last: When *case* is a name and several cases share it, select the
                last match if ``True`` (default), otherwise the first.

        Raises:
            TypeError: If *case* is neither a string nor a rips Case object
                exposing a string ``name`` attribute.
        """
        if isinstance(case, str):
            return self.get_case(case_name=case, find_last=find_last)
        validate_case(case)
        return case

    def get_case(self, case_name: str, find_last: bool = True) -> RipsCaseType | None:
        """Resolve target case from project by its name.

        The case name is not unique in ResInsight, by default it will find the last
        matching case name.
        """
        cases = self.get_project().cases()  # type: ignore[attr-defined]
        logger.debug(
            "Found %d cases in project: %s",
            len(cases),
            [case.name for case in cases],
        )
        if not cases:
            return None

        return select_by_name(cases, case_name, find_last=find_last)
