"""Shared base classes for ResInsight data readers/writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger

from .rips_utils import RipsApiUtils

if TYPE_CHECKING:
    from ._rips_package import (
        ResInsightInstanceOrPortType,
        RipsCaseType,
        RipsInstanceType,
        RipsProjectType,
    )

logger = null_logger(__name__)


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
        instance_or_port: ResInsightInstanceOrPortType | None,
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

        selected_case = None
        for case in cases:
            if case.name == case_name:
                selected_case = case
                if not find_last:
                    break
        return selected_case
