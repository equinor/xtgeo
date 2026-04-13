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
        RipsPolygonCollectionType,
        RipsPolygonType,
        RipsProjectType,
    )

logger = null_logger(__name__)


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
        if not hasattr(self.get_ripsapi_utils().instance, "project"):
            raise RuntimeError("Connected rips.Instance does not have a project API")
        return self.get_ripsapi_utils().project

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

    def get_polygon_collection(self) -> RipsPolygonCollectionType:
        """Return the first PolygonCollection in the active ResInsight project.

        Raises:
            RuntimeError: If no PolygonCollection is present in the project.
        """
        from ._rips_package import rips

        collections = self.get_project().descendants(rips.PolygonCollection)
        if not collections:
            raise RuntimeError(
                "No PolygonCollection found in the active ResInsight project"
            )
        logger.debug("Found %d PolygonCollection(s) in project", len(collections))
        return collections[0]

    @staticmethod
    def find_polygon(
        collection: RipsPolygonCollectionType, name: str, find_last: bool
    ) -> RipsPolygonType | None:
        """Return the first or last polygon with the given name, or ``None``.

        Args:
            collection: The ``rips.PolygonCollection`` to search.
            name: Exact polygon name to match.
            find_last: If ``True``, continue scanning after a match so the last
                match is returned; if ``False``, stop at the first match.
        """
        selected = None
        for poly in collection.polygons():
            if poly.name == name:
                selected = poly
                if not find_last:
                    break
        return selected
