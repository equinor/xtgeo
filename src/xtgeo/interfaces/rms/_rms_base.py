"""Shared base for RMS RegularSurface reader/writer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms._rmsapi_package import rmsapi
from xtgeo.interfaces.rms.rmsapi_utils import (
    RmsApiUtils,
    _StorageTypeRegularSurface as StorageType,
)

if TYPE_CHECKING:
    from xtgeo.interfaces.rms._rmsapi_package import (
        RmsProjectOrPathType,
        RmsProjectType,
    )

logger = null_logger(__name__)


class _BaseRmsDataRW:
    """
    Common init, input validation etc for read/write RMS RegularSurface, Points, etc.
    """

    def __init__(
        self,
        project_or_path: RmsProjectOrPathType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
        readonly: bool = True,
    ):
        # Check rmsapi availability early
        if rmsapi is None:
            raise RuntimeError(
                "rmsapi module is not available. "
                "This functionality requires RMS environment or rmsapi package."
            )
        self.project_or_path = project_or_path
        self.name = name
        self.category = category
        self.stype = stype
        self.realisation = realisation

        # Internal states
        self._readonly = readonly  # should be immutable
        self._rmsapi_utils: RmsApiUtils | None = None

        # name cannot be an empty string
        if not self.name:
            raise ValueError("The name is missing or empty.")

    @property
    def readonly(self) -> bool:
        """Read-only mode flag (immutable)."""
        return self._readonly

    def get_rmsapi_utils(self) -> RmsApiUtils:
        """Create RmsApiUtils with the selected readonly mode."""
        if self._rmsapi_utils is None:
            self._rmsapi_utils = RmsApiUtils(
                self.project_or_path, readonly=self.readonly
            )
        return self._rmsapi_utils

    def get_rms_project(self) -> RmsProjectType:
        """Get the RMS project instance of type rmsapi.Project."""
        return self.get_rmsapi_utils().project

    def get_stype_validated(self) -> StorageType:
        """Validate stype returns StorageType enum."""
        stype_lower = self.stype.lower()

        if stype_lower not in StorageType.values():
            raise ValueError(
                f"Given stype '{stype_lower}' is not supported. "
                f"Legal stypes are: {StorageType.values()}"
            )

        stype_enum = StorageType(stype_lower)

        if (
            stype_enum in (StorageType.HORIZONS, StorageType.ZONES)
            and not self.category
        ):
            raise ValueError(
                "Need to specify both name and category for horizons and zones"
            )

        return stype_enum

    def _cleanup(self) -> None:
        """Clean up RmsApiUtils instance."""
        if self._rmsapi_utils:
            self._rmsapi_utils.safe_close()
            self._rmsapi_utils = None
