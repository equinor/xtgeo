"""RMS API functions for RegularSurface."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms._rms_base import _BaseRmsDataRW
from xtgeo.interfaces.rms._rmsapi_package import rmsapi
from xtgeo.interfaces.rms.rmsapi_utils import (
    _DomainTypeClipBoardGeneral2D as DomainType,
    _StorageTypeRegularSurface as StorageType,
)

if TYPE_CHECKING:
    from xtgeo.interfaces.rms._rmsapi_package import (
        RmsProjectOrPathType,
        RmsSurfaceType,
        RmsVerticalDomainType,
    )


logger = null_logger(__name__)


@dataclass(frozen=True)
class RegularSurfaceDataRms:
    """Immutable data container for regular surface information."""

    name: str
    xori: float
    yori: float
    ncol: int
    nrow: int
    xinc: float
    yinc: float
    rotation: float
    values: np.ma.MaskedArray


class RegularSurfaceReader(_BaseRmsDataRW):
    """Handles reading (loading) regular surface data from RMS projects.

    This class handles all the complex loading logic, validation, and cleanup.
    """

    def __init__(
        self,
        project_or_path: RmsProjectOrPathType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
    ):
        super().__init__(
            project_or_path, name, category, stype, realisation, readonly=True
        )

        self._stype_enum = self.get_stype_validated()  # method in the base class

        # get the RMS project (rmsapi.Project) reference
        self._project = self.get_rmsapi_utils().project

    def load(self) -> RegularSurfaceDataRms:
        """Load surface data from RMS and return immutable data object."""
        try:
            return self._perform_load()
        except Exception as exc:
            self._cleanup()
            raise RuntimeError(f"Failed to load surface '{self.name}': {exc}") from exc

    def _perform_load(self) -> RegularSurfaceDataRms:
        """Internal method to handle the actual loading with subsequent cleanup."""

        # Extract surface data
        surface_data = self._extract_surface_from_rms()

        # Cleanup (close project if external)
        self._cleanup()

        return surface_data

    def _extract_surface_from_rms(self) -> RegularSurfaceDataRms:
        """Extract surface data from RMS project and create data object."""
        if TYPE_CHECKING:
            assert rmsapi is not None

        rms_item = None

        if self._stype_enum in (StorageType.HORIZONS, StorageType.ZONES):
            rms_item = self._get_horizon_zone_item()
        elif self._stype_enum in (StorageType.CLIPBOARD, StorageType.GENERAL2D_DATA):
            rms_item = self._get_clipboard_general2d_item()
        elif self._stype_enum == StorageType.TRENDS:
            rms_item = self._get_trends_item()
        else:
            raise ValueError(f"Unsupported storage type: {self._stype_enum}")

        # Validate surface object
        if not isinstance(rms_item, rmsapi.Surface):
            raise TypeError(
                f"Expected a Surface for '{self.name}', but got "
                f"{type(rms_item).__name__} for category '{self.category}'. "
                "Check that the item requested is indeed a surface."
            )

        # Extract grid data
        try:
            rmssurf = rms_item.get_grid(self.realisation)
            return self._create_surface_data(rmssurf)
        except KeyError as exc:
            raise RuntimeError(
                f"Could not load surface '{self.name}' from RMS API. "
                f"Realisation '{self.realisation}' may not exist."
            ) from exc

    def _create_surface_data(
        self,
        rms_surface: RmsSurfaceType,
    ) -> RegularSurfaceDataRms:
        """Create RegularSurfaceData from an RMSAPI RegularGrid2D.

        Args:
            rms_surface: A RegularGrid2D instance from RMS API

        Returns:
            Immutable surface data container
        """
        logger.info("Creating surface data from RMSAPI surface")

        return RegularSurfaceDataRms(
            name=self.name,
            xori=rms_surface.origin[0],
            yori=rms_surface.origin[1],
            ncol=rms_surface.dimensions[0],
            nrow=rms_surface.dimensions[1],
            xinc=rms_surface.increment[0],
            yinc=rms_surface.increment[1],
            rotation=rms_surface.rotation,
            # Wrap to ensure type correctness for values
            values=np.ma.array(rms_surface.get_values(), copy=False),
        )

    def _get_horizon_zone_item(self) -> RmsSurfaceType:  # rmsapi.Surface
        """Get surface item from horizons or zones."""
        if TYPE_CHECKING:
            assert rmsapi is not None
            assert isinstance(self._project, rmsapi.Project), (
                "Project must be initialized"
            )
            assert self._stype_enum is not None, "stype must be set"

        container = (
            self._project.horizons
            if self._stype_enum == StorageType.HORIZONS
            else self._project.zones
        )
        container_name = (
            "Horizons" if self._stype_enum == StorageType.HORIZONS else "Zones"
        )

        if self.name not in container:
            raise ValueError(f"Name '{self.name}' is not within {container_name}")
        if self.category not in container.representations:
            raise ValueError(
                f"Category '{self.category}' is not within {container_name} categories"
            )

        return container[self.name][self.category]

    def _get_clipboard_general2d_item(self) -> RmsSurfaceType:  # rmsapi.Surface
        """Get surface item from clipboard or general2d_data."""
        if TYPE_CHECKING:
            assert rmsapi is not None
            assert isinstance(self._project, rmsapi.Project), (
                "Project must be initialized"
            )

        try:
            container = getattr(self._project, self._stype_enum.value)
            if self.category:
                folders = self.category.split("|" if "|" in self.category else "/")
                return container.folders[folders][self.name]
            return container[self.name]
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Could not access '{self.name}' in {self._stype_enum.value}"
                + (f" with category '{self.category}'" if self.category else "")
            ) from exc

    def _get_trends_item(self) -> RmsSurfaceType:  # rmsapi.Surface
        """Get surface item from trends."""
        if self.category:
            warnings.warn(
                f"Ignoring category {self.category} for Trends storage type. Set "
                "it to empty or None to avoid this warning.",
                UserWarning,
            )

        assert rmsapi is not None
        assert isinstance(self._project, rmsapi.Project), "Project must be initialized"

        if self.name not in self._project.trends.surfaces:
            raise ValueError(f"Name '{self.name}' is not within Trends")
        return self._project.trends.surfaces[self.name]


class RegularSurfaceWriter(_BaseRmsDataRW):
    """Handles writing regular surface data to RMS API."""

    def __init__(
        self,
        project_or_path: RmsProjectOrPathType,
        name: str,
        category: str,
        stype: str,
        realisation: int = 0,
        domain: Literal["time", "depth", "unknown"] = "depth",  # clipboard/general2d
    ):
        super().__init__(
            project_or_path, name, category, stype, realisation, readonly=False
        )
        self._project = self.get_rms_project()  # rmsapi.Project
        self.domain = domain.lower()
        self._domain_enum: DomainType | None = None
        self._api_vertical_domain: Optional[RmsVerticalDomainType] = None
        self._stype_enum = self.get_stype_validated()  # method in the base class

    def save(self, data: RegularSurfaceDataRms) -> None:
        """Write surface data to RMS."""
        try:
            self._perform_save(data)
        except Exception as exc:
            raise RuntimeError(f"Failed to save surface '{self.name}': {exc}") from exc
        finally:
            self._cleanup()

    def _perform_save(self, data: RegularSurfaceDataRms) -> None:
        self._check_valid_domain()

        # Validate payload consistency
        validated_values = self._validate_payload(data)

        # Do the write
        self._write_to_rms(data, validated_values)

        # Save project if external
        if TYPE_CHECKING:
            assert rmsapi is not None
            assert isinstance(self._project, rmsapi.Project), (
                "Project must be initialized"
            )
        if getattr(self.get_rmsapi_utils(), "_rmsexternal", False):
            self._project.save()

    def _check_valid_domain(self) -> None:
        """Check that domain is valid for the given storage type."""
        if self.domain not in DomainType.values():
            raise ValueError(f"domain must be {DomainType.values()}")
        self._domain_enum = DomainType[self.domain.upper()]
        self._api_vertical_domain = self._resolve_api_vertical_domain()
        logger.debug("Using domain: %s", self._domain_enum)

    def _resolve_api_vertical_domain(self) -> RmsVerticalDomainType:
        """Map internal domain enum to RMS API VerticalDomain enum."""
        if TYPE_CHECKING:
            assert rmsapi is not None
        if self._domain_enum == DomainType.DEPTH:
            return rmsapi.VerticalDomain.depth
        if self._domain_enum == DomainType.TIME:
            return rmsapi.VerticalDomain.time
        return rmsapi.VerticalDomain.unknown

    @staticmethod
    def _validate_payload(data: RegularSurfaceDataRms) -> np.ma.MaskedArray:
        """Validate payload and return a normalized copy of values.

        This method validates the surface data and returns a copy of the values
        array normalized to MaskedArray format, without mutating the input.

        Args:
            data: The surface data to validate

        Returns:
            A copy of the values array as np.ma.MaskedArray

        Raises:
            ValueError: If validation fails
        """
        if data.ncol <= 0 or data.nrow <= 0:
            raise ValueError("ncol and nrow must be positive.")
        if data.xinc <= 0 or data.yinc <= 0:
            raise ValueError("xinc and yinc must be positive.")

        # Normalize to MaskedArray without mutating input
        if not isinstance(data.values, np.ma.MaskedArray):
            values = np.ma.array(data.values, copy=True)
        else:
            # Even if already MaskedArray, we'll make a copy to be safe
            values = data.values.copy()

        if values.shape != (data.ncol, data.nrow):
            raise ValueError(
                f"values shape {values.shape} does not match "
                f"(ncol, nrow)=({data.ncol}, {data.nrow})"
            )

        return values

    def _write_to_rms(
        self, data: RegularSurfaceDataRms, values: np.ma.MaskedArray
    ) -> None:
        if TYPE_CHECKING:
            assert rmsapi is not None
            assert isinstance(self._project, rmsapi.Project), (
                "Project must be initialized"
            )

        # Sanitize values for RMS API specifics (NaNs, etc.) before use
        values = self._sanitize_values_for_rms(values)

        # Build 2D grid (i.e. surface) geometry
        grid = rmsapi.RegularGrid2D.create(
            x_origin=data.xori,
            y_origin=data.yori,
            i_inc=data.xinc,
            j_inc=data.yinc,
            ni=data.ncol,
            nj=data.nrow,
            rotation=data.rotation,
        )
        grid.set_values(values)

        if self._stype_enum in (StorageType.HORIZONS, StorageType.ZONES):
            container = (
                self._project.horizons
                if self._stype_enum == StorageType.HORIZONS
                else self._project.zones
            )
            container_name = (
                "Horizons" if self._stype_enum == StorageType.HORIZONS else "Zones"
            )

            if self.name not in container:
                raise ValueError(f"Name '{self.name}' is not within {container_name}")
            if self.category not in container.representations:
                raise ValueError(
                    f"Category '{self.category}' is not within {container_name} "
                    "categories"
                )

            root = container[self.name][self.category]

            # Pass realisation where supported
            try:
                root.set_grid(grid, realisation=self.realisation)
            except TypeError:
                # Some APIs don't take realisation argument here
                root.set_grid(grid)

        elif self._stype_enum in (StorageType.CLIPBOARD, StorageType.GENERAL2D_DATA):
            styperef = getattr(self._project, self._stype_enum.value)

            def _get_current_item() -> tuple[Any, list[str]]:  # mixed type + list
                folders = []
                if self.category:
                    folders = self.category.split("|" if "|" in self.category else "/")
                    if folders:
                        styperef.folders.create(folders)
                current_item = styperef.folders[folders] if folders else styperef
                logger.debug(
                    "Current item: %s, type is %s", current_item, type(current_item)
                )
                logger.debug("Folders: %s, type is %s", folders, type(folders))
                return current_item, folders

            current_item, folders = _get_current_item()
            if self.name in current_item:
                root = current_item[self.name]
                if not isinstance(root, rmsapi.Surface):
                    raise TypeError(
                        f"Expected a Surface for '{self.name}', but got "
                        f"{type(root).__name__} for category '{self.category}'. "
                        "Check that the item requested is indeed a surface."
                    )
                logger.debug("Using domain: %s", self._domain_enum)
                current_domain = getattr(root, "vertical_domain", None)
                if current_domain != self._api_vertical_domain:
                    # RMS API does not allow changing domain of existing surface but
                    # we can brute-force remove the current surface and create a new one
                    logger.debug("Force remove current: %s", current_item[self.name])
                    del current_item[self.name]
                    logger.debug("Creating new surface with domain: %s", self.domain)
                    current_item, folders = _get_current_item()
                    root = styperef.create_surface(
                        self.name, folders, self._api_vertical_domain
                    )
            else:
                root = styperef.create_surface(
                    self.name, folders, self._api_vertical_domain
                )

            root.set_grid(grid)

        elif self._stype_enum == StorageType.TRENDS:
            # TRENDS do not have categories
            if self.category:
                warnings.warn(
                    f"Ignoring category {self.category} for Trends storage type. Set "
                    "it to empty or None to avoid this warning.",
                    UserWarning,
                )

            styperef = getattr(self._project, self._stype_enum.value)
            if self.name in styperef.surfaces:
                root = styperef.surfaces[self.name]
                if not isinstance(root, rmsapi.Surface):
                    raise TypeError(
                        f"Expected a Surface for '{self.name}', but got "
                        f"{type(root).__name__}. Check that the item requested is "
                        "indeed a surface."
                    )
            else:
                root = styperef.surfaces.create(self.name)

            root.set_grid(grid)

    @staticmethod
    def _sanitize_values_for_rms(
        values: np.ma.MaskedArray,
    ) -> np.ma.MaskedArray:
        """Sanitize array values for RMS, replacing NaN/Inf while preserving mask."""
        # Ensure dtype is float64
        if values.dtype != np.float64:
            mvalues: np.ma.MaskedArray = np.ma.array(
                values, dtype=np.float64, copy=False
            )
        else:
            mvalues = values

        # Preserve original mask
        original_mask = np.ma.getmaskarray(mvalues)

        # RMS API doesn't accept NaNs/Inf even behind mask; replace in data buffer
        applied_fill_value = np.finfo(np.float64).max
        filled = np.ma.filled(mvalues, fill_value=applied_fill_value)

        filled = np.where(
            np.isnan(filled) | np.isinf(filled), applied_fill_value, filled
        )

        # Create new masked array with combined mask:
        nan_inf_mask = np.isnan(mvalues.data) | np.isinf(mvalues.data)
        combined_mask = original_mask | nan_inf_mask

        return np.ma.array(filled, mask=combined_mask)
