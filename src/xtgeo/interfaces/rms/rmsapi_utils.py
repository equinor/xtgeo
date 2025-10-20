"""Module for various operation in the RMSAPI (former rmsapi) python interface."""

import enum
import pathlib
import warnings
from typing import Any, Literal

from packaging.version import parse as versionparse

from xtgeo.common.log import null_logger

from ._rmsapi_package import RmsProjectType, rmsapi

logger = null_logger(__name__)


class _DomainType(str, enum.Enum):
    """Valid 'domain' types for Roxar API operations, invoked with 'domain=...'"""

    DEPTH = "depth"
    TIME = "time"
    THICKNESS = "thickness"
    UNKNOWN = "unknown"

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid domain type values."""
        return [item.value for item in cls]


class _DomainTypeClipBoardGeneral2D(str, enum.Enum):
    """
    For clipboard/general2d_data where "thickness" is not valid according to API docs.
    """

    DEPTH = "depth"
    TIME = "time"
    UNKNOWN = "unknown"

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid domain type values."""
        return [item.value for item in cls]


class _StorageTypeRegularSurface(str, enum.Enum):
    """Valid 'storage' types for reg. surf in RMS, invoked with 'stype=...'"""

    HORIZONS = "horizons"
    ZONES = "zones"
    CLIPBOARD = "clipboard"
    GENERAL2D_DATA = "general2d_data"
    TRENDS = "trends"

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid surface type values."""
        return [item.value for item in cls]


class RmsApiUtils:
    """Class RmsApiUtils, for accessing project level methods::

     import xtgeo

     xr = xtgeo.RmsApiUtils(project)
     xr.create_horizon_category('DS_extracted_run3')
     xr.delete_horizon_category('DS_extracted_run2')

    The project itself can be a reference to an existing project, typically
    the magic ``project`` wording inside RMS python,
    or a file path to a RMS project (for external access).

    Args:
        project (rmsapi.Project or str): Reference to a RMS project
            either an existing instance or a RMS project folder path.
        readonly (bool). Default is False. If readonly, then it cannot be
            saved to this project (which is the case for "secondary" projects).

    Examples::

        import xtgeo
        path = '/some/path/to/rmsproject.rmsx'

        ext = xtgeo.RoxUtils(path, readonly=True)
        # ...do something
        ext.safe_close()

    """

    def __init__(self, project: RmsProjectType, readonly: bool = False) -> None:
        self._project = None

        if rmsapi is None:
            raise RuntimeError(
                "rmsapi package is not available. "
                "Please install it to use RMS features."
            )

        self._version = rmsapi.__version__

        if versionparse(self._version) < versionparse("1.10"):
            raise RuntimeError("XTGeo >= 4.0 requires rmsapi API >= 1.10")

        self._rmsexternal = True

        self._versions = {
            "1.0": ["10.0.x"],
            "1.1": ["10.1.0", "10.1.1", "10.1.2"],
            "1.1.1": ["10.1.3"],
            "1.2": ["11.0.0"],
            "1.2.1": ["11.0.1"],
            "1.3": ["11.1.0", "11.1.1", "11.1.2"],
            "1.4": ["12.0.0", "12.0.1", "12.0.2"],
            "1.5": ["12.1"],
            "1.6": ["13.0"],
            "1.7": ["13.1"],
            "1.7.1": ["13.1.1"],
            "1.7.2": ["13.1.2"],
            "1.8": ["14.0", "14.0.1"],
            "1.9": ["14.1"],
            "1.10": ["14.2"],
            "1.11": ["14.2.1", "14.2.2"],
            "1.12": ["14.5"],
            "1.12.1": ["14.5.0.1"],
            "1.13": ["15", "15.0.1.0"],
        }

        if project is not None and isinstance(project, (str, pathlib.Path)):
            projectname = str(project)
            self._project = rmsapi.Project.open(projectname, readonly=readonly)
            logger.info("Open RMS project from %s", projectname)

        elif isinstance(project, rmsapi.Project):
            # this will happen for _current_ project inside RMS or if
            # project is opened already e.g. by rmsapi.Project.open(). In the latter
            # case, the user should also close the project by project.close() as
            # an explicit action.

            self._rmsexternal = False
            self._project = project
            logger.info("RMS project instance is already open as <%s>", project)
        else:
            raise RuntimeError("Project is not valid")

    @property
    def rmsapiversion(self) -> str:
        """RMS API version (read only)"""
        return self._version

    roxversion = rmsapiversion

    @property
    def project(self) -> RmsProjectType:
        """The RMS project instance (read only)"""
        return self._project

    def safe_close(self) -> None:
        """Close the project but only if rms apps (external) mode, i.e.
        not current RMS project

        In case rmsapi.Project.open() is done explicitly, safe_close() will do nothing.

        """
        if self._rmsexternal and self._project is not None:
            try:
                self._project.close()
                logger.info("RMS project instance is closed")
            except TypeError as msg:
                warnings.warn(str(msg), UserWarning, stacklevel=2)
        else:
            logger.info("Close request, but skip for good reasons...")
            logger.debug(
                "... either in RMS GUI or in a sequence of running rmsapi apps"
            )

    def version_required(self, targetversion: str) -> bool:
        """Defines a minimum RMSAPI version for some feature (True or False).

        Args:
            targetversion (str): Minimum version to compare with.

        Example::

            rox = RmsApiUtils(project)
            if rox.version_required('1.5'):
                somefunction()
            else:
                print('Not supported in this version')

        """
        return versionparse(self._version) >= versionparse(targetversion)

    def rmsversion(self, apiversion: str) -> list[str] | None:
        """Get the actual RMS version(s) given an API version.

        Args:
            apiversion (str): ROXAPI version to ask for

        Returns:
            A list of RMS version(s) for the given API version, None if
                not any match.

        Example::

            rox = RmsApiUtils(project)
            rmsver = rox.rmsversion('1.5')
            print('The supported RMS version are {}'.format(rmsver))

        """

        return self._versions.get(apiversion, None)

    def _create_whatever_category(
        self,
        category: str | list[str],
        stype: str = "horizons",
        domain: Literal["depth", "time", "thickness", "unknown"] = "depth",
        htype: Literal["surface", "lines", "points"] = "surface",
    ) -> None:
        """Create one or more a Horizons/Zones... category entries.

        Args:
            category (str or list): Name(s) of category to make, either as
                a simple string or a list of strings.
            stype (str): 'Super type' in RMS (horizons or zones).
                Default is horizons
            domain (str): The vertical_domian in RMS, 'depth' (default) or 'time'
            htype (str): Horizon type: surface/lines/points
        """

        project = self.project
        # At this point, project is always a Project instance
        # (str/Path converted in __init__)
        assert rmsapi is not None
        assert isinstance(project, rmsapi.Project), "Project must be initialized"

        categories = []

        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for catg in categories:
            geom = rmsapi.GeometryType.surface
            if htype.lower() == "lines":
                geom = rmsapi.GeometryType.polylines
            elif htype.lower() == "points":
                geom = rmsapi.GeometryType.points

            dom = rmsapi.VerticalDomain.depth
            if domain.lower() == "time":
                dom = rmsapi.VerticalDomain.time
            elif domain.lower() == "thickness":
                dom = rmsapi.VerticalDomain.thickness
            elif domain.lower() == "unknown":
                dom = rmsapi.VerticalDomain.unknown

            if stype.lower() == "horizons":
                if catg not in project.horizons.representations:
                    try:
                        project.horizons.representations.create(catg, geom, dom)
                    except Exception as exmsg:
                        print(f"Error: {exmsg}")
                else:
                    print(f"Category <{catg}> already exists")

            elif stype.lower() == "zones":
                if catg not in project.zones.representations:
                    try:
                        project.zones.representations.create(catg, geom, dom)
                    except Exception as exmsg:
                        print(f"Error: {exmsg}")
                else:
                    print(f"Category <{catg}> already exists")

    def _delete_whatever_category(
        self,
        category: str | list[str],
        stype: Literal["horizons", "zones"] = "horizons",
    ) -> None:
        """Delete one or more horizons or zones categories.

        Args:
            category: Name(s) of category to make, either
                as a simple string or a list of strings.
            stype: 'Storage type', in RMS ('horizons' or 'zones').
                Default is 'horizons'
        """

        project = self._project

        assert project is not None
        categories = []

        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for catg in categories:
            stype_lower = stype.lower()
            if stype_lower == "horizons" or stype_lower == "zones":
                try:
                    del project.horizons.representations[catg]
                except KeyError as kerr:
                    if str(kerr) == catg:
                        print(f"Cannot delete {kerr}, does not exist")
            else:
                raise ValueError("Wrong stype applied")

    def _clear_whatever_category(
        self,
        category: str | list[str],
        stype: Literal["horizons", "zones"] = "horizons",
    ) -> None:
        """Clear (or make empty) the content of one or more horizon/zones... categories.

        Args:
            category (str or list): Name(s) of category to empty, either as
                a simple string or a list of strings.
            stype (str): 'Super type' in RMS (horizons or zones).
                Default is horizons

        .. versionadded:: 2.1
        """

        project = self._project
        assert project is not None

        categories = []
        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        xtype = project.horizons
        if stype.lower() == "zones":
            xtype = project.zones

        for catg in categories:
            for xitem in xtype:
                try:
                    item = xtype[xitem.name][catg]
                    item.set_empty()
                except KeyError as kmsg:
                    print(kmsg)

    def create_horizons_category(
        self,
        category: str | list[str],
        domain: Literal["depth", "time", "thickness", "unknown"] = "depth",
        htype: Literal["surface", "lines", "points"] = "surface",
    ) -> None:
        """Create one or more a Horizons category entries.

        Args:
            category: Name(s) of category to make, either as
                a simple string or a list of strings.
            domain: The vertical_domain as 'depth' (default) or 'time'
            htype: Horizon type: surface/lines/points
        """

        self._create_whatever_category(
            category, stype="horizons", domain=domain, htype=htype
        )

    def create_zones_category(
        self,
        category: str | list[str],
        domain: Literal["thickness", "unknown"] = "thickness",
        htype: Literal["surface", "lines", "points"] = "surface",
    ) -> None:
        """Create one or more a Zones category entries.

        Args:
            category (str or list): Name(s) of category to make, either as
                a simple string or a list of strings.
            domain (str): 'thickness' (default) or ...?
            htype (str): Horizon type: surface/lines/points
        """

        self._create_whatever_category(
            category, stype="zones", domain=domain, htype=htype
        )

    def delete_horizons_category(self, category: str | list[str]) -> None:
        """Delete on or more horizons or zones categories"""

        self._delete_whatever_category(category, stype="horizons")

    def delete_zones_category(self, category: str | list[str]) -> None:
        """Delete on or more horizons or zones categories. See previous"""

        self._delete_whatever_category(category, stype="zones")

    def clear_horizon_category(self, category: str | list[str]) -> None:
        """Clear (or make empty) the content of one or more horizon categories.

        Args:
            category (str or list): Name(s) of category to empty, either as
                 a simple string or a list of strings.

        .. versionadded:: 2.1
        """
        self._clear_whatever_category(category, stype="horizons")

    def clear_zone_category(self, category: str | list[str]) -> None:
        """Clear (or make empty) the content of one or more zone categories.

        Args:
            category (str or list): Name(s) of category to empty, either as
                 a simple string or a list of strings.

        .. versionadded:: 2.1
        """
        self._clear_whatever_category(category, stype="zones")


# Backward compatibility alias with deprecation warning. These PendingDeprecationWarning
# warnings can be upgraded to DeprecationWarning in future releases. Currently they are
# are not seen in RMS GUI for users due to warning filters, which is OK.
class RoxUtils(RmsApiUtils):
    """Deprecated: Use RmsApiUtils instead."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "RoxUtils is deprecated and will be removed in a future version. "
            "Use RmsApiUtils instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# For convenience, also create a simple alias without the warning for internal use
_RoxUtils = RmsApiUtils  # Internal alias without deprecation warning
