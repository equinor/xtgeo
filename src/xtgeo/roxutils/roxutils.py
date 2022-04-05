# -*- coding: utf-8 -*-
"""Module for simplifying various operation in the Roxar python interface."""


from distutils.version import StrictVersion

# from pkg_resources import parse_version as pver (ALT)

try:
    import _roxar
    import roxar
except ImportError:
    pass

from xtgeo.common import XTGeoDialog

from . import _roxutils_etc

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class RoxUtils(object):

    """Class RoxUtils, for accessing project level methods::

     import xtgeo

     xr = xtgeo.RoxUtils(project)
     xr.create_horizon_category('DS_extracted_run3')
     xr.delete_horizon_category('DS_extracted_run2')

    The project itself can be a reference to an existing project, typically
    the magic ``project`` wording inside RMS python,
    or a file path to a RMS project (for external access).

    Args:
        project (_roxar.Project or str): Reference to a RMS project
            either an existing instance or a RMS project folder path.
        readonly (bool). Default is False. If readonly, then it cannot be
            saved to this project (which is the case for "secondary" projects).

    Examples::

        import xgeo
        path = '/some/path/to/rmsprject.rmsx'

        ext = xtgeo.RoxUtils(path, readonly=True)
        # ...do something
        ext.safe_close()

    """

    def __init__(self, project, readonly=False):
        self._project = None

        self._version = roxar.__version__
        self._roxexternal = True

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
        }

        if project is not None and isinstance(project, str):
            projectname = project
            if readonly:
                self._project = roxar.Project.open_import(projectname)
            else:
                self._project = roxar.Project.open(projectname)
            logger.info("Open RMS project from %s", projectname)

        elif isinstance(project, _roxar.Project):
            # this will happen for _current_ project inside RMS or if
            # project is opened already e.g. by roxar.Project.open(). In the latter
            # case, the user should also close the project by project.close() as
            # an explicit action.

            self._roxexternal = False
            self._project = project
            logger.info("RMS project instance is already open as <%s>", project)
        else:
            raise RuntimeError("Project is not valid")

    @property
    def roxversion(self):
        """Roxar API version (read only)"""
        return self._version

    @property
    def project(self):
        """The Roxar project instance (read only)"""
        return self._project

    def safe_close(self):
        """Close the project but only if roxarapps (external) mode, i.e.
        not current RMS project

        In case roxar.Project.open() is done explicitly, safe_close() will do nothing.

        """
        if self._roxexternal:
            try:
                self._project.close()
                logger.info("RMS project instance is closed")
            except TypeError as msg:
                xtg.warn(msg)
        else:
            logger.info("Close request, but skip for good reasons...")
            logger.debug("... either in RMS GUI or in a sequence of running roxarapps")

    def version_required(self, targetversion):
        """Defines a minimum ROXAPI version for some feature (True or False).

        Args:
            targetversion (str): Minimum version to compare with.

        Example::

            rox = RoxUtils(project)
            if rox.version_required('1.2'):
                somefunction()
            else:
                print('Not supported in this version')

        """
        return StrictVersion(self._version) >= StrictVersion(targetversion)

    def rmsversion(self, apiversion):
        """Get the actual RMS version(s) given an API version.

        Args:
            apiversion (str): ROXAPI version to ask for

        Returns:
            A list of RMS version(s) for the given API version, None if
                not any match.

        Example::

            rox = RoxUtils(project)
            rmsver = rox.rmsversion('1.2')
            print('The supported RMS version are {}'.format(rmsver))

        """

        return self._versions.get(apiversion, None)

    def create_horizons_category(self, category, domain="depth", htype="surface"):
        """Create one or more a Horizons category entries.

        Args:
            category (str or list): Name(s) of category to make, either as
                a simple string or a list of strings.
            domain (str): 'depth' (default) or 'time'
            htype (str): Horizon type: surface/lines/points
        """

        _roxutils_etc.create_whatever_category(
            self, category, stype="horizons", domain=domain, htype=htype
        )

    def create_zones_category(self, category, domain="thickness", htype="surface"):
        """Create one or more a Horizons category entries.

        Args:
            category (str or list): Name(s) of category to make, either as
                a simple string or a list of strings.
            domain (str): 'thickness' (default) or ...?
            htype (str): Horizon type: surface/lines/points
        """

        _roxutils_etc.create_whatever_category(
            self, category, stype="zones", domain=domain, htype=htype
        )

    def delete_horizons_category(self, category):
        """Delete on or more horizons or zones categories"""

        _roxutils_etc.delete_whatever_category(self, category, stype="horizons")

    def delete_zones_category(self, category):
        """Delete on or more horizons or zones categories. See previous"""

        _roxutils_etc.delete_whatever_category(self, category, stype="zones")

    def clear_horizon_category(self, category):
        """Clear (or make empty) the content of one or more horizon categories.

        Args:
            category (str or list): Name(s) of category to empty, either as
                 a simple string or a list of strings.

        .. versionadded:: 2.1
        """
        _roxutils_etc.clear_whatever_category(self, category, stype="horizons")

    def clear_zone_category(self, category):
        """Clear (or make empty) the content of one or more zone categories.

        Args:
            category (str or list): Name(s) of category to empty, either as
                 a simple string or a list of strings.

        .. versionadded:: 2.1
        """
        _roxutils_etc.clear_whatever_category(self, category, stype="zones")
