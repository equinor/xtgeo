# -*- coding: utf-8 -*-
"""Module for simplifying various operation in the Roxar python interface."""

from __future__ import division, absolute_import
from __future__ import print_function

from distutils.version import StrictVersion

# from pkg_resources import parse_version as pver (ALT)

try:
    import roxar
    import _roxar
except ImportError:
    pass

from xtgeo.common import XTGeoDialog

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
            either a instance or a RMS project folder path.
        readonly (bool). Default is False. If readonly, then it cannot be
            saved to this project (which is the case for "secondary" projects).

    Examples::

        import xgeo
        path = '/some/path/to/rmsprject.rmsx'

        ext = xtgeo.RoxUtils(path, readonly=True)
        # ...do somthing
        ext.safe_close()

    """

    def __init__(self, project, readonly=False):
        self._project = None

        self._version = roxar.__version__
        self._roxarapps = True

        self._versions = {
            "1.0": ["10.0.x"],
            "1.1": ["10.1.0", "10.1.1", "10.1.2"],
            "1.1.1": ["10.1.3"],
            "1.2": ["11.0.0"],
            "1.2.1": ["11.0.1"],
            "1.3": ["11.1.0"],
        }

        if project is not None and isinstance(project, str):
            projectname = project
            if readonly:
                self._project = roxar.Project.open_import(projectname)
            else:
                self._project = roxar.Project.open(projectname)
            logger.info("Open RMS project from %s", projectname)

        elif isinstance(project, _roxar.Project):
            # this will only happen for _current_ project inside RMS
            self._roxarapps = False
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
        not current RMS project.
        """
        if self._roxarapps:
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

    def create_horizon_category(
        self, category, stype="horizons", domain="depth", htype="surface"
    ):
        """Create one or more a Horizons category entries.

        Args:
            category (str or list): Name(s) of category to make, either as
                 a simple string or a list of strings.
            stype (str): 'Super type' in RMS (horizons or zones).
                Default is horizons
            domain (str): 'depth' (default) or 'time'
            htype (str): Horizon type: surface/lines/points
        """

        project = self.project
        categories = []

        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for catg in categories:
            geom = roxar.GeometryType.surface
            if htype.lower() == "lines":
                geom = roxar.GeometryType.lines
            elif htype.lower() == "points":
                geom = roxar.GeometryType.points

            dom = roxar.VerticalDomain.depth
            if domain.lower() == "time":
                dom = roxar.GeometryType.lines

            if stype.lower() == "horizons":
                if catg not in project.horizons.representations:
                    try:
                        project.horizons.representations.create(catg, geom, dom)
                    except Exception as exmsg:  # pylint: disable=broad-except
                        print("Error: {}".format(exmsg))
                else:
                    print("Category <{}> already exists".format(catg))

            elif stype.lower() == "zones":
                if catg not in project.zones.representations:
                    try:
                        project.zones.representations.create(catg, geom, dom)
                    except Exception as exmsg:  # pylint: disable=broad-except
                        print("Error: {}".format(exmsg))
                else:
                    print("Category <{}> already exists".format(catg))

    def create_zones_category(self, category, domain="thickness", htype="surface"):
        """Same as create_horizon_category, but with stype='zones'."""

        self.create_horizon_category(
            category, stype="zones", domain=domain, htype=htype
        )

    def delete_horizon_category(self, category, stype="horizons"):
        """Delete onelayergrid or more horizons or zones categories.

        Args:
            category (str or list): Name(s) of category to make, either
                as a simple string or a list of strings.
            stype (str): 'Super type', in RMS ('horizons' or 'zones').
                Default is 'horizons'
        """

        project = self.project
        categories = []

        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for catg in categories:
            if stype.lower() == "horizons":
                try:
                    del project.horizons.representations[catg]
                except KeyError as kerr:
                    if kerr == catg:
                        print("Cannot delete {}, does not exist".format(kerr))
            elif stype.lower() == "zones":
                try:
                    del project.horizons.representations[catg]
                except KeyError as kerr:
                    if kerr == catg:
                        print("Cannot delete {}, does not exist".format(kerr))
            else:
                raise ValueError("Wrong stype applied")

    def delete_zones_category(self, category):
        """Delete on or more horizons or zones categories. See previous"""

        self.delete_horizon_category(category, stype="zones")
