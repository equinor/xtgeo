# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface"""
import numpy as np

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access


def import_horizon_roxapi(self, project, name, category, stype, realisation):
    """Import a Horizon surface via ROXAR API spec."""
    import roxar  # pylint: disable=import-error

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            _roxapi_import_surface(self, proj, name, category, stype, realisation)
    else:
        _roxapi_import_surface(self, project, name, category, stype, realisation)


def _roxapi_import_surface(self, proj, name, category, stype, realisation):

    self._name = name

    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError("Name {} is not within Horizons".format(name))
        if category not in proj.horizons.representations:
            raise ValueError(
                "Category {} is not within Horizons categories".format(category)
            )
        try:
            rox = proj.horizons[name][category].get_grid(realisation)
            _roxapi_horizon_to_xtgeo(self, rox)
        except KeyError as kwe:
            logger.error(kwe)
    elif stype == "zones":
        if name not in proj.zones:
            raise ValueError("Name {} is not within Zones".format(name))
        if category not in proj.zones.representations:
            raise ValueError(
                "Category {} is not within Zones categories".format(category)
            )
        try:
            rox = proj.zones[name][category].get_grid(realisation)
            _roxapi_horizon_to_xtgeo(self, rox)
        except KeyError as kwe:
            logger.error(kwe)
    else:
        raise ValueError("Invalid stype")


def _roxapi_horizon_to_xtgeo(self, rox):
    """Tranforming surfaces from ROXAPI to XTGeo object."""
    # local function
    logger.info("Surface from roxapi to xtgeo...")
    self._xori, self._yori = rox.origin
    self._ncol, self._nrow = rox.dimensions
    self._xinc, self._yinc = rox.increment
    self._rotation = rox.rotation

    # since XTGeo is F order, while RMS is C order...
    self._values = np.asanyarray(rox.get_values(), order="C")


def export_horizon_roxapi(self, project, name, category, stype, realisation):
    """Export (store) a Horizon surface to RMS via ROXAR API spec."""
    import roxar  # pylint: disable=import-error

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            _roxapi_export_surface(self, proj, name, category, stype, realisation)
    else:
        _roxapi_export_surface(self, project, name, category, stype, realisation)


def _roxapi_export_surface(self, proj, name, category, stype, realisation):
    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError("Name {} is not within Horizons".format(name))
        if category not in proj.horizons.representations:
            raise ValueError(
                "Category {} is not within Horizons categories".format(category)
            )
        try:
            roxroot = proj.horizons[name][category]
            rox = _xtgeo_to_roxapi_grid(self)
            rox.set_values(np.asanyarray(self.values, order="C"))
            roxroot.set_grid(rox, realisation=realisation)
        except KeyError as kwe:
            logger.error(kwe)

    elif stype == "zones":
        if name not in proj.zones:
            raise ValueError("Name {} is not within Zones".format(name))
        if category not in proj.zones.representations:
            raise ValueError(
                "Category {} is not within Zones categories".format(category)
            )
        try:
            roxroot = proj.zones[name][category]
            rox = _xtgeo_to_roxapi_grid(self)
            rox.set_values(np.asanyarray(self.values, order="C"))
            roxroot.set_grid(rox)
        except KeyError as kwe:
            logger.error(kwe)

    else:
        raise ValueError("Invalid stype")


def _xtgeo_to_roxapi_grid(self):
    # Create a 2D grid
    import roxar  # pylint: disable=import-error

    grid2d = roxar.RegularGrid2D.create(
        x_origin=self.xori,
        y_origin=self.yori,
        i_inc=self.xinc,
        j_inc=self.yinc,
        ni=self.ncol,
        nj=self.nrow,
        rotation=self.rotation,
    )

    return grid2d
