# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface"""

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()


def import_horizon_roxapi(self, project, name, category,
                          realisation):
    """Import a Horizon surface via ROXAR API spec."""
    import roxar

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            try:
                rox = proj.horizons[name][category].get_grid(realisation)
                _roxapi_horizon_to_xtgeo(self, rox)
            except KeyError as ke:
                logger.error(ke)
    else:
        rox = project.horizons[name][category].get_grid(realisation)
        _roxapi_horizon_to_xtgeo(self, rox)

    return self


def export_horizon_roxapi(self, project, name, category,
                          realisation):
    """Export (store) a Horizon surface to RMS via ROXAR API spec."""
    # import roxar

    raise NotImplementedError

    # if project is not None and isinstance(project, str):
    #     projectname = project
    #     with roxar.Project.open_import(projectname) as proj:
    #         try:
    #             rox = proj.horizons[name][category].get_grid(realisation)
    #             self._roxapi_horizon_to_xtgeo(rox)
    #         except KeyError as ke:
    #             logger.error(ke)
    # else:
    #     rox = project.horizons[name][category].get_grid(realisation)
    #     self._roxapi_horizon_to_xtgeo(rox)

    # return self


def _roxapi_horizon_to_xtgeo(self, rox):
    """Tranforming surfaces from ROXAPI to XTGeo object."""
    # local function
    logger.info('Surface from roxapi to xtgeo...')
    self._xori, self._yori = rox.origin
    self._ncol, self._nrow = rox.dimensions
    self._xinc, self._yinc = rox.increment
    self._rotation = rox.rotation
    self._values = rox.get_values()
