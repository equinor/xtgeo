# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface"""

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()


def import_horizon_roxapi(surf, project, name, category,
                          realisation):
    """Import a Horizon surface via ROXAR API spec."""
    import roxar

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            try:
                rox = proj.horizons[name][category].get_grid(realisation)
                _roxapi_horizon_to_xtgeo(surf, rox)
            except KeyError as ke:
                logger.error(ke)
    else:
        rox = project.horizons[name][category].get_grid(realisation)
        _roxapi_horizon_to_xtgeo(surf, rox)

    return surf


def export_horizon_roxapi(surf, project, name, category,
                          realisation):
    """Import a Horizon surface via ROXAR API spec."""
    import roxar

    pass

    # if project is not None and isinstance(project, str):
    #     projectname = project
    #     with roxar.Project.open_import(projectname) as proj:
    #         try:
    #             rox = proj.horizons[name][category].get_grid(realisation)
    #             surf._roxapi_horizon_to_xtgeo(rox)
    #         except KeyError as ke:
    #             logger.error(ke)
    # else:
    #     rox = project.horizons[name][category].get_grid(realisation)
    #     surf._roxapi_horizon_to_xtgeo(rox)

    # return surf


def _roxapi_horizon_to_xtgeo(surf, rox):
    """Tranforming surfaces from ROXAPI to XTGeo object."""
    # local function
    logger.info('Surface from roxapi to xtgeo...')
    surf._xori, surf._yori = rox.origin
    surf._ncol, surf._nrow = rox.dimensions
    surf._xinc, surf._yinc = rox.increment
    surf._rotation = rox.rotation
    surf._values = rox.get_values()
