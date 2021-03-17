# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface."""
import numpy as np
from xtgeo.common import XTGeoDialog
from xtgeo import RoxUtils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_horizon_roxapi(
    self, project, name, category, stype, realisation
):  # pragma: no cover
    """Import a Horizon surface via ROXAR API spec."""
    rox = RoxUtils(project, readonly=True)

    proj = rox.project

    _roxapi_import_surface(self, proj, name, category, stype, realisation)

    rox.safe_close()


def _roxapi_import_surface(
    self, proj, name, category, stype, realisation
):  # pragma: no cover

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

    elif stype == "clipboard":
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
            rox = proj.clipboard.folders[folders]
        else:
            rox = proj.clipboard
        roxsurf = rox[name].get_grid(realisation)
        _roxapi_horizon_to_xtgeo(self, roxsurf)

    else:
        raise ValueError("Invalid stype")


def _roxapi_horizon_to_xtgeo(self, rox):  # pragma: no cover
    """Tranforming surfaces from ROXAPI to XTGeo object."""
    # local function
    logger.info("Surface from roxapi to xtgeo...")
    self._xori, self._yori = rox.origin
    self._ncol, self._nrow = rox.dimensions
    self._xinc, self._yinc = rox.increment
    self._rotation = rox.rotation
    self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

    self._values = rox.get_values()
    logger.info("Surface from roxapi to xtgeo... DONE")


def export_horizon_roxapi(
    self, project, name, category, stype, realisation
):  # pragma: no cover
    """Export (store) a Horizon surface to RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

    logger.info("Surface from xtgeo to roxapi...")
    _roxapi_export_surface(self, rox.project, name, category, stype, realisation)

    if rox._roxexternal:
        rox.project.save()

    logger.info("Surface from xtgeo to roxapi... DONE")
    rox.safe_close()


def _roxapi_export_surface(
    self, proj, name, category, stype, realisation
):  # pragma: no cover
    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError("Name {} is not within Horizons".format(name))
        if category not in proj.horizons.representations:
            raise ValueError(
                "Category {} is not within Horizons categories".format(category)
            )
        try:
            roxroot = proj.horizons[name][category]
            roxg = _xtgeo_to_roxapi_grid(self)
            roxg.set_values(self.values)
            roxroot.set_grid(roxg, realisation=realisation)
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
            roxg = _xtgeo_to_roxapi_grid(self)
            roxg.set_values(self.values)
            roxroot.set_grid(roxg)
        except KeyError as kwe:
            logger.error(kwe)

    elif stype == "clipboard":
        folders = []
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
        if folders:
            proj.clipboard.folders.create(folders)

        roxroot = proj.clipboard.create_surface(name, folders)
        roxg = _xtgeo_to_roxapi_grid(self)
        roxg.set_values(self.values)
        roxroot.set_grid(roxg)
    else:
        raise ValueError("Invalid stype")


def _xtgeo_to_roxapi_grid(self):  # pragma: no cover
    # Create a 2D grid
    import roxar  # pylint: disable=import-error, import-outside-toplevel

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
