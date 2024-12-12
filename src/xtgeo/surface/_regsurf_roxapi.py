# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface."""

import os
import tempfile

import numpy as np

from xtgeo.common.log import null_logger
from xtgeo.roxutils import RoxUtils
from xtgeo.roxutils._roxar_loader import roxar

logger = null_logger(__name__)

VALID_STYPES = ["horizons", "zones", "clipboard", "general2d_data", "trends"]


def _check_stypes_names_category(roxutils, stype, name, category):
    """General check of some input values."""
    stype = stype.lower()

    if stype not in VALID_STYPES:
        raise ValueError(
            f"Given stype {stype} is not supported, legal stypes are: {VALID_STYPES}"
        )

    if not name:
        raise ValueError("The name is missing or empty.")

    if stype in ("horizons", "zones") and (name is None or not category):
        raise ValueError(
            "Need to spesify both name and category for horizons and zones"
        )

    if stype == "general2d_data" and not roxutils.version_required("1.6"):
        raise NotImplementedError(
            "API Support for general2d_data is missing in this RMS version"
            f"(current API version is {roxutils.roxversion} - required is 1.6"
        )


def import_horizon_roxapi(
    project, name, category, stype, realisation
):  # pragma: no cover
    """Import a Horizon surface via ROXAR API spec. to xtgeo."""
    roxutils = RoxUtils(project, readonly=True)

    _check_stypes_names_category(roxutils, stype, name, category)

    proj = roxutils.project
    args = _roxapi_import_surface(proj, name, category, stype, realisation)

    roxutils.safe_close()
    return args


def _roxapi_import_surface(
    proj, name, category, stype, realisation
):  # pragma: no cover
    args = {}
    args["name"] = name

    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError(f"Name {name} is not within Horizons")
        if category not in proj.horizons.representations:
            raise ValueError(f"Category {category} is not within Horizons categories")
        try:
            rox = proj.horizons[name][category].get_grid(realisation)
            args.update(_roxapi_horizon_to_xtgeo(rox))
        except KeyError as kwe:
            logger.error(kwe)

    elif stype == "zones":
        if name not in proj.zones:
            raise ValueError(f"Name {name} is not within Zones")
        if category not in proj.zones.representations:
            raise ValueError(f"Category {category} is not within Zones categories")
        try:
            rox = proj.zones[name][category].get_grid(realisation)
            args.update(_roxapi_horizon_to_xtgeo(rox))
        except KeyError as kwe:
            logger.error(kwe)

    elif stype in ("clipboard", "general2d_data"):
        styperef = getattr(proj, stype)
        if category:
            folders = category.split("|" if "|" in category else "/")
            rox = styperef.folders[folders]
        else:
            rox = styperef

        roxsurf = rox[name].get_grid(realisation)
        args.update(_roxapi_horizon_to_xtgeo(roxsurf))

    elif stype == "trends":
        if name not in proj.trends.surfaces:
            logger.info("Name %s is not present in trends", name)
            raise ValueError(f"Name {name} is not within Trends")
        rox = proj.trends.surfaces[name]

        roxsurf = rox.get_grid(realisation)
        args.update(_roxapi_horizon_to_xtgeo(roxsurf))

    else:
        raise ValueError(f"Invalid stype given: {stype}")  # should never reach here
    return args


def _roxapi_horizon_to_xtgeo(rox):  # pragma: no cover
    """Tranforming surfaces from ROXAPI to XTGeo object."""
    # local function
    args = {}
    logger.info("Surface from roxapi to xtgeo...")
    args["xori"], args["yori"] = rox.origin
    args["ncol"], args["nrow"] = rox.dimensions
    args["xinc"], args["yinc"] = rox.increment
    args["rotation"] = rox.rotation

    args["values"] = rox.get_values()
    logger.info("Surface from roxapi to xtgeo... DONE")
    return args


def export_horizon_roxapi(
    self, project, name, category, stype, realisation
):  # pragma: no cover
    """Export (store) a Horizon surface to RMS via ROXAR API spec."""
    roxutils = RoxUtils(project, readonly=False)

    _check_stypes_names_category(roxutils, stype, name, category)

    logger.info("Surface from xtgeo to roxapi...")
    use_srf = self.copy()  # avoid modifying the original instance
    if self.yflip == -1:
        # roxar API cannot handle negative increments
        use_srf.swapaxes()

    use_srf.values = use_srf.values.astype(np.float64)

    # Note that the RMS api does NOT accepts NaNs or Infs, even if behind the mask(!),
    # so we need to replace them with some other value
    if np.isnan(use_srf.values.data).any() or np.isinf(use_srf.values.data).any():
        logger.warning(
            "NaNs or Infs detected in the surface, replacing fill_value "
            "prior to RMS API usage."
        )
        applied_fill_value = np.finfo(np.float64).max  # aka 1.7976931348623157e+308
        all_values = np.ma.filled(use_srf.values, fill_value=applied_fill_value)
        # replace nan in all_values with applied_fill_value
        all_values = np.where(
            np.isnan(all_values) | np.isinf(all_values), applied_fill_value, all_values
        )
        # now remask the array; NB! use _values to avoid the mask being reset
        use_srf._values = np.ma.masked_equal(all_values, applied_fill_value)

    _roxapi_export_surface(
        use_srf, roxutils.project, name, category, stype, realisation
    )

    if roxutils._roxexternal:
        roxutils.project.save()

    logger.info("Surface from xtgeo to roxapi... DONE")
    roxutils.safe_close()


def _roxapi_export_surface(
    self, proj, name, category, stype, realisation
):  # pragma: no cover
    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError(f"Name {name} is not within Horizons")
        if category not in proj.horizons.representations:
            raise ValueError(f"Category {category} is not within Horizons categories")
        try:
            roxroot = proj.horizons[name][category]
            roxg = _xtgeo_to_roxapi_grid(self)
            roxg.set_values(self.values)
            roxroot.set_grid(roxg, realisation=realisation)
        except KeyError as kwe:
            logger.error(kwe)

    elif stype == "zones":
        if name not in proj.zones:
            raise ValueError(f"Name {name} is not within Zones")
        if category not in proj.zones.representations:
            raise ValueError(f"Category {category} is not within Zones categories")
        try:
            roxroot = proj.zones[name][category]
            roxg = _xtgeo_to_roxapi_grid(self)
            roxg.set_values(self.values)
            roxroot.set_grid(roxg)
        except KeyError as kwe:
            logger.error(kwe)

    elif stype in ("clipboard", "general2d_data"):
        folders = []
        if category:
            folders = category.split("|" if "|" in category else "/")
        styperef = getattr(proj, stype)
        if folders:
            styperef.folders.create(folders)

        roxroot = styperef.create_surface(name, folders)
        roxg = _xtgeo_to_roxapi_grid(self)
        roxg.set_values(self.values)
        roxroot.set_grid(roxg)

    elif stype == "trends":
        if name not in proj.trends.surfaces:
            logger.info("Name %s is not present in trends", name)
            raise ValueError(
                f"Name {name} is not within Trends (it must exist in advance!)"
            )
        # here a workound; trends.surfaces are read-only in Roxar API, but is seems
        # that load() in RMS is an (undocumented?) workaround...

        roxsurf = proj.trends.surfaces[name]
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            self.to_file(os.path.join(tmpdir, "gxx.gri"), fformat="irap_binary")

            roxsurf.load(os.path.join(tmpdir, "gxx.gri"), roxar.FileFormat.ROXAR_BINARY)

    else:
        raise ValueError(f"Invalid stype given: {stype}")  # should never reach here


def _xtgeo_to_roxapi_grid(self):  # pragma: no cover
    # Create a 2D grid

    return roxar.RegularGrid2D.create(
        x_origin=self.xori,
        y_origin=self.yori,
        i_inc=self.xinc,
        j_inc=self.yinc,
        ni=self.ncol,
        nj=self.nrow,
        rotation=self.rotation,
    )
