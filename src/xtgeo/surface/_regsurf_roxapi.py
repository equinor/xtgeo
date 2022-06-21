# coding: utf-8
"""Roxar API functions for XTGeo RegularSurface."""
import os
import tempfile

from xtgeo import RoxUtils
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

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
            raise ValueError("Name {} is not within Horizons".format(name))
        if category not in proj.horizons.representations:
            raise ValueError(
                "Category {} is not within Horizons categories".format(category)
            )
        try:
            rox = proj.horizons[name][category].get_grid(realisation)
            args.update(_roxapi_horizon_to_xtgeo(rox))
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
            args.update(_roxapi_horizon_to_xtgeo(rox))
        except KeyError as kwe:
            logger.error(kwe)

    elif stype in ("clipboard", "general2d_data"):
        styperef = getattr(proj, stype)
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
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
    _roxapi_export_surface(self, roxutils.project, name, category, stype, realisation)

    if roxutils._roxexternal:
        roxutils.project.save()

    logger.info("Surface from xtgeo to roxapi... DONE")
    roxutils.safe_close()


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

    elif stype in ("clipboard", "general2d_data"):
        folders = []
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
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
        try:
            import roxar  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "roxar not available, this functionality is not available"
            ) from err

        roxsurf = proj.trends.surfaces[name]
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            self.to_file(os.path.join(tmpdir, "gxx.gri"), fformat="irap_binary")

            roxsurf.load(os.path.join(tmpdir, "gxx.gri"), roxar.FileFormat.ROXAR_BINARY)

    else:
        raise ValueError(f"Invalid stype given: {stype}")  # should never reach here


def _xtgeo_to_roxapi_grid(self):  # pragma: no cover
    # Create a 2D grid
    try:
        import roxar  # pylint: disable=import-error, import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err

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
