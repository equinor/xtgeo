# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""
import os
import tempfile
import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access


def import_xyz_roxapi(self, project, name, category, stype, realisation, attributes):
    """Import a Points or Polygons item via ROXAR API spec.

    'Import' here means transfer of data from Roxar API memory space to
    XTGeo memory space.
    """

    rox = RoxUtils(project)

    if attributes:
        _roxapi_import_xyz_viafile(self, rox, name, category, stype, realisation)
    else:
        _roxapi_import_xyz(self, rox.project, name, category, stype, realisation)


def _roxapi_import_xyz_viafile(self, rox, name, category, stype, realisation):

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.critical("Cannot import module roxar")
        raise

    self._name = name
    proj = rox.project

    if not _check_category_etc(proj, name, category, stype, realisation):
        raise RuntimeError("Something is very wrong...")

    roxxyz = _get_roxitem(proj, name, category, stype, mode="get")

    try:
        # make a temporary folder and work within the with.. block
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            roxxyz.save(
                os.path.join(tmpdir, "generic.rmsattr"), roxar.FileFormat.RMS_POINTS
            )
            self.from_file(os.path.join(tmpdir, "generic.rmsattr"), fformat="rms_attr")

    except KeyError as kwe:
        logger.error(kwe)


def _roxapi_import_xyz(self, proj, name, category, stype, realisation):

    self._name = name

    if not _check_category_etc(proj, name, category, stype, realisation):
        raise RuntimeError("Something is very wrong...")

    roxxyz = _get_roxitem(proj, name, category, stype, mode="get")

    try:
        roxitem = roxxyz.get_values(realisation)
        _roxapi_xyz_to_xtgeo(self, roxitem)

    except KeyError as kwe:
        logger.error(kwe)


def _roxapi_xyz_to_xtgeo(self, roxxyz):
    """Tranforming some XYZ from ROXAPI to XTGeo object."""

    # In ROXAPI, polygons is a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")
    cnames = ["X_UTME", "Y_UTMN", "Z_TVDSS"]

    if self._ispolygons and isinstance(roxxyz, list):
        # polylines/-gons
        dfs = []
        for idx, poly in enumerate(roxxyz):
            dataset = pd.DataFrame.from_records(poly, columns=cnames)
            dataset["POLY_ID"] = idx
            dfs.append(dataset)

        dfr = pd.concat(dfs)

    elif not self._ispolygons and isinstance(roxxyz, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxxyz, columns=cnames)
        self._ispolygons = False

    else:
        raise RuntimeError("Unknown error in getting data from Roxar")

    self._df = dfr


def export_xyz_roxapi(
    self, project, name, category, stype, pfilter, realisation, attributes
):
    """Export (store) a XYZ item to RMS via ROXAR API spec."""

    rox = RoxUtils(project)

    if stype == "horizon_picks":
        _roxapi_export_xyz_hpicks(
            self, rox, name, category, stype, realisation, attributes
        )

    if attributes:
        _roxapi_export_xyz_viafile(
            self, rox, name, category, stype, pfilter, realisation, attributes
        )
    else:
        _roxapi_export_xyz(self, rox, name, category, stype, pfilter, realisation)


def _roxapi_export_xyz_hpicks(
    self, rox, name, category, stype, realisation, attributes
):
    """
    Export/store as RMS horizon picks; this is only valid if points belong to wells
    """
    # need to think on design!
    raise NotImplementedError


def _roxapi_export_xyz_viafile(
    self, rox, name, category, stype, pfilter, realisation, attributes
):
    """Set points/polys within RMS with attributes, using file workaround"""

    logger.warning("Realisation %s not in use", realisation)

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.critical("Cannot import roxar")
        raise

    proj = rox.project

    if not _check_category_etc:
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(proj, name, category, stype)

    # make a temporary folder and work within the with.. block
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Made a tmp folder: %s", tmpdir)
        ncount = self.to_file(
            os.path.join(tmpdir, "generic.rmsattr"),
            fformat="rms_attr",
            pfilter=pfilter,
            attributes=attributes,
        )

        if ncount:
            roxxyz.load(
                os.path.join(tmpdir, "generic.rmsattr"), roxar.FileFormat.RMS_POINTS
            )


def _roxapi_export_xyz(self, rox, name, category, stype, pfilter, realisation):

    logger.warning("Realisation %s not in use", realisation)

    proj = rox.project
    if not _check_category_etc:
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(proj, name, category, stype)

    # pylint: disable=len-as-condition
    if self.dataframe is None or len(self.dataframe.index) == 0:
        return

    df = self.dataframe.copy()
    # apply pfilter if any
    if pfilter:
        for key, val in pfilter.items():
            if key in df.columns:
                df = df.loc[df[key].isin(val)]
            else:
                raise KeyError(
                    "The requested pfilter key {} was not "
                    "found in dataframe. Valid keys are "
                    "{}".format(key, df.columns)
                )

    if self._ispolygons:
        arrxyz = []
        polys = df.groupby(self.pname)
        for _id, grp in polys:
            arr = np.stack([grp[self.xname], grp[self.yname], grp[self.zname]], axis=1)
            arrxyz.append(arr)
    else:
        xyz = df
        arrxyz = np.stack([xyz[self.xname], xyz[self.yname], xyz[self.zname]], axis=1)
    try:
        roxxyz.set_values(arrxyz)
    except KeyError as kwe:
        logger.error(kwe)


def _check_category_etc(
    proj, name, category, stype, realisation
):  # pylint: disable=too-many-branches
    """Helper to check if valid placeholder' whithin RMS."""

    logger.warning("Realisation %s not in use", realisation)

    if stype == "horizons":
        if name not in proj.horizons:
            raise ValueError("Name {} is not within Horizons".format(name))
        if category not in proj.horizons.representations:
            raise ValueError(
                "Category {} is not within Horizons categories".format(category)
            )
    elif stype == "zones":
        if name not in proj.zones:
            raise ValueError("Name {} is not within Zones".format(name))
        if category not in proj.zones.representations:
            raise ValueError(
                "Category {} is not within Zones categories".format(category)
            )
    elif stype == "faults":
        if name not in proj.faults:
            raise ValueError("Name {} is not within Faults".format(name))
        if category not in proj.zones.representations:
            raise ValueError(
                "Category {} is not within Faults categories".format(category)
            )
    elif stype == "clipboard":
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
            roxxyz = proj.clipboard.folders[folders]
        else:
            roxxyz = proj.clipboard
        if name not in roxxyz:
            raise ValueError("Name {} is not within Clipboard...".format(name))
    else:
        raise ValueError("Invalid stype")

    return True


def _get_roxitem(proj, name, category, stype, mode="set"):

    if stype == "horizons":
        roxxyz = proj.horizons[name][category]
    elif stype == "zones":
        roxxyz = proj.zones[name][category]
    elif stype == "faults":
        roxxyz = proj.faults[name][category]
    elif stype == "clipboard":
        if category:
            if "|" in category:
                folders = category.split("|")
            else:
                folders = category.split("/")
            roxxyz = proj.clipboard.folders[folders]
        else:
            roxxyz = proj.clipboard

        if mode == "get":
            roxxyz = roxxyz[name]

    else:
        roxxyz = None
        raise ValueError("Unsupported stype: {}".format(stype))

    return roxxyz
