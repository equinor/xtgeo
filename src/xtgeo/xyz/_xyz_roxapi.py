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


def import_xyz_roxapi(
    self, project, name, category, stype, realisation, attributes
):  # pragma: no cover
    """Import a Points or Polygons item via ROXAR API spec.

    'Import' here means transfer of data from Roxar API memory space to
    XTGeo memory space.
    """

    rox = RoxUtils(project)

    if stype == "clipboard" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        msg = (
            "Not supported in this ROXAPI version. Points/polygons access "
            "to clipboard requires RMS {}".format(minimumrms)
        )
        raise NotImplementedError(msg)

    if attributes:
        _roxapi_import_xyz_viafile(self, rox, name, category, stype, realisation)
    else:
        _roxapi_import_xyz(self, rox.project, name, category, stype, realisation)


def _roxapi_import_xyz_viafile(
    self, rox, name, category, stype, realisation
):  # pragma: no cover

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.critical("Cannot import module roxar")
        raise

    self._name = name
    proj = rox.project

    if not _check_category_etc(proj, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="get")

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


def _roxapi_import_xyz(
    self, proj, name, category, stype, realisation
):  # pragma: no cover
    """From RMS to XTGeo"""
    self._name = name

    if not _check_category_etc(proj, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="get")

    try:
        roxitem = roxxyz.get_values(realisation)
        _roxapi_xyz_to_xtgeo(self, roxitem)

    except KeyError as kwe:
        logger.error(kwe)


def _roxapi_xyz_to_xtgeo(self, roxxyz):  # pragma: no cover
    """Tranforming some XYZ from ROXAPI to XTGeo object."""

    # In ROXAPI, polygons is a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")
    cnames = [self._xname, self._yname, self._zname]

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
):  # pragma: no cover
    """Export (store) a XYZ item from XTGeo to RMS via ROXAR API spec."""

    rox = RoxUtils(project)

    if stype == "clipboard" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        msg = (
            "Not supported in this ROXAPI version. Points/polygons access "
            "to clipboard requires RMS {}".format(minimumrms)
        )
        raise NotImplementedError(msg)

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
):  # pragma: no cover
    """
    Export/store as RMS horizon picks; this is only valid if points belong to wells
    """
    # need to think on design!
    raise NotImplementedError


def _roxapi_export_xyz_viafile(
    self, rox, name, category, stype, pfilter, realisation, attributes
):  # pragma: no cover
    """Set points/polys within RMS with attributes, using file workaround"""

    logger.warning("Realisation %s not in use", realisation)

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.critical("Cannot import roxar")
        raise

    proj = rox.project

    if not _check_category_etc(proj, name, category, stype, realisation, mode="set"):
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="set")

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


def _roxapi_export_xyz(
    self, rox, name, category, stype, pfilter, realisation
):  # pragma: no cover

    logger.warning("Realisation %s not in use", realisation)

    proj = rox.project
    if not _check_category_etc(proj, name, category, stype, realisation, mode="set"):
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="set")

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
    proj, name, category, stype, realisation, mode="get"
):  # pylint: disable=too-many-branches  pragma: no cover

    """Helper to check if valid placeholder' whithin RMS."""

    logger.warning("Realisation %s not in use", realisation)

    stypedict = {"horizons": proj.horizons, "zones": proj.zones, "faults": proj.faults}

    if stype in stypedict.keys():
        if name not in stypedict[stype]:
            logger.error("Cannot access name in stype=%s: %s", stype, name)
            return False
        if category not in stypedict[stype].representations:
            logger.error("Cannot access category in stype=%s: %s", stype, category)
            return False

    elif stype == "clipboard" and mode == "get":
        folders = None
        if category:
            if isinstance(category, list):
                folders = category
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    "Cannot parse category: {}, see documentation!".format(category)
                )
            try:
                roxxyz = proj.clipboard.folders[folders]
            except KeyError as keyerr:
                logger.error(
                    "Cannot access clipboards folder (not existing?): %s", keyerr
                )
                return False
        else:
            roxxyz = proj.clipboard

        if name not in roxxyz.keys():
            raise ValueError("Name {} is not within Clipboard...".format(name))

    elif stype == "clipboard" and mode == "set":
        logger.info("No need to check clipboard while setting data")
    else:
        raise ValueError("Invalid stype")

    return True


def _get_roxitem(self, proj, name, category, stype, mode="set"):  # pragma: no cover
    # pylint: disable=too-many-branches
    if stype == "horizons":
        roxxyz = proj.horizons[name][category]
    elif stype == "zones":
        roxxyz = proj.zones[name][category]
    elif stype == "faults":
        roxxyz = proj.faults[name][category]
    elif stype == "clipboard":
        folders = None
        roxxyz = proj.clipboard
        if category:
            if isinstance(category, list):
                folders = category
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    "Cannot parse category: {}, see documentation!".format(category)
                )

            if mode == "get":
                roxxyz = proj.clipboard.folders[folders]

        if mode == "get":
            roxxyz = roxxyz[name]

        elif mode == "set":

            # clipboard folders will be created if not present, and overwritten else
            if self._ispolygons:
                roxxyz = proj.clipboard.create_polylines(name, folders)
            else:
                roxxyz = proj.clipboard.create_points(name, folders)

    else:
        roxxyz = None
        raise ValueError("Unsupported stype: {}".format(stype))

    return roxxyz
