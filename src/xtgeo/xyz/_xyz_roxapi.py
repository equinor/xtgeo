# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""
import os
import tempfile

import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog, _XTGeoFile
from xtgeo.roxutils import RoxUtils
from xtgeo.xyz import _xyz_io

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access


def import_xyz_roxapi(
    project, name, category, stype, realisation, attributes, is_polygons
):  # pragma: no cover
    """Import a Points or Polygons item via ROXAR API spec.

    'Import' means transfer of data from Roxar API memory space to XTGeo memory space.

    ~ a part of a classmethod, and it will return the following kwargs to __init__()::

        xname
        yname
        zname
        pname # optional for Polygons
        name
        dataframe
        values=None  # since dataframe is set
        filesrc
    """

    rox = RoxUtils(project, readonly=True)

    if stype == "clipboard" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        msg = (
            "Not supported in this ROXAPI version. Points/polygons access "
            "to clipboard requires RMS {}".format(minimumrms)
        )
        raise NotImplementedError(msg)

    if attributes:
        result = _roxapi_import_xyz_viafile(
            rox, name, category, stype, realisation, is_polygons
        )
    else:
        result = _roxapi_import_xyz(
            rox, name, category, stype, realisation, is_polygons
        )

    rox.safe_close()
    return result


def _roxapi_import_xyz_viafile(
    rox, name, category, stype, realisation, is_polygons
):  # pragma: no cover
    """Read XYZ from file due to a missing feature in Roxar API wrt attributes.

    However, attributes will be present in Roxar API from RMS version 12, and this
    routine should be replaced!
    """

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err

    if not _check_category_etc(rox.project, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    rox_xyz = _get_roxxyz(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    kwargs = {}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            tfile = os.path.join(tmpdir, "generic.rmsattr")
            rox_xyz.save(tfile, roxar.FileFormat.RMS_POINTS)
            pfile = _XTGeoFile(tfile)
            kwargs = _xyz_io.import_rms_attr(pfile)

    except KeyError as kwe:
        logger.error(kwe)

    return kwargs


def _roxapi_import_xyz(
    rox, name, category, stype, realisation, is_polygons
):  # pragma: no cover
    """From RMS to XTGeo"""
    kwargs = {}

    if not _check_category_etc(rox.project, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    kwargs["xname"] = "X_UTME"
    kwargs["yname"] = "Y_UTMN"
    kwargs["zname"] = "Z_TVDSS"

    if is_polygons:
        kwargs["pname"] = "POLY_ID"

    roxxyz = _get_roxxyz(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    values = _get_roxvalues(roxxyz, realisation=realisation)

    kwargs["values"] = _roxapi_xyz_to_dataframe(values, is_polygons=is_polygons)
    return kwargs


def _roxapi_xyz_to_dataframe(roxxyz, is_polygons=False):  # pragma: no cover
    """Tranforming some XYZ from ROXAPI to a Pandas dataframe."""

    # In ROXAPI, polygons/polylines are a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")
    cnames = ["X_UTME", "Y_UTMN", "Z_TVDSS"]

    if is_polygons and isinstance(roxxyz, list):
        # polylines/-gons
        dfs = []
        for idx, poly in enumerate(roxxyz):
            dataset = pd.DataFrame.from_records(poly, columns=cnames)
            dataset["POLY_ID"] = idx
            dfs.append(dataset)

        dfr = pd.concat(dfs)

    elif not is_polygons and isinstance(roxxyz, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxxyz, columns=cnames)

    else:
        raise RuntimeError(f"Unknown error in getting data from Roxar: {type(roxxyz)}")

    dfr.reset_index(drop=True, inplace=True)
    return dfr


def export_xyz_roxapi(
    self, project, name, category, stype, pfilter, realisation, attributes
):  # pragma: no cover
    """Export (store) a XYZ item from XTGeo to RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

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

    rox.safe_close()


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
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err

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

    if isinstance(self, xtgeo.Polygons):
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
):  # pylint: disable=too-many-branches  # pragma: no cover

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
            if isinstance(self, xtgeo.Polygons):
                roxxyz = proj.clipboard.create_polylines(name, folders)
            else:
                roxxyz = proj.clipboard.create_points(name, folders)

    else:
        raise ValueError("Unsupported stype: {}".format(stype))

    return roxxyz


def _get_roxxyz(
    rox, name, category, stype, mode="set", is_polygons=False
):  # pragma: no cover
    # pylint: disable=too-many-branches
    """Get the correct rox_xyz which is some pointer to a RoxarAPI structure."""
    if stype == "horizons":
        rox_xyz = rox.project.horizons[name][category]
    elif stype == "zones":
        rox_xyz = rox.project.zones[name][category]
    elif stype == "faults":
        rox_xyz = rox.project.faults[name][category]
    elif stype == "clipboard":
        folders = None
        rox_xyz = rox.project.clipboard
        if category:
            if isinstance(category, list):
                folders = category
                folders.append(category)
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    f"Cannot parse category: {category}, see documentation!"
                )

            if mode == "get":
                rox_xyz = rox.project.clipboard.folders[folders]

        if mode == "get":
            rox_xyz = rox_xyz[name]

        elif mode == "set":
            # clipboard folders will be created if not present, and overwritten else
            if is_polygons:
                rox_xyz = rox.project.clipboard.create_polylines(name, folders)
            else:
                rox_xyz = rox.project.clipboard.create_points(name, folders)

    else:
        raise TypeError(f"Unsupported stype: {stype}")

    return rox_xyz


def _get_roxvalues(rox_xyz, realisation=0):  # pragma: no cover
    """Return the values from the Roxar API, either numpy (Points) or list(Polygons)."""
    try:
        roxitem = rox_xyz.get_values(realisation)
        logger.info(roxitem)
    except KeyError as kwe:
        logger.error(kwe)

    return roxitem
