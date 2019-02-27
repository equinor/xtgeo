# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""
import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access


def import_xyz_roxapi(self, project, name, category,
                      stype, realisation):
    """Import a Points or Polygons item via ROXAR API spec."""

    rox = RoxUtils(project)

    _roxapi_import_xyz(self, rox.project, name, category, stype,
                       realisation)


def _roxapi_import_xyz(self, proj, name, category, stype, realisation):

    self._name = name

    if not _check_category_etc(self, proj, name, category, stype, realisation):
        raise RuntimeError('Something is very wrong...')

    try:
        if stype == 'horizons':
            roxxyz = proj.horizons[name][category].get_values(realisation)

        elif stype == 'zones':
            roxxyz = proj.zones[name][category].get_values(realisation)

        elif stype == 'faults':
            roxxyz = proj.faults[name][category].get_values(realisation)

        elif stype == 'clipboard':
            if category:
                if '|' in category:
                    folders = category.split('|')
                else:
                    folders = category.split('/')
                roxxyz = proj.clipboard.folders[folders]
            else:
                roxxyz = proj.clipboard
            roxxyz = roxxyz[name].get_values(realisation)

        else:
            roxxyz = None
            raise ValueError('Unsupported stype: {}'.format(stype))

        _roxapi_xyz_to_xtgeo(self, roxxyz)
    except KeyError as kwe:
        logger.error(kwe)


def _roxapi_xyz_to_xtgeo(self, roxxyz):
    """Tranforming some XYZ from ROXAPI to XTGeo object."""

    # In ROXAPI, polygons is a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info('Points/polygons/polylines from roxapi to xtgeo...')
    cnames = ['X_UTME', 'Y_UTMN', 'Z_TVDSS']

    if isinstance(roxxyz, list):
        # polylines/-gons
        dfs = []
        for id, poly in enumerate(roxxyz):
            dataset = pd.DataFrame.from_records(poly, columns=cnames)
            dataset['POLY_ID'] = id
            dfs.append(dataset)

        dfr = pd.concat(dfs)
        self._ispolygons = True

    elif isinstance(roxxyz, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxxyz, columns=cnames)
        self._ispolygons = False

    else:
        raise RuntimeError('Unknown error in getting data from Roxar')

    self._df = dfr


def export_xyz_roxapi(self, project, name, category, stype,
                      realisation):
    """Export (store) a XYZ item to RMS via ROXAR API spec."""
    import roxar  # pylint: disable=import-error

    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            _roxapi_export_xyz(self, proj, name, category, stype,
                               realisation)
    else:
        _roxapi_export_xyz(self, project, name, category, stype,
                           realisation)


def _roxapi_export_xyz(self, proj, name, category, stype, realisation):

    if not _check_category_etc:
        raise RuntimeError('Cannot access correct category or name in RMS')

    if stype == 'horizons':
        roxxyz = proj.horizons[name][category]
    elif stype == 'zones':
        roxxyz = proj.zones[name][category]
    elif stype == 'faults':
        roxxyz = proj.faults[name][category]
    elif stype == 'clipboard':
        if category:
            if '|' in category:
                folders = category.split('|')
            else:
                folders = category.split('/')
            roxxyz = proj.clipboard.folders[folders]
        else:
            roxxyz = proj.clipboard
    else:
        roxxyz = None
        raise ValueError('Unsupported stype: {}'.format(stype))

    if self._ispolygons:
        arrxyz = []
        polys = self.dataframe.groupby(self.pname)
        for id_, grp in polys:
            arr = np.stack([grp[self.xname], grp[self.yname],
                            grp[self.zname]], axis=1)
            arrxyz.append(arr)
    else:
        xyz = self.dataframe
        arrxyz = np.stack([xyz[self.xname], xyz[self.yname],
                           xyz[self.zname]], axis=1)
    try:
        roxxyz.set_values(arrxyz)
    except KeyError as kwe:
        logger.error(kwe)


def _check_category_etc(self, proj, name, category, stype, realisation):
    """Helper to check if valid placeholder' whithin RMS."""
    if stype == 'horizons':
        if name not in proj.horizons:
            raise ValueError('Name {} is not within Horizons'.format(name))
        if category not in proj.horizons.representations:
            raise ValueError('Category {} is not within Horizons categories'
                             .format(category))
    elif stype == 'zones':
        if name not in proj.zones:
            raise ValueError('Name {} is not within Zones'.format(name))
        if category not in proj.zones.representations:
            raise ValueError('Category {} is not within Zones categories'
                             .format(category))
    elif stype == 'faults':
        if name not in proj.faults:
            raise ValueError('Name {} is not within Faults'.format(name))
        if category not in proj.zones.representations:
            raise ValueError('Category {} is not within Faults categories'
                             .format(category))
    elif stype == 'clipboard':
        if category:
            if '|' in category:
                folders = category.split('|')
            else:
                folders = category.split('/')
            roxxyz = proj.clipboard.folders[folders]
        else:
            roxxyz = proj.clipboard
        if name not in roxxyz:
            raise ValueError('Name {} is not within Clipboard...'.format(name))
    else:
        raise ValueError('Invalid stype')

    return True
