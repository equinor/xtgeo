# -*- coding: utf-8 -*-
"""Private module for etc functions"""


try:
    import roxar
except ImportError:
    pass

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def create_whatever_category(
    self, category, stype="horizons", domain="depth", htype="surface"
):
    """Create one or more a Horizons/Zones... category entries.

    Args:
        category (str or list): Name(s) of category to make, either as
             a simple string or a list of strings.
        stype (str): 'Super type' in RMS (horizons or zones).
            Default is horizons
        domain (str): 'depth' (default) or 'time'
        htype (str): Horizon type: surface/lines/points
    """

    project = self.project
    categories = []

    if isinstance(category, str):
        categories.append(category)
    else:
        categories.extend(category)

    for catg in categories:
        geom = roxar.GeometryType.surface
        if htype.lower() == "lines":
            geom = roxar.GeometryType.lines
        elif htype.lower() == "points":
            geom = roxar.GeometryType.points

        dom = roxar.VerticalDomain.depth
        if domain.lower() == "time":
            dom = roxar.GeometryType.lines

        if stype.lower() == "horizons":
            if catg not in project.horizons.representations:
                try:
                    project.horizons.representations.create(catg, geom, dom)
                except Exception as exmsg:  # pylint: disable=broad-except
                    print("Error: {}".format(exmsg))
            else:
                print("Category <{}> already exists".format(catg))

        elif stype.lower() == "zones":
            if catg not in project.zones.representations:
                try:
                    project.zones.representations.create(catg, geom, dom)
                except Exception as exmsg:  # pylint: disable=broad-except
                    print("Error: {}".format(exmsg))
            else:
                print("Category <{}> already exists".format(catg))


def delete_whatever_category(self, category, stype="horizons"):
    """Delete one or more horizons or zones categories.

    Args:
        category (str or list): Name(s) of category to make, either
            as a simple string or a list of strings.
        stype (str): 'Super type', in RMS ('horizons' or 'zones').
            Default is 'horizons'
    """

    project = self.project
    categories = []

    if isinstance(category, str):
        categories.append(category)
    else:
        categories.extend(category)

    for catg in categories:
        if stype.lower() == "horizons":
            try:
                del project.horizons.representations[catg]
            except KeyError as kerr:
                if kerr == catg:
                    print("Cannot delete {}, does not exist".format(kerr))
        elif stype.lower() == "zones":
            try:
                del project.horizons.representations[catg]
            except KeyError as kerr:
                if kerr == catg:
                    print("Cannot delete {}, does not exist".format(kerr))
        else:
            raise ValueError("Wrong stype applied")


def clear_whatever_category(self, category, stype="horizons"):
    """Clear (or make empty) the content of one or more horizon/zones... categories.

    Args:
        category (str or list): Name(s) of category to empty, either as
             a simple string or a list of strings.
        stype (str): 'Super type' in RMS (horizons or zones).
            Default is horizons

    .. versionadded:: 2.1
    """

    project = self.project

    categories = []
    if isinstance(category, str):
        categories.append(category)
    else:
        categories.extend(category)

    xtype = project.horizons
    if stype.lower() == "zones":
        xtype = project.zones

    for catg in categories:
        for xitem in xtype:
            try:
                item = xtype[xitem.name][catg]
                item.set_empty()
            except KeyError as kmsg:
                print(kmsg)
