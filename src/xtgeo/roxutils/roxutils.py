# -*- coding: utf-8 -*-
"""Module for simplifying various operation in the Roxar python interface."""

from __future__ import division, absolute_import
from __future__ import print_function


try:
    import roxar
except ImportError:
    pass


class RoxUtils(object):

    """ Class RoxUtils
    E.g creating and deletion of Horizon folders, etc.

    import xtgeo

    xr = xtgeo.RoxUtils()
    xr.create_horizon_category(project, 'DS_extracted_run3')
    xr.delete_horizon_category(project, 'DS_extracted_run2')
    """

    def __init__(self):
        self._version = roxar.__version__

        self._versions = {'1.0': ['10.0.x'],
                          '1.1': ['10.1.0', '10.1.1', '10.1.2'],
                          '1.1.1': ['10.1.3'],
                          '1.2': ['11.0.0'],
                          '1.2.1': ['11.0.1'],
                          '1.3': ['11.1.0']}

    @property
    def roxversion(self):
        """Roxar API version (read only)"""
        return self._version

    def create_horizon_category(self, project, category, stype='horizons',
                                domain='depth', htype='surface'):
        """Create one or more a Horizons category entries.

        Args:
            project (str or special): Use project if within RMS project
            category (str or list): Name(s) of category to make, either as
                 a simple string or a list of strings.
            stype (str): Main folder in RMS (horizons or zones).
                Default is horizons
            domain (str): 'depth' (default) or 'time'
            htype (str): Horizon type: surface/lines/points
        """

        categories = []
        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for category in categories:
            geom = roxar.GeometryType.surface
            if htype.lower() == 'lines':
                geom = roxar.GeometryType.lines
            elif htype.lower() == 'points':
                geom = roxar.GeometryType.points

            dom = roxar.VerticalDomain.depth
            if domain.lower() == 'time':
                dom = roxar.GeometryType.lines

            if stype.lower() == 'horizons':
                if category not in project.horizons.representations:
                    try:
                        project.horizons.representations.create(category, geom,
                                                                dom)
                    except Exception as exmsg:
                        print('Error: {}'.format(exmsg))
                else:
                    print('Category <{}> already exists'.format(category))

            elif stype.lower() == 'zones':
                if category not in project.zones.representations:
                    try:
                        project.zones.representations.create(category, geom,
                                                             dom)
                    except Exception as exmsg:
                        print('Error: {}'.format(exmsg))
                else:
                    print('Category <{}> already exists'.format(category))

    def create_zones_category(self, project, category, domain='thickness',
                              htype='surface'):
        """Same as create_horizon_category, but with stype='zones'."""

        self.create_horizon_category(project, category, stype='zones',
                                     domain=domain, htype=htype)

    def delete_horizon_category(self, project, category, stype='horizons'):
        """Delete on or more horizons or zones categories.

        Args:
            project (str or special): Use project if within RMS project
            category (str or list): Name(s) of category to make, either
                as a simple string or a list of strings.
            stype (str): Main folder in RMS (horizons or zones).
                Default is horizons
        """

        categories = []
        if isinstance(category, str):
            categories.append(category)
        else:
            categories.extend(category)

        for category in categories:
            if stype.lower() == 'horizons':
                try:
                    del project.horizons.representations[category]
                except KeyError as kerr:
                    if kerr == category:
                        print('Cannot delete {}, does not exist'. format(kerr))
            elif stype.lower() == 'zones':
                try:
                    del project.horizons.representations[category]
                except KeyError as kerr:
                    if kerr == category:
                        print('Cannot delete {}, does not exist'. format(kerr))
            else:
                raise ValueError('Wrong stype applied')

    def delete_zones_category(self, project, category):
        """Delete on or more horizons or zones categories. See previous"""

        self.delete_horizon_category(project, category, stype='zones')
