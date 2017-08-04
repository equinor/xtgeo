# -*- coding: utf-8 -*-
"""
Metaclass for Grid and Properties.
"""

__author__ = 'Jan C. Rivenaes'
__copyright__ = 'Statoil property'
__credits__ = 'FMU team'
__licence__ = 'Propriatary'
__status__ = 'Development'
__version__ = 'see xtgeo/_version.py'

import logging


class Grid3D(object):

    def __init__(self):
        """
        The __init__ (constructor) method.

        """
        self._nx = 4
        self._ny = 3
        self._ny = 5

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

    @property
    def nx(self):
        """ Returns the NX (Ncolumns) number of cells"""
        return self._nx

    @nx.setter
    def nx(self, value):
        self.logger.warning("Cannot change the nx property")

    @property
    def ny(self):
        """ Returns the NY (Nrows) number of cells"""
        return self._ny

    @ny.setter
    def ny(self, value):
        self.logger.warning("Cannot change the ny property")

    @property
    def nz(self):
        """ Returns the NZ (Nlayers) number of cells"""
        return self._nz

    @nz.setter
    def nz(self, value):
        self.logger.warning("Cannot change the nz property")
