# -*- coding: utf-8 -*-
"""Metaclass for Grid and Properties, not to be used by itself"""

import logging


class Grid3D(object):

    def __init__(self):
        """The __init__ (constructor) method."""
        self._ncol = 4
        self._nrow = 3
        self._nlay = 5

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

    @property
    def ncol(self):
        """ Returns the NCOL (NX or Ncolumns) number of cells"""
        return self._ncol

    @ncol.setter
    def ncol(self, value):
        self.logger.warning("Cannot change the ncol property")

    @property
    def nrow(self):
        """ Returns the NROW (NY or Nrows) number of cells"""
        return self._nrow

    @nrow.setter
    def nrow(self, value):
        self.logger.warning("Cannot change the nrow property")

    @property
    def nlay(self):
        """ Returns the NLAY (NZ or Nlayers) number of cells"""
        return self._nlay

    @nlay.setter
    def nlay(self, value):
        self.logger.warning("Cannot change the nlay property")

    @property
    def nx(self):
        """ Returns the NX (Ncolumns) number of cells (deprecated; use ncol)"""
        self.logger.warning("Deprecated; use ncol instead")
        return self._ncol

    @nx.setter
    def nx(self, value):
        self.logger.warning("Cannot change the nx property")

    @property
    def ny(self):
        """ Returns the NY (Nrows) number of cells (deprecated; use nrow)"""
        self.logger.warning("Deprecated; use nrow instead")
        return self._nrow

    @ny.setter
    def ny(self, value):
        self.logger.warning("Cannot change the ny property")

    @property
    def nz(self):
        """ Returns the NZ (Nlayers) number of cells (deprecated; use nlay)"""
        self.logger.warning("Deprecated; use nlay instead")
        return self._nlay

    @nz.setter
    def nz(self, value):
        self.logger.warning("Cannot change the nz property")
