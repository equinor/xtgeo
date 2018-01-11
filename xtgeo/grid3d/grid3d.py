# -*- coding: utf-8 -*-
"""Abstract baseclass for Grid and GridProperties, not to be used by itself"""

from __future__ import print_function, absolute_import

import abc
import warnings
import six

import xtgeo


@six.add_metaclass(abc.ABCMeta)
class Grid3D(object):
    """Abstract base class for Grid3D."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self._xtg = xtgeo.common.XTGeoDialog()
        self.logger = self._xtg.functionlogger(clsname)

        self._ncol = 4
        self._nrow = 3
        self._nlay = 5

    @property
    def ncol(self):
        """ Returns the NCOL (NX or Ncolumns) number of cells"""
        return self._ncol

    @ncol.setter
    def ncol(self, value):
        warnings.warn(UserWarning("Cannot change the ncol property"))

    @property
    def nrow(self):
        """ Returns the NROW (NY or Nrows) number of cells"""
        return self._nrow

    @nrow.setter
    def nrow(self, value):
        warnings.warn(UserWarning("Cannot change the nrow property"))

    @property
    def nlay(self):
        """ Returns the NLAY (NZ or Nlayers) number of cells"""
        return self._nlay

    @nlay.setter
    def nlay(self, value):
        warnings.warn(UserWarning("Cannot change the nlay property"))

    @property
    def nx(self):
        """ Returns the NX (Ncolumns) number of cells (deprecated; use ncol)"""
        warnings.warn(DeprecationWarning("Deprecated; use ncol instead"))
        return self._ncol

    @nx.setter
    def nx(self, value):
        warnings.warn(UserWarning("Cannot change the nx property"))

    @property
    def ny(self):
        """ Returns the NY (Nrows) number of cells (deprecated; use nrow)"""
        warnings.warn(DeprecationWarning("Deprecated; use nrow instead"))
        return self._nrow

    @ny.setter
    def ny(self, value):
        warnings.warn("Cannot change the ny property")

    @property
    def nz(self):
        """ Returns the NZ (Nlayers) number of cells (deprecated; use nlay)"""
        warnings.warn(DeprecationWarning("Deprecated; use nlay instead"))
        return self._nlay

    @nz.setter
    def nz(self, value):
        warnings.warn(UserWarning("Cannot change the nz property"))
