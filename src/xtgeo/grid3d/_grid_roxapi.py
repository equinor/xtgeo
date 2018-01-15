# coding: utf-8
"""Roxar API functions for XTGeo Grid"""
import warnings

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()

# ROXAPI, some important properties that are NATIVE roxar API props:
#
# self._roxgrid -> proj.grid_models[gname].get_grid()
# self._roxindexer -> proj.grid_models[gname].get_grid().grid_indexer


def import_grid_roxapi(self, project, gname):
    """Import a Grid via ROXAR API spec."""
    try:
        import roxar
    except ImportError:
        warnings.warn('Cannot import roxar module!', RuntimeWarning)
        raise

    logger.info('Opening RMS project ...')
    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open_import(projectname) as proj:
            try:
                self._roxgrid = proj.grid_models[gname].get_grid()
                self._roxindexer = self._roxgrid.grid_indexer
            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    raise NotImplementedError('This task is not fully implemented')
