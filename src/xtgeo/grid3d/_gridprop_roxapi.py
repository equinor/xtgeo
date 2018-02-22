# coding: utf-8
"""Roxar API functions for XTGeo Grid Property"""
import warnings

import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the XTGeo GridProperty instance


def import_prop_roxapi(self, projectname, gname, pname, realisation):
    """Import a Property via ROXAR API spec."""
    import roxar

    self._roxprop = None

    logger.info('Opening RMS project ...')
    if projectname is not None and isinstance(projectname, str):
        # outside a RMS project
        with roxar.Project.open(projectname, readonly=True) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project

            try:
                roxgrid = proj.grid_models[gname]
                roxprop = roxgrid.properties[pname]
                self._roxorigin = True
                _convert_to_xtgeo_prop(self, pname, roxgrid, roxprop)

            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    else:
        # inside a RMS project
        try:
            roxgrid = projectname.grid_models[gname]
            roxprop = roxgrid.properties[pname]
            self._roxorigin = True
            _convert_to_xtgeo_prop(self, pname, roxgrid, roxprop)

        except KeyError as keyerror:
            raise RuntimeError(keyerror)

    return self


def export_prop_roxapi(self, projectname, gname, pname, saveproject=False,
                       realisation=0):
    """Export to a Property in RMS via ROXAR API spec."""
    import roxar

    logger.info('Opening RMS project ...')
    if projectname is not None and isinstance(projectname, str):
        # outside RMS project
        with roxar.Project.open(projectname) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project

            try:
                roxgrid = proj.grid_models[gname]
                _store_in_roxar(self, pname, roxgrid)

                if saveproject:
                    try:
                        proj.save()
                    except Exception as myexception:
                        warnings.warn('Could not save', myexception)

            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    else:
        # within RMS project
        try:
            roxgrid = projectname.grid_models[gname]
            _store_in_roxar(self, pname, roxgrid)
        except KeyError as keyerror:
            raise RuntimeError(keyerror)


def _convert_to_xtgeo_prop(self, pname, roxgrid, roxprop):

    # import roxar

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    pvalues = roxprop.get_values()
    logger.info('PVALUES is {}'.format(pvalues))

    # # test
    # properties = roxgrid.properties
    # rprop = properties.create('TEST',
    #                           property_type=roxar.GridPropertyType.continuous,
    #                           data_type=np.float32)

    # rprop.set_values(pvalues)

    logger.info('PVALUES {} {}'.format(pvalues, pvalues.flags))

    mybuffer = np.ndarray(indexer.dimensions, dtype=np.float64)

    mybuffer.fill(self.undef)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    mybuffer[iind, jind, kind] = pvalues[cellno]
    logger.info('BUFFER 0 is {}'.format(mybuffer))

    mybuffer = mybuffer.copy(order='F')
    mybuffer = mybuffer.ravel(order='K')

    mybuffer = ma.masked_greater(mybuffer, self.undef_limit)

    self._values = mybuffer

    self._cvalues = None
    self._ncol = indexer.dimensions[0]
    self._nrow = indexer.dimensions[1]
    self._nlay = indexer.dimensions[2]

    self._name = pname

    self._discrete = False

    logger.info('BUFFER 1 is {}'.format(mybuffer))


def _store_in_roxar(self, pname, roxgrid):

    import roxar

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    logger.info('Store in RMS...')

    val3d = self.values3d.copy(order='C')

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    pvalues = roxgrid.get_grid().generate_values(data_type=np.float32)
    pvalues[cellno] = val3d[iind, jind, kind]

    properties = roxgrid.properties
    rprop = properties.create(pname,
                              property_type=roxar.GridPropertyType.continuous,
                              data_type=np.float32)

    # values = ma.filled(values, self.undef)
    # values = values[values < self.undef_limit]
    # values = values.astype(np.float32)

    # rprop.set_values(values)
    rprop.set_values(pvalues)
