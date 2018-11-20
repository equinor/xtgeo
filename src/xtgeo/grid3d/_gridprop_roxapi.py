# coding: utf-8
"""Roxar API functions for XTGeo Grid Property"""

import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the XTGeo GridProperty instance


def import_prop_roxapi(self, project, gname, pname, realisation):
    """Import a Property via ROXAR API spec."""
    import roxar

    # self._roxprop = None

    logger.info('Opening RMS project ...')
    if project is not None and isinstance(project, str):
        # outside a RMS project
        with roxar.Project.open(project) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project
            _get_gridprop_data(self, roxar, proj, gname, pname)

    else:
        # inside a RMS project
        _get_gridprop_data(self, roxar, project, gname, pname)


def _get_gridprop_data(self, roxar, project, gname, pname):
    # inside a RMS project
    if gname not in project.grid_models:
        raise ValueError('No gridmodel with name {}'.format(gname))
    if pname not in project.grid_models[gname].properties:
        raise ValueError('No property in {} with name {}'.format(gname, pname))

    try:
        roxgrid = project.grid_models[gname]
        roxprop = roxgrid.properties[pname]

        if str(roxprop.type) == 'discrete':
            self._isdiscrete = True

        self._roxorigin = True
        _convert_to_xtgeo_prop(self, pname, roxgrid, roxprop)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)


def _convert_to_xtgeo_prop(self, pname, roxgrid, roxprop):

    # import roxar

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    pvalues = roxprop.get_values()
    self._roxar_dtype = pvalues.dtype

    logger.info('PVALUES is {}'.format(pvalues))

    logger.info('PVALUES {} {}'.format(pvalues, pvalues.flags))

    if self._isdiscrete:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)
    else:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.float64)

    mybuffer.fill(self.undef)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    mybuffer[iind, jind, kind] = pvalues[cellno]
    logger.info('BUFFER 0 is {}'.format(mybuffer))

    mybuffer = mybuffer.copy(order='C')

    mybuffer = ma.masked_greater(mybuffer, self.undef_limit)

    self._values = mybuffer

    self._ncol = indexer.dimensions[0]
    self._nrow = indexer.dimensions[1]
    self._nlay = indexer.dimensions[2]

    self._name = pname

    if self._isdiscrete:
        self.codes = roxprop.code_names.copy()

        tmpcode = self.codes.copy()
        for key, val in tmpcode.items():
            if val == '':
                val = 'unknown_' + str(key)
            tmpcode[key] = val
        self.codes = tmpcode

    logger.info('BUFFER 1 is {}'.format(mybuffer))


def export_prop_roxapi(self, project, gname, pname, saveproject=False,
                       realisation=0):
    """Export (i.e. store) to a Property in RMS via ROXAR API spec."""
    import roxar

    logger.info('Opening RMS project ...')
    if project is not None and isinstance(project, str):
        # outside RMS project
        with roxar.Project.open(project) as proj:

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
                    except RuntimeError:
                        xtg.warn('Could not save project!')

            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    else:
        # within RMS project
        try:
            roxgrid = project.grid_models[gname]
            _store_in_roxar(self, pname, roxgrid)
        except KeyError as keyerror:
            raise RuntimeError(keyerror)


def _store_in_roxar(self, pname, roxgrid):

    import roxar

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    logger.info('Store in RMS...')

    val3d = self.values.copy()

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    dtype = self._roxar_dtype
    logger.info('DTYPE is ', dtype)
    if self.isdiscrete:
        pvalues = roxgrid.get_grid().generate_values(data_type=dtype)
    else:
        pvalues = roxgrid.get_grid().generate_values(data_type=dtype)

    pvalues[cellno] = val3d[iind, jind, kind]

    properties = roxgrid.properties

    if self.isdiscrete:
        rprop = properties.create(
            pname, property_type=roxar.GridPropertyType.discrete,
            data_type=dtype)
        rprop.code_names = self.codes.copy()
    else:
        rprop = properties.create(
            pname, property_type=roxar.GridPropertyType.continuous,
            data_type=dtype)

    # values = ma.filled(values, self.undef)
    # values = values[values < self.undef_limit]
    # values = values.astype(np.float32)

    # rprop.set_values(values)
    rprop.set_values(pvalues.astype(dtype))

    if self.isdiscrete:
        rprop.code_names = self.codes.copy()
