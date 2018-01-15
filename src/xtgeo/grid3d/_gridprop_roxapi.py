# coding: utf-8
"""Roxar API functions for XTGeo Grid Property"""
import warnings

import numpy as np
import numpy.ma as ma

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_prop_roxapi(prop, project, gname, pname):
    """Import a Property via ROXAR API spec."""
    try:
        import roxar
    except ImportError:
        warnings.warn('Cannot import roxar module!', RuntimeWarning)
        raise

    logger.info('SELF 2a is {}'.format(prop))

    prop._roxprop = None

    logger.info('Opening RMS project ...')
    if project is not None and isinstance(project, str):
        projectname = project
        with roxar.Project.open(projectname, readonly=True) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project

            try:
                roxgrid = proj.grid_models[gname]
                roxprop = roxgrid.properties[pname]
                prop._roxorigin = True
                _convert_to_xtgeo_prop(prop, pname, roxgrid, roxprop)

            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    logger.info('SELF 2b is {}'.format(prop))
    print(prop._roxprop)

    return prop


def export_prop_roxapi(prop, project, gname, pname, saveproject=False):
    """Export to a Property in RMS via ROXAR API spec."""
    try:
        import roxar
    except ImportError:
        warnings.warn('Cannot import roxar module!', RuntimeWarning)
        raise

    logger.info('Opening RMS project ...')
    if project is not None and isinstance(project, str):
        projectname = project
        roxar.Project.unlock(projectname)
        with roxar.Project.open(projectname) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project

            try:
                roxgrid = proj.grid_models[gname]

                # tmp

                _store_in_roxar(prop, pname, roxgrid)

                if saveproject:
                    try:
                        proj.save()
                    except Exception as e:
                        warnings.warn('Could not save')

            except KeyError as keyerror:
                raise RuntimeError(keyerror)


def _convert_to_xtgeo_prop(prop, pname, roxgrid, roxprop):

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    pvalues = roxprop.get_values()
    logger.info('PVALUES is {}'.format(pvalues))

    # test
    properties = roxgrid.properties
    rprop = properties.create('TEST',
                              property_type=roxar.GridPropertyType.continuous,
                              data_type=np.float32)

    rprop.set_values(pvalues)

    logger.info('PVALUES {} {}'.format(pvalues, pvalues.flags))

    buffer = np.ndarray(indexer.dimensions, dtype=np.float64)

    buffer.fill(prop.undef)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    print(iind[0:30])
    print(jind)
    print(kind)

    buffer[iind, jind, kind] = pvalues[cellno]
    logger.info('BUFFER 0 is {}'.format(buffer))

    buffer = buffer.copy(order='F')
    buffer = buffer.ravel(order='K')

    buffer = ma.masked_greater(buffer, prop.undef_limit)

    prop._values = buffer

    prop._cvalues = None
    prop._ncol = indexer.dimensions[0]
    prop._nrow = indexer.dimensions[1]
    prop._nlay = indexer.dimensions[2]

    prop._name = pname

    prop._discrete = False

    logger.info('BUFFER 1 is {}'.format(buffer))


def _store_in_roxar(prop, pname, roxgrid):

    indexer = roxgrid.get_grid().grid_indexer

    logger.info(indexer.handedness)

    logger.info('Store in RMS...')

    val3d = prop.values3d.copy(order='C')

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

    # values = ma.filled(values, prop.undef)
    # values = values[values < prop.undef_limit]
    # values = values.astype(np.float32)

    # rprop.set_values(values)
    rprop.set_values(pvalues)
