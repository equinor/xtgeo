from __future__ import print_function, absolute_import

import logging

from xtgeo.common import XTGeoDialog

import cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def refine_vertically(self, rfactor, zoneprop=None):
    """Refine vertically, proportionally

    The rfactor can be a scalar or (todo:) a dictionary

    Input:
        self (object): A grid XTGeo object
        rfactor (scalar or dict): Refinement factor
        zoneprop (GridProperty): Zone property; must be defined if rfactor
            is a dict
    """

    # rfac is an array with length nlay, and has refinement per
    # layer
    rfac = _cxtgeo.new_intarray(self.nlay)
    i_index, j_index, k_index = self.get_indices()
    kval = k_index.values

    old_subgrids = self.subgrids
    new_subgrids = []

    if isinstance(rfactor, dict):
        newnlay = 0
        zprval = zoneprop.values
        for izone, rnfactor in rfactor.items():
            mininzn = int(kval[zprval == izone].min() - 1)  # 0 base
            maxinzn = int(kval[zprval == izone].max() - 1)  # 0 base
            logger.info('Zone %s: lay range %s %s', izone, mininzn, maxinzn)
            for ira in range(mininzn, maxinzn + 1):
                _cxtgeo.intarray_setitem(rfac, ira, rnfactor)
                newnlay = newnlay + rnfactor

            if old_subgrids is not None:
                new_subgrids.append(rnfactor * old_subgrids[izone - 1])
    else:
        newnlay = self.nlay * rfactor  # scalar case
        for i in range(self.nlay):
            _cxtgeo.intarray_setitem(rfac, i, rfactor)

        if old_subgrids is not None:
            for izone in range(len(old_subgrids)):
                new_subgrids.append(rfactor * old_subgrids[izone])

    logger.info('Old NLAY: %s, new NLAY: %s', self.nlay, newnlay)

    xtg_verbose_level = xtg.syslevel

    ref_num_act = _cxtgeo.new_intpointer()
    ref_p_zcorn_v = _cxtgeo.new_doublearray(self.ncol * self.nrow *
                                            (newnlay + 1) * 4)
    ref_p_actnum_v = _cxtgeo.new_intarray(self.ncol * self.nrow * newnlay)

    ier = _cxtgeo.grd3d_refine_vert(self.ncol,
                                    self.nrow,
                                    self.nlay,
                                    self._p_coord_v,
                                    self._p_zcorn_v,
                                    self._p_actnum_v,
                                    newnlay,
                                    ref_p_zcorn_v,
                                    ref_p_actnum_v,
                                    ref_num_act,
                                    rfac,
                                    0,
                                    xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('An error occured in the C routine '
                           'grd3d_refine_vert, code {}'.format(ier))

    # update instance:
    self._nlay = newnlay
    self._nactive = _cxtgeo.intpointer_value(ref_num_act)
    self._p_zcorn_v = ref_p_zcorn_v
    self._p_actnum_v = ref_p_actnum_v
    if old_subgrids is not None:
        self.subgrids = new_subgrids

    return self
