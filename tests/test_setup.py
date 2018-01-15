# coding: utf-8
# Setup common stuff for pytests...
import os

import pytest
import warnings

# SEGYIO ----------------------------------------------------------------------
no_segyio = False
try:
    import segyio  # pylint: disable=F401 # noqa:<Error No>
except ImportError:
    no_segyio = True

if no_segyio:
    warnings.warn('"segyio" library not found')

skipsegyio = pytest.mark.skipif(
    no_segyio,
    reason='Skip test with segyio (not present)')

# Roxar python-----------------------------------------------------------------
# Routines using matplotlib shall not ran if ROXENV=1
# use the @skipifroxar decorator

roxar = False
if 'ROXENV' in os.environ:
    roxenv = int(os.environ.get('ROXENV'))
    print(roxenv)
else:
    roxenv = 0
if roxenv == 1:
    roxar = True
    warnings.warn('Roxar is present')

skipifroxar = pytest.mark.skipif(roxar,
                                 reason='Skip test in Roxar python')

skipunlessroxar = pytest.mark.skipif(not roxar,
                                     reason='Skip if NOT Roxar python')


# Functions--------------------------------------------------------------------

def assert_equal(this, that, txt=''):
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=''):
    assert this == pytest.approx(that, abs=tol), txt


# =============================================================================
def test_dummy():
    one = 1
    assert one == 1
