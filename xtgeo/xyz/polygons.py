"""XTGeo Polygons class"""

# #############################################################################
#
# NAME:
#    polygons.py
#
# AUTHOR(S):
#    Jan C. Rivenaes
#
# DESCRIPTION:
#    Polygons (connected points), which is subset of Points
#    Stored as Pandas in Python.
# TODO/ISSUES/BUGS:
#
# LICENCE:
#    Statoil property
# #############################################################################

from __future__ import print_function
import logging

from .points import Points


class Polygons(Points):
    """
    Class for a points set in the XTGeo framework.
    """

    def __init__(self, *args, **kwargs):

        """The __init__ (constructor) method.

        The instance can be made either from file or by a spesification::

        >>> xp = Polygons()
        >>> xp.from_file('somefilename', fformat='xyz')

        Args:
            xxx (nn): to come


        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        Points.__init__(self)

        self.logger.debug('Ran __init__ method')

    # =========================================================================
    # Import and export
    # =========================================================================
    # =========================================================================
    # Get and Set properties
    # =========================================================================


    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import/Export methods for various formats
    # -------------------------------------------------------------------------

    # Import XYZ
    # -------------------------------------------------------------------------

    # Export RMS ascii
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Special methods for nerds
    # -------------------------------------------------------------------------
