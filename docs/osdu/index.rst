OSDU / RESQML Interface
=======================

.. versionadded:: 4.x

XTGeo provides a complete interface for reading and writing subsurface data
to `OSDU <https://community.opengroup.org/osdu>`_ Reservoir DDMS (RDDMS) servers
and RESQML 2.0.1 EPC+HDF5 file containers. This enables:

- **Seamless data exchange** between xtgeo and OSDU cloud platforms
- **File-based workflows** using standard RESQML EPC containers
- **Live protocol access** via ETP 1.2 WebSocket connections
- **Deep discovery** — traverse the full object graph in a dataspace
- **Change tracking** — detect when objects are created, modified, or deleted

.. grid:: 2

    .. grid-item-card:: User Guide
        :link: guide
        :link-type: doc

        Get started quickly. Connect, read, write, and discover data.

    .. grid-item-card:: API Reference
        :link: api
        :link-type: doc

        Complete function reference organised by use case.

    .. grid-item-card:: Design & Development
        :link: design
        :link-type: doc

        Architecture, protocols, and contribution guide.

    .. grid-item-card:: Demos & Testing
        :link: demos
        :link-type: doc

        Runnable examples, test setup, and references.

.. toctree::
   :maxdepth: 2
   :hidden:

   guide
   api
   design
   demos
