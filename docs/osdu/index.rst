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


For Users — Geologists & Geomodellers
--------------------------------------

Set up connections, read/write data in scripts and pipelines, and understand
property name mappings between Eclipse and OSDU.

- :doc:`guide` — Installation, connection setup, read/write recipes, property
  name mappings (Eclipse ↔ OSDU), EPC files, format compatibility
- :doc:`api` — Complete function reference (high-level functions first,
  low-level converters at the bottom)
- :doc:`demos` — Copy-paste runnable examples: local RDDMS, change tracking,
  bulk copy, EPC files
- :doc:`resqpy_interop` — Round-trip xtgeo → resqpy → xtgeo for advanced
  operations (sub-gridding, coarsening, fault analysis)


For Developers — Architecture & Internals
------------------------------------------

Understand the design, data model structures, protocol details, what is
supported (and what is not), and how to contribute.

- :doc:`developer` — Architecture, data model differences (xtgeo vs RESQML),
  supported vs unsupported features, ETP protocol details, pyetp, testing
  setup, Docker, test structure, contributing
- :doc:`api` — Provider classes, low-level converters, enums — the full
  programmatic interface

.. toctree::
   :maxdepth: 2
   :hidden:

   guide
   api
   demos
   resqpy_interop
   developer
