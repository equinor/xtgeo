Using resqpy with xtgeo
========================

`resqpy <https://github.com/bp/resqpy>`_ is a Python library for reading,
writing, and manipulating RESQML models. It provides powerful derived-model
operations (sub-gridding, coarsening, fault analysis) that complement xtgeo's
grid manipulation and format-conversion capabilities.

The two libraries interoperate through **RESQML EPC files**: xtgeo can export
grids and properties to EPC format that resqpy can load, and vice versa.
This enables an **xtgeo → resqpy → xtgeo** round-trip where you leverage
advanced resqpy functions that are not available in xtgeo itself.

Installation
------------

resqpy is **not** bundled with xtgeo. Install it separately:

.. code-block:: bash

    pip install xtgeo resqpy>=4.0

.. note::

    resqpy currently pins ``lxml<6.0``. If you also install xtgeo's OSDU extra
    (``pip install xtgeo[osdu]``), ensure that lxml version constraints are
    compatible in your environment.


Round-trip pattern
-------------------

The typical workflow is:

1. **Load** data in xtgeo (from ROFF, GRDECL, Eclipse, EPC, etc.)
2. **Export** to EPC via xtgeo's OSDU interface
3. **Process** with resqpy (extract sub-grid, coarsen, fault analysis, etc.)
4. **Import** results back into xtgeo from the resqpy-produced EPC
5. **Export** to simulation formats (GRDECL, ROFF, etc.) or continue in xtgeo

.. code-block:: text

    ┌─────────┐      EPC file       ┌──────────┐      EPC file       ┌─────────┐
    │  xtgeo  │ ──── export ──────▶ │  resqpy  │ ──── export ──────▶ │  xtgeo  │
    │  (load) │                      │(advanced)│                      │ (use)   │
    └─────────┘                      └──────────┘                      └─────────┘


Helper functions
-----------------

These helpers simplify the round-trip:

.. code-block:: python

    import xtgeo
    from xtgeo.interfaces.osdu import EpcFileProvider
    from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml


    def export_to_epc(epc_path, grid, properties=None, title="Grid", crs_epsg=23031):
        """Export an xtgeo grid (+ optional properties) to an EPC file."""
        provider = EpcFileProvider(epc_path, mode="w")
        provider.open()
        result = xtgeo_grid_to_resqml(
            provider,
            grid,
            title=title,
            crs_epsg=crs_epsg,
            properties=properties or [],
        )
        provider.close()
        return result


    def import_from_epc(epc_path, grid_uuid):
        """Import an xtgeo grid from an EPC file by UUID."""
        provider = EpcFileProvider(epc_path, mode="r")
        provider.open()
        result = ijk_grid_to_xtgeo(provider, grid_uuid)
        provider.close()
        return result[0] if isinstance(result, tuple) else result


Example 1 — Extract a sub-grid
--------------------------------

Use resqpy's ``extract_box`` to cut out a region of interest, then bring it
back into xtgeo:

.. code-block:: python

    import numpy as np
    import resqpy.derived_model as rdm

    # 1. Load grid in xtgeo
    grid = xtgeo.grid_from_file("model.roff")

    # 2. Export to EPC
    export_to_epc("work.epc", grid, title="FullField")

    # 3. Extract box with resqpy (kji min/max, inclusive)
    box = np.array([[0, 2, 3], [4, 7, 9]])
    sub = rdm.extract_box(
        epc_file="work.epc",
        box=box,
        inherit_properties=True,
        new_grid_title="WellModel",
        new_epc_file="subgrid.epc",
    )

    # 4. Import back into xtgeo
    subgrid = import_from_epc("subgrid.epc", str(sub.uuid))

    # 5. Use in xtgeo — e.g. export to GRDECL
    subgrid.to_file("subgrid.grdecl", fformat="grdecl")


Example 2 — Coarsen a grid
----------------------------

Upscale a fine grid for faster simulation using resqpy's coarsening:

.. code-block:: python

    import resqpy.derived_model as rdm
    import resqpy.olio.fine_coarse as fc

    grid = xtgeo.grid_from_file("fine_model.roff")
    export_to_epc("fine.epc", grid, title="Fine")

    # 2×2×1 coarsening (halve I and J, keep K layers)
    nk, nj, ni = grid.nlay, grid.nrow, grid.ncol
    fine_coarse = fc.FineCoarse((nk, nj, ni), (nk, nj // 2, ni // 2))
    fine_coarse.set_all_ratios_constant()

    coarse = rdm.coarsened_grid(
        epc_file="fine.epc",
        source_grid=None,
        fine_coarse=fine_coarse,
        new_grid_title="Coarse",
        new_epc_file="coarse.epc",
    )

    coarse_grid = import_from_epc("coarse.epc", str(coarse.uuid))
    coarse_grid.to_file("coarse.roff")


Example 3 — Fault transmissibility analysis
---------------------------------------------

Analyze fault connections and apply transmissibility multipliers:

.. code-block:: python

    import numpy as np
    import resqpy.grid as rqgrid
    import resqpy.fault as rf
    from resqpy.model import Model

    grid = xtgeo.grid_from_file("faulted.roff")
    export_to_epc("grid.epc", grid, title="FaultGrid")

    # Open in resqpy
    model = Model(epc_file="grid.epc")
    rq_grid = rqgrid.Grid(model, uuid=model.uuid(obj_type="IjkGridRepresentation"))

    # Define a fault plane at j=5
    j_faces = np.zeros((rq_grid.nk, rq_grid.nj - 1, rq_grid.ni), dtype=bool)
    j_faces[:, 5, :] = True

    gcs = rf.GridConnectionSet(
        model,
        grid=rq_grid,
        j_faces=j_faces,
        feature_name="MainFault",
        feature_type="fault",
        create_organizing_objects_where_needed=True,
        create_transmissibility_multiplier_property=True,
        fault_tmult_dict={"MainFault": 0.01},
    )

    print(f"Fault has {gcs.count} cell-face pairs with TMULT=0.01")


Limitations
-----------

- resqpy and xtgeo use different in-memory representations — the round-trip
  goes through EPC files on disk (or a temporary directory).
- Not all xtgeo grid features map 1:1 to RESQML (see :doc:`developer` for
  data model differences).
- resqpy currently requires ``lxml<6.0``, which may conflict with other
  dependencies. Use a dedicated virtual environment if needed.
