.. highlight:: python

===================
Use of XTGeo in RMS
===================

XTGeo can be incorporated within the RMS user interface and share
data with RMS. The integration will be continuosly improved.
Note that all these script examples are assumed to be ran inside
a python job within RMS.

Get and set data
----------------

In general, data are imported into XTGeo by a ``from_roxar()`` or by a
``xtgeo.xxx_from_roxar()`` (where xx is "surface", "grid", etc). Then the
altered instance can be stored in roxar/RMS by a ``to_roxar()`` method.

The ``to_roxar()`` method will not do a project save when inside RMS or when inside
a virtual project setting. However, if a project is applied as a file path, then
``to_roxar`` will save implicitly. Examples:

Inside RMS GUI
^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    surf = xtgeo.surface_from_roxar(project, "TopReek", "DS_extracted")
    surf.values += 100
    surf.to_roxar(project)

    # Note: project save needs to be done by user (GUI action)

Outside RMS, direct access
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    myproject = "/some/file/path/reek.rms11.1.1"

    surf = xtgeo.surface_from_roxar(myproject, "TopReek", "DS_extracted")
    surf.values += 100
    surf.to_roxar(myproject)

    # Note: project save is done automatic

Outside RMS, project mode
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    myproject = "/some/file/path/reek.rms11.1.1"
    rox = xtgeo.RoxUtils(myproject)

    surf = xtgeo.surface_from_roxar(rox.project, "TopReek", "DS_extracted")
    surf.values += 100
    surf.to_roxar(rox.project)

    # Note: project save is not done automatic, you need to:

    rox.project.save()
    rox.project.close()



Surface data
------------

Here are some simple examples on how to use XTGeo to interact with
RMS data, and e.g. do quick exports to files.

Export a surface in RMS to irap binary format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo and export
    surf = xtgeo.surface_from_roxar(project, "TopReek", "DS_extracted")

    surf.to_file("topreek.gri")

    # modify surface, add 1000 to all map nodes
    surf.values += 1000

    # store in RMS (category must exist)
    surf.to_roxar(project, "TopReek", "DS_whatever")


Export a surface in RMS to zmap ascii format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note here that an automatic resampling to a nonrotated regular
grid will be done in case the RMS map has a rotation.

.. code-block:: python

    import xtgeo as xt

    # surface names
    hnames = ["TopReek", "MiddleReek", "LowerReek"]

    # loop over stratigraphy
    for name in hnames:
        surf = xt.surface_from_roxar(project, name, "DS_extracted")
        fname = name.lower()  # lower case file name
        surf.to_file(fname + ".zmap", fformat="zmap_ascii")

    print("Export done")

Take a surface in RMS and multiply values with 2:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    surf = xtgeo.surface_from_roxar(project, "TopReek", "DS_tmp")

    surf.values *= 2  # values is the masked 2D numpy array property

    # store the surface back to RMS
    surf.to_roxar(project, "TopReek", "DS_tmp")


Do operations on surfaces, also inside polygons:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Find the diff maps in time domain, of the main surfaces. Also make a
a version where cut by polygons where surfaces has interp (minimum
common multiplum)

.. code-block:: python

   import xtgeo
   from fmu.config import utilities as ut

   CFG = ut.yaml_load("../../fmuconfig/output/global_variables.yml")["rms"]

   # ========= SETTINGS ===================================================================

   PRJ = project  # noqa

   # input
   TSCAT1 = "TS_interp_raw_ow"
   PCAT = "TL_interp_raw_approx_outline"


   # output
   ISCAT1 = "IS_twt_main_interp_raw_ow"
   ISCAT2 = "IS_twt_main_interp_raw_ow_cut"

   # ========= END SETTINGS ===============================================================


   def main():

       topmainzones = CFG["horizons"]["TOP_MAINRES"]
       mainzones = CFG["zones"]["MAIN_ZONES"]
       for znum, mzone in enumerate(mainzones):

           surf1 = xtgeo.surface_from_roxar(PRJ, topmainzones[znum], TSCAT1)
           surf2 = xtgeo.surface_from_roxar(PRJ, topmainzones[znum + 1], TSCAT1)

           diff = surf2.copy()
           diff.values -= surf1.values
           diff.to_roxar(PRJ, mzone, ISCAT1, stype="zones")
           print("Store {} at {}".format(mzone, ISCAT1))

           # extract differences inside a polygon and compute min/max values:

           poly = xtgeo.polygons_from_roxar(PRJ, topmainzones[znum], PCAT)
           surf1.eli_outside(poly)
           surf2.eli_outside(poly)
           diff2 = surf2.copy()
           diff2.values -= surf1.values
           print(
              "Min and max values inside polygons {} : {} (negative OK) for {}".format(
                    diff2.values.min(), diff2.values.max(), mzone
                    )
                )
           diff2.to_roxar(PRJ, mzone, ISCAT2, stype="zones")
           print("Store cut surface {} at {}".format(mzone, ISCAT2))


    if __name__ == "__main__":
        main()
        print("Done, see <{}> and <{}>".format(ISCAT1, ISCAT2))



3D grid data
------------

Exporting geometry to ROFF file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo and export
    mygrid = xtgeo.grid_from_roxar(project, "Geomodel")

    mygrid.to_file("topreek.roff")  # roff binary is default format


Edit a porosity in a 3D grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo
    myporo = xtgeo.gridproperty_from_roxar(project, "Geomodel", "Por")

    # now I want to limit porosity to 0.35 for values above 0.35:

    myporo.values[myporo.values > 0.35] = 0.35

    # store to another icon
    poro.to_roxar(project, "Geomodel", "PorNew")


Edit a permeability given a porosity cutoff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import xtgeo

   myporo = xtgeo.gridproperty_from_roxar(project, "Geomodel", "Por")
   myperm = xtgeo.gridproperty_from_roxar(project, "Geomodel", "Perm")

   # if poro < 0.01 then perm is 0.001, otherwise keep as is, illustrated with np.where()
   myperm.values = np.where(myporo.values < 0.1, 0.001, myperm.values)

   # store to another icon
   poro.to_roxar(project, "Geomodel", "PermEdit")


Edit a 3D grid porosity inside polygons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example where I want to read a 3D grid porosity, and set value
   # to 99 inside polygons

   import xtgeo

   mygrid = xtgeo.grid_from_roxar(project, "Reek_sim")
   myprop = xtgeo.gridproperty_from_roxar(project, "Reek_sim", "PORO")

   # read polygon(s), from Horizons, Faults, Zones or Clipboard
   mypoly = xtgeo.polygons_from_roxar(project, "TopUpperReek", "DL_test")

   # need to connect property to grid geometry when using polygons
   myprop.geometry = mygrid

   myprop.set_inside(mypoly, 99)

   # Save in RMS as a new icon
   myprop.to_roxar(project, "Reek_sim", "NEWPORO_setinside")

Make a hybrid grid
^^^^^^^^^^^^^^^^^^

XTGeo can convert a conventional grid to a so-called hybrid-grid where
a certain depth interval has horizontal layers.

.. code-block:: python

   import xtgeo

   PRJ = project  # noqa
   GNAME_INPUT = "Mothergrid"
   GNAME_HYBRID = "Simgrid"
   REGNAME = "Region"
   HREGNAME = "Hregion"

   NHDIV = 22
   REGNO = 1
   TOP = 1536
   BASE = 1580


   def hregion():
       """Make a custom region property for hybrid grid"""
       tgrid = xtgeo.grid_from_roxar(PRJ, GNAME_INPUT)
       reg = xtgeo.gridproperty_from_roxar(PRJ, GNAME_INPUT, REGNAME)

       reg.values[:, :, :] = 1
       reg.values[:, 193:, :] = 0  # remember 0 base in NP arrays

       reg.to_roxar(PRJ, GNAME_INPUT, HREGNAME)  # store for info/check

       return tgrid, reg


   def make_hybrid(grd, reg):
       """Convert to hybrid and store in RMS project"""
       grd.convert_to_hybrid(nhdiv=NHDIV, toplevel=TOP, bottomlevel=BASE, region=reg,
                             region_number=1)

       grd.inactivate_by_dz(0.001)
       grd.to_roxar(PRJ, GNAME_HYBRID)


   if __name__ == "__main__":

       print("Make hybrid...")
       grd, reg = hregion()
       make_hybrid(grd, reg)
       print("Make hybrid... done!")


.. figure:: images/hybridgrid.png

Cube data
---------

Slicing a surface in a cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Examples to come...

Well data
---------

Get average properties per zone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   import xtgeo

   PRJ = project  # noqa
   WELLNAME = "DC1-1V4_ref"
   TRAJNAME = "Imported trajectory"
   ZONELOGNAME = "ZONELOG"
   ZNAMES = {0: "UPPER", 1: "MIDDLE", 2: "LOWER"}


   def get_well():
       """Get XTGeo Well() object"""
       wll = xtgeo.well_from_roxar(PRJ, WELLNAME, trajectory=TRAJNAME)
       return wll


   def compute_avg_per_zone(wll):
       """Compute avg per zone without any other criteria"""

       df = wll.dataframe
       df_avgs = df.groupby(ZONELOGNAME).mean()
       df_avgs.rename(index=ZNAMES, inplace=True)  # rename zonelog numbers with true name

       print("Average properties per zone")
        print(df_avgs)

        # e.g. get avg PORO for MIDDLE, rounded to 3 decimals:
        print("\nAVG poro for MIDDLE is {:2.3f}\n".format(df_avgs.loc["MIDDLE", "PORO"]))


    def compute_avg_per_zone_smarter(wll):
        """Compute avg per zone by looking only at intervals that increase"""

        wll.zonelogname = ZONELOGNAME
        wll.make_zone_qual_log("QUAL")

        # This quality log will be 1 if zonelog is truly increasing, or 2 if truly
        # decreasing, so here I will only here filter on increasing (downward)
        # cf: https://xtgeo.readthedocs.io/en/latest/apiref/xtgeo.well.well1.html#
        # xtgeo.well.well1.Well.make_zone_qual_log

        df = wll.dataframe[wll.dataframe.QUAL == 1]  # only get the increasing part
        df_avgs = df.groupby(ZONELOGNAME).mean()
        df_avgs.rename(index=ZNAMES, inplace=True)  # rename zonelog numbers with name

        print("\n\nAverage properties per zone where penetrating zone downwards")
        print(df_avgs)


    def main():
        mywell = get_well()
        compute_avg_per_zone(mywell)
        compute_avg_per_zone_smarter(mywell)


    if __name__ == "__main__":
        main()


Line point data
---------------

Examples to comes...
