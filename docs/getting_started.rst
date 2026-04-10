Getting started
===============

.. code-block:: python

    import xtgeo

A first step is often to import some object from file. Here we create
an XTGeo :class:`xtgeo.RegularSurface` with the
:func:`xtgeo.surface_from_file` function. In this case it defaults to loading
the file as an IRAP binary file. 

.. code-block:: python

    mysurf = xtgeo.surface_from_file("myfile.gri")

Having an existing file to load from is not required to create
XTGeo objects, in most cases. They may also be created with directly with
Python.

.. code-block:: python

    mysurf = xtgeo.RegularSurface(
        ncol=30, nrow=50, xori=1234.5, yori=4321.0, xinc=30.0, yinc=50.0,
        rotation=30.0, values=vals, yflip=1,
    )

The :class:`xtgeo.RegularSurface` class provides many methods to 
retrieve information about the data contained in this object. Here we print 
the mean of the values it contains; where ``values`` is a 2D masked numpy array.

.. code-block:: python

    print(f"Mean is {mysurf.values.mean()}")

Because values are stored internally as numpy arrays they can also be operated
on as you would any numpy array. 

.. code-block:: python

    mysurface.values[mysurface.values < 2000] = 2000

XTGeo also allows data to be operated on as a Pandas dataframe. A robust set
of convenience methods are given in XTGeo class objects for the operations
common to reservoir modelling.

Once done you can save and export the modified file. XTGeo has support for
importing and exporting the most common file formats.

.. code-block:: python

    mysurface.to_file("newfile.gri")

XTGeo has many more capabilities beyond basic value manipulation. For more, 
check out the examples in the 
:doc:`Tutorial section <tutorial/tutorial_index>`.


ResInsight (RIPS) quick start
-----------------------------

XTGeo can connect to a running ResInsight instance through the optional
``rips`` package.

Install rips in the same environment as XTGeo::

    pip install rips

Launch ResInsight with default settings, communicating through port 50051.
You can either pass the path directly::

    import rips
    rips.Instance.launch(resinsight_executable="/path/to/ResInsight", launch_port=50051)

Or set the ``RESINSIGHT_EXECUTABLE`` environment variable and pass an empty
string:

.. code-block:: bash

    export RESINSIGHT_EXECUTABLE=/path/to/ResInsight

.. code-block:: python

    import rips
    rips.Instance.launch(resinsight_executable="", launch_port=50051)

Load a ResInsight corner-point grid into XTGeo by case name::

    import xtgeo

    grid = xtgeo.grid_from_resinsight(
        instance_or_port=50051,
        case_name="EXAMPLE",
    )
    print(grid.dimensions)

If multiple ResInsight cases share the same name, XTGeo selects the last
matching case by default. Set ``find_last=False`` to pick the first match::

    import xtgeo

    grid = xtgeo.grid_from_resinsight(
        instance_or_port=50051,
        case_name="EXAMPLE",
        find_last=False,
    )

Create a new ResInsight corner-point grid from XTGeo::

    import xtgeo
    from xtgeo.common.types import Dimensions

    grid = xtgeo.create_box_grid(Dimensions(10, 10, 5))
    grid.to_resinsight(
        instance_or_port=50051,
        gname="CPG from XTGeo",
    )
