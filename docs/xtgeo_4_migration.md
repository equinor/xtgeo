# XTGeo 4.0 migration guide

This document contains a set of instructions on how to update your code to work
with XTGeo 4.0.

## Importing from file or Roxar

One of the most significant changes in XTGeo 4.0 relates to how files are
imported. The following table lists the style that is now deprecated with its
replacement function alongside it.

Note that the example file extension given in the old imports are merely
suggestive and that _all_ valid file extensions are subject to the same manner
of deprecation.

| Class            | Deprecated import                        | Replacement                              |
|------------------|------------------------------------------|------------------------------------------|
| `Cube`           | `Cube("x.segy")`                         | `xtgeo.cube_from_file("x.segy")`
| `Cube`           | `Cube().from_file("x.segy")`             | `xtgeo.cube_from_file("x.segy")`
| `Cube`           | `Cube().from_roxar(...)`                 | `xtgeo.cube_from_roxar(...)`
| `Grid`           | `Grid("x.grdecl")`                       | `xtgeo.grid_from_file("x.grdecl")`
| `Grid`           | `Grid().from_file("x.roff")`             | `xtgeo.grid_from_file("x.roff)"`
| `Grid`           | `Grid().from_hdf("x.h5")`                | `xtgeo.grid_from_file("x.h5")`
| `Grid`           | `Grid().from_xtgf("x.xtg")`              | `xtgeo.grid_from_file("x.xtg")`
| `Grid`           | `Grid().from_roxar(...)`                 | `xtgeo.grid_from_roxar(...)`
| `GridProperties` | `GridProperties().from_file("x.roff")`   | `xtgeo.gridproperties_from_file("x.roff")`
| `GridProperty`   | `GridProperty("x.roff")`                 | `xtgeo.gridproperty_from_file("x.roff")`
| `GridProperty`   | `GridProperty().from_file("x.roff")`     | `xtgeo.gridproperty_from_file("x.roff")`
| `GridProperty`   | `GridProperty().from_roxar(...)`         | `xtgeo.gridproperty_from_roxar(...)`
| `Points`         | `Points("x.xyz")`                        | `xtgeo.points_from_file("x.xyz")`
| `Points`         | `Points().from_file("x.xyz")`            | `xtgeo.points_from_file("x.xyz")`
| `Points`         | `Points().from_roxar(...)`               | `xtgeo.points_from_roxar(...)`
| `Points`         | `Points().from_surface(...)`             | `xtgeo.points_from_surface(...)`
| `Points`         | `Points().from_wells(...)`               | `xtgeo.points_from_wells(...)`
| `Points`         | `Points().dfrac_from_wells(...)`         | `xtgeo.points_from_wells_dfrac(...)`
| `Polygons`       | `Polygons("x.xyz")`                      | `xtgeo.polygons_from_file("x.xyz")`
| `Polygons`       | `Polygons().from_file("x.xyz")`          | `xtgeo.polygons_from_file("x.xyz")`
| `RegularSurface` | `RegularSurface("x.gri")`                | `xtgeo.surface_from_file("x.gri")`
| `RegularSurface` | `RegularSurface().from_file("x.gri")`    | `xtgeo.surface_from_file("x.gri")`
| `RegularSurface` | `RegularSurface().from_hdf("x.h5")`      | `xtgeo.surface_from_file("x.h5")`
| `RegularSurface` | `RegularSurface().from_roxar(...)`       | `xtgeo.surface_from_roxar(...)`
| `RegularSurface` | `RegularSurface().from_cube("x.segy")`   | `xtgeo.surface_from_cube("x.segy")`
| `RegularSurface` | `RegularSurface().from_grid3d("x.roff")` | `xtgeo.surface_from_grid3d("x.roff")`
| `Surfaces`       | `Surfaces().from_grid3d("x.roff")`       | `xtgeo.surfaces_from_grid("x.roff")`
| `BlockedWell`    | `BlockedWell().from_roxar(...)`          | `xtgeo.blockedwell_from_roxar(...)`
| `BlockedWells`   | `BlockedWells().from_files(...)`         | `xtgeo.blockedwells_from_files(...)`
| `BlockedWells`   | `BlockedWells().from_roxar(...)`         | `xtgeo.blockedwells_from_roxar(...)`
| `Well`           | `Well("x.rmswell")`                      | `xtgeo.well_from_file("x.rmswell")`
| `Well`           | `Well().from_file("x.rmswell")`          | `xtgeo.well_from_file("x.rmswell")`
| `Well`           | `Well().from_hdf("x.h5")`                | `xtgeo.well_from_file("x.h5")`
| `Well`           | `Well().from_roxar(...)`                 | `xtgeo.well_from_roxar(...)`
| `Wells`          | `Wells(["x.rmswell"])`                   | `xtgeo.wells_from_files(["x.rmswell"])`
| `Wells`          | `Wells().from_files(["x.rmswell"])`      | `xtgeo.wells_from_files(["x.rmswell"])`

## Instantiating XTGeo objects

In addition to the deprecations related to creating XTGeo objects from file
some XTGeo objects have deprecated alternative ways of being initialized.

The general principle is that default initializations are no longer supported.

### Cube

Cubes have previously given default values for empty initializations. This
means that you could do something like

```python
import xtgeo

# ncol, nrow, nlay, xinc, yinc, zinc set to default values
mycube = xtgeo.Cube()  # ⛔️ no longer allowed!
```

This pattern is no longer allowed and explicit values must be provided for at
least the following keywords.

```python
import xtgeo

mycube = xtgeo.Cube(
    ncol=40,
    nrow=30,
    nlay=10,
    xinc=25.0,
    yinc=25.0,
    zinc=2.0,
)
```

### Grid

Grids have previously given default values for empty initializations. This
means that if you wanted to create a default box grid you could do

```python
import xtgeo

# Box grid with shape ncol=4, nrow=3, nlay=5
mygrid = xtgeo.Grid()  # ⛔️ no longer allowed!
```

This functionality has been replaced with `xtgeo.create_box_grid()`:

```python
import xtgeo

mybox = xtgeo.create_box_grid(4, 3, 5)
```

### GridProperties

Creating a `GridProperties()` instance with dimensions is deprecated. Dimensions
will be inferred from any provided properties.

```python
import xtgeo

gprops = xtgeo.GridProperties(3, 4, 5)  # ⛔️ no longer allowed!
```

### GridProperty

GridProperty's have previously given default values for empty initializations.
This is now deprecated.

```python
import xtgeo

# ncol=4, nrow=3, nlay=4, values=np.ndarray[...]
gprop = xtgeo.GridProperty()  # ⛔️ no longer allowed!
```

These values must now be provided explicitly.

```python
import xtgeo

# values=np.ndarray[...]
gprop = xtgeo.GridProperty(ncol=4, nrow=3, nlay=5)
```

### Points

It's previously been possible to initialize a `Points()` object from a
`RegularSurface`. This is now deprecated.

```python
import xtgeo

surf = xtgeo.surface_from_file("some.gri")
mypoints = xtgeo.Points(surf)  # ⛔️ no longer allowed!
```

This functionality has been replaced with an explicit function for this
purpose.

```python
import xtgeo

surf = xtgeo.surface_from_file("some.gri")
mypoints = xtgeo.points_from_surface(surf)
```

### RegularSurface

RegularSurface's have previously given default values for empty initializations.
This is now deprecated.

```python
import xtgeo

# ncol=4, nrow=3, xinc = 25.0, yinc = 25.0, values = [[..], ..]
surf = xtgeo.RegularSurface()  # ⛔️ no longer allowed!
```

These values must be provided explicitly. Note that if `values=...` is not
provided it defaults to an array of zeroes.

```python
import xtgeo

surf = xtgeo.RegularSurface(ncol=4, nrow=3, xinc=25.0, yinc=25.0)
```

### Well

Wells can no longer be initialized empty.

```python
import xtgeo

# rkb=0.0, xpos=0.0, ypos=0.0, wname="", df={..=[]}
well = xtgeo.Well()  # ⛔️ no longer allowed!
```

A well must be created with these values set explicitly.

```python
import xtgeo

well = xtgeo.Well(
    rkb=100.0,
    xpos=0.0,
    ypos=0.0,
    wname="OP_1",
    df=some_dataframe
)
```

## xtgeo.plot

The `xtgeo.plot` module is now deprecated. This functionality has been moved
to its own package called `xtgeoviz` which maintains the same API. To update
just change

```python
import xtgeo.plot  # ⛔️ no longer allowed! use:
import xtgeoviz.plot  # same functionality, new package
```

In the near term we no longer automatically expose the `plot` module directly
from the `xtgeo` namespace. This means that you cannot use a pattern like
this:

```python
import xtgeo

someplot = xtgeo.plot.Map()  # ⛔️ no longer allowed!
```

You must import `xtgeo.plot` explicitly:

```python
import xtgeo.plot

someplot = xtgeo.plot.Map()  # ✅
```

For more information about xtgeoviz feel free to check out its repository at
[https://github.com/equinor/xtgeoviz](https://github.com/equinor/xtgeoviz)
and its documentation at
[https://equinor.github.io/xtgeoviz/](https://equinor.github.io/xtgeoviz/).

## API Deprecations

This section describes method or property deprecations that apply to class
methods.

### Cube

segy scanning methods have been deprecated. Please use
[segyio](https://segyio.readthedocs.io/en/latest/) if you still need the
information these functions provided.

- `Cube.scan_segy_traces()` - deprecated
- `Cube.scan_segy_header()` - deprecated

### Grid

The following methods are deprecated but with replacements already in place.

- `Grid().create_box()` is deprecated. Use `xtgeo.create_box_grid()` instead.
- `Grid().dataframe` is deprecated. Use `Grid().get_dataframe()` instead.
- `Grid().get_gridproperties()` is deprecated. Use `Grid().gridprops` instead.
- `Grid().get_dxdy()` is deprecated. Use `Grid().get_dx()` and
  `Grid().get_dy()` instead.

These methods are deprecating the `mask` argument that they can receive. This
argument has been changed to `asmasked` which is already in place.

- `Grid().get_actnum(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `Grid().get_actnum(asmasked=True)`.
- `Grid().get_dz(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `Grid().get_dz(asmasked=True)`.
- `Grid().get_ijk(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `Grid().get_ijk(asmasked=True)`.
- `Grid().get_xyz(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `Grid().get_xyz(asmasked=True)`.

Additionally,

- `Grid().report_zone_mismatch()` has deprecated the `onelayergrid` option.
  This option is redundant and unneeded.
- `Grid().grid_from_roxar()` has deprecated the `dimensions_only` option.
  This option is redundant and unneeded.

### GridProperties

- Setting `GridProperties().names` has been deprecated. Names can still be set
  in the following manner:
  ```python
  for prop in gridprops:
      prop.name = newname
  ```
- `GridProperties().dataframe()` has been deprecated. Use
  `GridProperties().get_dataframe()` instead.
- `GridProperties.scan_keywords()` has been deprecated. Use
  `xtgeo.list_gridproperties()` instead.

These methods are deprecating the `mask` argument that they can receive. This
argument has been changed to `asmasked` which is already in place.

- `GridProperties().get_actnum(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `GridProperties().get_actnum(asmasked=True)`.
- `GridProperties().get_ijk(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `GridProperties().get_ijk(asmasked=True)`.

### GridProperty

- `GridProperty().values3d` has been deprecated. Use `GridProperty().values`
  instead.
- `GridProperty().get_actnum(mask=True)` is deprecated.
  Use `asmasked` instead, e.g. `GridProperty().get_actnum(asmasked=True)`.

### Points

- `Points().from_list()` is deprecated. Use a direct initialization instead as
  `Points(values=plist)`.
- `Points().from_dataframe()` is deprecated. Use
  `Points(values=df[[east, north, tvdmsl]], xname=east, yname=north, zname=tvdmsl)`
  instead.

### Polygons

- `Polygons().from_list()` is deprecated. Use a direct initialization instead as
  `Polygons(plist)`.

### RegularSurface

- `RegularSurface().quickplot(colortable=...)` the `colortable=` argument is
  deprecated. Use `quickplot(colormap=...)` instead.
