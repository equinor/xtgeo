# ResInsight Folder Traversal Adapters

## Overview

XTGeo resolves nested ResInsight folders from paths such as `Regional/Depth Maps`.
The traversal algorithm is simple: split the path, inspect each node's direct
children, select a matching child, and optionally create a missing child.

The difficulty is that the `rips` hierarchies do not expose one uniform folder
API. Surface and polygon folders differ in their child access methods, recursive
node types, name attributes, and runtime return types. The traversal adapters
make those differences explicit while retaining one implementation of the path
algorithm.

The design has four parts:

- `_resolve_folder()` implements hierarchy-independent path traversal.
- `_FolderOps[_NodeT]` defines the two operations traversal needs: `children()`
  and `add()`.
- `_SurfaceFolderOps` translates the surface collection API into that contract.
- `_PolygonFolderOps` translates the polygon container API into that contract.

The existing `resolve_folder()` function remains the surface-specific entry
point. `resolve_polygon_folder()` is a separate polygon-specific entry point.
This preserves the established surface API and gives each hierarchy an accurate
return type.

## Why the previous design was suboptimal

### It mixed the algorithm with one hierarchy's API

The previous `resolve_folder()` directly called
`SurfaceCollection.sub_collections()` and `SurfaceCollection.add_folder()`.
Path parsing, name lookup, missing-folder behavior, and surface-specific access
were all implemented in the same function.

That was adequate while surfaces were the only supported hierarchy. It became a
poor extension point once polygon traversal was needed. Reusing the function for
polygons would have required conditionals inside the traversal loop or a broad
protocol pretending that every hierarchy had the same methods.

For example, these child operations are not interchangeable:

```python
# Surface hierarchy
children = surface_folder.sub_collections()

# Polygon hierarchy
children = polygon_folder.children("SubCollections", rips.RimPolygonContainer)
```

Adding checks such as `if polygon: ... else: ...` to the traversal loop would
make every new hierarchy increase the complexity of the shared algorithm. The
adapters instead keep `_resolve_folder()` unaware of how children are obtained.

### The unchecked cast hid missing runtime evidence

The previous surface implementation effectively did this when creating a
folder:

```python
child = cast(
    "RipsSurfaceCollectionType",
    folder.add_folder(
        folder_name=segment,
        on_name_conflict=NameConflictPolicy.FAIL,
    ),
)
```

A `cast()` changes what a static type checker believes; it does not inspect or
convert the object at runtime. The code therefore promised that `add_folder()`
returned a `SurfaceCollection` without proving it. If `rips` returned a broader
PDM object or an unexpected object, the error would surface later as an
unrelated missing-attribute failure.

The surface adapter now establishes that evidence at the integration boundary:

```python
child = node.add_folder(
    folder_name=name,
    on_name_conflict=NameConflictPolicy.FAIL,
)
if not isinstance(child, rips.SurfaceCollection):
    raise RuntimeError("ResInsight returned an invalid surface folder type")
return child
```

This both narrows the static type and validates the runtime contract. A bad
`rips` result fails immediately with an error that identifies the violated
hierarchy contract.

### A root-preserving generic was not valid for every hierarchy

Surface traversal is root-preserving:

```text
SurfaceCollection -> SurfaceCollection -> SurfaceCollection
```

Polygon traversal is different. Its root is a `PolygonCollection`, while direct
and nested children are represented by the broader `RimPolygonContainer` type:

```text
PolygonCollection -> RimPolygonContainer -> RimPolygonContainer
```

A generic function of the form below would therefore be unsound for polygons:

```python
def resolve(root: T, ...) -> T:
    ...
```

With a `PolygonCollection` root, that signature claims every resolved child is
also a `PolygonCollection`. The documented recursive type is
`RimPolygonContainer`, so the claim is too narrow.

The adapter design chooses the recursive node type independently for each
hierarchy:

- Surface operations use `RipsSurfaceCollectionType` as `_NodeT`.
- Polygon operations use `RipsRimPolygonContainerType` as `_NodeT`.
- `resolve_polygon_folder()` accepts the narrower polygon root but returns the
  broader, correct recursive node type.

### A universal folder protocol would overstate the `rips` API

A structural protocol containing methods such as `sub_collections()` and
`add_folder()` would appear convenient, but polygon traversal does not use the
same child access contract as surface traversal. Making the protocol broad
enough for both would either require optional methods or weaken types to `Any`.
Both approaches move hierarchy knowledge into the generic algorithm and reduce
the value of static checking.

`_FolderOps` instead describes what the algorithm needs, not what all `rips`
objects supposedly provide:

```python
class _FolderOps(Protocol[_NodeT]):
    def children(self, node: _NodeT) -> Iterable[_NodeT]: ...
    def add(self, node: _NodeT, name: str) -> _NodeT: ...
```

The protocol is deliberately small. Each adapter is responsible for satisfying
it using the real API of one hierarchy.

## How traversal now works

The generic core owns only behavior common to nested paths:

1. Ignore empty path segments.
2. Ask the adapter for the current node's direct children.
3. Select the first child with the requested hierarchy-specific name.
4. Raise `RuntimeError` when a segment is missing and creation is disabled.
5. Ask the adapter to create the child when creation is enabled.
6. Continue traversal using the typed child returned by the adapter.

Name attributes remain wrapper configuration because the object models differ:

- Surface folders use `surface_user_description`.
- Polygon containers use `polygon_collection_name`.

Folders are created with `NameConflictPolicy.FAIL`. Existing folders are found
before creation, so resolving the same path repeatedly reuses its nodes rather
than replacing folders and potentially deleting their contents.

## Concrete examples

### Example 1: creating and reusing a surface path

```python
folder = resolve_folder(
    surface_root,
    "Regional/Depth Maps",
    "surface_user_description",
    create=True,
)
```

The core first asks `_SurfaceFolderOps.children(surface_root)`, which delegates
to `surface_root.sub_collections()`. If `Regional` is absent, the adapter creates
it and verifies that the result is a `rips.SurfaceCollection`. The same process
then resolves `Depth Maps`.

Calling the function again with the same path finds both existing collections.
No `add_folder()` call is made, and the existing `Depth Maps` object is returned.

### Example 2: traversing polygon containers

```python
folder = resolve_polygon_folder(
    polygon_root,
    "Boundaries/Approved",
    create=True,
)
```

The polygon adapter obtains children with:

```python
node.children("SubCollections", rips.RimPolygonContainer)
```

The class filter matters because polygon subcollections may (or may not)
have different concrete types,
all represented through the `RimPolygonContainer` base type. The resolved
object is therefore typed as `RipsRimPolygonContainerType`, not incorrectly as
the root's `RipsPolygonCollectionType`.

The caller does not need to know the PDM field keyword or repeat runtime class
filtering. Those details belong to `_PolygonFolderOps`.

### Example 3: rejecting an invalid creation result

Suppose an incompatible `rips` version or API defect causes a surface
`add_folder()` call to return an unrelated PDM object. The old cast would accept
it silently:

```python
# Static assertion only; no runtime check occurs.
child = cast("RipsSurfaceCollectionType", unexpected_object)
```

The adapter rejects it at once:

```text
RuntimeError: ResInsight returned an invalid surface folder type (...)
```

The same boundary validation exists for polygon creation using
`rips.RimPolygonContainer`. Unit tests deliberately return `object()` from
`add_folder()` to verify both failures.

### Example 4: why cases should not get a folder adapter

Cases are selected from the flat result of `Project.cases()`. They do not form a
recursive path hierarchy and do not expose folder creation. Their lookup is
already handled by the generic `select_by_name()` primitive and the dedicated
case resolver:

```python
case = select_by_name(project.cases(), "MODEL", find_last=True)
```

Forcing cases into `_FolderOps` would require invented `children()` and `add()`
semantics. This example illustrates an important boundary of the design: add an
adapter only for an actual recursive hierarchy. Share the smaller lookup helper
for flat named collections.

## Benefits and tradeoffs

The adapters provide:

- Accurate static types for each hierarchy.
- Runtime validation where the generated API has broad return types.
- One path traversal algorithm without hierarchy conditionals.
- A stable surface resolver signature for existing callers.
- Localized knowledge of PDM child fields and concrete `rips` classes.
- Focused fake-object tests plus live ResInsight integration coverage.

The design adds a small amount of indirection. A reader must move from the
public wrapper to an adapter and then to `_resolve_folder()`. That cost is
intentional: it makes external API assumptions visible and testable instead of
hiding them in casts or conditionals. A new adapter should be introduced only
when another real recursive hierarchy has different access or validation rules.
