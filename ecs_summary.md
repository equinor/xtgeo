Main issue:
The `SurfaceCollection` and `PolygonCollection` hierarchies are not compatible.
Although `SurfaceCollection` and `PolygonCollection` share a common base class,
their folder traversal APIs and recursive return types differ.

When methods like `add_folder()` and `children()` live at different levels
in the class hierarchy, things become troublesome.
Surface folders recurse as `SurfaceCollection`, while polygon folders recurse
through the broader `RimPolygonContainer` type. Their child-access APIs also
differ, making one directly typed traversal implementation difficult.


Surface traversal is root-preserving:

```text
SurfaceCollection -> SurfaceCollection -> SurfaceCollection
```

Polygon traversal is different. Its root is a `PolygonCollection`, while direct
and nested children are represented by the broader `RimPolygonContainer` type:

```text
PolygonCollection -> RimPolygonContainer -> RimPolygonContainer
```

Is it an option to constrain ourselves to this?
```text
PolygonCollection -> PolygonCollection -> PolygonCollection
```
A) If ResInsight actually guaranteed that polygon children were `PolygonCollection`,
a root-preserving generic could simplify the typing.
ResInsight's documented contract does not provide that guarantee.

B) xtgeo could raise an error if the polygon children were not `PolygonCollection`.
Then the xtgeo interface to ResInsight is not compatible with the ResInsight API



## Alternatives to traversal adapters

- **Use `Any` or broad base-class types.**
    A single traversal function can call
	hierarchy-specific methods with casts or suppressed type errors. This is the
	simplest option, but static checking can no longer detect invalid method calls
	or return types. Incompatible `rips` objects may then fail later, far from the
	integration boundary where they were produced.

- **Put hierarchy conditionals in the traversal loop.**
    The resolver can branch
	between surface and polygon operations at each step. This avoids adapters but
	mixes path traversal with external API details; every additional hierarchy
	adds more branches and makes the shared algorithm harder to test and extend.

- **Maintain separate traversal functions.**
    Each hierarchy can have direct,
	loosely typed code tailored to its API. This is easy to understand locally,
	but duplicates path parsing, lookup, creation, and error handling, increasing
	the risk that behavior diverges between implementations.

