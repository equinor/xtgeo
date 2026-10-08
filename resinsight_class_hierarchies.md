# ResInsight Class Hierarchies

## Python Inheritance

```mermaid
classDiagram
    direction TB

    object <|-- PdmObjectBase
    PdmObjectBase <|-- PdmNestedCollectionBase

    PdmNestedCollectionBase <|-- SurfaceCollection
    PdmNestedCollectionBase <|-- RimPolygonContainer
    RimPolygonContainer <|-- PolygonCollection

    class PdmObjectBase {
        children(field, class)
        descendants(class)
    }

    class PdmNestedCollectionBase {
        add_folder(name)
    }

    class SurfaceCollection {
        surface_user_description
        sub_collections() List~SurfaceCollection~
    }

    class RimPolygonContainer {
        polygon_collection_name
    }

    class PolygonCollection {
        polygons()
        sub_collections() List~RimPolygonContainer~
    }
```

## Traversal Shapes

```mermaid
flowchart LR
    subgraph S["Surface folders: type-preserving"]
        S0["SurfaceCollection<br/>root"]
        S1["SurfaceCollection<br/>Regional"]
        S2["SurfaceCollection<br/>Depth Maps"]

        S0 -->|"sub_collections()"| S1
        S1 -->|"sub_collections()"| S2
    end

    subgraph P["Polygon folders: type broadening"]
        P0["PolygonCollection<br/>root"]
        P1["RimPolygonContainer<br/>Boundaries"]
        P2["RimPolygonContainer<br/>Approved"]

        P0 -->|"SubCollections"| P1
        P1 -->|"SubCollections"| P2
    end
```

The crucial point is:

```text
PolygonCollection is-a RimPolygonContainer
RimPolygonContainer contains RimPolygonContainer children
```

Thus a `PolygonCollection` root satisfies the container interface, but traversing
into it produces the broader `RimPolygonContainer` type. Surfaces do not broaden:
every level remains `SurfaceCollection`.

This is what the adapters encode in
[`_resinsight_base.py`](src/xtgeo/interfaces/resinsight/_resinsight_base.py#L62)
and
[`_resinsight_base.py`](src/xtgeo/interfaces/resinsight/_resinsight_base.py#L85).


Sources: [PolygonCollection](https://api.resinsight.org/en/main/api/rips.PolygonCollection.html),
[RimPolygonContainer](https://api.resinsight.org/en/main/api/rips.RimPolygonContainer.html),
and [SurfaceCollection](https://api.resinsight.org/en/main/api/rips.SurfaceCollection.html).