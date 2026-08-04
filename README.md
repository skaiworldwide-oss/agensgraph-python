# AgensGraph Python Driver

A driver for [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), built on
[psycopg 3](https://www.psycopg.org/psycopg3/). It reads the graph types the server
returns — `graphid`, `vertex`, `edge` and `graphpath` — as Python values, in both the text
and the binary wire format.

Requires Python 3.11 or later and AgensGraph 2.16 or later.

## Install

```sh
pip install agensgraph-python
```

## Graph values

A vertex and an edge are values rather than handles. Each is immutable, compares and
hashes on its identity alone, and so can be used as a dictionary key or a set member.

```python
from agensgraph import Edge, GraphId, Vertex

v = Vertex(GraphId(3, 1), "person", {"name": "Arthur"})
v.id.labid, v.id.locid       # (3, 1)
v.label                      # 'person'
v.properties["name"]         # 'Arthur'
v.get("nickname", "none")    # 'none'
{v: "seen"}                  # hashable
```

A property map is decoded on first access rather than when the value is built, so a
result whose properties are never read is never parsed.

A path is a sequence of alternating vertices and edges. Indexing and iteration walk the
elements in the order the server wrote them, and `len` counts elements, so a path of one
vertex and no edges has length one and is truthy. The number of hops is `path.length`.

```python
p.vertices    # (Vertex, ...)
p.edges       # (Edge, ...)
p.start, p.end
p[0], p[1]    # first vertex, first edge
p.length      # hop count
```

A dynamic label or property key cannot be bound as a parameter, so the driver requires a
`Label` at any call site that places one into a query. That makes an unquoted
interpolation impossible to write by accident.

## Development

```sh
uv sync --group dev
uv run pytest
uv run mypy
uv run ruff check
```

The test suite runs against no server. Tests that need a live AgensGraph carry the
`server` marker and read their connection string from `AGENSGRAPH_TEST_DSN`.

## Versions

`2.0` is a rewrite on psycopg 3 and shares no API with the `1.x` releases, which were a
type-extension module for psycopg2. `1.x` remains available on the `v1.0.2` tag.
