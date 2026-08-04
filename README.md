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

## Two wire formats

The driver never rewrites a query. `RETURN n` is sent as written, the server answers in
its text rendering, and the driver reads that — which is the default and needs nothing
from the server beyond the answer itself.

The same query can be asked for in the composite rendering instead, per statement. That
form leaves out the label name, so it needs a label table for the connection, and it is
worth asking for where a result carries paths or element arrays: on those, measured over
2,000 paths, it reads `nodes(p)` in 32 ms against 78 ms and `RETURN p` in 61 ms against
97 ms. On whole vertices the two are within 2% of each other, so it buys nothing there.

Both renderings produce the same objects, and the test suite asserts that against values
the server itself produced.

## Versions

The driver reads AgensGraph 2.16 and later. It learns which it is talking to from the
`agversion` parameter the server sends at startup, so the check costs no round trip.

Everything above works on every supported version. Four features do not exist before
2.18, and asking for one on an older server says so rather than letting the server fail
on syntax it has never seen:

```python
caps = agensgraph.Capabilities.of(conn)
caps.has_property_promotion()   # a property stored in a column of its own
caps.has_gql_clauses()          # LET, NEXT, FINISH, FILTER, FOR, CALL
caps.has_element_ordering()     # ORDER BY on a vertex or an edge
caps.has_endpoint_elision()     # visible in a plan
```

## Failures

What kind of error something is comes from psycopg, so a graph failure is caught by the
same PEP-249 class as any other. What to *do* about it is a separate question the class
cannot answer, and `agensgraph.errors` answers it:

```python
from agensgraph.errors import Retryability, retryability

recovery = retryability(exc, wrote=True)
recovery.is_retryable
recovery.needs_new_connection
```

A graph write conflict is reported as `40001`, the same code an ordinary row conflict
uses, and it is the ordinary outcome of two writers touching one element rather than an
exotic one. Six recoveries are distinguished, because a boolean cannot tell a lost
connection from a stale statement cache, and anything unrecognised is treated as fatal.

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
