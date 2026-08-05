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

A label or a property key cannot be bound as a parameter — the grammar has no place for one
there — so a statement naming one dynamically has to carry it in its text. `agensgraph.cypher`
is where that quoting lives:

```python
from agensgraph.cypher import quote_identifier

quote_identifier("Person")      # 'Person'
quote_identifier("my label")    # '"my label"'
quote_identifier('a"b')         # '"a""b"'
quote_identifier("MATCH")       # '"MATCH"'
```

A name holding a null byte is refused rather than quoted, because the server's lexer stops at
one and the statement would end somewhere other than where it appears to.

## Connecting

```python
import agensgraph

with agensgraph.connect("host=localhost dbname=graph") as conn:
    conn.graph("social")
    result = conn.execute_query("MATCH (n:Person) RETURN n")
    for (person,) in result.records:
        print(person.properties["name"])
```

The awaiting interface is the same, awaited:

```python
conn = await agensgraph.AsyncConnection.connect("host=localhost dbname=graph")
async with conn:
    await conn.graph("social")
    result = await conn.execute_query("MATCH (n:Person) RETURN n")
```

Both are built on psycopg's own connection, so cursors, server cursors, `COPY`, pipelines,
`LISTEN`, transactions and plain SQL are all there unchanged. The transaction model is
psycopg's: a statement outside a transaction opens one, `commit()` and `rollback()` are on
the connection, closing rolls back, and `autocommit=True` is how a statement that cannot run
inside a transaction gets to run.

`conn.graph(name)` selects a graph and fills the label table the composite rendering needs.
The name is quoted rather than bound, because the grammar has no place for a parameter there.

## Parameters

Parameters are psycopg's — `%s`, or `%(name)s`, never string formatting. Pass plain Python
values:

```python
conn.execute_query("MATCH (n:Person) WHERE n.name = %s RETURN n", ("Arthur",))
conn.execute_query("MATCH (n:Person) RETURN n LIMIT %s", (10,))
conn.execute_query("CREATE (:Person %s)", ({"name": "Arthur", "age": 42},))
conn.execute_query("MATCH (n) WHERE id(n) = %s RETURN n", (vertex.id,))
```

A string is sent as `text` rather than with its type left for the server to work out. That
matters more than it sounds. Cypher reads almost every parameter as `jsonb`, and an untyped
one is handed straight to jsonb's parser — so `"123"` would arrive as the *number* 123,
`"null"` as a JSON null, and a search for either would quietly match nothing. Saying the type
reaches the server's own `cypher_to_jsonb` conversion instead, which keeps a string a string.

The price is one thing: in plain SQL, a string standing in for some other type now wants a
cast, because the server no longer guesses.

```python
conn.execute("select * from t where d = %s::date", ("2026-08-05",))   # cast
conn.execute("select * from t where d = %s", (date(2026, 8, 5),))     # or the real type
conn.execute("select * from t where d = %s",
             (agensgraph.Unspecified("2026-08-05"),))                 # or opt back out
```

Passing the value's own type — `date`, `UUID`, `int`, `Decimal`, `bytes` — behaves exactly as
psycopg does, as do all `text`, `varchar` and `name` positions. When a cast is missing the
server says so while parsing, naming both types, and the driver attaches a note pointing at
the fix. This is what the PostgreSQL JDBC driver does by default, so AgensGraph's own Java
driver already works this way.

A `list` stays a PostgreSQL array, since plain SQL on the same connection needs it to be one.
Wrap it as `Jsonb([...])` for a JSON array.

One shape is refused before it is sent:

```python
conn.execute_query("MATCH (a)-[r*1..%s]->(b) RETURN a", (3,))   # ValueError
```

The server accepts that and reads the parameter as a property map, so the statement prepares,
reports its parameter as `jsonb`, and matches a walk of *any* length. Every other position
that cannot take a parameter reports a syntax error of its own and is left to the server.

## What a write changed

```python
result = conn.execute_query("CREATE (:Person {name: 'a'})", counts_=True)
result.counts.inserted_vertices     # 1
result.counts.complete              # True
```

The server resets its counters unevenly. A write with no `RETURN` zeroes all five before it
runs, so all five belong to it. A write *with* a `RETURN` zeroes only the counters for the
clauses it has, and the rest still hold whatever an earlier statement left. So:

```python
conn.execute_query("CREATE (:Person {name: 'a'}), (:Person {name: 'b'})")
result = conn.execute_query("MATCH (n:Person) SET n.x = 1 RETURN n", counts_=True)
result.counts.updated_properties    # 2
result.counts.inserted_vertices     # None -- not 2, which is the earlier statement's
result.counts.complete              # False
```

`None` means the counter was not answered for. It is never reported as zero, and a total that
is missing one of its terms is not reported at all.

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

## Relationship to 1.x

`2.0` is a rewrite and shares no API with the `1.x` releases, which were a type-extension
module for psycopg2. `1.x` remains available on the `v1.0.2` tag.
