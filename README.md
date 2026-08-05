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

## A pool

```python
pool = agensgraph.ConnectionPool("host=localhost dbname=graph", graph="social",
                                 min_size=4, max_size=16)
pool.open(wait=True)          # fails here if the server is wrong, not later under load

with pool.connection(deadline=agensgraph.Deadline(5.0)) as conn:
    result = conn.execute_query("MATCH (n:Person) RETURN n")

pool.close()
```

It wraps psycopg's pool rather than reimplementing it, and adds four things psycopg has no
equivalent for. A server this driver cannot read is refused by `open()` before any worker
starts. `invalidate()` retires every connection now held in one call, so a server restart costs
one event rather than one failure per connection. `configure` runs once per new connection and
`setup` once per handout — psycopg has only the first, and a graph or a statement timeout belongs
to the second. And a `Deadline` threaded through covers both the wait for a connection and the
statement that runs on it, reaching the server as `statement_timeout`.

`get_stats()` returns psycopg's sixteen counters plus `generation` and `connections_retired`.

## Reading a large result

```python
with conn.transaction():
    for (person,) in conn.stream("MATCH (n:Person) RETURN n", size=500):
        ...
```

The rows stay on the server. `DECLARE ... CURSOR FOR MATCH` is a syntax error, so the statement
is placed where a subquery goes — which takes only the read-only subset. A statement that writes
is refused by name before anything is sent; a trailing `LIMIT` or `ORDER BY` is fine, one in the
middle is not, and the server says so. A transaction is required, and that is also what makes
abandoning the iterator safe: leaving the transaction closes the cursor with it.

## Loading a lot at once

```python
conn.load_vertices("Doc", [{"key": "a", "title": "..."}, ...])
by_key = conn.identity_map("Doc", "key")
conn.load_edges("Cites", [(by_key["a"], by_key["b"], {"weight": 1})])
```

Copying rather than a statement per row: measured at **223,000 vertices a second**, against
140,000 for a single `UNWIND ... CREATE` and 47,000 one at a time. No identity is supplied — the
column's default produces exactly the identities a `CREATE` would. Edges need the two they join,
which is what `identity_map` reads, in one statement for the whole label.

## What is in the database

There is no `\d` for a graph, so this is the way:

```python
conn.graphs()                     # every graph, with its schema and label count
conn.labels()                     # id, name, kind, and what it inherits
conn.indexes("Person")            # property indexes
conn.constraints("Person")        # including uniqueness, which the index view hides
conn.declared_properties()        # properties with a column of their own
conn.element_counts()             # per label, reading no property at all
```

`constraints()` reads the constraint catalog rather than `ag_property_indexes`, because that view
filters exclusion constraints out and a uniqueness assertion is kept as one — so it would
otherwise report a graph as having none while it has them.

## Embedding vectors

Vectors need pgvector, and unlike every other type here their oid comes from the extension rather
than the server — so it differs between databases and has to be looked up:

```python
if conn.has_vectors():
    conn.register_vectors()      # returns ('vector', 'halfvec')
```

**Registering matters more than it sounds.** A vector left in the property map is JSON, so it
arrives as a list of numbers. Give the property a column of its own and it arrives as that
column's type — and with no loader for it, as the *string* `'[1,2,3,4]'`. So a driver that reads
vectors correctly without promotion reads them wrongly with it. Both are asserted in the suite.

Two ways to index one, and only one of them is a trap:

```python
from agensgraph.vector import generated_column, expression_index, nearest

# the dimension lives on the column, so a wrong-length value is refused when written
conn.execute(f"CREATE VLABEL Emb ({generated_column('v', 1024)})")
conn.execute('CREATE INDEX ON social."Emb" USING hnsw (v vector_l2_ops)')

# or index the property where it is -- but the cast must carry the dimension
conn.execute(expression_index("social", "Doc", "v", 1024))
#   (properties->>'v')::vector      -> "column does not have dimensions"
#   (properties->>'v')::vector(1024) -> indexes

rows = conn.execute(nearest("social", "Emb", "v", limit=10), (query_vector_text,)).fetchall()
```

### Sparse vectors

`sparsevec` gets a value of its own rather than a list, and the reason is size: three non-zero
entries in a million dimensions is **36 bytes on the wire against roughly 8 MiB** as a list of
Python floats. Reading it densely would expand it 230,000-fold and discard the entire reason the
type exists.

```python
from agensgraph.vector import SparseVector

v = SparseVector({0: 1.0, 3: 2.0}, 6)   # indices, then how many dimensions
len(v)            # 2   -- what is stored
v.dimensions      # 6   -- how long it is
v.to_dict()       # {0: 1.0, 3: 2.0}
v.to_dense()      # [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]  -- asked for, never done by default
SparseVector.from_dense([1, 0, 0, 2, 0, 0]) == v
```

**Indices count from zero here.** The server's text form counts from one — `{1:9}/3` is the first
of three entries — while its binary form counts from zero. Two renderings disagreeing about the
base give an off-by-one rather than an error, so the conversion happens once, at the text boundary,
and everything in Python is zero-based like the language it's in. `to_dense()[i]` and `indices`
agree.

Whatever the server would refuse is refused on construction, where you can still do something about
it: a repeated index, an index outside the dimension, a dimension below one, `NaN`, infinity. An
explicit zero is dropped, because the server drops it.

A Cypher *list* cannot be written into a sparse column — the server refuses it, where a dense
column accepts one — so passing the value is the way in, and it works in both a Cypher property
position and a SQL cast:

```python
conn.execute("CREATE (:Emb {v: %s})", (v,))
conn.execute('SELECT id, v <-> %s::sparsevec AS d FROM social."Emb" ORDER BY d LIMIT 5', (v,))
```

### Sending dense vectors fast

Reading a dense vector needs nothing special — it arrives as a list of numbers. *Sending* one does,
because every other route formats each number as decimal text:

| sending one 1536-dimension embedding | |
|---|---|
| a `list` with a `::vector(1536)` cast | 2.55 ms |
| a string built by hand | 0.79 ms |
| **`DenseVector(values)`** | **0.32 ms** |

and in bulk, loading 20,000 embeddings of 768 dimensions:

| | rows/s |
|---|---|
| one statement at a time | 2,002 |
| `COPY` in text | 3,282 |
| **`COPY` binary with `DenseVector`** | **31,396** |

```python
from agensgraph.vector import DenseVector

conn.execute("INSERT INTO docs VALUES (%b)", (DenseVector(embedding),))

with conn.cursor().copy("COPY docs (v) FROM STDIN (FORMAT BINARY)") as copy:
    copy.set_types(["vector"])
    for embedding in embeddings:
        copy.write_row([DenseVector(embedding)])
```

The wire carries 6,148 bytes instead of 17,595, and the value survives exactly.

### Distances by name

All six of pgvector's distance operators, named — two of them differ by one character and mean
entirely different things:

```python
from agensgraph.vector import Distance

Distance.L2             # <->   Distance.L1        # <+>
Distance.COSINE         # <=>   Distance.HAMMING   # <~>  (bit strings)
Distance.INNER_PRODUCT  # <#>   Distance.JACCARD   # <%>  (bit strings)

Distance.COSINE.operator_class   # 'vector_cosine_ops', for the index
Distance.HAMMING.is_for_bits     # True
```

`conn.vector_version()` reports pgvector's version as numbers rather than a bare yes/no, because
pgvector gates its own features on it — sparse vectors and half precision arrived in 0.7.0,
iterative index scans in 0.8.0.

### Reading: ask for binary

Reading 200 embeddings of 1536 dimensions: **text 81.9 ms, binary 13.0 ms** — binary is 6.3×
faster, and the two are now asserted to produce identical values on randomly drawn floats.

## Watching it work

```python
def log(record):
    print(record.statement, record.elapsed, record.rows, record.failed)

agensgraph.add_query_logger(log)
agensgraph.enable_tracing()      # needs agensgraph-python[otel]
```

Off costs a boolean test — measured at 165 µs per statement with a logger attached against 165
with none. A clock is not read unless something is going to ask for the number. Spans are per
statement and never per row, the tracing API is imported only when asked for and the SDK never,
the record carries an opaque connection number rather than the connection's settings, and no
parameter value ever reaches a span.

## Using it from a generic tool

`agensgraph.dbapi` provides the PEP 249 names, so an ORM or migration tool can drive the driver
without knowing anything about graphs — and `connect()` there still returns a connection that
reads a vertex as a vertex. `connection.closed` and `connection.broken` are psycopg's entire
judgement about a lost connection, so nothing has to match on an error message.

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
uv run python -X dev -W error -m pytest      # -X dev is how an unclosed socket is found
uv run mypy
uv run ruff check src tests tools
uv run python tools/async_to_sync.py --check  # the blocking interface is generated
```

Most of the suite runs against no server. Tests that need a live AgensGraph carry the `server`
marker and read their connection string from `AGENSGRAPH_TEST_DSN`; without it they are skipped,
so CI asserts that they were collected rather than trusting a green run.

The blocking interface is **generated** from the awaiting one by `tools/async_to_sync.py`. Edit
`connection_async.py` and `pool_async.py`, never `connection.py` or `pool.py`, and run the tool.
Three separate checks enforce it: that a generated file matches its source, that no awaiting
module was added without being listed for conversion, and that nothing awaiting survives in a
generated file.

## Relationship to 1.x

`2.0` is a rewrite and shares no API with the `1.x` releases, which were a type-extension
module for psycopg2. `1.x` remains available on the `v1.0.2` tag.
