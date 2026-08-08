# agensgraph-python

[![PyPI](https://img.shields.io/pypi/v/agensgraph-python.svg)](https://pypi.org/project/agensgraph-python/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/agensgraph-python/)
[![AgensGraph](https://img.shields.io/badge/AgensGraph-2.17%2B-1f6feb.svg)](https://github.com/skaiworldwide-oss/agensgraph)
[![Tests](https://github.com/skaiworldwide-oss/agensgraph-python/actions/workflows/python-driver-test.yaml/badge.svg)](https://github.com/skaiworldwide-oss/agensgraph-python/actions/workflows/python-driver-test.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Read and write an [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph) graph from Python,
in blocking or awaiting code, over [psycopg 3](https://www.psycopg.org/psycopg3/).

A vertex, an edge and a path come back as values you can use rather than strings you have to parse.
Cypher and SQL run on the same connection, in the same transaction, so a graph traversal and a join
are one query when you want them to be. Everything psycopg does (cursors, server-side cursors,
`COPY`, pipelines, `LISTEN`, two-phase commit) keeps working, because the connection *is* a psycopg
connection.

```python
import agensgraph

with agensgraph.connect("host=localhost dbname=graph") as conn:
    conn.graph("social")
    result = conn.execute_query("MATCH (n:Person) RETURN n")
    for (person,) in result.records:
        print(person.properties["name"])
```

## Install

```console
pip install agensgraph-python
```

Python 3.11 or later. AgensGraph 2.17 or later.

Optional extras, imported only where they are used, so they never load on a read path:

```console
pip install "agensgraph-python[arrow]"     # to_arrow
pip install "agensgraph-python[pandas]"    # to_pandas
pip install "agensgraph-python[polars]"    # to_polars
pip install "agensgraph-python[otel]"      # spans through opentelemetry-api
```

## Contents

**Basics**
[Connecting](#connecting) ·
[Running a statement](#running-a-statement) ·
[Graph values](#graph-values) ·
[Parameters](#parameters) ·
[Identifiers](#identifiers)

**Reading**
[Two wire formats](#two-wire-formats) ·
[Streaming a large result](#streaming-a-large-result) ·
[Numbers](#numbers-in-a-property-map) ·
[Columnar export](#columnar-export)

**Writing**
[What a write changed](#what-a-write-changed) ·
[Bulk loading](#bulk-loading) ·
[A burst of statements](#a-burst-of-statements)

**Schema**
[Reading the catalogs](#reading-the-catalogs) ·
[Declaring what should exist](#declaring-what-should-exist)

**Vectors**
[Storing and reading](#embedding-vectors) ·
[Indexing and search](#indexing-and-searching) ·
[Sparse vectors](#sparse-vectors) ·
[Tuning](#tuning-a-search)

**Running it in production**
[Pooling](#pooling) ·
[Deadlines](#deadlines) ·
[Retrying](#retrying) ·
[Losing a connection mid-commit](#losing-a-connection-mid-commit) ·
[Cancellation](#cancellation) ·
[When the network goes quiet](#when-the-network-goes-quiet)

**Everything else**
[Observability](#observability) ·
[Untrusted statements](#running-a-statement-you-did-not-write) ·
[Errors](#errors) ·
[LISTEN and NOTIFY](#listen-and-notify) ·
[Two-phase commit](#two-phase-commit) ·
[Generic tools and SQLAlchemy](#generic-tools-and-sqlalchemy) ·
[Server versions](#server-versions) ·
[Performance](#performance) ·
[API reference](#api-reference) ·
[Development](#development)

## Connecting

```python
import agensgraph

with agensgraph.connect("host=localhost dbname=graph") as conn:
    conn.graph("social")
    ...
```

The awaiting interface is the same interface, awaited:

```python
conn = await agensgraph.AsyncConnection.connect("host=localhost dbname=graph")
async with conn:
    await conn.graph("social")
    result = await conn.execute_query("MATCH (n:Person) RETURN n")
```

Only the methods that wait for the server are written twice. Everything else (adapters, the label
table, statement checks, building a result) exists once and is shared, and the blocking interface is
generated from the awaiting one, so the two cannot drift.

`conn.graph(name)` selects a graph and fills the label table that the composite rendering needs. The
name is quoted rather than bound, because the grammar has no place for a parameter there. The table
is dropped again automatically when anything moves the session somewhere else, including a rollback
that undoes the move.

**Transactions are psycopg's**, unchanged: a statement outside a transaction opens one, `commit()`
and `rollback()` are on the connection, closing rolls back, and `autocommit=True` is how a statement
that cannot run inside a transaction gets to run.

**The server is checked at connect.** The version arrives in the startup packet, so refusing a
server this driver cannot read costs no round trip and happens before the first statement rather
than at whichever later one first wanted a catalog the server has never had.

## Running a statement

```python
result = conn.execute_query("MATCH (n:Person) WHERE n.name = %s RETURN n", ("Arthur",))

result.records     # a list of tuples
result.keys        # the column names, read once per result
result.counts      # what the statement changed, if you asked
result.oids        # the type of each column, as the server described it
```

`execute_query` takes the statement, the parameters as a sequence, and reserved arguments ending in
an underscore:

| | |
|---|---|
| `binary_=True` | ask for the composite rendering, see [Two wire formats](#two-wire-formats) |
| `counts_=True` | read what the statement changed, see [What a write changed](#what-a-write-changed) |
| `prepare_=` | override when psycopg prepares the statement |
| `row_=` | a psycopg row factory |

Three methods run a statement, and each returns one kind of thing, so none of them returns a union:
`execute_query` reads it all and gives a result, [`stream`](#streaming-a-large-result) gives an
iterator that keeps the rows on the server, and `execute` is psycopg's own and gives a cursor.

There is no `graph_` and no `timeout_`, and both absences have a reason. Selecting a graph is a
statement, is undone by a rollback, and drops the label table, so it belongs on the connection or on
the pool, once, rather than once per statement. Bounding a statement's time is two more statements,
one to set the limit and one to put it back, which measures over twice a bare `RETURN 1` on a local
server and is two more round trips against one a network away. A deadline therefore belongs where it
is paid once for many statements: see [Deadlines](#deadlines).

## Graph values

```python
(v,) = conn.execute_query("MATCH (n:Person) RETURN n LIMIT 1").records[0]

v.id                 # GraphId(labid=3, locid=1), prints as 3.1
v.label              # 'person', lower case: the server folded it, see Identifiers below
v.properties         # {'age': 42, 'name': 'Arthur'}, in the order the server stores them

(e,) = conn.execute_query("MATCH ()-[r:KNOWS]->() RETURN r LIMIT 1").records[0]
e.start, e.end       # the GraphIds of the two vertices

(p,) = conn.execute_query("MATCH p = (:Person)-[:KNOWS]->() RETURN p LIMIT 1").records[0]
p.vertices, p.edges  # in order
list(p)              # interleaved: vertex, edge, vertex, ...
len(p)               # element count
p.length             # hop count
```

A vertex and an edge are values rather than handles. Each is immutable, compares and hashes on its
**identity alone** (which is what the server does too), and so can be a dictionary key or a set
member.

**A property map is decoded when you first read it**, not when the row arrives. A query that ranks
or counts vertices without looking inside them never pays for the map at all, and what that is worth
grows with the map: parsing a vertex whose properties are never touched is about 1.4 times cheaper
on a map of three keys, 2.8 times on a map of sixty, and 3.5 times on a 1536-dimension embedding.

It also means the map is decoded under whatever
[number setting](#numbers-in-a-property-map) is in force at the moment you touch it, rather than at
the moment the row was fetched.

They are structs the garbage collector does not track, which is most of what makes a large result
cheap: nothing here can take part in a reference cycle, since a property map holds only what JSON
can. The public surface is read-only, and it is read-only by having no setters rather than by
refusing writes; that is not only taste, because routing every field through a `__setattr__` that
raises costs more than storing the value does, on every element of every result.

`label` is the label a vertex was created with, and it is in the value the server sent. Its
**ancestry is not**, and is not carried: `labels(n)` is a Cypher function the server answers with the
row, own label first. Carrying it would put a label table lookup on the text rendering, which is the
one reading that needs no table at all.

A row is a tuple, and the column names are read once per result into `result.keys`, so there is no
dict per row and no name scanned per lookup. The same vertex appearing in many rows of a join is
many objects that compare equal rather than one shared object; collapsing them would save one
property decode per repeat, which on a join of a few hundred rows with kilobyte maps is under a
hundredth of the read, against a pass over every result to discover whether there are any repeats at
all.

## Parameters

Placeholders are psycopg's `%s`. Cypher's own `$n` also works where the server accepts it.

```python
conn.execute_query("MATCH (n:Person) WHERE n.name = %s RETURN n", ("Arthur",))
conn.execute_query("MATCH (n:Person) RETURN n LIMIT %s", (10,))
conn.execute_query("CREATE (:Person %s)", ({"name": "Arthur", "age": 42},))
conn.execute_query("MATCH (n) WHERE id(n) = %s RETURN n", (vertex.id,))
```

**A Python `str` is sent as `text`.** That matters more than it sounds. Left untyped, the server
parses the bytes as JSON, so `"Arthur"` was an error and `"123"`, `"null"`, `"true"` and `"1.5"` were
accepted and **silently matched nothing**. Declared as text, the server's own text to JSON coercion
runs instead and every one of those becomes a correct answer. The plans and index conditions are
unchanged by this, which was checked across equality, ranges, `STARTS WITH`, `IN`, `ORDER BY` and
`<>`, including under a forced generic plan.

What it costs: a `str` passed where the type is *not* text-like no longer resolves, and says so while
the statement is parsed, naming both types. The fix is psycopg's own documented idiom, `%s::date`.
Passing the value's real type (`date`, `UUID`, `int`, `Decimal`, `bytes`) is untouched, and
`concat(%s, %s)`, which fails without this, works. To send a value the old way for one parameter:

```python
from agensgraph import Unspecified
conn.execute_query("SELECT %s", (Unspecified("123"),))
```

**A mapping is sent as jsonb**, because psycopg cannot adapt a bare `dict` at all and nearly every
parameter a Cypher statement takes is read as jsonb. A `list` stays a PostgreSQL array, because plain
SQL on the same connection needs `WHERE x = ANY(%s)` to keep working.

### Where a parameter is allowed

| position | binds as |
|---|---|
| a general expression, `WHERE`, `RETURN`, list and map values, `ORDER BY` | jsonb |
| `LIMIT`, `SKIP`, `OFFSET` | bigint |
| a pattern property map, `SET a = %s`, `CREATE (:l %s)` | jsonb |
| `a[%s]` | jsonb |
| `id(a) = %s` | graphid |
| `size(%s)` | text (2.18 and later; earlier parsers take no parameter there) |
| a label, `a.%s`, `{%s: v}`, the lower bound of a walk | not allowed, syntax error |
| `UNWIND %s`, `x IN %s` | needs an explicit `::jsonb` |

**One shape is refused by the driver before it is sent.** A parameter as the *upper* bound of a
variable-length relationship, `[r*1..%s]`, is not a syntax error: the server reads it as an unbounded
walk plus a property map, prepares without complaint, and quietly returns the wrong rows. All three
spellings of it are refused, in both placeholder styles:

```python
conn.execute_query("MATCH (a)-[r*1..%s]->(b) RETURN a", (3,))   # ValueError
```

### Sending a property map

Rendered with msgspec, which is where most of the cost of a write is: several times faster than the
standard library on a small map, and about ten times on an embedding, where the numbers dominate.

`NaN` and the infinities are refused, wherever they appear. jsonb has no way to store one, and
encoding it would write `null` instead, which is the wrong value rather than an error. Sent alone
rather than in a map, a non-finite float reaches the server as a `float8` and converting one to jsonb
stores the *text* `"NaN"`, so what came back would be a string where a number was sent. Both are
refused.

Values that are rendered to rather than stored as what was sent:

| written | stored as | reads back as |
|---|---|---|
| `datetime`, `date`, `time` | its ISO text | `str` |
| `UUID` | its text | `str` |
| `bytes` | base64 text | `str` |
| `set` | a JSON array | `list` |

A `Decimal` is **not** in that table: it is stored as a JSON number and keeps every digit, because
jsonb stores a number as `numeric`, which is arbitrary precision.

## Identifiers

A label or a property key cannot be bound as a parameter, since the grammar has no place for one
there, so a statement naming one dynamically has to carry it in its text. That quoting lives in one
place:

```python
from agensgraph.cypher import quote_identifier, quote_string

quote_identifier("person")      # person
quote_identifier("Person")      # "Person", since the server folds an unquoted name to lower case
quote_identifier("my label")    # "my label"
quote_identifier('a"b')         # "a""b"
quote_identifier("MATCH")       # "MATCH"
```

A name holding a null byte is refused rather than quoted, because the server's lexer stops at one and
the statement would end somewhere other than where it appears to. Everything in this driver that
builds a statement around a name (the index and constraint builders, the vector helpers, the identity
map, `LISTEN`) goes through this.

## Two wire formats

The driver never rewrites a query. `RETURN n` is sent as written, the server answers in its text
rendering, and the driver reads that. It needs nothing from the server beyond the answer itself.

The same query can be asked for in the composite rendering instead, per-statement, with
`binary_=True`. That form leaves the label name out of the value, so it needs a label table for the
connection, which `conn.graph(name)` fills.

**Ask for it when the result is large and the link is not the bottleneck.** It trades bandwidth for
parsing: every element repeats its column oids, its lengths and a tuple id the text form never
writes, so a vertex is about a fifth more bytes and an edge several times more. In exchange the
driver reads lengths instead of measuring where each value ends.

How much it pays depends on the shape. Over a few thousand rows it reads edges about 1.45 times
faster, paths about 1.3, and whole vertices only about 1.15, since a vertex is mostly its property
map and both renderings hand that to the same decoder. It stops paying on a short result, where the
round trip costs more than either parse: at a few hundred rows, whole vertices are about 0.8 times,
which is to say slower. [Embeddings](#embedding-vectors) are the clearest case for it, because a
vector's text is decimal and its binary is the numbers themselves.

When in doubt, leave it off and turn it on for the one query that is slow.

Both renderings produce the same objects, and the suite asserts that against values the server itself
produced, comparing them strictly rather than with `==`, which would pass on disagreement and fail on
agreement in five separate ways.

## What a write changed

```python
result = conn.execute_query("CREATE (:Person {name: 'a'})", counts_=True)
result.counts.inserted_vertices     # 1
```

Five counters: `inserted_vertices`, `inserted_edges`, `deleted_vertices`, `deleted_edges`,
`updated_properties`.

A counter is `None` rather than `0` when it cannot be attributed to this statement, which is a real
case rather than a formality. The counters live on the session, and the server zeroes them per
*clause group*, so a write that returns rows leaves the groups its own clauses do not touch holding
whatever an earlier write left there. The driver reads them before and after and reports only what it
can account for: a counter that changed can only have been changed here, one that did not and was
already zero is zero either way, and the one case where a stale number and a real one look alike is
left unanswered instead of guessed.

That extra reading is only taken where it is needed. A write that returns no rows zeroes all five on
the server, so one reading afterwards is the whole answer.

## Streaming a large result

```python
with conn.transaction():
    for (person,) in conn.stream("MATCH (n:Person) RETURN n"):
        ...
```

A server-side cursor, so the rows stay on the server. A hundred rows are fetched at a time by
default, because the standard's own default of one is a round trip per row.

The grammar has no Cypher arm for `DECLARE ... CURSOR`, so the statement is wrapped in
`SELECT * FROM (...) t`, and that wrap accepts only the read-only subset. A write is refused by name
before anything is sent. A trailing `LIMIT` or `ORDER BY` is fine; one in the middle is not, and the
server says so rather than the driver keeping a second copy of the server's grammar (the two releases
disagree about that case, which is exactly why).

A transaction is required, and not as a formality: it is what makes abandoning the iterator safe,
because leaving the transaction closes the cursor with it. Each stream names its own cursor, so two
open at once do not collide.

## Numbers in a property map

jsonb keeps an arbitrary precision decimal; Python's float does not. Integers of any length survive
exactly. Only non-integers lose anything, past about seventeen significant digits.

`1e400` is worth calling out: the server stores it as an exact 401 digit integer, not an infinity.

For the cases where the lost digits matter:

```python
agensgraph.read_numbers_exactly()      # once, at startup
```

Every non-integer then reads as a `Decimal`, keeping whatever the server holds, `1e-400` included.
What it costs depends on what the map holds, since only a non-integer takes the slower path: a map of
integers is unchanged, a mixed one costs twice as much, one of non-integers about four times, and an
embedding of a thousand floats closer to six. It applies to a property map and to a bare jsonb column
alike, so `RETURN n` and `RETURN n.p` agree about the same stored value.

It is process-wide and meant to be chosen once at startup, which is also the only moment it is
unambiguous: a map on the text path is decoded when it is first read, so two rows of one result can
disagree if the setting changed between touching them.

## Columnar export

```python
from agensgraph.columnar import to_arrow, to_pandas, to_polars

table = to_arrow(conn.execute_query("MATCH (n:Person) RETURN n.name, n.age"))
```

Each column is built as a column, with its type declared rather than inferred, which is about an
order of magnitude faster than assembling one Python value at a time, for Arrow, pandas and polars
alike.

**A whole vertex becomes a struct** of its identity, its label and its property map, and the map is
the JSON text taken from the bytes it arrived in, so it is never decoded into a dict:

```python
table = to_arrow(conn.execute_query("MATCH (n:Person) RETURN n"))
# n: struct<id: uint64, label: dictionary<values=string, indices=int32, ordered=0>, properties: string>
```

An edge carries `start` and `end` as the same `uint64` a vertex's `id` is, so a join between them is
an integer join. A path is a struct of a list of vertices and a list of edges.

`Layout` decides the cases where more than one answer is defensible:

```python
from agensgraph.columnar import Layout

Layout(elements="columns")            # spread a vertex over n.id, n.label, n.properties
Layout(identity="text")               # '3.1' rather than the packed integer
Layout(properties=some_struct_type)   # pull named fields out of the map
Layout(properties="skip")             # leave the map out
Layout(labels="text")                 # not dictionary encoded
Layout(vectors="list")                # not fixed-size
Layout(sparse="dense")                # expand a sparse vector
```

**An embedding becomes `FixedSizeList<float32>`** straight from the wire bytes, with no Python float
in between. Ask for the composite rendering and it exports dramatically faster, in half the memory:

```python
result = conn.execute_query("MATCH (n:Emb) RETURN n.v AS v", binary_=True)
table = to_arrow(result)          # v: fixed_size_list<item: float>[384]
```

**A large result is exported a chunk at a time.** `batches()` and `reader()` take a server-side
cursor and never hold more than one chunk of Python objects. The first chunk settles the schema and
every later one is held to it, so the chunks concatenate:

```python
import pyarrow.dataset
from agensgraph import columnar

with conn.transaction(), conn.cursor(name="export") as cursor:
    cursor.execute('SELECT id, v FROM "graph".emb')
    pyarrow.dataset.write_dataset(columnar.reader(cursor, size=8192), "out", format="parquet")
```

`columns(records, keys)` is the same transposition without a backend, for a caller who wants plain
lists. It keeps every column even when two share a name.

## Bulk loading

```python
conn.load_vertices("Doc", [{"key": "a", "title": "..."}, ...])
by_key = conn.identity_map("Doc", "key")
conn.load_edges("Cites", [(by_key["a"], by_key["b"], {"weight": 1})])
```

`COPY` in binary rather than a statement per row: about thirty times a statement at a time, and
better than half again on top of the best a single `UNWIND ... CREATE` can do.

No identity is supplied. The label table's `id` column has a default that builds the graph id from
the label's own id and its sequence, so copying only the property map produces exactly the identities
a `CREATE` would have. That removes the whole business of generating identities client-side and
keeping them unique.

Edges need the identities of the two vertices they join, which is what `identity_map` reads, in one
statement for the whole label. It raises rather than guessing if the key is not unique, or if an
element does not have it, because silently collapsing two vertices into one would attach every edge
of both to whichever survived.

A frame goes in directly, without becoming Python objects on the way:

```python
conn.load_vertex_frame("Person", table)      # each column is a property
conn.load_edge_frame("KNOWS", edges)         # start and end are packed identities
```

That is about half again faster than the same table handed to `load_vertices()` as mappings, and no
more than that, because almost all of a load is the server's own ingest. The client's share is what
falls, by about eight times.

## A burst of statements

For a batch whose cost is round trips rather than work:

```python
conn.pipeline_batch([f"CREATE (:Event {{n: {n}}})" for n in range(1000)])
```

**A failure is attributed to the batch, not to a statement**, and the signature says so. That is not
caution, it is what the server does: with four statements where only the second is bad, the *first*
raised and the rest reported no SQLSTATE at all. `BatchFailed` carries the statements that were sent,
in order, and the server's own error as its cause. Re run them one at a time to find out which.

## Reading the catalogs

```python
conn.graphs()                    # [Graph(name, schema, labels)]
conn.labels()                    # [Label(id, name, kind, parent)]
conn.declared_properties()       # [DeclaredProperty(label, name, type, nullable)]
conn.indexes()                   # [Index(label, name, unique, definition)]
conn.constraints()               # [Constraint(label, name, unique, definition)]
conn.element_counts()            # {'Person': 2, 'KNOWS': 1}
```

Each takes an optional label to narrow to, and an optional graph to read about instead of the one the
session is on.

Counting per label reads no property: the label id is part of every element's identity, so the group
key is a function of the id column alone. Whether the *heap* is read anyway is the planner's to
decide, and often yes, because against a narrow label it prefers a sequential scan to an index only
one. An edge can never avoid it, because the engine creates an edge's id index as BRIN, which carries
no tuple pointers.

Two things about uniqueness are worth knowing, because they are not where you would look. A
uniqueness constraint in AgensGraph is an **exclusion** constraint, so it is filtered *out* of the
property index view, and a driver reporting from that view alone would show a graph as having no
uniqueness constraints when it has them. `constraints()` reads the constraint catalog as well, so it
finds them.

`declared_properties()` asks the server whether it can promote a property at all before reading the
catalog that records one, because that catalog does not exist on every server and the version cannot
tell you: the 2.18 release branch and main both report `2.18-devel`, and only one of them has it.

## Declaring what should exist

Say what the schema should be, and let the driver work out the difference:

```python
from agensgraph import DesiredIndex, Unique, Check

conn.ensure_indexes([
    DesiredIndex("person", ["email"], unique=True),
    DesiredIndex("doc", ["title"], method="gin"),
])
conn.ensure_constraints([
    Unique("person", "email"),
    Check("person", "age > 0", "person_age_positive"),   # a check needs a name of its own
])
```

Both return the statements they ran, take `dry_run=True` to return them without running any, and
`drop_extra=True` to remove what is there and not asked for. Running the same declaration twice is a
no-op the second time.

`DesiredIndex` and `Unique` derive a name when you do not give one, from the label and the
properties. `Check` cannot, since an expression gives nothing to derive from, so its name is
required.

**Name a label the way the server stores it**, which `conn.labels()` will tell you. Every name here
is quoted, so it is taken literally: `CREATE VLABEL Person` stores `person`, and asking for
`DesiredIndex("Person", ...)` then looks for a label that does not exist. Write `"person"`, or create
the label quoted in the first place so the capital survives. This is the one place the folding rule
in [Identifiers](#identifiers) reaches out and bites, because everywhere else both sides fold
together and agree.

The matching is deliberately conservative, because of how the server stores these. A name is derived
from the columns, truncated, then given a counter on collision, so names are not the key. A
definition is printed with defaults omitted. A predicate is stored normalised, so `age > 0` comes
back as something that does not look like what you wrote. Three consequences: name an operator class
only when it differs from the default, expect a partial index to be compared on its normalised
predicate, and let anything the reconciler cannot match confidently alone rather than dropping it.

## Embedding vectors

Needs [pgvector](https://github.com/pgvector/pgvector) in the database. It is created per database,
not per server, so ask the connection rather than the version:

```python
conn.has_vectors()          # is the extension created here
conn.register_vectors()     # read vector, halfvec and sparsevec on this connection
conn.vector_version()       # (0, 8, 6)
```

Two routes, and both are indexed the same way:

```python
# a property left in the map, with the cast carrying the dimension
conn.execute("CREATE PROPERTY INDEX ON movie USING hnsw ((embedding::vector(4)) vector_cosine_ops)")

# or a column of its own, which needs no cast and refuses a wrong-length value on write
from agensgraph.vector import generated_column
conn.execute(f"CREATE VLABEL doc ({generated_column('embedding', 1024)})")
```

A promoted column is the stronger of the two: the dimension lives on the column, so `vector_in`
enforces it at the moment a value is written rather than at search time.

### Storing and reading

```python
from agensgraph.vector import Vector

conn.execute("CREATE (:doc {embedding: %s})", (Vector(values),))

(v,) = conn.execute_query("MATCH (n:doc) RETURN n.embedding").records[0]
len(v)              # free, the dimension is in the first two bytes
v.values            # array('f'), which numpy and torch take without copying
v.tolist()          # an ordinary list
v[0], v[-1], v[0:2], list(v), sum(v), 2.0 in v, v.index(2.0)
v == [1.0, 2.0]     # True when the numbers match
```

A `Vector` keeps the bytes the server sent and turns them into numbers only if you look. That is the
same bargain the driver makes with a property map, and it pays here for the same reason: a vector
search asks the *server* for the distance, so the components of the vectors it ranked are often never
read. It is not a `list` (`isinstance(v, list)` is `False`, `json.dumps(v)` raises) but it compares
equal to one, which is the part that would otherwise go wrong quietly.

**Ask for `binary_=True` when you read vectors.** This is the one place the rendering makes a large
difference rather than a small one: a vector of 1536 dimensions prints as roughly fifteen kilobytes
of decimal against six of wire bytes, so the text form costs about three times as much before
anything is parsed and seven times once the numbers are read.

**Send a `Vector`, not a list and not a string you built.** Every other route formats each number as
decimal for the server to parse back: a list cast to `vector(1536)` costs about eight times as much,
a hand built string about two and a half. In bulk the gap is wider, and `COPY` in binary with
`Vector` loads an order of magnitude faster than `COPY` in text.

Half precision is `halfvec`, and works the same way. Binary quantisation is available in SQL, where
`bit` is spellable; Cypher has no syntax for that cast, which is worth knowing before you look for
one.

### Indexing and searching

The index statement and the search statement come from the same function, so they cannot disagree:

```python
from agensgraph.vector import vector_index, nearest, Distance

conn.execute(vector_index("doc", "embedding", dimensions=1024,
                          operator_class=Distance.COSINE.operator_class))
# create property index on doc using hnsw ((embedding::vector(1024)) vector_cosine_ops)

rows = conn.execute_query(
    nearest("doc", "embedding", dimensions=1024, operator=Distance.COSINE, limit=10),
    (query_vector,),
).records
# match (n:doc) return n order by n.embedding::vector(1024) <=> %s::vector(1024) limit 10
```

`Distance` is a `StrEnum` whose value *is* the operator, so it goes straight into `operator=`, and
`.operator_class` gives the class the matching index needs.

That sharing exists because of two ways to lose the index silently:

- **The typmod must match exactly.** Against a `vector(4)` index, a search casting to a bare `vector`
  or to `vector(3)` sorts a sequential scan and tells you nothing.
- **An operator class serves one operator.** A `vector_cosine_ops` index answers `<=>` alone, and
  ordering by `<->` against it scans.

Distances by name, because two of them differ by a single character and mean entirely different
things: `L2` (`<->`), `INNER_PRODUCT` (`<#>`, negated so that smaller is nearer as it is for the
others), `COSINE` (`<=>`), `L1` (`<+>`), and for bit strings `HAMMING` (`<~>`) and `JACCARD` (`<%>`).
`hnsw` and `ivfflat` are both available, and `vector_index` takes `options=` for things like
`WITH (lists=10)`.

Writing the search by hand is fine too, and on an unpromoted property the cast is required rather
than optional, since `jsonb <-> vector` is not an operator:

```python
conn.execute_query(
    "MATCH (m:movie) RETURN m ORDER BY m.embedding::vector(4) <=> %s::vector(4) LIMIT 4",
    (query_vector,),
)
```

### Sparse vectors

```python
from agensgraph.vector import SparseVector

v = SparseVector({0: 1.0, 3: 2.0}, 6)   # indices, then how many dimensions
v.to_dense()                            # when you really want the zeros
```

It is a value of its own rather than a dense list, and the reason is size: three non zeros in a
million dimensions is a few dozen bytes on the wire against roughly eight mebibytes as a list of
floats, and expanding it would discard the entire reason the type exists.

**Indices are zero-based in Python**, everywhere, including `to_dense()`. The wire's text form is one
based and its binary form is zero-based, which is a genuine trap: two renderings that disagree about
the base, decoded without accounting for it, give a plausible off-by-one rather than an error. The
conversion happens in one place, on the text boundary only, and is tested from both directions.

What the server refuses, the constructor refuses: duplicate indices, an index out of range, a
dimension below one, a non-finite value. An explicit zero is dropped, because the server drops it,
and mirroring that is what keeps a round trip stable.

A Cypher list cannot be written into a sparse column at all, so the dumper is not a convenience here;
it is the reasonable way in.

### Tuning a search

```python
conn.vector_search_options({"hnsw.ef_search": 100})
```

Seven settings, listed with the type each takes in `agensgraph.vector.SEARCH_OPTIONS`. Applied to the
transaction by default, so they do not leak into the next caller on a pooled connection; pass
`local=False` to set them for the session.

## Pooling

```python
pool = agensgraph.ConnectionPool("host=localhost dbname=graph", graph="social",
                                 min_size=4, max_size=16)
pool.open(wait=True)          # fails here if the server is wrong, not later under load

with pool.connection(deadline=agensgraph.Deadline(5.0)) as conn:
    result = conn.execute_query("MATCH (n:Person) RETURN n")

pool.close()
```

It wraps psycopg's pool rather than reimplementing one, and adds five things psycopg has no
equivalent for.

**A server this driver cannot read is refused by `open()`** before any worker starts. That check is
its own connection, made and closed first, because a refusal raised inside the pool would be counted
as a connection error and retried for the whole reconnect timeout instead of reported.

**`invalidate()` retires every connection now held**, in one call, and returns the generation now in
force. Nothing is closed and nothing waits: a connection in use stays usable until its holder is
finished, and is closed when it comes back. So a server restart costs one call rather than one
failure per connection.

**`drain()` is the other half and does a different job.** It closes every connection and opens the
same number again, for a change a connection carries from the moment it is made: a registration on
the adapters map, a setting asked for in `configure`, a rotated password behind a callable connection
string. The generation is untouched, because those connections are not wrong to be reused; they are
replaced so the replacements are built the new way.

**`configure` runs once per new connection and `setup` once per handout.** psycopg has only the
first, and a graph or a statement timeout belongs to the second.

**A connection whose statement was interrupted is closed rather than lent again.** psycopg does try
to leave it clean, and when that finishes the connection really is reusable, but it cannot promise it
finished: the read it re-enters is bounded by nothing, and because an `asyncio` timeout works *by*
cancelling, no timeout on that connection can fire either. Such a connection is replaced, which costs
one connection on an event that is rare by definition. A statement that merely *failed* keeps its
connection, since a value the driver refused never reached the socket and a server's own refusal
leaves the connection idle and answering.

`get_stats()` returns psycopg's own counters plus `generation`, `connections_retired` and
`connections_interrupted`; `pop_stats()` is the same with the accumulating ones reset, for reporting
an interval. `resize()` and `check()` pass through.

For a process that handles one request and exits, or a serverless one that may be frozen between
requests, **`NullConnectionPool`** keeps nothing: one connection per caller, closed when they are
done. Everything else is the same, so moving between the two is a change of class and nothing else. A
borrow and one statement costs about eight times what it costs from a pool that keeps its
connections, which is what connecting costs and is the reason not to choose it for a process that
lives long enough to reuse one.

Both pools have an awaiting twin: `AsyncConnectionPool` and `AsyncNullConnectionPool`.

## Deadlines

```python
from agensgraph import Deadline

budget = Deadline(5.0)
with pool.connection(deadline=budget) as conn:
    conn.execute_query("MATCH (n) RETURN count(n)")
```

One budget, threaded through the wait for a connection *and* the statement that runs on it, so time
spent waiting is time the statement no longer has. It reaches the server as `statement_timeout`, set
once per borrow and reset when the connection goes back, so it cannot be inherited by the next
caller.

`statement_timeout` is deliberately set a little inside the budget. Arriving at `COMMIT` with no time
left manufactures the one failure that cannot be retried safely, so the gap is reserved rather than
spent.

It never uses `transaction_timeout` or `idle_in_transaction_session_timeout` as a query deadline,
because those terminate the *session*. `Expired` is a `TimeoutError`, so a caller that already
catches one keeps working, and running out of your own budget is classified differently from a pool
with nothing to give.

## Retrying

`RetryPolicy` decides; it does not drive. You keep the loop, because only you know what is safe to
run again and whether the transaction had already written:

```python
import time
from agensgraph import RetryPolicy

policy = RetryPolicy(attempts=3)     # the first try counts, so this is one try and two retries
previous = []

for number in range(1, policy.attempts + 1):
    try:
        result = pool.execute_query("MATCH (n) RETURN count(n)")
        policy.succeeded()           # pays back into the shared allowance
        break
    except Exception as exc:
        attempt = policy.decide(exc, number=number)   # .retry .delay .recovery .reason
        if not attempt.retry:
            raise
        previous.append(exc)
        time.sleep(attempt.delay)
else:
    raise policy.exhausted(previous[-1], attempts=policy.attempts, previous=previous)
```

`decide()` also takes `wrote=True`, which says the transaction had written something, and that is
what turns a lost connection from "reconnect and try again" into "the outcome is now unknown", and
`remaining=` from your budget, so a wait that would not leave time for another attempt is not taken.

**Retryability is six categories with a recovery action, not a boolean**, because a boolean gets
`26000` catastrophically wrong: reconnecting appears to fix it while hiding a broken reset hook that
keeps firing.

| | |
|---|---|
| `SAFE` | retry on the same connection. `40001`, `40P01`. **The common case for a graph write.** |
| `RECONNECT` | new connection, then retry. Class 08, `57P01`, `57P02`, `57P03`, `25P03` |
| `BACKPRESSURE` | retry, but wait longer. `53300`, `53200`, `55P03`, `55006` |
| `RESET_STATE` | clear the statement cache, keep the connection. `26000` |
| `UNKNOWN` | resolve it, do not guess. `08007`, `40003` |
| `FATAL` | never retry. `57014`, classes 42, 22, 23 |

Concurrent graph writes surface as `40001` by design, so a conflict is the *expected* path rather
than an exotic one, and it is safe to retry.

Backoff is full jitter, capped before the jitter is drawn rather than after, which is a distinction
worth keeping: capping afterwards makes the cap reachable only by chance and the distribution not the
one the name describes. A backpressure retry doubles the ceiling before capping, so it waits longer
without breaking the cap.

**A counter is not a budget.** Four layers each willing to try three times is sixty four attempts
from one user action, so the allowance is a token bucket shared across the process: a transient
failure costs more than a rejection does, because a connection failure is usually the whole service
rather than this one request. When it is exhausted, the message says so, which saves a great deal of
support time.

The connection's fate is decided in one place on every exit path, including cancellation, rather than
delegated to a callback that some paths skip.

## Losing a connection mid-commit

The one failure that cannot be retried blindly is a commit whose acknowledgement never arrived: the
write may have landed or may not, and doing it again could apply it twice.

```python
with conn.transaction():
    xid = conn.transaction_id(assign=True)     # keep this
    conn.execute_query("CREATE (:Order {id: 1})")

# if the connection is lost while committing, ask a different connection what became of it
outcome = other_conn.resolve_commit(xid)
```

`CommitOutcome.COMMITTED` means it landed, so do not retry. `ABORTED` means it did not, so retry as
you would a conflict. `IN_PROGRESS` means back off and ask again. `UNKNOWN` means the server can no
longer tell, which is not the same as aborted and has to be surfaced rather than guessed either way.

Stashing the id is nearly free, since any `CREATE` assigns one anyway.

## Cancellation

The connection is discarded. Always.

That is the majority position among drivers, and the alternatives have a bad record: the two cleverer
attempts in one widely used client produced a CVE each, one of which could send one request's
response to a different request. The rule here is to decide the connection's fate synchronously
first, with no awaits, and only then bound and isolate any cleanup.

`57014` has four separate causes (a client cancel, `statement_timeout`, a recovery conflict, and a
tie broken toward lock timeout) and its message is localised, so the driver tracks "I cancelled this"
in its own state and never parses message text. A successful cancel dispatch is not proof either: the
request matches on process id and key and returns silently on a mismatch, and process ids are reused.
The only evidence a cancel landed is `57014` on the original connection.

## When the network goes quiet

Keepalive is asked for by default: without it, a connection whose network stops carrying packets
waits for the kernel, which is two hours and a quarter on Linux.

Each setting is filled in on its own, so naming one does not silently leave the others at the
system's values. `keepalives=0` turns them all off.

`tcp_user_timeout` is **not** among them, and that is worth saying, because it is the setting usually
recommended for this. It bounds how long *transmitted* data may go unacknowledged, and a connection
waiting for a reply has transmitted nothing. Measured against a server whose traffic was dropped:

| | outcome |
|---|---|
| nothing set | still waiting when the test gave up |
| **`tcp_user_timeout` alone** | **still waiting when the test gave up** |
| keepalive, with or without `tcp_user_timeout` | failed within seconds |

So it is useful *alongside* keepalive and useless instead of it. It is left unset because it also
applies while sending, where too small a value would end a healthy connection.

## Observability

```python
def log(record):
    print(record.statement, record.elapsed, record.rows, record.failed)

agensgraph.add_query_logger(log)
agensgraph.enable_tracing()      # needs agensgraph-python[otel]
```

**Every statement is reported**, not only the ones `execute_query` sent: `conn.execute`, a cursor you
took yourself, and the driver's own catalog reads all arrive at the same place, because a caller
asking what the driver sends wants the round trips they did not write. `elapsed` is the round trip,
sending the statement until the server has answered, and not the reading of the rows afterwards.

Off costs a boolean test. A statement takes the same time with a logger attached as with none, and
the reporting adds about a quarter of a per cent to a statement's round trip. A clock is not read,
and a timer is not even allocated, unless something is going to ask for the number. Spans are per
statement and never per row, the tracing API is imported only when asked for and the SDK never, the
record carries an opaque connection number rather than the connection's settings (which hold the
password), and no parameter value ever reaches a span.

**Notices are structured**, drained per-statement so they do not leak into the next result:

```python
agensgraph.observability.add_notice_listener(lambda n: print(n.severity, n.message))
```

## Running a statement you did not write

Model output, most often. The rule is that the **server** refuses a write rather than the driver,
because a driver that reads the text to decide has to recognise every way of writing, and does not:
this is PostgreSQL underneath, so `INSERT`, `TRUNCATE` and `DROP` are all available and none of them
is Cypher.

```python
with conn.read_only_transaction():
    result = conn.execute_query(whatever_the_model_said)
```

Measured inside that transaction: every Cypher write, `INSERT`, `TRUNCATE`, `DROP` and `CREATE
VLABEL` are refused with `25006`, and a plain read runs. The transaction characteristic is
psycopg's own `connection.read_only`, which is the thing to set for a whole session; what this adds
is the check below.

**What a read-only transaction does not stop, and it is worth knowing.** `COPY ... TO PROGRAM` runs
a command on the server's host and is allowed, because it takes rows *out* of the database rather
than putting any in, so there is no write to refuse. Reading the statement does not stop it either:
a second statement after a semicolon, and a leading comment, both get one past. What stops it is
not holding the privilege, so that is what is asked about:

```python
conn.can_run_server_programs()      # asked once per connection, and kept
```

A role that holds it is refused the transaction, with a message saying what to change, rather than
left to find out. Pass `allow_server_programs=True` to accept it. **Connect as a role that is
neither a superuser nor a member of `pg_execute_server_program`**, and the boundary holds:

| | refused by |
|---|---|
| a Cypher write, `INSERT`, `TRUNCATE`, `DROP`, DDL | the transaction, `25006` |
| `COPY ... TO PROGRAM`, including one hidden behind another statement | the privilege, `42501` |
| a plain read | nothing, it runs |

`cypher.check_can_wrap()` is a reasonable hint to show a user early, and it already beats a
write-detecting regex on `RETURN n.set`. It is not a boundary and is not used as one.

## Errors

They are psycopg's exception classes, raised by psycopg, so `except psycopg.errors.UniqueViolation`
keeps working and no wrapping layer stands between you and them. The engine mints no SQLSTATE of its
own, which was checked against its source rather than assumed, so there is nothing to register and
nothing that reaches into psycopg's global tables for unrelated connections.

Three of the server's refusals cannot be told apart by SQLSTATE, and are translated into named
classes so you do not have to match on message text:

| | |
|---|---|
| `ConfigurationError` | `enable_graph_dml` or `enable_eager` is off. Both arrive as `XX000` |
| `ReadOnlyGraphWrite` | a graph write in a read-only transaction, which the server reports as `cannot execute ??? ...` |
| `CapabilityError` | a feature this server does not have, naming the version that does |

Plus the driver's own: `BatchFailed`, `StaleLabelCache`, `ReleasedConnection`, `StaleGeneration`,
`UnresolvedCommit`, `NoEnclosingTransaction`, `Expired`.

### What a failure prints as

PostgreSQL puts row data in a failure's DETAIL. A uniqueness failure reads
`Key (email)=(alice@example.com) already exists`, so a plain `logger.exception()` writes into the log
a value the driver never saw as a parameter. **DETAIL and CONTEXT are therefore not part of the
message by default.** What the message loses, `exc.diag.message_detail` and `exc.diag.context` still
hold, so the data stays for a post-mortem and only the rendering is cut.

```python
agensgraph.show_error_details(True)   # process-wide, so it cannot be forgotten at a call site
```

It cuts DETAIL and CONTEXT and nothing else. A primary message can still echo the value it was
handed, because that is the server quoting its own input rather than a value belonging to another
row.

`agensgraph.errors.mask_dsn()` is there for a caller who holds a connection string and wants to write
it somewhere. Nothing in this driver calls it, and that was checked rather than assumed: no message,
log record or `__repr__` it produces carries one.

## LISTEN and NOTIFY

```python
conn.listen("changes")
conn.notify("changes", "payload")
conn.listening()                       # what this connection is subscribed to

for note in conn.notifications(timeout=5.0):
    print(note.channel, note.payload)
```

Channel names are quoted, and announcing goes through `pg_notify` with the name as a parameter, so a
name that needs quoting behaves the same in both directions.

**Prefer one style or the other**, not both. psycopg changed the semantics of mixing a handler with
the iterator twice and then made it a warning; pick the handler as primary and treat the iterator as
the alternative.

## Two-phase commit

psycopg's own methods work unchanged, including for a write that returned rows:

```python
conn.tpc_begin(conn.xid(0, "gid", "branch"))
conn.execute_query("CREATE (:Thing)")
conn.tpc_prepare()
conn.tpc_commit()
```

Two things to know. The connection must not be in autocommit. And `max_prepared_transactions` is
**0 by default**, so on an untouched server `tpc_prepare()` raises `NotSupportedError` naming the
setting, which is the server telling you what to change.

## Generic tools and SQLAlchemy

The PEP 249 names are on `agensgraph.dbapi`, so anything that drives databases generically drives
this: `connect`, `apilevel`, `threadsafety`, `paramstyle`, the type constructors, and the exception
hierarchy re exported so `DBAPIError.instance()` keys off the right classes.

`connection.closed` and `connection.broken` are exposed, which is what lets a dialect decide whether
a connection is dead without matching on error strings.

A whole SQLAlchemy dialect is **ten lines**:

```python
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg

class AgensGraphDialect(PGDialect_psycopg):
    driver = "agensgraph"

    @classmethod
    def import_dbapi(cls):
        import agensgraph.dbapi
        return agensgraph.dbapi
```

Version detection is free, because `PostgreSQL 18beta1 (AgensGraph 2.18-devel)` already matches the
PostgreSQL dialect's own regex. One thing to know if you write Cypher through SQLAlchemy:
`(:Label)` collides with SQLAlchemy's `:param` syntax, so use `text()` with bound parameters or
escape the colon.

## Server versions

The driver reads AgensGraph 2.17 and later, and refuses anything older at connect rather than at
whichever later query first wants a catalog it does not have. It learns which it is talking to from
the `agversion` parameter the server sends at startup, so the check costs no round trip.

The floor is what the build tests against: every release named here has the whole suite run against
it on every change, so support is a thing that is checked rather than a thing that is claimed.

```python
caps = conn.capabilities
caps.version                      # (2, 18)
caps.reported                     # '2.18-devel'
caps.has_property_promotion()     # a property stored in a column of its own
caps.has_gql_clauses()            # LET, NEXT, FINISH, FILTER, FOR, CALL
caps.has_element_ordering()       # ORDER BY on a vertex or an edge
caps.has_endpoint_elision()       # visible in a plan
```

Each takes `check=True` to raise `CapabilityError` naming the required version instead of returning
`False`.

**A capability read from the version describes a release, not the build in front of you.** Two
servers reporting `2.18-devel` were found to differ, one carrying the catalog a promoted property is
recorded in and one not. Where the answer has to be right rather than free, ask the server:

```python
conn.can_promote_properties()     # asks the catalog, once per connection, and keeps the answer
```

The wire and type layer is identical across every supported release, even though they sit on
different PostgreSQL bases (2.17 on 17, 2.18 and main on 18), so binary graph ids, the composite
decode, write counters, introspection and the commit resolution path need no version gating at all.
What is 2.18 only is property promotion, the GQL clause set, and ordering by a graph element.

## Performance

The design decisions worth knowing, since they shape what is fast:

- **A property map is decoded on first touch**, not on arrival. A read that never looks inside a
  vertex never pays for its map.
- **Rows are structs the collector does not track.** A large result is close to invisible to it.
  `agensgraph.paused_collection()` pauses collection around a bulk read anyway, and
  `agensgraph.freeze_after_import()` moves everything alive at startup out of the collector's way.
- **Parsing is not the round trip.** Everything above makes turning bytes into Python objects
  cheaper, and against a database on another machine that is a small share of what a query costs:
  the round trip dominates, and after it, whether the query uses an index. Measured over a couple of
  thousand vertices on a local server, skipping the property decode entirely is worth about 1.3
  times the parse and close to nothing end to end. Expect these to show on large results and on
  embedding-sized maps, where the parse is the bottleneck, and to be invisible on a query returning
  twenty rows.
- **`COPY` for bulk, `stream` for large reads, `Vector` for embeddings.** Each is an order of
  magnitude over the obvious alternative.

## API reference

Everything exported from `agensgraph`. The submodules `agensgraph.columnar`, `agensgraph.vector`,
`agensgraph.errors` and `agensgraph.dbapi` carry the rest.

**Connecting**

| | |
|---|---|
| `connect(conninfo, **kwargs)` | open a blocking connection |
| `Connection` / `AsyncConnection` | the connection classes, both psycopg subclasses |
| `Capabilities` | what one server can do, built from the version at no round trip |
| `MINIMUM_VERSION` | `(2, 17)`, the oldest server this driver reads |

**Pools**

| | |
|---|---|
| `ConnectionPool` / `AsyncConnectionPool` | keeps connections, with generations, drain and deadlines |
| `NullConnectionPool` / `AsyncNullConnectionPool` | keeps none, for one request per process |

**Values read from the wire**

| | |
|---|---|
| `Vertex`, `Edge`, `Path` | what a graph query returns |
| `GraphId` | an identity, `labid` and `locid`; `LABID_MAX`, `LOCID_MAX` are their bounds |
| `Vector`, `SparseVector` | embeddings, lazy and sparse |
| `Jsonb` | psycopg's wrapper, re-exported for a value you want sent as jsonb explicitly |

**Values you pass in**

| | |
|---|---|
| `Label` | a label or property name to be placed into a statement as an identifier |
| `Unspecified` | a string sent with no type, which is psycopg's own default behaviour |
| `Distance` | `L2`, `INNER_PRODUCT`, `COSINE`, `L1`, `HAMMING`, `JACCARD` |
| `DesiredIndex`, `Unique`, `Check` | what you want to exist, for the reconcilers |

**What a statement gives back**

| | |
|---|---|
| `Result` | `records`, `keys`, `counts`, `oids` |
| `GraphWriteCounts` | the five counters, each `None` when it cannot be attributed |
| `CommitOutcome` | `COMMITTED`, `ABORTED`, `IN_PROGRESS`, `UNKNOWN` |

**What the catalogs give back**

| | |
|---|---|
| `Graph` | `name`, `schema`, `labels` |
| `LabelInfo` | `id`, `name`, `kind`, `parent` |
| `DeclaredProperty` | `label`, `name`, `type`, `nullable` |
| `Index`, `Constraint` | `label`, `name`, `unique`, `definition` |
| `IndexElement` | one property an index is keyed on, with its operator class and order |

**Reliability**

| | |
|---|---|
| `connection.read_only_transaction()` | a transaction the server will not let write |
| `connection.can_run_server_programs()` | whether this role could run a command on the host |
| `Deadline`, `Expired` | one budget across the wait and the statement; `Expired` is a `TimeoutError` |
| `RetryPolicy` | decides whether and when to try again; you keep the loop |
| `Retryability`, `retryability(exc)`, `is_retryable(exc)` | the six categories, and how to ask |
| `TokenBucket` | the process-wide allowance a counter cannot multiply |

**Numbers**

| | |
|---|---|
| `read_numbers_exactly(bool)` | read a non-integer as a `Decimal` |
| `reading_numbers_exactly()` | whether that is currently on |

**Observability**

| | |
|---|---|
| `add_query_logger(fn)`, `remove_query_logger(fn)` | every statement, with `QueryRecord` |
| `enable_tracing(tracer=None)`, `disable_tracing()` | a span per statement |
| `Notice` | something the server said during a statement |
| `Notify` | an asynchronous notification, from `notifications()` |

**Errors**

| | |
|---|---|
| `BatchFailed` | a pipeline failed as a batch, carrying what was sent |
| `show_error_details(bool)`, `showing_error_details()` | whether DETAIL and CONTEXT reach the message |
| `redact_details(exc)` | cut them from one exception |
| `errors` | the whole hierarchy, plus `translate`, `mask_dsn`, `attach_query` |

**Bulk and memory**

| | |
|---|---|
| `paused_collection()` | pause the collector around a bulk read |
| `freeze_after_import()` | move what is already alive out of its way |

**Submodules**

| | |
|---|---|
| `columnar` | `to_arrow`, `to_pandas`, `to_polars`, `batches`, `reader`, `columns`, `Layout` |
| `vector` | `Vector`, `SparseVector`, `vector_index`, `nearest`, `generated_column`, `Distance`, `SEARCH_OPTIONS` |
| `dbapi` | the PEP 249 surface |
| `errors` | the exception hierarchy and the classification function |

## Development

```console
uv sync --group dev --all-extras
uv run python -X dev -W error -m pytest -q          # offline; the live tests skip themselves
AGENSGRAPH_TEST_DSN="host=localhost dbname=test" uv run python -m pytest -q
```

The blocking interface is generated from the awaiting one:

```console
uv run python tools/async_to_sync.py           # regenerate
uv run python tools/async_to_sync.py --check   # fail if it is out of date
```

Continuous integration runs the offline suite on Python 3.11 through 3.14, the full suite against
AgensGraph v2.17, v2.18 and main (each with pgvector built against it), the linter and formatter,
`mypy --strict`, a resolution at the lowest declared versions, a fuzzer over the text reader, and an
install of the built wheel and source distribution into clean environments.

Many tests pin engine behaviour the driver works around, deliberately: if a later release fixes one,
that test fails and names the workaround to drop.

## Relationship to 1.x

2.0 is a rewrite with no backwards compatibility. 1.x was a psycopg2 type-extension shim of about
nine hundred lines, and it was incorrect on output the engine legitimately produces: an empty path
raised, `NULL` elements had no representation, a label containing `{` collapsed a whole path into one
token, an unanchored graph id match turned `7.9.5` into `(7,9)`, every returned value was unhashable,
and `len(path)` made a valid single-vertex path falsy.

The distribution name and the import name are unchanged. 1.x remains on a maintenance branch for
security only, and the psycopg2 line does not migrate.

## License

Apache 2.0.
