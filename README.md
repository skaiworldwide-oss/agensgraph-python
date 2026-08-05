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

A vertex, an edge and a path are read from the wire. They are structs the garbage collector does not
track, which is most of what makes a large result cheap:

| building 200,000 vertices | |
|---|---|
| as an ordinary object with `__slots__` | 176 ms |
| **as an untracked struct** | **33 ms** |

Reading 200,000 vertices end to end went from **1564 ms to 986 ms** for the same reason. Nothing here
can take part in a reference cycle — a property map holds only what JSON can — so going untracked
costs nothing.

The public surface is read-only: `id`, `label`, `properties`, and `start`/`end` on an edge have no
setters.


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

### Sending a property map

Rendered with msgspec, which is where most of the cost of a write is:

| | `json.dumps` | this |
|---|---|---|
| a 1024-float embedding | 681 µs | **61 µs** |
| a small map | 2.22 µs | **0.52 µs** |

`NaN` and the infinities are refused. jsonb has no way to store one, and encoding it would write
`null` instead — which is the wrong value rather than an error. The check runs only when the output
holds a `null` at all, so a map of numbers never pays for it.

A `datetime`, `date`, `UUID`, `Decimal`, `bytes` or `set` is rendered rather than refused —
respectively an ISO string, a string, a string, a base64 string and a list.

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

### Pausing the collector

```python
with agensgraph.paused_collection():
    records = conn.execute_query("MATCH (n:Person) RETURN n").records
```

Worth about **1.03×** on a read of 200,000 vertices — and that is the point: a row is a struct the
collector does not track, so a result is almost entirely invisible to it. The same read against rows
built as ordinary objects was **1.69×**, which is the cost this avoids by not being tracked at all.
Reference counting still frees as usual; only the collection of cycles waits.

`agensgraph.freeze_after_import()` is the other half — call it once at startup and every later
collection skips the module-level objects that were never going to be collected anyway.

## Handing a result to something columnar

An escape hatch, not how results are held — taking a table apart into Python objects costs more than
never having built one:

```python
from agensgraph.columnar import to_arrow, to_pandas, to_polars

result = conn.execute_query("MATCH (n:Person) RETURN n.name AS name, n.age AS age")
frame = to_pandas(result.records, result.keys)
```

Installed as needed, and imported only where used: `agensgraph-python[arrow]`, `[pandas]`,
`[polars]`.

**A whole vertex is refused**, because it is an identity, a label and a map rather than one value —
with a message saying to return the parts wanted instead. A `GraphId` becomes its text form, a
`Vector` a list of numbers, a sparse vector its text form.

## Numbers in a property map

jsonb keeps an arbitrary-precision decimal; Python's float does not. Where they part company,
measured:

| written | read back |
|---|---|
| an integer of any size, `1e400` included | **exactly**, as an `int` |
| `3.141592653589793238462643383279` | `3.141592653589793` |
| `1e-400` | `0.0` |
| `-0.0` | `0.0` — the **server** drops the sign, before the driver sees it |

`1e400` is worth calling out: the server stores it as an exact 401-digit integer, not an infinity.

For the cases where the lost digits matter:

```python
agensgraph.read_numbers_exactly()      # once, at startup
```

Every non-integer then reads as a `Decimal` keeping whatever the server holds — `1e-400` included.
It costs about **3.7×** to decode a map of numbers, and integers are exact either way. Writing a
`Decimal` back stores it as a JSON string, so that round trip is not identity.

## Sending a burst of statements

For a batch whose cost is round trips rather than work:

```python
conn.pipeline_batch([f"CREATE (:Event {{n: {n}}})" for n in range(1000)])
```

**A pipeline reports a failure against the wrong statement.** Measured with four statements of which
only the second was bad: the *first* raised the error, and the other three raised with no SQLSTATE at
all. So a failure here raises `BatchFailed`, carrying every statement sent, with the server's error
as its `__cause__`:

```python
try:
    conn.pipeline_batch(statements)
except agensgraph.BatchFailed as failed:
    for statement in failed.statements:   # run them one at a time to find the culprit
        ...
```

Running them serially is left to you rather than done automatically, because replaying a write would
apply it twice. psycopg's `conn.pipeline()` remains available for the cases where you want to read
results back.

## Committing in two phases

Works, with graph writes, unchanged from psycopg — including a write that returned rows, which the
server plans differently:

```python
conn.tpc_begin("order-4711")
conn.execute("CREATE (:Order {id: 4711})")
conn.tpc_prepare()
...
conn.tpc_commit()          # or from any other connection, via tpc_recover()
```

Two things to know. The connection must not be in autocommit. And `max_prepared_transactions` is
**0 by default**, so on an untouched server `tpc_prepare()` raises `NotSupportedError` carrying the
server's own message and the name of the setting to change.

A prepared transaction holds its locks until it is committed or rolled back, so one left behind
blocks a later `DROP GRAPH`. `tpc_recover()` lists what is waiting — but ask it from another
connection, since one that already has something prepared cannot run the query.

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

### Saying what should exist, rather than what to do

Creating an index that is already there is an error, not a no-op. Hand over the list instead and
get back the statements that made it so — empty when nothing had to change:

```python
from agensgraph import Check, DesiredIndex, IndexElement, Unique

conn.ensure_indexes([
    DesiredIndex("Person", ("name",)),
    DesiredIndex("Person", ("email",), unique=True),
    DesiredIndex("Person", (IndexElement("surname"), IndexElement("age", descending=True))),
    DesiredIndex("Person", (IndexElement("tags", "jsonb_path_ops"),), method="gin"),
    DesiredIndex("Person", ("email",), name="person_active", where="active = true"),
])
conn.ensure_constraints([
    Unique("Person", "email"),
    Check("Person", "age > 0", "person_age_positive"),
])
```

`dry_run=True` returns the statements without running them; `drop_extra=True` also removes what was
not asked for.

An index is matched by its access method and the elements it keys, read off the definition the
server printed. A constraint is matched by name. Both follow from how the server stores them:

| | |
|---|---|
| a name | derived from the columns, then truncated, then a counter on collision — so names are not the key |
| a definition | printed with defaults omitted (`ASC` never, `NULLS LAST` only with `DESC`, an operator class only when not the default) |
| a predicate | stored normalised — `age > 0` prints as `(age) > cypher_to_jsonb(0)` |

Three consequences worth knowing. **Name an operator class only when it differs from the default**,
since the server omits a default when printing and the two would never compare equal — asking for
one anyway raises after the first run rather than rebuilding the index for ever. **A partial index
needs a `name` and is matched by it**, so a change to its predicate is not noticed; change the name
to force a rebuild. And an index over a nested path, an expression, or with `INCLUDE` columns is not
describable here — it is never matched and never dropped, so write those as DDL.

A `Check` needs a name of its own because the server names an unnamed one `<label>_properties_check`
and then adds a counter.

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

**Two ways to hold a searchable vector**, and both are indexed as a *property* index. Leave it in
the property map and cast it, or give it a column of its own:

```python
from agensgraph.vector import Distance, generated_column, nearest, vector_index

# in the property map -- the cast carries the dimension, in the index and in the search alike
conn.execute(vector_index("Doc", "v", dimensions=1024))
result = conn.execute_query(nearest("Doc", "v", dimensions=1024, limit=10), (query_text,))

# or with a column of its own, which needs no cast and refuses a wrong-length value on write
conn.execute(f"CREATE VLABEL Emb ({generated_column('v', 1024)})")
conn.execute(vector_index("Emb", "v", operator_class=Distance.L2.operator_class))
result = conn.execute_query(nearest("Emb", "v", operator=Distance.L2, limit=10), (query_text,))
```

**It has to be a property index.** `CREATE PROPERTY INDEX` compiles its expression through the
Cypher parser, so the index holds the same expression a Cypher query builds. An index written as
plain SQL over `(properties ->> 'v')::vector(1024)` holds the *SQL* expression instead and is
**never matched** — measured with sequential scans switched off, where the planner chose a
penalised sequential scan rather than the index it could not use.

Two more things cost the index silently, both measured, and the second is why `vector_index` and
`nearest` share one spelling of the cast:

| | plan |
|---|---|
| the search casts to `vector(1024)`, as the index does | **Index Scan** |
| the search casts to a bare `vector`, or to `vector(3)` | Sort over Seq Scan |
| the operator class does not serve the operator asked | Sort over Seq Scan |

`Distance.operator_class` gives the right class for an operator. `hnsw` and `ivfflat` both work;
pass `options={"lists": 100}` for the `WITH` clause an ivfflat index wants.

### Tuning a search

```python
conn.vector_search_options({"hnsw.ef_search": 100})     # the transaction, not the session
```

Seven settings exist, and `agensgraph.vector.SEARCH_OPTIONS` lists them with the type each takes:
`hnsw.ef_search` (40), `hnsw.iterative_scan` (off), `hnsw.max_scan_tuples` (20000),
`hnsw.scan_mem_multiplier` (1), `ivfflat.probes` (1), `ivfflat.iterative_scan` (off),
`ivfflat.max_probes` (32768). A name that is not one of them is refused before it is sent — the
server accepts an unknown `hnsw.` name without complaint, so a typo would otherwise look applied.

### Binary quantisation

`binary_quantize()` runs in Cypher and returns a bit string. **The cast it needs does not.** Cypher's
cast grammar takes `::vector(n)` and `::halfvec(n)`, but `::bit(n)` is a syntax error and there is no
`cast(… as …)` form — so hamming (`<~>`) and jaccard (`<%>`) search needs plain SQL against the
label's own table. Both halves are asserted in the suite.

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

### Dense vectors

A dense vector arrives as a `Vector`, which keeps the bytes the server sent and turns them into
numbers only if you look. That is the same bargain the driver makes with a property map, and it pays
here for the same reason: a vector search asks the *server* for the distance, so the components of
the vectors it ranked are often never read.

```python
(v,) = conn.execute("SELECT v FROM docs LIMIT 1", binary=True).fetchone()

len(v)              # free -- the dimension is in the first two bytes
v == [1.0, 2.0]     # True when the numbers match
v[0], v[-1], v[0:2], list(v), sum(v), 2.0 in v, v.index(2.0)
v.values            # an array('f') that numpy and torch take without copying
v.tolist()          # an ordinary list, if you want one
```

It is not a `list` — `isinstance(v, list)` is `False` and `json.dumps(v)` raises — but it compares
equal to one, which is the part that would otherwise go wrong quietly.

| reading 200 embeddings of 1536 dimensions | |
|---|---|
| binary, values untouched | **5.8 ms** |
| binary, one value read from each | 10.5 ms |
| text, values untouched | 16.9 ms |

Ask for `binary=True` where you can: even with the parse deferred, text costs 3× more.

Sending is where the largest saving is, because every other route formats each number as decimal
text:

| sending one 1536-dimension embedding | |
|---|---|
| a `list` with a `::vector(1536)` cast | 2.55 ms |
| a string built by hand | 0.79 ms |
| **`Vector(values)`** | **0.32 ms** |

and in bulk, loading 20,000 embeddings of 768 dimensions:

| | rows/s |
|---|---|
| one statement at a time | 2,002 |
| `COPY` in text | 3,282 |
| **`COPY` binary with `Vector`** | **31,396** |

```python
from agensgraph.vector import Vector

conn.execute("INSERT INTO docs VALUES (%b)", (Vector(embedding),))

with conn.cursor().copy("COPY docs (v) FROM STDIN (FORMAT BINARY)") as copy:
    copy.set_types(["vector"])
    for embedding in embeddings:
        copy.write_row([Vector(embedding)])
```

The same type goes both ways, so a vector read from one place can be sent to another without being
unpacked at all.

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

Both renderings are asserted to produce identical values on floats drawn from the whole of single
precision — not just on small whole numbers, which survive every conversion and so prove nothing.

## Being told when the graph changes

A trigger on a label table can announce a change, which makes this the change feed for a graph:

```python
conn.listen("graph_changed")
conn.add_notify_handler(lambda notice: print(notice.channel, notice.payload))
...
conn.notify("graph_changed", "doc")     # from anywhere
conn.listening()                        # ['graph_changed']
conn.unlisten()                         # all of them
```

The channel is quoted into the statement, because neither `LISTEN` nor `UNLISTEN` takes a parameter
for it and `LISTEN` cannot be prepared at all. `notify()` goes through `pg_notify`, which is a
function and does take one, so a channel held in a variable is bound rather than quoted.

There is also `conn.notifications(timeout=…, stop_after=…)` to read them as they arrive. **Prefer the
handler.** The iterator holds the connection's lock while it is being read, so a caller who stops
part way leaves the connection unusable until it is collected. Using both at once raises rather than
delivering each announcement to whichever route happens to be looking.

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
