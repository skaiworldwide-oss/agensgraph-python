# Changelog

## 2.0.0 (unreleased)

A rewrite on [psycopg 3](https://www.psycopg.org/psycopg3/). Nothing from 1.x is carried over: the
distribution name and the import name are the same, and no code written against 1.x runs against
this. See [Relationship to 1.x](https://github.com/skaiworldwide-oss/agensgraph-python#relationship-to-1x)
for why.

The connection *is* a psycopg connection, so cursors, server-side cursors, `COPY`, pipelines,
`LISTEN` and two-phase commit keep working, and Cypher and SQL run in the same transaction.

### Breaking

- **psycopg 3 instead of psycopg2.** `psycopg[binary]>=3.3.4` and `psycopg-pool>=3.3.1`.
- **AgensGraph 2.17 is the oldest server read.** The version arrives in the startup packet, so a
  server below the floor is refused at connect rather than on a missing catalog later.
- **Python 3.11 is the floor.**
- **A Python `str` parameter is declared `text`.** On 1.x a string went out untyped, and the server
  parsed it as JSON: `'123'`, `'null'`, `'true'` and `'1.5'` were accepted and matched nothing, so a
  lookup by order number or postcode silently returned no rows. It now finds its row. The cost is
  that a string passed where the column is a `date`, `uuid`, `integer` or `bytea` no longer resolves
  and says so at parse time naming both types; `%s::date` fixes it, and `Unspecified("...")` restores
  the old behaviour for one value.
- **`Vertex`, `Edge`, `Path` and `GraphId` are new types**, hashable, with metadata and user
  properties in separate namespaces. On 1.x every returned value was unhashable, so a vertex could
  not be a dict key or a set member, and a property named `label` overwrote the real label. Their
  fields are renamed: `.vid` and `.eid` are `.id`, and `.props` is `.properties`. `len(path)` is the
  number of elements, and `path.length` is the number of hops, which is what `len` used to give.
- **A property map is a plain `dict`, sent as a parameter.** 1.x wrapped one in `Property({...})`,
  which inlined it into the statement as Cypher literal text. Placeholders are psycopg's `%s`, as
  they were under psycopg2.
- **Nothing is registered on anyone else's connection.** Importing 1.x registered its casters into
  psycopg2's global map, so every psycopg2 connection in the process read graph types. This driver
  keeps its adapters on its own connections. There is no psycopg2 path at all; for a psycopg 3
  connection this driver did not make, `agensgraph.adapters.register_text()` does it explicitly.

### Fixed, from 1.x

Each of these was demonstrated against output the engine legitimately produces:

- An empty path, which is legal, raised `InterfaceError` with an empty message.
- `NULL` elements had no representation, so `[NULL,NULL,NULL]` raised.
- A label containing `{` collapsed a whole path into one token and leaked a `JSONDecodeError`,
  because the reader tracked brace depth but not string state.
- An unanchored graph id match turned `7.9.5` into `(7,9)`, and combined with the `][` pivot in the
  edge pattern a property value could have start and end ids fabricated out of its text.
- `len(path)` was the edge count, so a valid single-vertex path was falsy.

### Reading a result

- Vertices, edges and paths come back as values, from either rendering the server emits. The text
  rendering is the default; `binary_=True` asks for the composite one per statement. Measured over
  five thousand rows it is worth asking for on edges, at a median of 1.29, and run-to-run variation
  swamps the difference on paths and on whole vertices.
- A list in the text rendering whose label holds an element rendering, as `v[1.1]{},w`, reads two
  ways. A path is read the way that holds together; a list of vertices or edges the way the label
  table confirms, once `graph()` or `refresh_labels()` has filled it.
- A property map is decoded on first access, so a row whose properties are never read never pays for
  them.
- `read_numbers_exactly()` for a property map whose numbers must keep every digit.
- `stream()` reads a large result a chunk at a time through a server-side cursor, and says which
  statements cannot be wrapped in one rather than letting the server produce a bare syntax error.
- `agensgraph.columnar` exports to Arrow, pandas and polars, with the backends imported only where
  they are used.
- `to_builtins()` and a `default=` hook, so a graph value can be handed to anything wanting JSON.

### Writing

- `load_vertices()` and `load_edges()` copy elements in.
- `upsert_vertices()` and `upsert_edges()` write only what is not already there, keyed on a property
  or on the pair of elements an edge joins, and refuse a key that nothing keeps unique.
- `identity_map()` reads what the server called each element.
- Dense and sparse embedding vectors, `agensgraph.vector.nearest()`, and property indexes built the
  way a Cypher query matches them.

### Connections

- A pool with a generation counter, so one server restart retires the whole set instead of handing
  N failures to N callers; `drain()`; and a pool that keeps nothing, for a per-request process.
- `deadline()` bounds a block on a connection with no pool.
- A retry policy with six recoveries rather than a boolean, a token allowance that time refills, and
  full jitter under a cap.
- If the connection is lost while committing, a different connection is asked what became of the
  transaction rather than the write being retried blind.
- Keepalive is asked for by default, because a connection waiting for a reply has transmitted
  nothing and so `tcp_user_timeout` does not bound the wait.
- A connection whose statement was interrupted is closed rather than lent out again.
- Where the server reports `graph_path` (AgensGraph 2.18.6 and later), the graph the session is
  reading is read off the connection, so a describing method given no graph sends only its own
  statement, and the label table follows the path back across a commit, a rollback and a
  `transaction()` block. Elsewhere the statements that go past are read for it.

### Reading the catalogs

- `graphs()`, `labels()`, `properties()`, `indexes()` and `constraints()`, which is the only
  introspection there is: `psql` has no describe support for these catalogs.
- `describe()` builds a graph's shape from the catalogs, installing nothing in the database.
- `ensure_labels()`, `ensure_indexes()` and `ensure_constraints()` diff what should exist against
  what does, and will write the DDL out for a caller with no connection. A constraint name derived
  for a label and property too long for the server ends in a digest of the whole, so two long names
  stay two.
- A `DesiredLabel` declares the properties it keeps in columns of their own, as `PromotedProperty`;
  on a label already there they are added with `ALTER`, and one the label has, its own or inherited,
  is left alone. `declared_properties()` reports an inherited column too, marked `inherited`.
- A vector index reconciles: an `IndexElement` takes a `cast`, so the index over a property kept in
  the map is described as the server prints it. Casting a property that has a column of its own is
  refused, since a search on the column never uses such an index. `vector_indexes()` reads them back
  in parts as `VectorIndex`.

### Everything else

- A read-only transaction the server enforces, for running a statement you did not write, and a
  refusal to send more than one statement at a time.
- Write counters per statement, credited to the clauses that could have moved them.
- Spans behind a module-level gate, a query logger, and the server's notices as structured values.
- `LISTEN`/`NOTIFY`, with a handler or an iterator and never both at once.
- `pipeline_query()` reads a burst of statements in one round trip; `pipeline_batch()` writes one,
  and reports failure as the batch's rather than pretending to know which statement it was.
- The PEP 249 surface, proved by a SQLAlchemy dialect in the test suite.
- Failures carry the statement and the parameters as attributes and keep row data out of what the
  failure prints as, because PostgreSQL puts row data in `DETAIL`.
- A graph write refused in a read-only transaction is `ReadOnlyGraphWrite` on 2.17 and 2.18 alike:
  from 2.18.4 the server names the command `CYPHER` where it printed `???` before.
- Every version string a release has reported parses: `2.16`, `2.17.0`, `2.18.4.0`, `2.18.6.0-rc1`
  and `2.18-devel`. The two leading numbers are the AgensGraph line and the PostgreSQL major under
  it, and nothing after them is read. Parsing is not acceptance: `2.16` is read and then refused,
  since the floor is 2.17.
- Ships `py.typed`, checked under `mypy --strict` and pyright strict.

## 1.0.2 and earlier

A psycopg2 type-extension module. See the [`v1.0` branch](https://github.com/skaiworldwide-oss/agensgraph-python/tree/v1.0).
