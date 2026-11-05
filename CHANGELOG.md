# Changelog

## 2.1.0 (2026-11-06)

AgensGraph 2.14 to 2.18, a read-only boundary that holds, a safe way to run model-written Cypher,
and a choice of where vectors live. Every 2.0.0 signature is unchanged, and so is every default but
the three noted under *Changed*.

### Security

- **`read_only_transaction()` could be left by the statement it was guarding.** In 2.0.0, on every
  server, text sent with no parameters went over the simple query protocol, so
  `...; COMMIT; DROP GRAPH g CASCADE` ended the read-only transaction and ran the rest read-write.
  Inside the block every statement now goes over the extended protocol, which takes one statement
  (`42601`); text that would end or split the transaction is refused before it is sent; the
  transaction is pinned read-only before the first statement; and this holds on every route:
  `execute`, `execute_query`, a cursor, `executemany`, `COPY`, `stream` and a pipeline. Anything
  that ran model output through 2.0.0's block should upgrade.
- **AgensGraph 2.14 and 2.15 run a Cypher write in a read-only transaction**, and skip privileges
  for it: a role allowed only `SELECT` committed a `SET` and a `DETACH DELETE`. The fix is in 2.16.0
  and was not backported. On those releases the block refuses a statement holding a Cypher write
  clause before sending it, and a write the reading misses is undone by the rollback.
  `read_only_is_enforced()` asks the server which kind it is.

### Server releases

- **AgensGraph 2.14 is the oldest server read**, down from 2.17. Every release from 2.14.1 to
  2.18.6.1 has the whole suite run against it on every change.
- What differs between releases is one declarative table, `capabilities.FEATURES`, asked through
  `Capabilities.has(name, check=False)`. It reads the patch as well as the line, and a release
  candidate or development build answers as the first release of its line. Nothing else compares
  versions.
- Identifiers are quoted against the reserved words of every supported grammar.
- The error for a plain SQL write to a label's table names `enable_graph_dml` only where it exists
  (2.17 and later), and 2.15's internal error for a write to a missing label is `MissingLabel`, as
  2.14 and 2.16 report it.

### Added

- **`run_untrusted(text, max_rows=, timeout=)`** for one model-written statement. The text is
  checked before it is sent (one Cypher read, an allow-list of functions, bounded paths, no SQL,
  comments or dollar quotes, a size limit); it runs in a read-only transaction that is always rolled
  back, through a cursor in one round trip (0.42–0.44 ms against 0.30 bare); the role is checked
  (not a superuser, no host or session reach, not the application's role or a member of it); and the
  result must have the columns its `RETURN` declares. A call is checked wherever it stands, a map's
  value included, and one in a named schema is refused; names in any script are read as names.
  `allow_writes=True` needs 2.16 or later, and lets a write start with `CREATE` or `MERGE`.
  `untrusted_role_problems()` says what is wrong with a role.
- **`VectorStorage`: where a label's vectors live.** `storage="properties"`, the default, is the
  2.0.0 design and needs 2.17; below that it is refused with a note naming the other design.
  `storage="table"` keeps them in a table keyed by the vertex with a cascading key, on every release
  from 2.14, and is the faster design for filtered search (4–11 ms against 375–930 at 22,000 vectors).
  `ensure_vectors`, `write_vectors`, `vectors_of`, `delete_vectors`, `nearest`, `build_vector_index`,
  `drop_vectors` and `vector_storage_state` do the same things in either design, and a table whose
  key a `DROP ... CASCADE` took raises `StaleVectorStorage` rather than attach old vectors to new
  vertices. `ensure_vectors` makes only what is missing, so a role allowed only to read and write
  can call it once the storage is there; an index built for another distance is reported as a
  mismatch rather than passing for ready.
- **A deadline held by the client as well as the server.** A block with a budget is watched by one
  thread; if a statement is running and the server has sent nothing for half a second longer than
  its own limit allows, the socket is shut down and the caller gets `Stalled`, an `Expired` that is
  classified as a lost connection. Through a stalled network a one-second budget returns in 1.44 s,
  where 2.0.0 waited past fifteen. Nothing else sets it off: a block that runs past its budget keeps
  its connection. Arming it costs a third of a microsecond.
- `execute_query(settings_={...})`: settings for one statement, sent in its round trip (0.25 ms
  against 1.03 for the same settings in a transaction of their own).
- `pipeline_query` takes `binary_`, `prepare_` and `row_`.
- `reserve_ids`, `load_vertices_with_ids` and `merge_vertices`: bulk writes that hand back each
  vertex's identity, reserved through the label's own sequence.
- `run_transaction(work, policy=, deadline=)` on a pool: runs a transaction again where that is
  safe, and settles a commit whose answer was lost, or raises `UnresolvedCommit`.
- `schema_lock()`, taken by every `ensure_*`, so two processes reconciling one graph take turns
  rather than one failing with `42P07`.
- `describe(counts=False)`, and `meta_is_current()` and `label_positions()`, which say whether
  `graphmeta` is current without counting edges.
- `connect(idle_in_transaction_timeout=seconds)`, as a startup setting.
- `connection.recover(exc)`: what a failure's category asks of the connection, done before another
  attempt. A lost prepared statement (`26000`) failed the same way on every attempt in autocommit
  until the statement cache was emptied; `run_transaction` now does this before handing the
  connection back.
- The vector types are sent in every format: `%t` and arrays of vectors, binary `COPY` into
  `halfvec` and `sparsevec` columns, and `%b` for a `SparseVector` into a SQL column of its type.
  `Vector(numpy_array)` copies the buffer (0.5 µs for 1536 values against 46), and
  `register_vectors()` is one statement.
- `Bits`, a bit string: what binary quantisation produces and Hamming and Jaccard distance measure.
  After `register_vectors()`, `bit` reads as `Bits` and a `Bits` is sent as `bit`, so a search
  needs no cast on its parameter. It is a `str` of `0` and `1`, so it equals what a bit string read
  as before, and a connection that never registered reads one exactly as it did.
- Spans for what no one statement covers: `COPY`, `PIPELINE` (per batch, with its size), `STREAM`
  (never made current, since it yields between rows) and `POOL WAIT`.

### Changed

- **psycopg 3.3.6 or later**, up from 3.3.4. `recover()` empties the statement cache by sending
  `DEALLOCATE ALL`, and 3.3.6 is the first release that empties its own copy when it sees one; on
  3.3.4 and 3.3.5 a lost prepared statement kept failing after recovery. Every integration that
  depends on this driver allows 3.3.6.
- A statement's span is named for its operation (`MATCH`, `SELECT`, ...) with `db.operation.name`
  set, rather than `query` for every statement.
- `vector_search_options(local=True)` in autocommit, outside a transaction or pipeline, did nothing
  without saying so; it now warns and sends nothing, and 3.0 will raise. It sends every option in
  one statement.
- The gap a deadline leaves below its budget is half a second or a fifth of the budget, whichever
  is less, so a budget of half a second or less is no longer refused. Budgets over 2.5 s are
  unchanged.
- A pool skips the health check for a connection returned less than 30 seconds ago, and checks every
  borrow for 30 seconds after a failure that may mean the server went away.
- `has_element_ordering()` is deprecated in favour of `has_element_comparison()`.
- `StaleLabelCache` is classified `Retryability.REFRESH_LABELS`, a seventh category, rather than
  `RESET_STATE`: emptying the statement cache does nothing for a label table out of date, and a
  caller following the old category did the wrong thing. `RESET_STATE` is now `26000` alone.
- psycopg's `Jsonb` goes through the driver's encoder: faster, and NaN is refused rather than sent.

### Fixed

- After `register_vectors()`, a text `COPY` with `set_types(["text"])` handed its strings to the
  sparse vector's dumper and raised `AttributeError`.
- A binary result naming a label created after the label table was read raised; it is now read
  again after one refresh, without running the statement again.
- The strings `"nan"`, `"inf"` and `"infinity"` as keys became non-finite numbers and failed.
- `describe()` scanned every edge on 2.14 to 2.17 to decide whether `graphmeta` was gathered.
- `to_json(result)` wrote each vertex's private fields (`_id`, `_raw`, `_props`) rather than its
  identity, label and properties, and so did a row read as a named tuple: both are tuple subclasses,
  which the walk did not enter. A `Result` now becomes one object per row, keyed by its columns, and
  a named-tuple row an object. `to_builtins` takes the same values, where it raised before.
- A blocking pool's `configure`, `setup` and `reset`, and the work `run_transaction` takes, were
  typed as returning something to wait for, so a type checker refused a plain function.

## 2.0.0 (2026-09-18)

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

A psycopg2 type-extension module.
