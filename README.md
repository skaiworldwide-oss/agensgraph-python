# AgensGraph Python Driver

AgensGraph Python Driver allows Python programs to connect to an AgensGraph database. Since it is [Psycopg2](http://initd.org/psycopg/) type extension module for AgensGraph, it supports additional data types such as `Vertex`, `Edge`, and `Path` to represent graph data.

## Features
- Cypher query support for Psycopg2 PostgreSQL Python driver (enables cypher queries directly)
- Deserialize AgensGraph results (AGType) to Vertex, Edge, Path

## Install for PYPI
```sh
pip install agensgraph-python
```
## Build From Source

```sh
git clone https://github.com/skaiworldwide-oss/agensgraph-python
cd agensgraph-python
python setup.py install
```

## Example

```python
import psycopg2
import agensgraph

conn = psycopg2.connect("dbname=test host=127.0.0.1 user=agens")
cur = conn.cursor()
cur.execute("DROP GRAPH IF EXISTS t CASCADE")
cur.execute("CREATE GRAPH t")
cur.execute("SET graph_path = t")

cur.execute("CREATE (:v {name: 'AgensGraph'})")
conn.commit()

cur.execute("MATCH (n) RETURN n")
v = cur.fetchone()[0]
print(v.props['name'])
```

## Test

You may run the following command to test AgensGraph Python Driver.

```sh
python setup.py test
```

Before running the command, set the following environment variables to specify which database you will use for the test.

Variable Name                | Meaning
---------------------------- | ---------------------------
`AGENSGRAPH_TESTDB`          | database name to connect to
`AGENSGRAPH_TESTDB_HOST`     | database server host
`AGENSGRAPH_TESTDB_PORT`     | database server port
`AGENSGRAPH_TESTDB_USER`     | database user name
`AGENSGRAPH_TESTDB_PASSWORD` | user password
