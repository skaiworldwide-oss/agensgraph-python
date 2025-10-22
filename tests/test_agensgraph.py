'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import unittest

import psycopg2

import agensgraph

dbname = os.environ.get('AGENSGRAPH_TESTDB', 'agensgraph_test')
dbhost = os.environ.get('AGENSGRAPH_TESTDB_HOST', None)
dbport = os.environ.get('AGENSGRAPH_TESTDB_PORT', None)
dbuser = os.environ.get('AGENSGRAPH_TESTDB_USER', None)
dbpass = os.environ.get('AGENSGRAPH_TESTDB_PASSWORD', None)

dsn = 'dbname=%s' % dbname
if dbhost is not None:
    dsn += ' host=%s' % dbhost
if dbport is not None:
    dsn += ' port=%s' % dbport
if dbuser is not None:
    dsn += ' user=%s' % dbuser
if dbpass is not None:
    dsn += ' password=%s' % dbpass

class TestConnection(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg2.connect(dsn)
        self.cur = self.conn.cursor()
        self.cur.execute('DROP GRAPH IF EXISTS t CASCADE')
        self.cur.execute('CREATE GRAPH t')
        self.cur.execute('SET graph_path = t')
        self.conn.commit()

    def tearDown(self):
        self.cur.execute('DROP GRAPH t CASCADE')
        self.cur.close()
        self.conn.close()

class TestGraphId(TestConnection):
    def test_graphid(self):
        self.cur.execute('CREATE (n {}) RETURN id(n)')
        self.conn.commit()

        gid0 = self.cur.fetchone()[0]

        self.cur.execute("MATCH (n) WHERE id(n) = %s RETURN id(n)", (gid0,))
        gid1 = self.cur.fetchone()[0]

        self.assertEqual(gid1, gid0)

class TestVertex(TestConnection):
    def test_vertex(self):
        self.cur.execute(
                "CREATE (n:v {s: '', i: 0, b: false, a: [], o: {}}) RETURN n")
        self.conn.commit()

        v = self.cur.fetchone()[0]
        self.assertEqual('v', v.label)
        self.assertEqual('', v.props['s'])
        self.assertEqual(0, v.props['i'])
        self.assertFalse(v.props['b'])
        self.assertEqual([], v.props['a'])
        self.assertEqual({}, v.props['o'])

        self.cur.execute("MATCH (n) WHERE id(n) = %s RETURN count(*)", (v.vid,))
        self.assertEqual(1, self.cur.fetchone()[0])

class TestEdge(TestConnection):
    def test_edge(self):
        self.cur.execute(
                "CREATE (n)-[r:e {s: '', i: 0, b: false, a: [], o: {}}]->(m)\n"
                'RETURN n, r, m')
        self.conn.commit()

        t = self.cur.fetchone()
        v0 = t[0]
        e = t[1]
        v1 = t[2]

        self.assertEqual('e', e.label)
        self.assertEqual(e.start, v0.vid)
        self.assertEqual(e.end, v1.vid)
        self.assertEqual('', e.props['s'])
        self.assertEqual(0, e.props['i'])
        self.assertFalse(e.props['b'])
        self.assertEqual([], e.props['a'])
        self.assertEqual({}, e.props['o'])

        self.cur.execute("MATCH ()-[r]->() WHERE id(r) = %s RETURN count(*)",
                         (e.eid,))
        self.assertEqual(1, self.cur.fetchone()[0])

class TestPath(TestConnection):
    def test_path(self):
        self.cur.execute("CREATE p=({s: '[}\\\\\"'})-[:e]->() RETURN p")
        self.conn.commit()

        p = self.cur.fetchone()[0]
        self.assertEqual(1, len(p))

        for v in p.vertices:
            self.cur.execute("MATCH (n) WHERE id(n) = %s RETURN count(*)",
                             (v.vid,))
            self.assertEqual(1, self.cur.fetchone()[0])

        for e in p.edges:
            self.cur.execute(
                    "MATCH ()-[r]->() WHERE id(r) = %s RETURN count(*)",
                    (e.eid,))
            self.assertEqual(1, self.cur.fetchone()[0])

class TestParam(TestConnection):
    def setUp(self):
        super(TestParam, self).setUp()
        self.name = "'Agens\"Graph'"

    def test_param_dict(self):
        d = {'name': self.name, 'since': 2016}
        p = agensgraph.Property(d)
        self.cur.execute('CREATE (n %s) RETURN n', (p,))
        self.conn.commit()

        v = self.cur.fetchone()[0]
        self.assertEqual(self.name, v.props['name'])
        self.assertEqual(2016, v.props['since'])

    def test_param_list_and_tuple(self):
        a = [self.name, 2016]
        t = (self.name, 2016)
        pa = agensgraph.Property(a)
        pt = agensgraph.Property(t)
        self.cur.execute('CREATE (n {a: %s, t: %s}) RETURN n', (pa, pt))
        self.conn.commit()

        v = self.cur.fetchone()[0]
        va = v.props['a']
        self.assertEqual(self.name, va[0])
        self.assertEqual(2016, va[1])
        vt = v.props['t']
        self.assertEqual(self.name, vt[0])
        self.assertEqual(2016, vt[1])

if __name__ == '__main__':
    unittest.main()
