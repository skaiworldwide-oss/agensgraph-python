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

import unittest

from psycopg2.extras import json

from agensgraph._edge import Edge, cast_edge
from agensgraph._graphid import GraphId

class TestEdge(unittest.TestCase):
    def setUp(self):
        out = 'e[5.7][7.3,7.9]{"s": "", "i": 0, "b": false, "a": [], "o": {}}'
        self.e = cast_edge(out, None)

    def test_label(self):
        self.assertEqual('e', self.e.label)

    def test_eid(self):
        self.assertEqual(GraphId((5, 7)), self.e.eid)

    def test_start(self):
        self.assertEqual(GraphId((7, 3)), self.e.start)

    def test_end(self):
        self.assertEqual(GraphId((7, 9)), self.e.end)

    def test_props(self):
        self.assertEqual('', self.e.props['s'])
        self.assertEqual(0, self.e.props['i'])
        self.assertFalse(self.e.props['b'])
        self.assertEqual([], self.e.props['a'])
        self.assertEqual({}, self.e.props['o'])

    def test_eq(self):
        self.assertEqual(self.e, self.e)

    def test_str(self):
        props = '{"s": "", "i": 0, "b": false, "a": [], "o": {}}'
        out = "e[5.7][7.3,7.9]%s" % json.dumps(json.loads(props))
        self.assertEqual(out, str(self.e))

    def test_repr(self):
        self.assertEqual("%s(%s)" % (Edge.__name__, self.e), repr(self.e))
