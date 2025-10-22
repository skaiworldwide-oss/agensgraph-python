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

from agensgraph._graphid import GraphId
from agensgraph._vertex import Vertex, cast_vertex

class TestVertex(unittest.TestCase):
    def setUp(self):
        out = 'v[7.9]{"s": "", "i": 0, "b": false, "a": [], "o": {}}'
        self.v = cast_vertex(out, None)

    def test_label(self):
        self.assertEqual('v', self.v.label)

    def test_vid(self):
        self.assertEqual(GraphId((7, 9)), self.v.vid)

    def test_props(self):
        self.assertEqual('', self.v.props['s'])
        self.assertEqual(0, self.v.props['i'])
        self.assertFalse(self.v.props['b'])
        self.assertEqual([], self.v.props['a'])
        self.assertEqual({}, self.v.props['o'])

    def test_eq(self):
        self.assertEqual(self.v, self.v)

    def test_str(self):
        props = '{"s": "", "i": 0, "b": false, "a": [], "o": {}}'
        out = "v[7.9]%s" % json.dumps(json.loads(props))
        self.assertEqual(out, str(self.v))

    def test_repr(self):
        self.assertEqual("%s(%s)" % (Vertex.__name__, self.v), repr(self.v))
