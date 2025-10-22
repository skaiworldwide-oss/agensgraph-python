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

from agensgraph._graphpath import Path, cast_graphpath

class TestPath(unittest.TestCase):
    def setUp(self):
        self.o0 = '[n[7.3]{}]'
        self.p0 = cast_graphpath(self.o0, None)
        self.o1 = '[n[7.3]{},r[5.7][7.3,7.9]{},n[7.9]{}]'
        self.p1 = cast_graphpath(self.o1, None)

    def test_vertices(self):
        self.assertEqual(1, len(self.p0.vertices))
        self.assertEqual('n[7.3]{}', str(self.p0.vertices[0]))
        self.assertEqual(2, len(self.p1.vertices))
        self.assertEqual('n[7.3]{}', str(self.p1.vertices[0]))
        self.assertEqual('n[7.9]{}', str(self.p1.vertices[1]))

    def test_edges(self):
        self.assertEqual(0, len(self.p0.edges))
        self.assertEqual(1, len(self.p1.edges))
        self.assertEqual('r[5.7][7.3,7.9]{}', str(self.p1.edges[0]))

    def test_eq(self):
        self.assertEqual(self.p0, self.p0)
        self.assertEqual(self.p1, self.p1)

    def test_len(self):
        self.assertEqual(0, len(self.p0))
        self.assertEqual(1, len(self.p1))

    def test_str(self):
        self.assertEqual(self.o0, str(self.p0))
        self.assertEqual(self.o1, str(self.p1))

    def test_repr(self):
        self.assertEqual("%s(%s)" % (Path.__name__, self.p0), repr(self.p0))
        self.assertEqual("%s(%s)" % (Path.__name__, self.p1), repr(self.p1))
