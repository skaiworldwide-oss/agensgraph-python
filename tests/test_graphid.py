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

from agensgraph._graphid import GraphId, cast_graphid, adapt_graphid

class TestGraphId(unittest.TestCase):
    def setUp(self):
        self.out = '7.9'
        self.gid = cast_graphid(self.out, None)

    def test_getId(self):
        self.assertEqual((7, 9), self.gid.getId())

    def test_eq(self):
        self.assertEqual(self.gid, self.gid)

    def test_str(self):
        self.assertEqual(self.out, str(self.gid))

    def test_repr(self):
        self.assertEqual("%s(%s)" % (GraphId.__name__, self.gid),
                         repr(self.gid))

    def test_adapt(self):
        self.assertEqual(b"'7.9'", adapt_graphid(self.gid).getquoted())
