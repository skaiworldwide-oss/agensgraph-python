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

from agensgraph._property import Property

from psycopg2.extensions import QuotedString, adapt

class TestProperty(unittest.TestCase):
    def test_string(self):
        self.assertEqual(r"'\"'", Property('"').getquoted())
        self.assertEqual(r"''''", Property("'").getquoted())

    def test_number(self):
        self.assertEqual('0', Property(0).getquoted())
        self.assertEqual('-1', Property(-1).getquoted())
        self.assertEqual('3.14159', Property(3.14159).getquoted())

    def test_boolean(self):
        self.assertEqual('true', Property(True).getquoted())
        self.assertEqual('false', Property(False).getquoted())

    def test_null(self):
        self.assertEqual('null', Property(None).getquoted())

    def test_array(self):
        a = ["'\\\"'", 3.14159, True, None, (), {}]
        e = "['''\\\\\\\"''',3.14159,true,null,[],{}]"
        self.assertEqual(e, Property(a).getquoted())

    def test_object(self):
        self.assertEqual("{'\\\"':'\\\"'}", Property({'"': '"'}).getquoted())
        self.assertEqual("{'3.14159':3.14159}",
                         Property({3.14159: 3.14159}).getquoted())
        self.assertEqual("{'true':false}", Property({True: False}).getquoted())
        self.assertEqual("{'null':null}", Property({None: None}).getquoted())
        self.assertEqual("{'a':[]}", Property({'a': []}).getquoted())
        self.assertEqual("{'o':{}}", Property({'o': {}}).getquoted())
        self.assertRaises(TypeError, Property({(): None}).getquoted)
