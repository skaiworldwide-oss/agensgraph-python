"""Renderings the server actually produces, and the ones that break a naive reader.

The first group is copied from the engine's own regression output and from the earlier
driver's tests. The second group is everything those two sources miss: values the
server emits happily and that a reader built around delimiter search gets wrong.
"""

from __future__ import annotations

VERTICES: list[tuple[bytes, str, int, int, dict]] = [
    (b"ag_vertex[1.1]{}", "ag_vertex", 1, 1, {}),
    (b'v[5.2]{"id": 1}', "v", 5, 2, {"id": 1}),
    (
        b'repo[3.1]{"name": "agens-graph", "year": 2016}',
        "repo",
        3,
        1,
        {"name": "agens-graph", "year": 2016},
    ),
    (
        b'v[7.9]{"s": "", "i": 0, "b": false, "a": [], "o": {}}',
        "v",
        7,
        9,
        {"s": "", "i": 0, "b": False, "a": [], "o": {}},
    ),
    # A label may hold any of the characters the format itself uses, because the server
    # writes label names with no escaping at all.
    (b"a,b[7.9]{}", "a,b", 7, 9, {}),
    (b"a}b[7.9]{}", "a}b", 7, 9, {}),
    (b"a{b[7.9]{}", "a{b", 7, 9, {}),
    (b"a]b[7.9]{}", "a]b", 7, 9, {}),
    (b'a"b[7.9]{}', 'a"b', 7, 9, {}),
    (b"my label[7.9]{}", "my label", 7, 9, {}),
    (b'\xec\x82\xac\xeb\x9e\x8c[7.9]{"n": 1}', "사람", 7, 9, {"n": 1}),
    # The documented maxima, which read as a negative number if the packed value is
    # treated as signed.
    (b"v[65535.281474976710655]{}", "v", 65535, 281474976710655, {}),
    (b"v[32768.1]{}", "v", 32768, 1, {}),
    # Structural characters inside property values.
    (b'v[5.1]{"k": "a,b"}', "v", 5, 1, {"k": "a,b"}),
    (b'v[5.1]{"a": "],["}', "v", 5, 1, {"a": "],["}),
    (b'v[5.1]{"a": "}{"}', "v", 5, 1, {"a": "}{"}),
    (b'v[5.1]{"a": {"b": "},{"}}', "v", 5, 1, {"a": {"b": "},{"}}),
    (b'v[5.1]{"a": [1,2]}', "v", 5, 1, {"a": [1, 2]}),
    (b'n[7.3]{"s": "[}\\""}', "n", 7, 3, {"s": '[}"'}),
    (b'v[7.9]{"a": "][1.1,2.2]"}', "v", 7, 9, {"a": "][1.1,2.2]"}),
]

EDGES: list[tuple[bytes, str, tuple[int, int], tuple[int, int], tuple[int, int], dict]] = [
    (b"e1[3.1][1.1,1.2]{}", "e1", (3, 1), (1, 1), (1, 2), {}),
    (b'lib[4.1][3.1,3.2]{"lang": "java"}', "lib", (4, 1), (3, 1), (3, 2), {"lang": "java"}),
    (b'e[6.8][5.2,5.3]{"weight": 4}', "e", (6, 8), (5, 2), (5, 3), {"weight": 4}),
    (
        b'directed[22.1][23.1,23.2]{"weight": 1.0, "keywords": "develop, produce"}',
        "directed",
        (22, 1),
        (23, 1),
        (23, 2),
        {"weight": 1.0, "keywords": "develop, produce"},
    ),
    (b"r[5.7][7.3,7.9]{}", "r", (5, 7), (7, 3), (7, 9), {}),
    (b"a,b[5.7][7.3,7.9]{}", "a,b", (5, 7), (7, 3), (7, 9), {}),
]

PATHS: list[tuple[bytes, int, int]] = [
    (b"[]", 0, 0),
    (b"[n[7.3]{}]", 1, 0),
    (b"[n[7.3]{},r[5.7][7.3,7.9]{},n[7.9]{}]", 2, 1),
    (
        b'[repo[3.1]{"name": "a"},lib[4.1][3.1,3.2]{"lang": "java"},repo[3.2]{"name": "b"}]',
        2,
        1,
    ),
    (b'[v[5.1]{"k": "a,b"},e[6.1][5.1,5.5]{},v[5.5]{}]', 2, 1),
    (b'[v[5.1]{"a": "},{"},e[6.1][5.1,5.5]{},v[5.5]{}]', 2, 1),
    (b"[a{b[7.3]{},r[5.7][7.3,7.9]{},a{b[7.9]{}]", 2, 1),
    (b"[a,b[7.3]{},r[5.7][7.3,7.9]{},a,b[7.9]{}]", 2, 1),
    (
        b'[v[5.1]{"id": 0},e[6.1][5.1,5.2]{},v[5.2]{"id": 1},e[6.2][5.2,5.3]{},v[5.3]{"id": 2}]',
        3,
        2,
    ),
]

ELEMENT_ARRAYS: list[tuple[bytes, int]] = [
    (b"[]", 0),
    (b"[NULL]", 1),
    (b"[NULL,NULL,NULL]", 3),
    (b"[v[5.1]{},NULL]", 2),
    (b'[v[5.1]{"id": 0},v[5.5]{"id": 4}]', 2),
    (b"[r[5.7][7.3,7.9]{},r[5.8][7.9,7.3]{}]", 2),
    (b'[a[3.1]{"v": "[5.5]{},B[7.7]{},C[9.9]{}"},a,b[4.1]{}]', 2),
    (b'[a[3.1]{"v": "[5.5]{},B[7.7]{"},a,b[4.1]{}]', 2),
    (b'[v[5.1]{"v": "x[1.1]{}"},NULL,v[5.2]{}]', 3),
]

# Text that must be rejected rather than quietly turned into something plausible.
REJECTED_GRAPHIDS: list[bytes] = [
    b"7.9.5",
    b"7.9x",
    b"x7.9",
    b"7.",
    b".9",
    b"79",
    b"",
    b"7 . 9",
    b"-7.9",
    b"+7.9",
    b"0x7.0x9",
    "٧.٩".encode(),  # Arabic-Indic digits, which int() would otherwise accept
    b"65536.1",
    b"1.281474976710656",
]

REJECTED_VERTICES: list[bytes] = [
    b"v[7.9]",
    b"v[7.9]{}extra",
    b"v[7.9]null",
    b"v[7.9]0",
    b"[7.9]{}",
    b"v{7.9}{}",
    b"",
    b"NULL",
]
