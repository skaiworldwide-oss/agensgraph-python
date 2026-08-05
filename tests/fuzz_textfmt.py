"""Fuzzing what reads the text rendering.

Run under Atheris, which is Linux x86_64 and cp312 upward:

    .venv/bin/python tests/fuzz_textfmt.py -atheris_runs=200000
    .venv/bin/python tests/fuzz_textfmt.py corpus/          # replay a saved corpus

A finding is any exception the readers are not documented to raise. ``ValueError`` is what they raise
for input that is not what it claims to be, and a ``UnicodeDecodeError`` is one of those, so both are
allowed through; anything else -- an index error, an attribute error, a recursion error -- means the
reader trusted something it should have checked.

The corpus below seeds the shapes already known to be hazardous, so a run starts from them rather than
having to discover them.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

with atheris.instrument_imports():
    from agensgraph._protocol.graphid import parse_text
    from agensgraph._protocol.textfmt import parse_edge, parse_vertex, split_elements

ALLOWED = (ValueError, UnicodeDecodeError)

SEEDS = [
    # An empty path, and a path of nothing but nulls.
    b"[]",
    b"[NULL]",
    b"[NULL,NULL,NULL]",
    # A label holding each character the rendering itself uses.
    b'a[1.1]{"k": 1}',
    b'a,b[1.1]{"k": 1}',
    b'a{b[1.1]{"k": 1}',
    b'a}b[1.1]{"k": 1}',
    b'a[b[1.1]{"k": 1}',
    b'a]b[1.1]{"k": 1}',
    b'a"b[1.1]{"k": 1}',
    # A property value holding what looks like a boundary between two elements.
    b'a[1.1]{"k": "},Company["}',
    b'a[1.1]{"k": "][" }',
    # Identities at their limits, and one with a third part.
    b"a[65535.281474976710655]{}",
    b"a[0.0]{}",
    b"a[7.9.5]{}",
    b"a[-1.1]{}",
    # An edge, which carries two more identities.
    b'e[2.1][1.1,1.2]{"k": 1}',
    b"e[2.1][1.1,1.2]{}",
    # An array of elements, which is where a boundary has to be measured rather than guessed.
    b'[a[1.1]{"k": 1},b[1.2]{"k": 2}]',
    b'[a[1.1]{"k": "],b["},b[1.2]{}]',
    # Truncations of each of those.
    b"a[1.1]{",
    b"a[1.1",
    b"a[",
    b"[a[1.1]{}",
    b"",
]


def one(data: bytes) -> None:
    """Hand the same bytes to each reader, and let only a stated failure through."""
    for read in (split_elements, parse_vertex, parse_edge, parse_text):
        with contextlib.suppress(*ALLOWED):
            read(data)  # type: ignore[arg-type]


def main() -> None:
    atheris.Setup(sys.argv, one)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
