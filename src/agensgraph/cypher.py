"""Handling the text of a statement, in the two places it cannot be avoided.

A statement is never rewritten. It is sent as it was written, and its results are read as
the server chose to send them. Two things still have to be done to the text, and both are
here so that neither is done anywhere else.

A label or a property key cannot be bound as a parameter -- the grammar has no place for
one there -- so a statement that names either dynamically has to carry it in its text. That
is the one place where quoting is the driver's job, and getting it wrong is an injection.

And three shapes are accepted by the server and read as something other than what they say.
A parameter where the length of a variable-length relationship belongs is read as a property
map instead, so the statement prepares without complaint, reports a parameter of the wrong
type, and matches walks of any length. Every other position that cannot take a parameter
fails with a syntax error, which needs nothing from a driver; these three do not, so they
are refused here before the statement is sent.
"""

from __future__ import annotations

import re

__all__ = [
    "WRAP",
    "check_bindable_positions",
    "check_can_wrap",
    "quote_identifier",
    "quote_string",
    "without_literals",
    "wrap_for_cursor",
]

# A parameter standing where a walk length belongs: directly after the star, after the
# range, or after a lower bound and the range. All three prepare without complaint.
#
# Both spellings are looked for. A statement written for this driver marks its parameters
# the way psycopg does, with %s or %(name)s, and those are turned into $1 and $2 on the way
# out -- so a check that knew only the server's spelling would never fire on anything anyone
# actually wrote.
_PLACEHOLDER = r"(?:\$\d+|%(?:\([^)]*\))?[sbt])"
_LENGTH_PARAMETER = re.compile(r"\*\s*(?:\d+\s*)?(?:\.\.\s*)?" + _PLACEHOLDER)

# Lower case only: the lexer lowers an unquoted name, so anything holding a capital has to
# be quoted to reach the server as it was written.
_UNQUOTED_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*\Z")

_RESERVED = frozenset(
    {
        "all",
        "and",
        "any",
        "as",
        "asc",
        "ascending",
        "by",
        "case",
        "create",
        "delete",
        "desc",
        "descending",
        "detach",
        "distinct",
        "else",
        "end",
        "ends",
        "exists",
        "false",
        "filter",
        "for",
        "in",
        "is",
        "limit",
        "load",
        "match",
        "merge",
        "not",
        "null",
        "on",
        "optional",
        "or",
        "order",
        "remove",
        "return",
        "set",
        "skip",
        "starts",
        "then",
        "true",
        "union",
        "unwind",
        "when",
        "where",
        "with",
        "xor",
    }
)


def quote_identifier(name: str) -> str:
    """Quote a label or a property key for placing into a statement.

    A name already in lower case and holding nothing but letters, digits and underscores is
    left bare. Everything else is quoted, and a quote inside the name is doubled, which is
    how the server's own lexer reads one back.

    An unquoted name is lowered by the lexer, so ``MixedKey`` left bare reaches the server
    as ``mixedkey``: it would name a different property from the one written, and two keys
    differing only in case would become one.

    A name holding a null byte is refused rather than quoted. The server's lexer stops at
    one, so quoting it would produce a statement that ends somewhere other than where it
    appears to.
    """
    if "\x00" in name:
        raise ValueError("an identifier cannot hold a null byte")
    if _UNQUOTED_IDENTIFIER.match(name) and name.lower() not in _RESERVED:
        return name
    return '"' + name.replace('"', '""') + '"'


def quote_string(value: str) -> str:
    """Quote a string for placing into a statement, for the rare case that needs it.

    Almost nothing needs this: a value belongs in a parameter, where the server never reads
    it as syntax at all. It is here for the positions the grammar will not take a parameter
    in, and it refuses a null byte for the same reason as an identifier.
    """
    if "\x00" in value:
        raise ValueError("a string cannot hold a null byte")
    return "'" + value.replace("'", "''") + "'"


def without_literals(statement: str) -> str:
    """The statement with everything the lexer does not read as syntax blanked out.

    Strings, quoted identifiers, dollar-quoted bodies and comments are replaced by spaces
    of the same length, so that positions still line up and a scan of what is left cannot
    be fooled by something a person wrote inside a string.
    """
    out = list(statement)
    length = len(statement)
    pos = 0
    while pos < length:
        ch = statement[pos]
        if ch in "'\"":
            end = _end_of_quoted(statement, pos, ch)
            _blank(out, pos, end)
            pos = end
        elif ch == "$":
            tag_end = _dollar_tag_end(statement, pos)
            if tag_end < 0:
                pos += 1
                continue
            tag = statement[pos:tag_end]
            close = statement.find(tag, tag_end)
            end = length if close < 0 else close + len(tag)
            _blank(out, pos, end)
            pos = end
        elif statement.startswith("--", pos):
            end = statement.find("\n", pos)
            end = length if end < 0 else end
            _blank(out, pos, end)
            pos = end
        elif statement.startswith("/*", pos):
            end = _end_of_block_comment(statement, pos)
            _blank(out, pos, end)
            pos = end
        else:
            pos += 1
    return "".join(out)


def _blank(out: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if out[i] != "\n":
            out[i] = " "


def _end_of_quoted(statement: str, start: int, quote: str) -> int:
    """The offset just past a quoted run, where a doubled quote does not end it."""
    pos = start + 1
    length = len(statement)
    while pos < length:
        if statement[pos] != quote:
            pos += 1
        elif statement.startswith(quote * 2, pos):
            pos += 2
        else:
            return pos + 1
    return length


def _dollar_tag_end(statement: str, start: int) -> int:
    """The offset just past a dollar-quote tag, or -1 if this is not one.

    ``$$`` and ``$tag$`` open a body; ``$1`` is a parameter and is left alone.
    """
    pos = start + 1
    length = len(statement)
    while pos < length and (statement[pos].isalnum() or statement[pos] == "_"):
        if statement[pos].isdigit() and pos == start + 1:
            return -1
        pos += 1
    if pos < length and statement[pos] == "$":
        return pos + 1
    return -1


def _end_of_block_comment(statement: str, start: int) -> int:
    """The offset just past a block comment, which nests."""
    depth = 0
    pos = start
    length = len(statement)
    while pos < length:
        if statement.startswith("/*", pos):
            depth += 1
            pos += 2
        elif statement.startswith("*/", pos):
            depth -= 1
            pos += 2
            if depth == 0:
                return pos
        else:
            pos += 1
    return length


def check_bindable_positions(statement: str) -> None:
    """Refuse a statement whose parameter would be read as something else.

    Only the length of a variable-length relationship needs this. A parameter there is read
    as a property map: the statement prepares, reports its parameter as jsonb, and matches
    a walk of any length, so nothing later in the round trip reveals that the length was
    never applied. Every other position the grammar will not take a parameter in reports a
    syntax error of its own, and is left to the server.
    """
    found = _LENGTH_PARAMETER.search(without_literals(statement))
    if found is None:
        return
    raise ValueError(
        f"a parameter cannot give the length of a variable-length relationship: "
        f"{found.group(0)!r}. The server accepts this and reads the parameter as a "
        f"property map, matching a walk of any length. Write the length into the "
        f"statement instead."
    )


# A clause that changes something. A statement holding one cannot be read in chunks, because
# the wrap a server-side cursor needs takes only the read-only subset.
#
# Each of these words is also a legal property name, label name and map key. So the word is taken
# for a clause only where one could stand: not after a dot or a quote or a colon, which is a
# property read, a quoted name and a label; and not before a colon, which is a key in a map.
# Missing a write here costs nothing, since the server refuses one from the wrap anyway. Refusing
# a read that never wrote anything is the mistake worth avoiding.
_WRITE_CLAUSE = re.compile(
    r'(?<![A-Za-z0-9_.":])(create|merge|set|delete|remove|detach)(?![A-Za-z0-9_]|\s*:)',
    re.IGNORECASE,
)

WRAP = "select * from ({statement}) as {alias}"
"""How a Cypher statement is made readable by a server-side cursor.

``DECLARE ... CURSOR FOR MATCH`` is a syntax error -- the grammar has no arm for it -- so the
only way to read a Cypher result in chunks is to put it where a subquery goes. The alias is not
optional: without one the server reports that Cypher in a FROM needs an alias.
"""


def wrap_for_cursor(statement: str, *, alias: str = "t") -> str:
    """The statement as a server-side cursor can read it.

    What the wrap takes, from the grammar and confirmed against a server: a chain of ``MATCH``,
    ``WITH``, ``LET``, ``LOAD``, ``UNWIND``, ``FOR`` and ``CALL { }`` ending in ``RETURN``, with a
    trailing ``ORDER BY``, ``SKIP`` or ``LIMIT``; ``FINISH`` in place of ``RETURN``; and ``UNION``,
    ``INTERSECT`` or ``EXCEPT`` between parenthesised reads. The alias is required.

    What it refuses: any write, ``FILTER``, a ``LIMIT`` or ``ORDER BY`` that is not last, and
    ``CALL func() YIELD``, which the grammar allows only as a top-level clause.
    """
    check_can_wrap(statement)
    return WRAP.format(statement=statement.rstrip().rstrip(";"), alias=quote_identifier(alias))


def check_can_wrap(statement: str) -> None:
    """Refuse a statement that cannot be read in chunks, saying which part is the reason.

    Only a write is caught here. The subtler refusals -- a ``LIMIT`` that is not last, an
    ``ORDER BY`` that is not final -- the server reports clearly by itself, and repeating that
    judgement client-side would mean keeping a copy of the grammar in step with it.
    """
    found = _WRITE_CLAUSE.search(without_literals(statement))
    if found is None:
        return
    raise ValueError(
        f"a statement that writes cannot be read in chunks: it holds "
        f"{found.group(0).upper()}, and reading in chunks needs the statement placed where a "
        f"subquery goes, which takes only the read-only subset. Read it whole instead."
    )
