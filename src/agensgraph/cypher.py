"""Handling the text of a statement, in the two places it cannot be avoided.

A statement is never rewritten. It is sent as it was written, and its results are read as
the server chose to send them. Three things still have to be done to the text, and all three
are here so that none of them is done anywhere else.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "WRAP",
    "changes_graph_path",
    "check_bindable_positions",
    "check_can_wrap",
    "quote_identifier",
    "quote_string",
    "without_literals",
    "wrap_for_cursor",
]

# What clears the graph path without naming it. Measured against a server, along with the
# statements that do name it: `SET ROLE`, `SET SESSION AUTHORIZATION`, `DISCARD PLANS` and
# `DISCARD SEQUENCES` leave it alone.
_CLEARS_EVERYTHING = re.compile(r"\b(?:reset|discard)\s+all\b")

# A parameter standing where a walk length belongs: directly after the star, after the
# range, or after a lower bound and the range. All three prepare without complaint.
#
# Both spellings are looked for. A statement written for this driver marks its parameters
# the way psycopg does, with %s or %(name)s, and those are turned into $1 and $2 on the way
# out -- so a check that knew only the server's spelling would never fire on anything anyone
# actually wrote.
#
# The star has to stand where the grammar puts a walk length, which is inside the brackets of
# a relationship pattern: a dash, then a bracket holding at most a variable and a label. A
# star anywhere else is multiplication or a count, and `return 2 * $1` is a statement the
# server runs and answers.
_PLACEHOLDER = r"(?:\$\d+|%(?:\([^)]*\))?[sbt])"
_LENGTH_PARAMETER = re.compile(
    r"-\s*\[\s*(?:[A-Za-z_]\w*)?\s*(?::[\w\s]*)?\*\s*(?:\d+\s*)?(?:\.\.\s*)?" + _PLACEHOLDER
)

# The first character of anything the lexer does not read as syntax. A class of single
# characters is scanned in one pass, so `--` and `/*` are found by their first character and
# the second is read where the scan stops.
_LITERAL_START = re.compile(r"['\"$/-]")
_NOT_NEWLINE = re.compile(r"[^\n]")

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

    One scan finds where a run can begin, and what is kept is joined from slices, so the cost
    follows the number of literals a statement holds.
    """
    out: list[str] = []
    append = out.append
    length = len(statement)
    pos = 0
    for found in _LITERAL_START.finditer(statement):
        start = found.start()
        if start < pos:
            continue  # Inside a run already taken.
        ch = statement[start]
        if ch in "'\"":
            end = _end_of_quoted(statement, start, ch)
        elif ch == "$":
            tag_end = _dollar_tag_end(statement, start)
            if tag_end < 0:
                continue  # A parameter, which is syntax and stays.
            tag = statement[start:tag_end]
            close = statement.find(tag, tag_end)
            end = length if close < 0 else close + len(tag)
        elif ch == "-":
            if not statement.startswith("--", start):
                continue  # A minus, or the arrow of a relationship pattern.
            end = statement.find("\n", start)
            end = length if end < 0 else end
        else:
            if not statement.startswith("/*", start):
                continue  # A division.
            end = _end_of_block_comment(statement, start)
        append(statement[pos:start])
        run = statement[start:end]
        # A newline is kept, so that a line comment does not swallow the lines after it.
        append(" " * len(run) if "\n" not in run else _NOT_NEWLINE.sub(" ", run))
        pos = end
    if not pos:
        return statement
    append(statement[pos:])
    return "".join(out)


def _end_of_quoted(statement: str, start: int, quote: str) -> int:
    """The offset just past a quoted run, where a doubled quote does not end it."""
    pos = start + 1
    length = len(statement)
    while pos < length:
        found = statement.find(quote, pos)
        if found < 0:
            return length
        if statement.startswith(quote * 2, found):
            pos = found + 2
            continue
        return found + 1
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


def changes_graph_path(statement: str) -> bool:
    """Whether this statement may leave the session reading a different graph.

    ``SET``, ``RESET`` and ``set_config`` name the setting; ``RESET ALL`` and ``DISCARD ALL``
    clear it without naming it. Read from the statement as written, so a mention inside a
    string counts. Saying yes to a statement that changes nothing costs a reload of the label
    table; saying no to one that does costs a wrong label on every binary read after it.

    Every statement is read, so what it costs matters. Lowering once and then asking whether
    two literals appear is what a substring search does in one pass, and the pattern is
    reached only by a statement holding the word it needs.
    """
    lowered = statement.lower()
    if "graph_path" in lowered:
        return True
    return "all" in lowered and _CLEARS_EVERYTHING.search(lowered) is not None


def check_bindable_positions(statement: str) -> None:
    """Refuse a statement whose parameter would be read as something else.

    Only the length of a variable-length relationship needs this. A parameter there is read
    as a property map: the statement prepares, reports its parameter as jsonb, and matches
    a walk of any length, so nothing later in the round trip reveals that the length was
    never applied. Every other position the grammar will not take a parameter in reports a
    syntax error of its own, and is left to the server.

    A statement holding no star cannot name a walk length, and blanking a literal only ever
    takes a star away, so a statement without one is let through on the first read of it.
    """
    if "*" not in statement:
        return
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

# Which of the five write counters a clause can move, read from `initGraphWRStats`: a pattern
# zeroes the two insert counters, a delete expression the two delete counters, and a set list
# the property counter. `MERGE` carries a pattern and a set list, and `INSERT` is a synonym of
# `CREATE`.
WRITE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "create": (0, 1),
    "insert": (0, 1),
    "merge": (0, 1, 4),
    "delete": (2, 3),
    "detach": (2, 3),
    "set": (4,),
    "remove": (4,),
}

_WRITE_WORD = re.compile(
    r'(?<![a-z0-9_.":])(create|insert|merge|set|delete|remove|detach)(?![a-z0-9_]|\s*:)'
)
_WRITE_WORDS = tuple(WRITE_GROUPS)

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


def writable_counters(statement: str) -> frozenset[int]:
    """Which write counters this statement's clauses could move.

    The server zeroes a counter only for a clause that can write it, so a counter no clause
    here names was neither zeroed nor written and its answer for this statement is nought.
    A statement naming no write clause at all can move none of them.

    Read from the statement as written, with strings and comments blanked out, and the words
    are the ones a clause is spelled with. A statement that is not Cypher and happens to hold
    one of them is read as though it could write, which costs an unanswered counter rather
    than a wrong one.

    A statement holding none of the words is answered by substring tests over the text as
    written, which is what an ordinary read is. Blanking a literal writes spaces, so it can
    only take a word away and never add one, and a statement without one has nothing here to
    find whatever its strings hold.
    """
    if not any(word in statement.lower() for word in _WRITE_WORDS):
        return frozenset()
    lowered = without_literals(statement).lower()
    groups: set[int] = set()
    for found in _WRITE_WORD.finditer(lowered):
        groups.update(WRITE_GROUPS[found.group(1)])
    return frozenset(groups)


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
