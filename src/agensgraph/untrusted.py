"""Reading a statement nobody here wrote, before it is sent.

A model's Cypher is run by :meth:`~agensgraph.Connection.run_untrusted`, which holds it to a
role, a read-only transaction that always rolls back, a time limit and a row cap. What this
module adds is the reading that has to happen before any of that: things the server would run
that none of those stop.

- **Leaving the wrapper.** The fast path places the text where a subquery goes,
  ``SELECT * FROM (<text>) AS t``. Text that closes that bracket can append SQL of its own and
  read whatever the role can. So brackets must balance and never go below zero, and the words
  that only SQL uses there are refused.
- **Calling a function.** Cypher calls SQL functions by name, and several have effects a
  rollback does not undo: ``pg_advisory_lock`` survives it, ``pg_sleep`` holds a connection,
  ``set_config`` and ``pg_terminate_backend`` reach past the statement. Only Cypher's own
  functions are allowed; a quoted or commented name is still a name.
- **Size.** One statement with a thousand patterns grew a backend to 31 GB and took the server
  down. A time limit does not bound memory, so the text, its patterns and its walks are bounded.

The tokenizer follows the server's quoting rules -- ``''`` and ``\\'`` and ``""`` escapes,
``E'…'`` strings and dollar quotes -- so a quote cannot be used to hide a bracket or a name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "ALLOWED_FUNCTIONS",
    "MAX_HOPS",
    "MAX_PATTERN_PARTS",
    "MAX_TEXT_BYTES",
    "UNTRUSTED_ROLE_QUERY",
    "CheckedText",
    "UntrustedResult",
    "check_columns",
    "check_untrusted",
    "role_problems",
]

UNTRUSTED_ROLE_QUERY = """
select r.rolsuper,
       pg_has_role(current_user, 'pg_execute_server_program', 'member'),
       pg_has_role(current_user, 'pg_read_server_files', 'member'),
       pg_has_role(current_user, 'pg_write_server_files', 'member'),
       pg_has_role(current_user, 'pg_signal_backend', 'member'),
       case when %(app)s::text is null then false
            when not exists (select 1 from pg_catalog.pg_roles a where a.rolname = %(app)s) then false
            else current_user = %(app)s or pg_has_role(current_user, %(app)s, 'member') end,
       exists (
           select 1 from pg_catalog.pg_class c
           join pg_catalog.pg_namespace n on n.oid = c.relnamespace
           where n.nspname = any(%(schemas)s::text[]) and c.relkind in ('r', 'p')
             and (has_table_privilege(c.oid, 'insert') or has_table_privilege(c.oid, 'update')
                  or has_table_privilege(c.oid, 'delete') or has_table_privilege(c.oid, 'truncate')))
from pg_catalog.pg_roles r where r.rolname = current_user
"""
"""What the connected role could do that model-written Cypher must not be able to.

Being a member of a role is enough to use it -- a role can ``SET ROLE`` to any role it belongs to,
inside a read-only transaction -- so membership is what is asked about, not what is held now.
"""

_ROLE_PROBLEMS = (
    "it is a superuser",
    "it can run programs on the server's host (pg_execute_server_program)",
    "it can read files on the server (pg_read_server_files)",
    "it can write files on the server (pg_write_server_files)",
    "it can cancel and end other sessions of its role (pg_signal_backend)",
    "it is the application's role or a member of it",
    "it may change tables in the schemas it was asked about",
)


def role_problems(row: tuple[bool, ...], *, writes_enforced: bool) -> tuple[str, ...]:
    """What makes a role unfit to run model-written Cypher, from :data:`UNTRUSTED_ROLE_QUERY`.

    Write privileges count only where the server lets Cypher writes skip them, 2.14 and 2.15.
    """
    found = [problem for problem, held in zip(_ROLE_PROBLEMS, row, strict=True) if held]
    if writes_enforced and _ROLE_PROBLEMS[-1] in found:
        found.remove(_ROLE_PROBLEMS[-1])
    return tuple(found)


@dataclass(frozen=True)
class UntrustedResult:
    """What a model-written statement returned.

    ``truncated`` is true when the statement had more rows than were asked for; ``rows`` then
    holds the first of them.
    """

    columns: tuple[str, ...]
    rows: list[tuple[object, ...]]
    truncated: bool

    @classmethod
    def capped(
        cls, columns: tuple[str, ...], rows: Sequence[tuple[object, ...]], max_rows: int
    ) -> UntrustedResult:
        """The result from rows read one past the cap, which is how a truncation is seen."""
        return cls(columns, list(rows[:max_rows]), len(rows) > max_rows)


MAX_TEXT_BYTES = 8192
MAX_PATTERN_PARTS = 32
MAX_HOPS = 5

ALLOWED_FUNCTIONS = frozenset(
    {
        # Aggregates.
        "count", "sum", "avg", "min", "max", "collect", "stdev", "stdevp",
        # Elements, paths and lists.
        "id", "label", "labels", "type", "properties", "keys", "startnode", "endnode", "start",
        "end", "length", "size", "nodes", "relationships", "head", "last", "tail", "range",
        "shortestpath", "allshortestpaths",
        # Predicates and conditionals.
        "coalesce", "exists", "all", "any", "none", "single",
        # Numbers.
        "abs", "ceil", "floor", "round", "sign", "sqrt", "exp", "log", "log10", "pi", "sin",
        "cos", "tan", "asin", "acos", "atan", "atan2", "degrees", "radians",
        # Conversions and strings.
        "tostring", "tointeger", "tofloat", "toboolean", "toupper", "tolower", "upper", "lower",
        "trim", "ltrim", "rtrim", "substring", "replace", "reverse", "left", "right", "split",
    }
)  # fmt: skip
"""Cypher's own functions. Every other name followed by ``(`` is refused."""

# The clauses a read is made of, and so the words it may start with.
_CLAUSES = frozenset({"match", "optional", "with", "unwind", "return"})
_FIRST_WRITE_KEYWORDS = frozenset({"create", "merge"})
_LIMITS = frozenset({"order", "skip", "limit"})

# Words a read never needs, and words only SQL uses where the text sits inside the wrapper.
_REFUSED_WORDS = frozenset(
    {
        "explain", "select", "load", "call", "copy", "commit", "rollback", "table", "values",
        "from", "join", "lateral", "insert", "union", "into",
    }
)  # fmt: skip

# Words that open a clause and may be followed by a bracket without being a function call.
_KEYWORDS = frozenset(
    {
        "match", "optional", "where", "and", "or", "not", "xor", "in", "return", "with",
        "unwind", "as", "distinct", "case", "when", "then", "else", "is", "null", "starts",
        "ends", "contains", "order", "by", "skip", "limit", "asc", "desc", "ascending",
        "descending", "true", "false", "create", "merge", "set", "delete", "detach", "remove",
        "on",
    }
)  # fmt: skip

_WRITE_WORDS = frozenset({"create", "merge", "set", "delete", "detach", "remove"})

# Letters of any script, as the server's scanner takes them: a label in Korean is a name.
_IDENTIFIER = re.compile(r"[^\W\d]\w*")
_DOLLAR_QUOTE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")
_HOPS = re.compile(r"\*\s*(\d+)?\s*(\.\.\s*(\d+)?)?")


@dataclass(frozen=True)
class CheckedText:
    """What the reading found, for the executor to act on.

    ``text`` is the statement with one trailing semicolon removed. ``wrappable`` is false for a
    statement the cursor wrap's grammar refuses on every server -- one starting with ``WITH`` --
    so it goes straight to the slower path rather than spending a round trip to be told so.
    ``limits_mid_statement`` marks an ``ORDER BY``, ``SKIP`` or ``LIMIT`` that is not in the last
    clause, which 2.18's wrap refuses. ``columns`` holds the alias of each item of the final
    ``RETURN``, ``None`` for an item without one, or is ``None`` when the statement returns ``*``.
    """

    text: str
    writes: bool
    wrappable: bool
    limits_mid_statement: bool
    columns: tuple[str | None, ...] | None
    words: frozenset[str]


class _Token(NamedTuple):
    kind: str  # "word", "quoted", "punct", "number"
    value: str
    pos: int


def _with_depth(tokens: Sequence[_Token]) -> Iterator[tuple[int, int, _Token]]:
    """Each token with its index and the bracket depth it stands at.

    An opening bracket counts at the depth outside it, and a closing one at the depth inside the
    brackets it ends -- so a word at depth 0 is at the top level of the statement.
    """
    depth = 0
    for at, token in enumerate(tokens):
        if token.kind == "punct" and token.value in ")]}":
            depth -= 1
        yield at, depth, token
        if token.kind == "punct" and token.value in "([{":
            depth += 1


class _Refused(ValueError):
    pass


def _tokens(text: str, *, escaping: bool) -> list[_Token]:
    """Split the text the way the server would, refusing what a model has no reason to send.

    ``escaping`` reads a backslash in a plain string as escaping the quote after it, as a server
    with ``standard_conforming_strings`` off does. The two readings disagree about where such a
    string ends, so the caller reads the text both ways.
    """
    out: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif text.startswith("--", i) or text.startswith("/*", i):
            raise _Refused("a comment, which can hide the rest of a statement")
        elif ch == "$":
            if _DOLLAR_QUOTE.match(text, i):
                raise _Refused("a dollar-quoted string")
            raise _Refused("a $ outside a string")
        elif ch == "'" or (ch in "eE" and text.startswith("'", i + 1)):
            backslash_escapes = escaping or ch in "eE"
            i += 2 if ch in "eE" else 1
            while True:
                if i >= n:
                    raise _Refused("a string that is never closed")
                if text[i] == "\\" and backslash_escapes:
                    i += 2
                    continue
                if text[i] == "'":
                    if text.startswith("'", i + 1):
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(_Token("string", "", i))
        elif ch == '"':
            start = i
            i += 1
            name: list[str] = []
            while True:
                if i >= n:
                    raise _Refused("a quoted name that is never closed")
                if text[i] == '"':
                    if text.startswith('"', i + 1):
                        name.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                name.append(text[i])
                i += 1
            out.append(_Token("quoted", "".join(name), start))
        elif found := _IDENTIFIER.match(text, i):
            out.append(_Token("word", found.group(0).lower(), i))
            i = found.end()
        elif ch.isdigit():
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            out.append(_Token("number", text[i:j], i))
            i = j
        else:
            out.append(_Token("punct", ch, i))
            i += 1
    return out


def _bare_words(tokens: list[_Token]) -> set[str]:
    """The words used as words rather than as names.

    A word after a dot or a colon is a property or a label, one before a colon is a map key, and
    one after ``AS`` is an alias -- ``n.from``, ``:Table``, ``{set: 1}`` and ``AS values`` are all
    names. A quoted name is never a word.
    """
    bare = set()
    for at, token in enumerate(tokens):
        if token.kind != "word":
            continue
        previous = tokens[at - 1] if at else None
        following = tokens[at + 1] if at + 1 < len(tokens) else None
        if previous is not None and (
            (previous.kind == "punct" and previous.value in ".:")
            or (previous.kind == "word" and previous.value == "as")
        ):
            continue
        if following is not None and following.kind == "punct" and following.value == ":":
            continue
        bare.add(token.value)
    return bare


def _check_brackets(tokens: list[_Token]) -> None:
    depth = 0
    for token in tokens:
        if token.kind != "punct":
            continue
        if token.value in "([{":
            depth += 1
        elif token.value in ")]}":
            depth -= 1
            if depth < 0:
                raise _Refused("a bracket closed that the statement never opened")
    if depth != 0:
        raise _Refused("a bracket that is never closed")


def _check_functions(tokens: list[_Token]) -> None:
    """Refuse every call to a function that is not one of Cypher's own.

    A name followed by ``(`` is a call wherever it stands: a label or a property is never followed
    by one, so a name after ``:`` -- a map's value -- is checked like any other. A name after ``.``
    is a function in a named schema, which no Cypher function needs.
    """
    for at, token in enumerate(tokens[:-1]):
        following = tokens[at + 1]
        if following.kind != "punct" or following.value != "(":
            continue
        if token.kind not in ("word", "quoted"):
            continue
        previous = tokens[at - 1] if at else None
        if previous is not None and previous.kind == "punct" and previous.value == ".":
            raise _Refused(f"a call to {token.value!r} in a named schema")
        name = token.value.lower()
        if token.kind == "word" and name in _KEYWORDS:
            continue
        if name not in ALLOWED_FUNCTIONS:
            raise _Refused(
                f"a call to {token.value!r}, which is not one of Cypher's own functions"
            )


def _check_walks(text: str, tokens: list[_Token]) -> None:
    """Refuse too many patterns, and a walk with no upper bound or too many steps.

    Read forward by index and refused as soon as a limit is passed, since the text is a model's
    and the cost of reading it should not grow faster than it does.
    """
    parts = 0
    for at, token in enumerate(tokens):
        if token.kind != "punct":
            continue
        if token.value == "(" and at:
            previous = tokens[at - 1]
            opens_a_node = (previous.kind == "punct" and previous.value in "-<>,=") or (
                previous.kind == "word" and previous.value in ("match", "merge", "create")
            )
            if opens_a_node:
                parts += 1
                if parts > MAX_PATTERN_PARTS:
                    raise _Refused(f"more than {MAX_PATTERN_PARTS} patterns")
        elif token.value == "[" and at and tokens[at - 1].value in ("-", "<"):
            close = next(
                (
                    tokens[later].pos
                    for later in range(at + 1, len(tokens))
                    if tokens[later].kind == "punct" and tokens[later].value == "]"
                ),
                None,
            )
            detail = text[token.pos + 1 : close] if close is not None else ""
            star = detail.find("*")
            if star < 0:
                continue
            found = _HOPS.match(detail, star)
            assert found is not None
            low, dotted, high = found.group(1), found.group(2), found.group(3)
            bound = high if dotted else low
            if bound is None:
                raise _Refused("a walk with no upper bound")
            if int(bound) > MAX_HOPS:
                raise _Refused(f"a walk of up to {bound} steps; at most {MAX_HOPS} are allowed")


def _return_columns(tokens: list[_Token]) -> tuple[str | None, ...] | None:
    """The aliases of the final RETURN's items, read at the top level of the statement."""
    last_return = None
    for at, depth, token in _with_depth(tokens):
        if depth == 0 and token.kind == "word" and token.value == "return":
            last_return = at
    if last_return is None:
        return ()
    items: list[str | None] = []
    current: list[_Token] = []
    for _, depth, token in _with_depth(tokens[last_return + 1 :]):
        if depth == 0 and token.kind == "word" and token.value in _LIMITS:
            break
        if depth == 0 and token.kind == "punct" and token.value == ",":
            items.append(_alias(current))
            current = []
            continue
        current.append(token)
    if current:
        items.append(_alias(current))
    if len(items) == 1 and _is_star(tokens, last_return):
        return None
    return tuple(items)


def _limits_mid_statement(tokens: list[_Token]) -> bool:
    """Whether an ORDER BY, SKIP or LIMIT at the top level is followed by another clause."""
    limited = False
    for _, depth, token in _with_depth(tokens):
        if depth == 0 and token.kind == "word":
            if token.value in _LIMITS:
                limited = True
            elif token.value in _CLAUSES and limited:
                return True
    return False


def _is_star(tokens: list[_Token], at: int) -> bool:
    """Whether the RETURN at *at* returns ``*``, a ``DISTINCT`` before it aside."""
    for later in range(at + 1, len(tokens)):
        token = tokens[later]
        if not (token.kind == "word" and token.value == "distinct"):
            return token.kind == "punct" and token.value == "*"
    return False


def _alias(item: list[_Token]) -> str | None:
    if len(item) >= 2 and item[-2].kind == "word" and item[-2].value == "as":
        name = item[-1]
        return name.value if name.kind in ("word", "quoted") else None
    return None


def check_columns(expected: tuple[str | None, ...] | None, names: tuple[str, ...]) -> None:
    """Refuse a result whose columns are not the ones the statement's ``RETURN`` asked for.

    A statement that left the cursor wrapper would bring columns of its own. The reading above
    should make that impossible; this is the check that does not rely on it.
    """
    if expected is None:
        return
    if len(names) != len(expected):
        raise ValueError(
            f"the result has {len(names)} columns and the statement returns {len(expected)}"
        )
    for alias, name in zip(expected, names, strict=True):
        if alias is not None and alias.lower() != name.lower():
            raise ValueError(
                f"the result has a column {name!r} where the statement returns {alias!r}"
            )


def check_untrusted(text: str, *, allow_writes: bool = False) -> CheckedText:
    """Refuse model-written text that is not plainly what it should be, before it is sent.

    Raises :class:`ValueError` naming the reason. ``allow_writes`` admits Cypher's write
    clauses; everything else is refused either way.
    """
    try:
        return _check(text, allow_writes=allow_writes)
    except _Refused as reason:
        raise ValueError(
            f"model-written Cypher refused before it was sent: it holds {reason}"
        ) from None


def _check(text: str, *, allow_writes: bool) -> CheckedText:
    if len(text.encode()) > MAX_TEXT_BYTES:
        raise _Refused(f"more than {MAX_TEXT_BYTES} bytes")
    body = text.strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    # Read the way every current server does, and where a backslash could change where a string
    # ends, the other way as well: either reading refusing is enough.
    if "\\" in body:
        _check_tokens(body, _tokens(body, escaping=True), allow_writes=allow_writes)
    tokens = _tokens(body, escaping=False)
    return _check_tokens(body, tokens, allow_writes=allow_writes)


def _check_tokens(body: str, tokens: list[_Token], *, allow_writes: bool) -> CheckedText:
    if not tokens:
        raise _Refused("nothing to run")
    if any(t.kind == "punct" and t.value == ";" for t in tokens):
        raise _Refused("more than one statement")
    first = tokens[0]
    starts = _CLAUSES | _FIRST_WRITE_KEYWORDS if allow_writes else _CLAUSES
    if first.kind != "word" or first.value not in starts:
        raise _Refused(
            f"a first word of {first.value!r}; a read starts with MATCH, OPTIONAL, "
            f"WITH, UNWIND or RETURN"
            + (", and a write may also start with CREATE or MERGE" if allow_writes else "")
        )
    words = _bare_words(tokens)
    refused = sorted(words & _REFUSED_WORDS)
    if refused:
        raise _Refused(f"{refused[0].upper()}, which a Cypher read has no use for")
    writes = bool(words & _WRITE_WORDS)
    if writes and not allow_writes:
        clause = sorted(words & _WRITE_WORDS)[0].upper()
        raise _Refused(f"{clause}, a write, and this statement may only read")
    _check_brackets(tokens)
    _check_functions(tokens)
    _check_walks(body, tokens)
    return CheckedText(
        text=body,
        writes=writes,
        wrappable=first.value != "with" and not writes,
        limits_mid_statement=_limits_mid_statement(tokens),
        columns=_return_columns(tokens),
        words=frozenset(words),
    )
