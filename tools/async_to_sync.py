#!/usr/bin/env python3
"""Write the blocking interface from the awaiting one.

The two interfaces are the same code with the awaiting taken out, so one of them is written
and the other is produced. Keeping both by hand is what goes wrong: the two copies of one
module in redis-py have drifted until they differ on more lines than either one has, and its
awaiting test suite covers sixteen fewer files than its blocking one. Nothing about that was
a decision; it is what happens to two files that are supposed to say the same thing.

What this does is small on purpose. It removes ``await``, turns ``async def``, ``async with``
and ``async for`` into their blocking forms, and renames the handful of names that differ
between the two. Two escape valves cover what a rename cannot express: ``@only_async`` drops
a definition from the blocking file entirely, and ``if IS_ASYNC:`` keeps only the branch that
belongs, so a method whose two forms genuinely differ can say so in one place instead of
being written twice.

Run it with ``--check`` to fail when a generated file is not what the source says it should
be, which is what keeps an edit to the wrong file from surviving.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "agensgraph"

# Every file produced, and the file it is produced from. A new pair goes here and nowhere
# else, so a module that escapes conversion is a line missing from one visible list.
MODULES: dict[str, str] = {
    "connection_async.py": "connection.py",
}

# Names that differ between the two interfaces. The convention keeps this short: a class
# whose name starts with Async loses it, and so does a function whose name starts with
# async_, so most names need no entry at all.
RENAMES: dict[str, str] = {
    "AsyncConnection": "Connection",
    "AsyncCursor": "Cursor",
    "AsyncServerCursor": "ServerCursor",
    "AsyncClientCursor": "ClientCursor",
    "AsyncRowFactory": "RowFactory",
    "AsyncConnectionPool": "ConnectionPool",
    "AsyncNullConnectionPool": "NullConnectionPool",
    "AsyncTransaction": "Transaction",
    "AsyncCopy": "Copy",
    "AsyncPipeline": "Pipeline",
    "AsyncGenerator": "Iterator",
    "AsyncIterator": "Iterator",
    "AsyncIterable": "Iterable",
    "asynccontextmanager": "contextmanager",
    "aclose": "close",
    "anext": "next",
    "IS_ASYNC": "IS_ASYNC",
}

HEADER = """\
# This file is generated from {source} by tools/async_to_sync.py.
# Edit that file and run the tool; edits made here are lost on the next run.
"""

ONLY_ASYNC = "only_async"
ASYNC_FLAG = "IS_ASYNC"


class Blocking(ast.NodeTransformer):
    """Rewrite an awaiting module as the blocking one it corresponds to."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if _is_only_async(node):
            return None
        plain = ast.FunctionDef(
            name=_rename(node.name),
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
            type_params=list(node.type_params),
        )
        # Visited as one node afterwards rather than child by child, so that nothing --
        # an argument's annotation above all -- can be left out by being forgotten here.
        return ast.copy_location(self.generic_visit(plain), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if _is_only_async(node):
            return None
        node.name = _rename(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = _rename(node.name)
        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return self.visit(node.value)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        plain = ast.With(items=node.items, body=node.body, type_comment=node.type_comment)
        return ast.copy_location(self.generic_visit(plain), node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        plain = ast.For(
            target=node.target,
            iter=node.iter,
            body=node.body,
            orelse=node.orelse,
            type_comment=node.type_comment,
        )
        return ast.copy_location(self.generic_visit(plain), node)

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        """Keep one branch of ``if IS_ASYNC:``, which is the blocking one.

        A method whose two forms genuinely differ says so here rather than being written out
        twice, and the branch that does not apply is not carried into the generated file at
        all -- so nothing in it can be read as code that runs.
        """
        if isinstance(node.test, ast.Name) and node.test.id == ASYNC_FLAG:
            kept = [self.visit(child) for child in node.orelse]
            return kept or [ast.copy_location(ast.Pass(), node)]
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = _rename(node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = _rename(node.attr)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node.name = _rename(node.name)
        if node.asname:
            node.asname = _rename(node.asname)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Rename inside a docstring or a forward reference, which are both strings."""
        if isinstance(node.value, str):
            node.value = _rename_text(node.value)
        return node


def _is_only_async(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == ONLY_ASYNC)
        or (isinstance(d, ast.Attribute) and d.attr == ONLY_ASYNC)
        for d in node.decorator_list
    )


def _rename(name: str) -> str:
    """The blocking name for an awaiting one.

    The table is consulted first, then the two conventions: a leading ``Async`` and a leading
    ``async_`` are dropped. Keeping the conventions is what stops the table from growing into
    the hundred-odd hand-maintained entries that a driver without them ends up with.
    """
    if name in RENAMES:
        return RENAMES[name]
    if name.startswith("Async") and len(name) > 5 and name[5].isupper():
        return name[5:]
    if name.startswith("async_"):
        return name[6:]
    return name


def _rename_text(text: str) -> str:
    """Rename every known name inside a run of prose."""
    for source, target in RENAMES.items():
        if source != target:
            text = text.replace(source, target)
    return text


def convert(source: Path, target_name: str) -> str:
    """The blocking form of one module, formatted the way the rest of the tree is."""
    tree = ast.parse(source.read_text(), filename=str(source))
    converted = ast.fix_missing_locations(Blocking().visit(tree))
    body = ast.unparse(converted)
    text = HEADER.format(source=source.name) + "\n" + body + "\n"
    return _formatted(text, target_name)


def _formatted(text: str, name: str) -> str:
    """Run the tree's own formatter over the output, so a diff is never about layout."""
    for command in (
        ["ruff", "check", "--fix-only", "--quiet", "--stdin-filename", name, "-"],
        ["ruff", "format", "--quiet", "--stdin-filename", name, "-"],
    ):
        result = subprocess.run(
            command, input=text, capture_output=True, text=True, cwd=ROOT, check=False
        )
        if result.returncode != 0:
            raise SystemExit(f"{command[0]} failed on the generated {name}:\n{result.stderr}")
        text = result.stdout
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change instead of writing it, and fail if anything would",
    )
    args = parser.parse_args()

    stale: list[str] = []
    for source_name, target_name in MODULES.items():
        source = PACKAGE / source_name
        target = PACKAGE / target_name
        if not source.exists():
            raise SystemExit(f"{source} is listed for conversion and does not exist")
        wanted = convert(source, target_name)
        found = target.read_text() if target.exists() else ""
        if wanted == found:
            continue
        if not args.check:
            target.write_text(wanted)
            print(f"wrote {target.relative_to(ROOT)}")
            continue
        stale.append(target_name)
        print(f"{target.relative_to(ROOT)} is not what {source_name} says it should be:")
        sys.stdout.writelines(
            difflib.unified_diff(
                found.splitlines(keepends=True),
                wanted.splitlines(keepends=True),
                fromfile=f"{target_name} (on disk)",
                tofile=f"{target_name} (from {source_name})",
            )
        )
    if stale:
        print(
            f"\nrun {Path(__file__).relative_to(ROOT)} to bring {', '.join(stale)} up to date"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
