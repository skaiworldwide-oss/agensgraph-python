"""Keeping the generated interface honest.

Three separate things can go wrong and each needs its own check. The generated file can be
edited directly, so it is compared against what its source says it should be. A new awaiting
module can be added without being listed for conversion, so the list is compared against what
is on disk. And the generated file can be left behind by an edit to its source, which is the
same check as the first one -- it is the one that catches the ordinary mistake.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "async_to_sync.py"
PACKAGE = ROOT / "src" / "agensgraph"


def modules() -> dict[str, str]:
    """The conversion list, read from the tool itself rather than repeated here."""
    namespace: dict[str, object] = {}
    source = TOOL.read_text()
    start = source.index("MODULES: dict[str, str] = {")
    end = source.index("}", start) + 1
    exec(compile(source[start:end], str(TOOL), "exec"), namespace)
    return namespace["MODULES"]  # type: ignore[return-value]


def test_every_generated_file_is_what_its_source_says() -> None:
    """The check that catches an edit to the wrong file, and a stale generated one."""
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_awaiting_module_escapes_conversion() -> None:
    """A module added without being listed would silently have no blocking counterpart."""
    on_disk = {path.name for path in PACKAGE.glob("*_async.py")}
    listed = set(modules())
    assert on_disk == listed, f"not listed for conversion: {sorted(on_disk - listed)}"


@pytest.mark.parametrize("generated", sorted(modules().values()))
def test_a_generated_file_says_so_at_the_top(generated: str) -> None:
    """So that nobody edits one without being told where it came from."""
    head = (PACKAGE / generated).read_text()[:400]
    assert "generated" in head
    assert "tools/async_to_sync.py" in head


@pytest.mark.parametrize("generated", sorted(modules().values()))
def test_nothing_awaiting_survives_in_a_generated_file(generated: str) -> None:
    """The blocking interface has to be blocking throughout.

    A construct the tool does not know about would otherwise be carried across unchanged and
    left as a coroutine nobody awaits, which raises nothing and does nothing.
    """
    text = (PACKAGE / generated).read_text()
    lines = [
        line
        for line in text.splitlines()
        if "generated from" not in line and "Edit that file" not in line
    ]
    body = "\n".join(lines)
    for construct in ("await ", "async def ", "async with ", "async for ", "Async"):
        assert construct not in body, f"{generated} still contains {construct!r}"


def convert(text: str) -> str:
    """One snippet through the tool, without touching anything on disk."""
    import ast

    sys.path.insert(0, str(TOOL.parent))
    try:
        import async_to_sync
    finally:
        sys.path.pop(0)
    tree = ast.parse(text)
    return ast.unparse(ast.fix_missing_locations(async_to_sync.Blocking().visit(tree)))


class TestTheConversionItself:
    """What the tool does to each construct, checked without touching the tree."""

    def test_awaiting_is_removed(self) -> None:
        assert convert("async def f():\n    await g()") == "def f():\n    g()"

    def test_an_awaiting_block_becomes_a_plain_one(self) -> None:
        assert convert("async def f():\n    async with a() as b:\n        pass") == (
            "def f():\n    with a() as b:\n        pass"
        )

    def test_an_awaiting_loop_becomes_a_plain_one(self) -> None:
        assert convert("async def f():\n    async for x in y:\n        pass") == (
            "def f():\n    for x in y:\n        pass"
        )

    def test_a_name_loses_its_prefix(self) -> None:
        assert "class Connection:" in convert("class AsyncConnection:\n    pass")

    def test_a_function_loses_its_prefix(self) -> None:
        assert "def connect(" in convert("async def async_connect():\n    pass")

    def test_an_annotation_is_renamed_too(self) -> None:
        """The place a rename is easiest to forget, and hardest to notice missing."""
        got = convert("async def f(r: AsyncRowFactory) -> AsyncConnection:\n    pass")
        assert "AsyncRowFactory" not in got
        assert "r: RowFactory" in got
        assert "-> Connection" in got

    def test_a_name_that_merely_starts_with_async_is_left_alone(self) -> None:
        assert "Asynchrony" in convert("class Asynchrony:\n    pass")

    def test_a_definition_can_be_left_out_altogether(self) -> None:
        got = convert(
            "class A:\n    @only_async\n    async def f(self):\n        pass\n    def g(self):\n        pass"
        )
        assert "def f" not in got
        assert "def g" in got

    def test_one_branch_can_be_kept(self) -> None:
        got = convert("if IS_ASYNC:\n    a = 1\nelse:\n    a = 2")
        assert "a = 2" in got
        assert "a = 1" not in got

    def test_a_kept_branch_that_is_empty_still_parses(self) -> None:
        assert convert("if IS_ASYNC:\n    a = 1") == "pass"

    def test_prose_is_renamed(self) -> None:
        got = convert('async def f():\n    """Returns an AsyncConnection."""')
        assert "Returns a Connection" in got or "Connection." in got
        assert "AsyncConnection" not in got
