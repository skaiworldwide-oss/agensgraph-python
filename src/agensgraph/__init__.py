"""AgensGraph driver for Python."""

from __future__ import annotations

from ._protocol.graphid import LABID_MAX as LABID_MAX
from ._protocol.graphid import LOCID_MAX as LOCID_MAX
from .types import Edge as Edge
from .types import GraphId as GraphId
from .types import Label as Label
from .types import Path as Path
from .types import Vertex as Vertex

__all__ = [
    "LABID_MAX",
    "LOCID_MAX",
    "Edge",
    "GraphId",
    "Label",
    "Path",
    "Vertex",
]


def _installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("agensgraph-python")
    except PackageNotFoundError:
        return "0.0.0.dev0"


__version__: str = _installed_version()
