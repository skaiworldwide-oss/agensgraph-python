"""AgensGraph driver for Python."""

from __future__ import annotations

from . import errors as errors
from ._protocol.graphid import LABID_MAX as LABID_MAX
from ._protocol.graphid import LOCID_MAX as LOCID_MAX
from .capabilities import MINIMUM_VERSION as MINIMUM_VERSION
from .capabilities import Capabilities as Capabilities
from .errors import Retryability as Retryability
from .errors import is_retryable as is_retryable
from .errors import retryability as retryability
from .types import Edge as Edge
from .types import GraphId as GraphId
from .types import Label as Label
from .types import Path as Path
from .types import Vertex as Vertex

__all__ = [
    "LABID_MAX",
    "LOCID_MAX",
    "MINIMUM_VERSION",
    "Capabilities",
    "Edge",
    "GraphId",
    "Label",
    "Path",
    "Retryability",
    "Vertex",
    "errors",
    "is_retryable",
    "retryability",
]


def _installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("agensgraph-python")
    except PackageNotFoundError:
        return "0.0.0.dev0"


__version__: str = _installed_version()
