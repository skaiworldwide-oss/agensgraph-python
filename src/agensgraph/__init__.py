"""AgensGraph driver for Python."""

from __future__ import annotations

from psycopg.types.json import Jsonb as Jsonb

from . import errors as errors
from ._core import Result as Result
from ._protocol.graphid import LABID_MAX as LABID_MAX
from ._protocol.graphid import LOCID_MAX as LOCID_MAX
from .adapters import Unspecified as Unspecified
from .capabilities import MINIMUM_VERSION as MINIMUM_VERSION
from .capabilities import Capabilities as Capabilities
from .connection import Connection as Connection
from .connection_async import AsyncConnection as AsyncConnection
from .deadline import Deadline as Deadline
from .deadline import Expired as Expired
from .errors import Retryability as Retryability
from .errors import is_retryable as is_retryable
from .errors import retryability as retryability
from .pool import ConnectionPool as ConnectionPool
from .pool_async import AsyncConnectionPool as AsyncConnectionPool
from .retry import RetryPolicy as RetryPolicy
from .retry import TokenBucket as TokenBucket
from .summary import GraphWriteCounts as GraphWriteCounts
from .types import Edge as Edge
from .types import GraphId as GraphId
from .types import Label as Label
from .types import Path as Path
from .types import Vertex as Vertex

__all__ = [
    "LABID_MAX",
    "LOCID_MAX",
    "MINIMUM_VERSION",
    "AsyncConnection",
    "AsyncConnectionPool",
    "Capabilities",
    "Connection",
    "ConnectionPool",
    "Deadline",
    "Edge",
    "Expired",
    "GraphId",
    "GraphWriteCounts",
    "Jsonb",
    "Label",
    "Path",
    "Result",
    "RetryPolicy",
    "Retryability",
    "TokenBucket",
    "Unspecified",
    "Vertex",
    "connect",
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


def connect(conninfo: str = "", **kwargs: object) -> Connection[object]:
    """Open a blocking connection to a graph server.

    The same as :meth:`Connection.connect`, under the name a caller reaches for first.
    """
    return Connection.connect(conninfo, **kwargs)
