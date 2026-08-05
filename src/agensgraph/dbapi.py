"""The interface a tool expects of a database module.

Anything that talks to databases generically -- an ORM, a migration tool, a dashboard -- looks
for the names in PEP 249 rather than for this driver's own. Providing them is what makes a
SQLAlchemy dialect a few dozen lines instead of a few hundred, and it is the whole migration path
for anyone on the psycopg2 releases.

Almost all of it is psycopg's already, so almost all of it is re-exported rather than written.
What differs is one thing: :func:`connect` returns a graph connection, so a tool driving this
module reads a vertex as a vertex.

``adapters`` here is **this driver's** template, not psycopg's global map, and that matters: a tool
that derives its own map from it and hands the result back as a connection's context would otherwise
hand over a map with no graph types in it, and every vertex would arrive as the text it prints as.
SQLAlchemy's psycopg dialect does exactly that.

``__version__`` here is **psycopg's**, not this driver's, and deliberately: a tool reading it off a
database module is asking which of that module's features it may use, and every feature reachable
through this one is psycopg's. This driver's own version is ``agensgraph.__version__``.

Two attributes are worth naming for what they replace. ``closed`` and ``broken`` on a connection
are the entirety of psycopg's own judgement about whether a connection has been lost -- so a tool
that asks those two never has to match on the text of an error message, which is what tools did
before they existed and what breaks whenever a message is reworded or translated.
"""

from __future__ import annotations

from psycopg import (
    BINARY as BINARY,
)
from psycopg import (
    DATETIME as DATETIME,
)
from psycopg import (
    NUMBER as NUMBER,
)
from psycopg import (
    ROWID as ROWID,
)
from psycopg import (
    STRING as STRING,
)
from psycopg import (
    Binary as Binary,
)
from psycopg import (
    Date as Date,
)
from psycopg import (
    DateFromTicks as DateFromTicks,
)
from psycopg import (
    Time as Time,
)
from psycopg import (
    TimeFromTicks as TimeFromTicks,
)
from psycopg import (
    Timestamp as Timestamp,
)
from psycopg import (
    TimestampFromTicks as TimestampFromTicks,
)
from psycopg import __version__ as __version__
from psycopg import (
    apilevel as apilevel,
)
from psycopg import (
    paramstyle as paramstyle,
)
from psycopg import (
    threadsafety as threadsafety,
)

from ._core import GRAPH_ADAPTERS as adapters
from .connection import Connection
from .errors import (
    DatabaseError as DatabaseError,
)
from .errors import (
    DataError as DataError,
)
from .errors import (
    Error as Error,
)
from .errors import (
    IntegrityError as IntegrityError,
)
from .errors import (
    InterfaceError as InterfaceError,
)
from .errors import (
    InternalError as InternalError,
)
from .errors import (
    NotSupportedError as NotSupportedError,
)
from .errors import (
    OperationalError as OperationalError,
)
from .errors import (
    ProgrammingError as ProgrammingError,
)
from .errors import (
    Warning as Warning,
)

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "Binary",
    "DataError",
    "DatabaseError",
    "Date",
    "DateFromTicks",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
    "Warning",
    "__version__",
    "adapters",
    "apilevel",
    "connect",
    "paramstyle",
    "threadsafety",
]


def connect(dsn: str = "", **kwargs: object) -> Connection[object]:
    """Open a connection, under the name a generic tool looks for.

    The one thing here that is not psycopg's: what comes back reads the graph types, so a tool
    driving this module gets a vertex rather than the text a vertex prints as.
    """
    return Connection.connect(dsn, **kwargs)
