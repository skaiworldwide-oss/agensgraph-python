"""Subscribing to what the server announces.

A graph trigger can announce a change, which makes this the change feed for a graph.

The channel is an identifier in ``LISTEN`` and ``UNLISTEN``, so it is quoted into the statement --
neither takes a parameter, and ``LISTEN`` cannot be prepared at all. Announcing is different:
``pg_notify`` is a function, so there the channel is a parameter and nothing is quoted.
"""

from __future__ import annotations

from .cypher import quote_identifier

__all__ = [
    "LISTENING_QUERY",
    "NOTIFY_QUERY",
    "listen_statement",
    "unlisten_statement",
]

NOTIFY_QUERY = "select pg_notify(%s, %s)"
"""Announce something. The channel is a parameter here, unlike in ``LISTEN``."""

LISTENING_QUERY = "select pg_listening_channels()"
"""Which channels this connection is subscribed to."""


def listen_statement(channel: str) -> str:
    """Subscribe to a channel."""
    if not channel:
        raise ValueError("a channel has a name")
    return f"listen {quote_identifier(channel)}"


def unlisten_statement(channel: str | None = None) -> str:
    """Stop listening to a channel, or to every channel when given none."""
    if channel is None:
        return "unlisten *"
    if not channel:
        raise ValueError("a channel has a name")
    return f"unlisten {quote_identifier(channel)}"
