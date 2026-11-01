"""One thread that frees a caller whose budget ran out on a network that stopped answering.

A server's ``statement_timeout`` is the right way to stop a statement that is merely slow: the
server gives up, says so, and the connection is fine. It cannot help when the network stops -- a
proxy that stops forwarding, a load balancer that stops passing packets but still acknowledges
them -- because the server's answer never arrives, and keepalive probes are answered by whatever
is in the way. Measured through a proxy that stopped forwarding, on 2.16: a statement under a
one-second ``statement_timeout`` was still waiting after 15 seconds, and an awaiting call under
``wait_for(..., 2)`` came back after 12, psycopg having waited five seconds for its cancel and
five more for a reply.

So a block with a budget is watched from the client too. If a statement is running and the server
has sent nothing for ``GRACE`` longer than its own limit allows, the connection's socket is shut
down -- not closed: closing a descriptor another thread is reading from is undefined, while
shutting it down makes that read return at once, in a blocking connection and an awaiting one
alike. The caller is told the server stopped answering, and the connection, broken by then, is
discarded by a pool. A statement that is merely slow is ended by the server first, so silence that
long is the network; nothing else sets this off -- not a block that runs past its budget, nor a
caller doing its own work between statements, nor a long batch whose answers keep arriving.

One thread for the process, sleeping until the earliest moment it holds. Arming is a push onto a
heap under a lock, and disarming marks the watch rather than searching for it. A watch may carry a
``look``, asked at its moment whether to shut the connection down or when to ask again.
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg import BaseConnection

__all__ = ["GRACE", "WATCHDOG", "Look", "Watch", "Watchdog"]

GRACE = 0.5
"""How much longer than the server's own limit it may go unheard before its socket is shut down.

Room for the server's own cancellation to arrive; a stalled network is the only thing that should
ever use this up.
"""


Look = Callable[[Any, float], "float | None"]
"""Asked with the connection and the time: ``None`` shuts it down, a moment is when to ask again.

It runs on the watchdog's thread under its lock, so it should read and do nothing else. It is
handed the connection rather than holding it, so a disarmed watch keeps nothing alive.
"""


class Watch:
    """One armed watch: a connection, and what decides when it is given up on."""

    __slots__ = ("cancelled", "conn", "fired", "look")

    def __init__(self, conn: BaseConnection[Any], look: Look) -> None:
        self.conn: BaseConnection[Any] | None = conn
        self.look = look
        self.cancelled = False
        self.fired = False


class Watchdog:
    """The heap of watches and the thread that keeps it."""

    def __init__(self) -> None:
        self._reset()
        os.register_at_fork(after_in_child=self._reset)

    def _reset(self) -> None:
        # A child of a fork inherits the heap but not the thread; it starts again empty, and
        # never touches a socket it shares with its parent.
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._heap: list[tuple[float, int, Watch]] = []
        self._order = itertools.count()
        self._cancelled = 0
        self._thread: threading.Thread | None = None

    def arm(self, at: float, conn: BaseConnection[Any], look: Look) -> Watch:
        """Watch a connection, asking *look* about it at *at* on the monotonic clock."""
        watch = Watch(conn, look)
        with self._lock:
            sooner = not self._heap or at < self._heap[0][0]
            heapq.heappush(self._heap, (at, next(self._order), watch))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="agensgraph-watchdog", daemon=True
                )
                self._thread.start()
            elif sooner:
                self._wake.notify()
        return watch

    def disarm(self, watch: Watch) -> bool:
        """Stop watching; whether the watch had already fired.

        Under the same lock the firing happens under, so a block that ends in time is never
        shut down after the fact.
        """
        with self._lock:
            if not watch.fired and not watch.cancelled:
                watch.cancelled = True
                watch.conn = None
                self._cancelled += 1
                if self._cancelled > 1024 and self._cancelled * 2 > len(self._heap):
                    self._heap = [entry for entry in self._heap if not entry[2].cancelled]
                    heapq.heapify(self._heap)
                    self._cancelled = 0
            return watch.fired

    def pending(self) -> int:
        """How many watches are armed and neither fired nor disarmed."""
        with self._lock:
            return sum(
                1 for _, _, watch in self._heap if not watch.cancelled and not watch.fired
            )

    def _run(self) -> None:
        with self._lock:
            while True:
                heap = self._heap
                while heap and heap[0][2].cancelled:
                    heapq.heappop(heap)
                    self._cancelled -= 1
                if not heap:
                    self._wake.wait()
                    continue
                now = time.monotonic()
                wait = heap[0][0] - now
                if wait > 0:
                    self._wake.wait(wait)
                    continue
                _, _, watch = heapq.heappop(heap)
                later = watch.look(watch.conn, now)
                if later is not None:
                    heapq.heappush(heap, (later, next(self._order), watch))
                    continue
                watch.fired = True
                conn, watch.conn = watch.conn, None
                if conn is not None:
                    _shut_down(conn)


def _shut_down(conn: BaseConnection[Any]) -> None:
    """End a connection's reads and writes without closing the descriptor under its reader.

    A connection closed meanwhile is left alone: its descriptor may already belong to another
    socket, and shutting that one down would break a stranger.
    """
    if conn.closed:
        return
    try:
        descriptor = os.dup(conn.pgconn.socket)
    except Exception:
        return
    try:
        held = socket.socket(fileno=descriptor)
    except OSError:
        os.close(descriptor)
        return
    with held, contextlib.suppress(OSError):
        held.shutdown(socket.SHUT_RDWR)


WATCHDOG = Watchdog()
"""The process's one watchdog."""
