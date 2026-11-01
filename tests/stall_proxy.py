"""A TCP proxy that can stop forwarding without closing anything.

What a stalled network looks like from both ends: the connection is up, nothing reports an error,
and nothing arrives. A proxy that stops reading, a load balancer that stops passing packets -- the
server never hears the statement, so its own timeout never starts, and keepalive probes are
answered by the thing in the way.
"""

from __future__ import annotations

import contextlib
import select
import socket
import threading
import time

from psycopg.conninfo import conninfo_to_dict, make_conninfo


class StallProxy:
    """Forward 127.0.0.1:<port> to a server until :meth:`stall` is called."""

    def __init__(self, dsn: str) -> None:
        params = conninfo_to_dict(dsn)
        self._upstream = (
            str(params.get("host") or "127.0.0.1"),
            int(params.get("port") or 5432),
        )
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self.dsn = make_conninfo(dsn, host="127.0.0.1", port=str(self.port))
        self._stalled = threading.Event()
        self._closed = threading.Event()
        self._sockets: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def stall(self) -> None:
        """Stop forwarding in both directions; every socket stays open.

        Returns once every pump has seen it: a pump waits a twentieth of a second at a time, so
        without the pause a statement sent straight after could still slip through.
        """
        self._stalled.set()
        time.sleep(0.12)

    def resume(self) -> None:
        self._stalled.clear()

    def close(self) -> None:
        self._closed.set()
        for held in [self._listener, *self._sockets]:
            with contextlib.suppress(OSError):
                held.close()

    def __enter__(self) -> StallProxy:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            server = socket.create_connection(self._upstream)
            self._sockets += [client, server]
            for source, target in ((client, server), (server, client)):
                threading.Thread(target=self._pump, args=(source, target), daemon=True).start()

    def _pump(self, source: socket.socket, target: socket.socket) -> None:
        while not self._closed.is_set():
            if self._stalled.is_set():
                self._closed.wait(0.01)
                continue
            try:
                readable, _, _ = select.select([source], [], [], 0.05)
                if not readable or self._stalled.is_set():
                    continue
                data = source.recv(65536)
            except (OSError, ValueError):
                return
            if not data:
                with contextlib.suppress(OSError):
                    target.shutdown(socket.SHUT_WR)
                return
            try:
                target.sendall(data)
            except OSError:
                return
