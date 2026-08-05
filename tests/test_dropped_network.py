"""What happens when the network stops carrying packets without saying so.

This is the failure users report as "the driver hung", and it is reachable only by *dropping*
traffic. Rejecting it gives a prompt reset, which is a shallow path that a suite testing only that
would mistake for coverage.

So the assertions here are on wall clock rather than on an exception having arrived: an exception that
arrives after fifteen minutes is the bug, not the absence of one.

Needs to insert a firewall rule, so it skips where it cannot. It only ever names the port the tests
were pointed at, and removes what it added even when a test fails -- a rule left behind would drop
every later connection to that server.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time

import pytest

import agensgraph

pytestmark = pytest.mark.server

COMMENT = "agensgraph-test-drop"
"""Stamped on every rule, so they can be found and removed without guessing."""

CAP = 20.0
"""How long a wait is watched before it is called unbounded."""


def can_drop() -> bool:
    return shutil.which("sudo") is not None and (
        subprocess.run(
            ["sudo", "-n", "/usr/sbin/iptables", "-S"], capture_output=True
        ).returncode
        == 0
    )


def rules(action: str, port: str) -> None:
    """Add or remove the pair that silences a port in both directions."""
    for chain, flag in (("INPUT", "--dport"), ("OUTPUT", "--sport")):
        subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/sbin/iptables",
                action,
                chain,
                "-p",
                "tcp",
                flag,
                port,
                "-j",
                "DROP",
                "-m",
                "comment",
                "--comment",
                COMMENT,
            ],
            check=False,
            capture_output=True,
        )


def remove_every_rule() -> None:
    """Remove until none is left, so a repeated insert cannot strand one."""
    for _ in range(8):
        listed = subprocess.run(
            ["sudo", "-n", "/usr/sbin/iptables", "-S"], capture_output=True, text=True
        ).stdout
        if COMMENT not in listed:
            return
        for line in listed.splitlines():
            if COMMENT in line and line.startswith("-A "):
                subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "/usr/sbin/iptables",
                        *line.replace("-A ", "-D ", 1).split(),
                    ],
                    check=False,
                    capture_output=True,
                )
    raise AssertionError("could not remove the drop rules, so the port is still silenced")


@pytest.fixture
def dropping(dsn: str):  # type: ignore[no-untyped-def]
    """Hands back something that silences the server's port, and unsilences it afterwards."""
    if not can_drop():
        pytest.skip("needs 'sudo -n iptables' to drop packets")
    from psycopg.conninfo import conninfo_to_dict

    port = str(conninfo_to_dict(dsn).get("port") or 5432)
    if port in ("5432", ""):
        pytest.skip("refusing to touch the default port, which is not this suite's server")
    remove_every_rule()
    try:
        yield lambda: rules("-I", port)
    finally:
        remove_every_rule()


class Waiting:
    """A connection with a thread blocked inside it, and what ended that wait.

    Closing it while the thread is still inside ``execute`` is safe and is what :meth:`release` does:
    the thread is handed an ``OperationalError`` saying the socket closed. Leaving it unclosed is what
    is not safe -- psycopg reports an unclosed connection when it is collected, and under ``-W error``
    that ends whichever unrelated test happens to be running at the time.
    """

    def __init__(self, conn) -> None:  # type: ignore[no-untyped-def]
        self.conn = conn
        self.outcome: list[tuple[str, float]] = []
        self.worker = threading.Thread(target=self._ask, daemon=True)

    def _ask(self) -> None:
        started = time.monotonic()
        try:
            self.conn.execute("select pg_sleep(40)")
            self.outcome.append(("returned", time.monotonic() - started))
        except BaseException as exc:
            self.outcome.append((type(exc).__name__, time.monotonic() - started))

    def run(self, silence) -> tuple[str, float]:  # type: ignore[no-untyped-def]
        self.worker.start()
        time.sleep(0.4)
        silence()
        self.worker.join(timeout=CAP)
        return self.outcome[0] if self.outcome else ("still waiting", CAP)

    def release(self) -> None:
        """Close it, and let the thread notice."""
        with contextlib.suppress(Exception):
            self.conn.close()
        self.worker.join(timeout=CAP)


class TestADroppedNetwork:
    def test_it_is_bounded_rather_than_left_to_the_kernel(self, dsn, dropping) -> None:  # type: ignore[no-untyped-def]
        """With what this driver asks for by default, the wait ends in about a minute rather than
        the two and a quarter hours the kernel would take. Shortened here so it can be measured."""
        waiting = Waiting(
            agensgraph.Connection.connect(
                dsn,
                autocommit=True,
                keepalives_idle=1,
                keepalives_interval=1,
                keepalives_count=3,
            )
        )
        what, seconds = waiting.run(dropping)
        assert what != "still waiting", f"the wait was not bounded within {CAP:.0f}s"
        assert what != "returned"
        assert seconds < 15.0, (
            f"it took {seconds:.1f}s, which is not a bound anyone would notice"
        )
        # The two a tool asks instead of matching on the text of an error.
        assert waiting.conn.broken
        waiting.release()
        assert waiting.conn.closed

    def test_the_setting_usually_recommended_does_not_bound_it_alone(
        self, dsn, dropping
    ) -> None:  # type: ignore[no-untyped-def]
        """``tcp_user_timeout`` bounds how long *transmitted* data may go unacknowledged, and a
        connection waiting for a reply has transmitted nothing. So it alone leaves this unbounded,
        which is the reason keepalive is asked for instead of it."""
        waiting = Waiting(
            agensgraph.Connection.connect(
                dsn, autocommit=True, keepalives=0, tcp_user_timeout=2000
            )
        )
        what, _ = waiting.run(dropping)
        waiting.release()
        assert what == "still waiting", (
            f"tcp_user_timeout alone ended the wait with {what}, so this driver's reason for "
            f"asking for keepalive instead needs revisiting"
        )

    def test_with_nothing_asked_for_it_is_unbounded(self, dsn, dropping) -> None:  # type: ignore[no-untyped-def]
        """What a caller gets by turning off what this driver asks for."""
        waiting = Waiting(agensgraph.Connection.connect(dsn, autocommit=True, keepalives=0))
        what, _ = waiting.run(dropping)
        waiting.release()
        assert what == "still waiting"

    def test_a_rejected_connection_is_the_shallow_path(self, dsn, dropping) -> None:  # type: ignore[no-untyped-def]
        """Kept to show the contrast the module docstring claims: a refusal is prompt, and a suite
        asserting only this would cover none of the above."""
        from psycopg.conninfo import conninfo_to_dict

        port = str(conninfo_to_dict(dsn).get("port"))
        subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/sbin/iptables",
                "-I",
                "OUTPUT",
                "-p",
                "tcp",
                "--dport",
                port,
                "-j",
                "REJECT",
                "--reject-with",
                "tcp-reset",
                "-m",
                "comment",
                "--comment",
                COMMENT,
            ],
            check=False,
            capture_output=True,
        )
        started = time.monotonic()
        with pytest.raises(agensgraph.errors.OperationalError):
            agensgraph.Connection.connect(dsn, connect_timeout=10)
        assert time.monotonic() - started < 5.0


class TestNothingIsLeftBehind:
    def test_the_server_is_reachable_again_afterwards(self, dsn, dropping) -> None:  # type: ignore[no-untyped-def]
        """The assertion that protects every later test in the suite."""
        dropping()
        remove_every_rule()
        with agensgraph.Connection.connect(dsn, autocommit=True) as conn:
            assert conn.execute_query("select 1").records == [(1,)]

    def test_no_rule_of_ours_survives(self) -> None:
        if not can_drop():
            pytest.skip("needs 'sudo -n iptables' to inspect the rules")
        listed = subprocess.run(
            ["sudo", "-n", "/usr/sbin/iptables", "-S"], capture_output=True, text=True
        ).stdout
        assert COMMENT not in listed


class TestWhatIsAskedForOnEveryConnection:
    """The settings are filled in one at a time, because they do nothing useful apart."""

    @staticmethod
    def asked(dsn: str, **kwargs: object) -> dict[str, str]:
        from psycopg.conninfo import conninfo_to_dict

        with agensgraph.Connection.connect(dsn, **kwargs) as conn:  # type: ignore[arg-type]
            given = conninfo_to_dict(conn.info.dsn)
            return {
                key: str(value)
                for key, value in given.items()
                if "keepal" in key or "tcp_user" in key
            }

    def test_nothing_given_gets_all_of_them(self, dsn: str) -> None:
        assert self.asked(dsn) == {
            "keepalives": "1",
            "keepalives_idle": "30",
            "keepalives_interval": "10",
            "keepalives_count": "3",
        }

    def test_naming_one_does_not_leave_the_others_at_the_system_values(self, dsn: str) -> None:
        """An interval with the system's idle time of two hours is never reached, so a caller who
        names only the interval would have asked for nothing."""
        asked = self.asked(dsn, keepalives_interval=5)
        assert asked["keepalives_interval"] == "5"
        assert asked["keepalives_idle"] == "30"

    def test_naming_the_idle_time_keeps_it(self, dsn: str) -> None:
        assert self.asked(dsn, keepalives_idle=60)["keepalives_idle"] == "60"

    def test_the_setting_that_does_not_bound_a_read_does_not_count_as_deciding(
        self, dsn: str
    ) -> None:
        """A caller who sets only ``tcp_user_timeout`` has asked for something that leaves a hung
        read unbounded, so keepalive is still filled in for them."""
        asked = self.asked(dsn, tcp_user_timeout=5000)
        assert asked["tcp_user_timeout"] == "5000"
        assert asked["keepalives_idle"] == "30"

    @pytest.mark.parametrize("off", [0, "0"])
    def test_turning_keepalive_off_turns_all_of_it_off(self, dsn: str, off: object) -> None:
        assert self.asked(dsn, keepalives=off) == {"keepalives": "0"}

    def test_the_pool_hands_out_connections_with_them_too(self, dsn: str) -> None:
        from psycopg.conninfo import conninfo_to_dict

        pool = agensgraph.ConnectionPool(dsn, min_size=1, max_size=1)
        pool.open(wait=True)
        try:
            with pool.connection() as conn:
                given = conninfo_to_dict(conn.info.dsn)
                assert str(given.get("keepalives_idle")) == "30"
        finally:
            pool.close()
