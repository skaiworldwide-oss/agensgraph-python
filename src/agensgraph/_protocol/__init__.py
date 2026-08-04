"""Reading and writing graph values, with no I/O.

Everything here takes bytes and returns values, or takes values and returns bytes. No
module in this package opens a socket, imports an I/O library or knows what a
connection is. That keeps the part most likely to hold a subtle bug testable at memory
speed, fuzzable without a server, and reproducible from a seed.
"""

from __future__ import annotations
