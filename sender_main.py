#!/usr/bin/env python3
"""Prototype sender entry for the offline ProGVC loopback.

The current phase-4 sender still runs in one process with the receiver so it can
be regression-tested without an external signaling server. The packet payloads
and ABR decisions are the same pieces that will move over a live DataChannel.
"""

from scripts.run_offline_loopback import main


if __name__ == "__main__":
    raise SystemExit(main())
