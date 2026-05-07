#!/usr/bin/env python3
"""Prototype receiver entry for the offline ProGVC loopback.

For now this shares the same one-process loopback runner as `sender_main.py`.
Keeping both entry points in place makes the next split into live sender and
receiver processes mechanical once signaling is available.
"""

from scripts.run_offline_loopback import main


if __name__ == "__main__":
    raise SystemExit(main())
