#!/usr/bin/env python3
"""Apply the pinned LiteX 2024.12 macOS simulation compatibility fixes."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"unexpected LiteX source in {path}; refusing to patch")
    path.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    core = args.venv / "lib/python3.10/site-packages/litex/build/sim/core"

    replace_once(
        core / "sim.c",
        """  if (!evtimer_pending(ev, NULL)) {
    event_del(ev);
    evtimer_add(ev, &tv);
  }
""",
        """  /* main() drives simulation; no timer re-arm is required. */
""",
    )
    replace_once(
        core / "sim.c",
        """  tv.tv_sec = 0;
  tv.tv_usec = 0;
  ev = event_new(base, -1, EV_PERSIST, cb, vsim);
  event_add(ev, &tv);
  event_base_dispatch(base);
""",
        """  /* A zero-time persistent timer does not fire with the macOS/Conda
   * libevent combination used by this workspace.  Drive simulation directly;
   * module tick callbacks still provide clock and serial-console behavior. */
  for (;;) {
    event_base_loop(base, EVLOOP_NONBLOCK);
    cb(-1, 0, vsim);
  }
""",
    )
    replace_once(
        core / "veril.cpp",
        """  int finished;
  tfp->flush();
  if(finished = Verilated::gotFinish()) {
    tfp->close();
  }
""",
        """  int finished;
  if(finished = Verilated::gotFinish()) {
    tfp->flush();
    tfp->close();
  }
""",
    )
    print(f"LiteX simulation compatibility patch ready: {core}")


if __name__ == "__main__":
    main()
