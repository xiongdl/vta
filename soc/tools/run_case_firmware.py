#!/usr/bin/env python3
"""Run a case firmware simulator and turn its UART verdict into an exit code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} simulator sim_config.js", file=sys.stderr)
        return 2
    simulator = Path(sys.argv[1]).resolve()
    config = Path(sys.argv[2]).resolve()
    process = subprocess.Popen(
        [str(simulator), config.name], cwd=config.parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    verdict = None
    assert process.stdout is not None
    try:
        for line in iter(process.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            if b"VTA case PASS" in line:
                verdict = 0
                break
            if b"VTA case FAIL" in line:
                verdict = 1
                break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if verdict is None:
        print(f"simulator exited without a VTA case verdict (status {process.returncode})",
              file=sys.stderr)
        return process.returncode or 1
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
