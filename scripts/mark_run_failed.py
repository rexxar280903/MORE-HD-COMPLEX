#!/usr/bin/env python3
"""Mark a crashed run (status still RUNNING) as FAILED so that a new attempt may start (G3-04)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.run_status import mark_failed  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run_dir")
p.add_argument("--reason", required=True)
a = p.parse_args()
mark_failed(a.run_dir, a.reason)
print(f"{a.run_dir}: marked FAILED ({a.reason})")
