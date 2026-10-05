#!/usr/bin/env python3
"""Jalur A: one primary run (MORE-HD or MORE-HD-C) for one condition (pseudocode §9.1)."""

from core.environment import lock_threads

lock_threads()  # G4-01: one BLAS/OpenMP thread per process, set before numpy is imported

import argparse  # noqa: E402

from core import constants as C  # noqa: E402
from core.cli import add_common_run_args, make_config, run_and_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", required=True, choices=C.PRIMARY_ARCHITECTURES)
    add_common_run_args(p)
    args = p.parse_args()
    run_and_report(make_config(args, args.architecture), write_workbook=not args.no_workbook)


if __name__ == "__main__":
    main()
