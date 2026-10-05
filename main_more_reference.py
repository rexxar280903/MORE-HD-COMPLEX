#!/usr/bin/env python3
"""MORE reproduction track (MORE without loss adjuster R, Wu et al. 2023) under the identical protocol."""

from core.environment import lock_threads

lock_threads()  # G4-01

import argparse  # noqa: E402

from core import constants as C  # noqa: E402
from core.cli import add_common_run_args, make_config, run_and_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(p)
    args = p.parse_args()
    run_and_report(make_config(args, C.ARCH_MORE_REPRO), write_workbook=not args.no_workbook)


if __name__ == "__main__":
    main()
