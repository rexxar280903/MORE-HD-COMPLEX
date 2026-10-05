#!/usr/bin/env python3
"""Targeted ablation G1-05/G1-07: model B (MORE-HD-60P) or C (MORE-HD-C-FixedRZ) at K in {3,6,10}."""

from core.environment import lock_threads

lock_threads()  # G4-01

import argparse  # noqa: E402

from core import constants as C  # noqa: E402
from core.cli import add_common_run_args, make_config, run_and_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=C.ABLATION_MODELS)
    add_common_run_args(p)
    args = p.parse_args()
    run_and_report(make_config(args, args.model), write_workbook=not args.no_workbook)


if __name__ == "__main__":
    main()
