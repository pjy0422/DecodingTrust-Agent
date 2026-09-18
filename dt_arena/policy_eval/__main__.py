"""Command dispatcher for DTAP policy evaluation."""

from __future__ import annotations

import argparse
import sys


COMMANDS = ("run", "matrix", "holdout")


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dt_arena.policy_eval")
    parser.add_argument("command", choices=COMMANDS)
    return parser


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _help_parser().print_help()
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        _help_parser().error(f"invalid command: {command!r}")

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "run":
        from .scripts.run_policy_e2e import main as target
    elif command == "matrix":
        from .scripts.run_domain_matrix import main as target
    else:
        from .scripts.run_holdout_matrix import main as target
    target()


if __name__ == "__main__":
    main()
